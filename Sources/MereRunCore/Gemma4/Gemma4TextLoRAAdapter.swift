import Foundation
import MLX
import MLXNN

enum Gemma4TextLoRAAdapterError: Error, LocalizedError, Sendable {
    case modelIsNotModule
    case noMatchingWeights([String])

    public var errorDescription: String? {
        switch self {
        case .modelIsNotModule:
            return "Gemma4 text LoRA adapter can only be applied to a Module-backed Gemma4 model."
        case .noMatchingWeights(let keys):
            return "Gemma4 text LoRA adapter did not match injected LoRA layers. Weight keys: \(keys.prefix(8).joined(separator: ", "))"
        }
    }
}

enum Gemma4TextLoRAAdapter {
    static func apply(
        _ lora: LoRA,
        to model: any Gemma4CausalModel,
        targetSuffixes: [String] = Gemma4TextLoRAInjector.defaultTargetSuffixes
    ) async throws -> [String: TrainableLoRALayer] {
        guard let module = model as? Module else {
            throw Gemma4TextLoRAAdapterError.modelIsNotModule
        }
        let weights = try await LoRAWeightLoader.load(from: lora)
        let scale = scale(for: lora)
        let layers = try Gemma4TextLoRAInjector.inject(
            into: module,
            rank: weights.rank,
            alpha: weights.alpha * scale,
            targetSuffixes: targetSuffixes,
            zeroInitUp: false
        )

        var matched = 0
        for (path, pair) in weights.weights {
            guard let layer = layers[path] else { continue }
            layer.loraDown = pair.down.asType(.float32)
            layer.loraUp = pair.up.asType(.float32)
            layer.role = .assistant
            layer.isActive = true
            matched += 1
        }
        guard matched > 0 else {
            throw Gemma4TextLoRAAdapterError.noMatchingWeights(weights.weights.keys.sorted())
        }

        eval(layers.values.flatMap { [$0.loraDown, $0.loraUp] })
        return layers
    }

    private static func scale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale), .remote(_, let scale):
            return Float(scale)
        }
    }
}
