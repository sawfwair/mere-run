import Foundation
import MLX

enum LagunaTextLoRAAdapterError: Error, LocalizedError, Sendable {
    case noMatchingWeights([String])

    var errorDescription: String? {
        switch self {
        case .noMatchingWeights(let keys):
            return "Laguna text LoRA adapter did not match injected LoRA layers. Weight keys: \(keys.prefix(8).joined(separator: ", "))"
        }
    }
}

struct LagunaTextLoRAApplyReport: Sendable, Equatable {
    let matchedLayerCount: Int
    let injectedLayerCount: Int
}

enum LagunaTextLoRAAdapter {
    static func apply(
        _ lora: LoRA,
        to model: LagunaCausalLM,
        targetSuffixes: [String]? = nil
    ) async throws -> LagunaTextLoRAApplyReport {
        let weights = try await LoRAWeightLoader.load(from: lora)
        let resolvedTargets = targetSuffixes ?? weights.weights.keys.sorted()
        let layers = try LagunaTextLoRAInjector.inject(
            into: model,
            rank: weights.rank,
            alpha: weights.alpha * scale(for: lora),
            targetSuffixes: resolvedTargets,
            zeroInitUp: true
        )
        let layersByAdapterKey = Dictionary(uniqueKeysWithValues: layers.map { path, layer in
            (adapterKey(for: path), layer)
        })
        var matched = 0
        for (path, pair) in weights.weights {
            guard let layer = layersByAdapterKey[path] else { continue }
            layer.loraDown = pair.down.asType(.float32)
            layer.loraUp = pair.up.asType(.float32)
            layer.role = .assistant
            layer.isActive = true
            matched += 1
        }
        guard matched > 0 else {
            throw LagunaTextLoRAAdapterError.noMatchingWeights(
                weights.weights.keys.sorted()
            )
        }
        model.invalidateTextLoRAUnsafeAcceleration()
        eval(layers.values.flatMap { [$0.loraDown, $0.loraUp] })
        return LagunaTextLoRAApplyReport(
            matchedLayerCount: matched,
            injectedLayerCount: layers.count
        )
    }

    private static func adapterKey(for modulePath: String) -> String {
        modulePath.hasPrefix("model.")
            ? String(modulePath.dropFirst("model.".count))
            : modulePath
    }

    private static func scale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale), .remote(_, let scale):
            return Float(scale)
        }
    }
}
