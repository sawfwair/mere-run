import Foundation
import MLX

enum LFM2TextLoRAAdapterError: Error, LocalizedError, Sendable {
    case noMatchingWeights([String])

    var errorDescription: String? {
        switch self {
        case .noMatchingWeights(let keys):
            return "LFM2 text LoRA adapter did not match injected attention layers. "
                + "Weight keys: \(keys.prefix(8).joined(separator: ", "))"
        }
    }
}

struct LFM2TextLoRAApplyReport: Sendable, Equatable {
    let matchedLayerCount: Int
    let injectedLayerCount: Int
}

enum LFM2TextLoRAAdapter {
    static func apply(
        _ lora: LoRA,
        to model: LFM2Model,
        targetSuffixes: [String]? = nil
    ) async throws -> LFM2TextLoRAApplyReport {
        let weights = try await LoRAWeightLoader.load(from: lora)
        let resolvedTargets = targetSuffixes ?? weights.weights.keys.sorted()
        let layers = try LFM2TextLoRAInjector.inject(
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
            guard let layer = layersByAdapterKey[adapterKey(for: path)] else { continue }
            layer.loraDown = pair.down.asType(.float32)
            layer.loraUp = pair.up.asType(.float32)
            layer.role = .assistant
            layer.isActive = true
            matched += 1
        }
        guard matched > 0 else {
            throw LFM2TextLoRAAdapterError.noMatchingWeights(weights.weights.keys.sorted())
        }

        eval(layers.values.flatMap { [$0.loraDown, $0.loraUp] })
        return LFM2TextLoRAApplyReport(
            matchedLayerCount: matched,
            injectedLayerCount: layers.count
        )
    }

    static func adapterKey(for modulePath: String) -> String {
        var key = modulePath
        while key.hasPrefix("model.") {
            key = String(key.dropFirst("model.".count))
        }
        return key.replacingOccurrences(of: ".self_attn.o_proj", with: ".self_attn.out_proj")
    }

    private static func scale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale), .remote(_, let scale):
            return Float(scale)
        }
    }
}
