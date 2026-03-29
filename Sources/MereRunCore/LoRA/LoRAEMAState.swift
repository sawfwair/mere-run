import Foundation
import MLX

/// Exponential moving average (EMA) state for LoRA weights.
///
/// Designed to be shared across trainers and usable with `MLX.asyncEval`.
public final class LoRAEMAState: Evaluatable {
    public var weights: [String: (down: MLXArray, up: MLXArray)] = [:]
    public let decay: Float

    public init(decay: Float) {
        self.decay = decay
    }

    public func initialize(from loraLayers: [String: TrainableLoRALayer]) {
        for (path, layer) in loraLayers {
            weights[path] = (down: layer.loraDown, up: layer.loraUp)
        }
    }

    public func update(from loraLayers: [String: TrainableLoRALayer]) {
        let d = MLXArray(decay)
        let oneMinusD = MLXArray(1.0 - decay)

        for (path, layer) in loraLayers {
            if let ema = weights[path] {
                let newDown = d * ema.down + oneMinusD * layer.loraDown
                let newUp = d * ema.up + oneMinusD * layer.loraUp
                weights[path] = (down: newDown, up: newUp)
            } else {
                weights[path] = (down: layer.loraDown, up: layer.loraUp)
            }
        }
    }

    public func apply(to loraLayers: [String: TrainableLoRALayer]) {
        for (path, layer) in loraLayers {
            if let ema = weights[path] {
                layer.loraDown = ema.down
                layer.loraUp = ema.up
            }
        }
    }

    public func innerState() -> [MLXArray] {
        // Order doesn't matter for eval; we just want to force materialization.
        weights.values.flatMap { [$0.down, $0.up] }
    }
}

