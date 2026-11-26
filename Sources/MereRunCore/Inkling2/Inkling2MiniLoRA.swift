import Foundation
import MLX
import MLXNN
import MLXRandom
import MereRunInkling2Model

final class Inkling2MiniLoRALayer: Module, TrainableLoRALayer {
    let loraRank: Int
    let loraAlpha: Float
    var loraDown: MLXArray
    var loraUp: MLXArray
    var loraDownM: MLXArray?
    var loraDownV: MLXArray?
    var loraUpM: MLXArray?
    var loraUpV: MLXArray?
    var role: LoRARole = .train
    var isActive = true

    init(down: MLXArray, up: MLXArray, alpha: Float) {
        loraRank = down.dim(0)
        loraAlpha = alpha
        loraDown = down
        loraUp = up
    }

    convenience init(config: Inkling2TextConfiguration, rank: Int, alpha: Float, seed: UInt64) {
        let bound = 1 / sqrt(Float(config.hiddenSize))
        self.init(down: MLXRandom.uniform(low: -bound, high: bound, [rank, config.hiddenSize], key: MLXRandom.key(seed)),
                  up: MLXArray.zeros([config.unpaddedVocabSize, rank]), alpha: alpha)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        matmul(matmul(hidden.asType(.float32), loraDown.T), loraUp.T) * (loraAlpha / Float(loraRank))
    }

    static func load(from url: URL, for config: Inkling2Configuration, scale: Double) throws -> Inkling2MiniLoRALayer {
        guard config.modelType == "inkling2_loop" else {
            throw ChatRequestIssue("--lora", "Inkling 2 output adapters require a Mini checkpoint")
        }
        let (weights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard metadata["format"] == TextLoRATrainingManifest.inkling2MiniFormat,
              let rank = metadata["lora_rank"].flatMap(Int.init), rank > 0,
              let alpha = metadata["lora_alpha"].flatMap(Float.init), alpha.isFinite,
              let down = weights["unembed.lora_down.weight"], down.shape == [rank, config.textConfig.hiddenSize],
              let up = weights["unembed.lora_up.weight"], up.shape == [config.textConfig.unpaddedVocabSize, rank] else {
            throw LoRAError.invalidFormat("Expected an Inkling 2 Mini output-head adapter matching this checkpoint.")
        }
        return Inkling2MiniLoRALayer(down: down.asType(.float32), up: up.asType(.float32), alpha: alpha * Float(scale))
    }
}

final class Inkling2MiniTrainingHead: Module {
    let unembed: Inkling2MiniLoRALayer

    init(_ layer: Inkling2MiniLoRALayer) { unembed = layer }
}
