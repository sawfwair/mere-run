import Foundation
import MLX
import MLXNN

// MARK: - Z-Image-i2L Model
//
// Converts concatenated SigLIP2 + DINOv3 embeddings (5632-dim) into a Z-Image Turbo LoRA.
//
// DiffSynth-Studio implementation (ImageEmbeddingToLoraMatrix):
// - proj_a: CompressedMLP(in_dim -> compress_dim -> (in_dim_target * rank))
// - proj_b: CompressedMLP(in_dim -> compress_dim -> (out_dim_target * rank))
// - lora_down = proj_a(x).view(rank, in_dim_target)
// - lora_up   = proj_b(x).view(out_dim_target, rank)

private struct ZImageI2LTarget: Sendable {
    let name: String
    let inDim: Int
    let outDim: Int
}

private enum ZImageI2LTargets {
    static let attention: [ZImageI2LTarget] = [
        .init(name: "attention.to_q", inDim: 3840, outDim: 3840),
        .init(name: "attention.to_k", inDim: 3840, outDim: 3840),
        .init(name: "attention.to_v", inDim: 3840, outDim: 3840),
        .init(name: "attention.to_out.0", inDim: 3840, outDim: 3840),
    ]

    static let feedForward: [ZImageI2LTarget] = [
        .init(name: "feed_forward.w1", inDim: 3840, outDim: 10240),
        .init(name: "feed_forward.w2", inDim: 10240, outDim: 3840),
        .init(name: "feed_forward.w3", inDim: 3840, outDim: 10240),
    ]

    static let groups: [[ZImageI2LTarget]] = [
        attention,
        feedForward,
    ]
}

// MARK: - Compressed MLP (proj_in -> proj_out)

final class ZImageI2LCompressedMLP: Module {
    @ModuleInfo(key: "proj_in") private var projIn: Linear
    @ModuleInfo(key: "proj_out") private var projOut: Linear

    init(inDim: Int, midDim: Int, outDim: Int) {
        self._projIn.wrappedValue = Linear(inDim, midDim, bias: false)
        self._projOut.wrappedValue = Linear(midDim, outDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projOut(projIn(x))
    }
}

// MARK: - Image Embedding -> LoRA matrices (A/B)

final class ZImageI2LImageEmbeddingToLoraMatrix: Module {
    @ModuleInfo(key: "proj_a") private var projA: ZImageI2LCompressedMLP
    @ModuleInfo(key: "proj_b") private var projB: ZImageI2LCompressedMLP

    let rank: Int
    let inDim: Int
    let outDim: Int

    init(embeddingDim: Int, compressDim: Int, inDim: Int, outDim: Int, rank: Int) {
        self.rank = rank
        self.inDim = inDim
        self.outDim = outDim

        self._projA.wrappedValue = ZImageI2LCompressedMLP(
            inDim: embeddingDim,
            midDim: compressDim,
            outDim: inDim * rank
        )
        self._projB.wrappedValue = ZImageI2LCompressedMLP(
            inDim: embeddingDim,
            midDim: compressDim,
            outDim: outDim * rank
        )
    }

    func callAsFunction(_ x: MLXArray) -> (down: MLXArray, up: MLXArray) {
        let downFlat = projA(x)
        let upFlat = projB(x)
        let down = downFlat.reshaped(rank, inDim)
        let up = upFlat.reshaped(outDim, rank)
        return (down, up)
    }
}

// MARK: - Trainer Block (targets for a single transformer layer id)

final class ZImageI2LTrainerBlock: Module {
    @ModuleInfo(key: "layers") private var layers: [ZImageI2LImageEmbeddingToLoraMatrix]

    private let targets: [ZImageI2LTarget]
    private let prefix: String
    private let blockId: Int

    fileprivate init(targets: [ZImageI2LTarget], prefix: String, blockId: Int, config: ZImageI2LConfig) {
        self.targets = targets
        self.prefix = prefix
        self.blockId = blockId
        self._layers.wrappedValue = targets.map { target in
            ZImageI2LImageEmbeddingToLoraMatrix(
                embeddingDim: config.inputDim,
                compressDim: config.compressDim,
                inDim: target.inDim,
                outDim: target.outDim,
                rank: config.loraRank
            )
        }
    }

    func generateLoRA(_ x: MLXArray) -> [String: MLXArray] {
        var dict: [String: MLXArray] = [:]
        dict.reserveCapacity(targets.count * 2)

        for (target, layer) in zip(targets, layers) {
            let (down, up) = layer(x)
            let base = "\(prefix).\(blockId).\(target.name)"
            // Match DiffSynth naming: `.lora_A.default.weight` / `.lora_B.default.weight`
            dict["\(base).lora_A.default.weight"] = down
            dict["\(base).lora_B.default.weight"] = up
        }

        return dict
    }
}

// MARK: - LoRA Component (layers / context_refiner / noise_refiner)

final class ZImageI2LComponent: Module {
    @ModuleInfo(key: "blocks") private var blocks: [ZImageI2LTrainerBlock]

    init(prefix: String, numBlocks: Int, config: ZImageI2LConfig) {
        let groups = ZImageI2LTargets.groups
        self._blocks.wrappedValue = groups.flatMap { group in
            (0..<numBlocks).map { blockId in
                ZImageI2LTrainerBlock(targets: group, prefix: prefix, blockId: blockId, config: config)
            }
        }
    }

    func generateLoRA(_ x: MLXArray) -> [String: MLXArray] {
        // Each transformer layer emits 7 LoRA pairs (attention: 4, ffn: 3) => 14 tensors.
        let perLayerTensorCount = (ZImageI2LTargets.attention.count + ZImageI2LTargets.feedForward.count) * 2
        var dict: [String: MLXArray] = [:]
        dict.reserveCapacity(blocks.count * perLayerTensorCount)

        for block in blocks {
            dict.merge(block.generateLoRA(x), uniquingKeysWith: { _, new in new })
        }

        return dict
    }
}

// MARK: - Z-Image-i2L Model

public final class ZImageI2LModel: Module {
    public let config: ZImageI2LConfig

    @ModuleInfo(key: "layers_lora") private var layersLoRA: ZImageI2LComponent
    @ModuleInfo(key: "context_refiner_lora") private var contextRefinerLoRA: ZImageI2LComponent
    @ModuleInfo(key: "noise_refiner_lora") private var noiseRefinerLoRA: ZImageI2LComponent

    public init(config: ZImageI2LConfig) {
        self.config = config

        self._layersLoRA.wrappedValue = ZImageI2LComponent(prefix: "layers", numBlocks: config.numLayers, config: config)
        self._contextRefinerLoRA.wrappedValue = ZImageI2LComponent(prefix: "context_refiner", numBlocks: config.numRefinerLayers, config: config)
        self._noiseRefinerLoRA.wrappedValue = ZImageI2LComponent(prefix: "noise_refiner", numBlocks: config.numRefinerLayers, config: config)
    }

    /// Generate LoRA weights from image embeddings
    public func callAsFunction(_ embeddings: MLXArray) -> ZImageI2LOutput {
        // embeddings: [batch, inputDim] (preferred) or [inputDim]
        let embedding: MLXArray
        switch embeddings.ndim {
        case 1:
            embedding = embeddings
        case 2:
            embedding = embeddings.dim(0) > 1 ? MLX.mean(embeddings, axis: 0) : embeddings.squeezed(axis: 0)
        default:
            embedding = embeddings.reshaped(-1)
        }

        var dict: [String: MLXArray] = [:]
        dict.reserveCapacity(476)  // 238 LoRA pairs * 2 tensors each
        dict.merge(layersLoRA.generateLoRA(embedding), uniquingKeysWith: { _, new in new })
        dict.merge(contextRefinerLoRA.generateLoRA(embedding), uniquingKeysWith: { _, new in new })
        dict.merge(noiseRefinerLoRA.generateLoRA(embedding), uniquingKeysWith: { _, new in new })

        return ZImageI2LOutput(weights: dict, rank: config.loraRank)
    }
}

// MARK: - I2L Output

public struct ZImageI2LOutput: @unchecked Sendable {
    public let weights: [String: MLXArray]
    public let rank: Int

    /// Convert to dictionary format for saving as safetensors
    public func toDict() -> [String: MLXArray] {
        weights
    }
}
