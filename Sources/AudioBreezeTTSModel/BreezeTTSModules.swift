// Adapted from Blaizzy/mlx-audio-swift at 01dec7c9bdce3088a6b6b7ab9f2e403458195efb.
// Copyright (c) 2025 Prince Canuma. Licensed under MIT; see THIRD_PARTY_NOTICES.md.
@preconcurrency import MLX
import MLXNN
import MereRunKVCache
import Foundation

final class BreezeAudioEmbedding: Module {
    let numCodebooks: Int
    let vocabSize: Int

    @ModuleInfo(key: "embed_audio_tokens") var embedAudioTokens: Embedding
    @ModuleInfo(key: "audio_embeds_projector") var audioEmbedsProjector: Linear?

    init(numCodebooks: Int, vocabSize: Int, audioEmbedSize: Int, hiddenSize: Int) {
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        _embedAudioTokens.wrappedValue = Embedding(
            embeddingCount: numCodebooks * vocabSize,
            dimensions: audioEmbedSize
        )
        _audioEmbedsProjector.wrappedValue = audioEmbedSize == hiddenSize
            ? nil
            : Linear(audioEmbedSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ codebooks: MLXArray) -> MLXArray {
        precondition(
            codebooks.ndim == 3 && codebooks.dim(-1) == numCodebooks,
            "Breeze audio codebooks must have shape [batch, time, numCodebooks]"
        )
        let offsets = MLX.arange(numCodebooks).reshaped([1, 1, numCodebooks]) * vocabSize
        var hidden = embedAudioTokens(codebooks + offsets)
        if let audioEmbedsProjector {
            hidden = audioEmbedsProjector(hidden)
        }
        return hidden.sum(axis: -2)
    }
}

final class BreezeBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeAudioEmbedding
    @ModuleInfo var layers: [BreezeTransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeTTSConfig) {
        let backbone = config.backboneConfig
        let shape = BreezeTransformerShape(backbone, ropeScaling: config.ropeScaling)
        _embedTokens.wrappedValue = BreezeAudioEmbedding(
            numCodebooks: config.numCodebooks,
            vocabSize: config.audioVocabSize,
            audioEmbedSize: config.audioEmbedSize,
            hiddenSize: backbone.hiddenSize
        )
        _layers.wrappedValue = (0..<backbone.numHiddenLayers).map { _ in BreezeTransformerBlock(config: shape) }
        _norm.wrappedValue = RMSNorm(dimensions: backbone.hiddenSize, eps: backbone.rmsNormEps)
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        precondition((inputIDs == nil) != (inputEmbeddings == nil), "Pass input IDs or embeddings")
        var hidden = inputEmbeddings ?? embedTokens(inputIDs!)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
        }
        return norm(hidden)
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}

final class BreezeDepthModel: Module {
    let config: BreezeDepthDecoderConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "backbone_hidden_state_projector") var backboneProjector: Linear?
    @ModuleInfo(key: "inputs_embeds_projector") var inputProjector: Linear
    @ModuleInfo var layers: [BreezeTransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeDepthDecoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.vocabSize,
            dimensions: config.audioEmbedSize
        )
        _backboneProjector.wrappedValue = config.backboneHiddenSize == config.audioEmbedSize
            ? nil
            : Linear(config.backboneHiddenSize, config.audioEmbedSize, bias: false)
        _inputProjector.wrappedValue = Linear(config.audioEmbedSize, config.hiddenSize, bias: false)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in BreezeTransformerBlock(config: BreezeTransformerShape(config)) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        precondition(tokenIDs.ndim == 2, "Depth token IDs must have shape [batch, time]")
        precondition(backboneHiddenState.ndim == 2, "Backbone hidden state must have shape [batch, hidden]")
        precondition(tokenIDs.dim(0) == backboneHiddenState.dim(0), "Depth batch sizes must match")
        let length = tokenIDs.dim(1)
        let positions = MLX.maximum(MLX.arange(length) - 1, MLXArray(0))
        var embeddings = embedTokens(tokenIDs + positions.reshaped([1, length]) * config.vocabSize)
        var backbone = backboneHiddenState
        if let backboneProjector {
            backbone = backboneProjector(backbone)
        }
        embeddings = concatenated([
            backbone.expandedDimensions(axis: 1),
            embeddings[0..., 1..., 0...],
        ], axis: 1)
        var hidden = inputProjector(embeddings)
        for layer in layers {
            hidden = layer(hidden)
        }
        return norm(hidden)
    }
}

final class BreezeCodebooksHead: Module {
    @ParameterInfo var weight: MLXArray

    init(headCount: Int, hiddenSize: Int, vocabSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([headCount, hiddenSize, vocabSize])
    }
}

final class BreezeDepthDecoder: Module {
    @ModuleInfo var model: BreezeDepthModel
    @ModuleInfo(key: "codebooks_head") var codebooksHead: BreezeCodebooksHead

    init(config: BreezeDepthDecoderConfig) {
        _model.wrappedValue = BreezeDepthModel(config: config)
        _codebooksHead.wrappedValue = BreezeCodebooksHead(
            headCount: config.numCodebooks - 1,
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize
        )
    }

    func nextLogits(tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        let headIndex = tokenIDs.dim(1) - 2
        precondition(headIndex >= 0 && headIndex < codebooksHead.weight.dim(0), "Invalid depth head")
        let hidden = model(tokenIDs, backboneHiddenState: backboneHiddenState)[0..., -1, 0...]
        return matmul(hidden, codebooksHead.weight[headIndex])
    }
}
