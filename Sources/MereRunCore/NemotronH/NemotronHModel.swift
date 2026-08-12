import Foundation
import MLX
import MLXFast
import MLXNN

final class NemotronHAttention: NemotronHMixer {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDimensions: Int
    private let scale: Float

    init(config: NemotronHConfig) {
        headCount = config.numAttentionHeads
        keyValueHeadCount = config.numKeyValueHeads
        headDimensions = config.headDim
        scale = 1 / Foundation.sqrt(Float(config.headDim))
        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: false
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    override func callAsFunction(_ x: MLXArray, cache: NemotronHLayerCache?) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let attentionCache: Gemma4AttentionCache?
        if case .attention(let value)? = cache {
            attentionCache = value
        } else {
            attentionCache = nil
        }
        let offset = attentionCache?.offset ?? 0
        let queries = queryProjection(x)
            .reshaped(batch, length, headCount, headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(x)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        var values = valueProjection(x)
            .reshaped(batch, length, keyValueHeadCount, headDimensions)
            .transposed(0, 2, 1, 3)
        if let attentionCache {
            let state = attentionCache.attentionState(appending: keys, values: values)!
            keys = state.0
            values = state.1
        }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length == 1 {
            mask = .none
        } else if offset == 0 {
            mask = .causal
        } else {
            let queryPositions = MLXArray(
                Int32(offset)..<Int32(offset + length)
            ).reshaped(length, 1)
            let keyPositions = MLXArray(0..<Int32(keys.dim(2))).reshaped(1, keys.dim(2))
            let allowed = keyPositions .<= queryPositions
            let zeros = MLXArray.zeros([length, keys.dim(2)], dtype: x.dtype)
            mask = .array(MLX.where(allowed, zeros, zeros - 1e9))
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        return outputProjection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, length, headCount * headDimensions)
        )
    }
}

final class NemotronHBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMixer

    let blockType: String

    init(config: NemotronHConfig, blockType: String) {
        self.blockType = blockType
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEps
        )
        switch blockType {
        case "mamba": self._mixer.wrappedValue = NemotronHMamba(config: config)
        case "attention": self._mixer.wrappedValue = NemotronHAttention(config: config)
        case "moe": self._mixer.wrappedValue = NemotronHMoE(config: config)
        default: preconditionFailure("Unknown Nemotron-H block type: \(blockType)")
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: NemotronHLayerCache?) -> MLXArray {
        x + mixer(norm(x), cache: cache)
    }
}

final class NemotronHBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var finalNorm: RMSNorm

    init(config: NemotronHConfig) {
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = config.layersBlockType.map {
            NemotronHBlock(config: config, blockType: $0)
        }
        self._finalNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEps
        )
        super.init()
    }

    func inputEmbeddings(_ inputIDs: MLXArray) -> MLXArray {
        embeddings(inputIDs.dtype == .int32 ? inputIDs : inputIDs.asType(.int32))
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?]?,
        captureLayerIndices: Set<Int>
    ) -> (hidden: MLXArray, captures: [Int: MLXArray]) {
        precondition(cache == nil || cache?.count == layers.count)
        var hidden = inputEmbeddings(inputIDs)
        var captures: [Int: MLXArray] = [:]
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index] ?? nil)
            if captureLayerIndices.contains(index) {
                captures[index] = hidden
            }
        }
        return (finalNorm(hidden), captures)
    }
}

struct NemotronHForwardOutput {
    let hidden: MLXArray
    let logits: MLXArray
    let capturedHiddenStates: [Int: MLXArray]
}

public final class NemotronHCausalLM: Module, @unchecked Sendable {
    @ModuleInfo(key: "backbone") var backbone: NemotronHBackbone
    @ModuleInfo(key: "lm_head") var lmHead: NemotronHNVFP4Linear

    public let config: NemotronHConfig

    public init(config: NemotronHConfig) {
        self.config = config
        self._backbone.wrappedValue = NemotronHBackbone(config: config)
        self._lmHead.wrappedValue = NemotronHNVFP4Linear(
            inputDimensions: config.hiddenSize,
            outputDimensions: config.vocabSize
        )
        super.init()
    }

    func inputEmbeddings(_ inputIDs: MLXArray) -> MLXArray {
        backbone.inputEmbeddings(inputIDs)
    }

    func logits(from hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    func forward(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?]?,
        captureLayerIndices: Set<Int> = []
    ) -> NemotronHForwardOutput {
        let output = backbone(
            inputIDs,
            cache: cache,
            captureLayerIndices: captureLayerIndices
        )
        return NemotronHForwardOutput(
            hidden: output.hidden,
            logits: logits(from: output.hidden),
            capturedHiddenStates: output.captures
        )
    }

    func prefill(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?],
        captureLayerIndices: Set<Int>
    ) -> NemotronHForwardOutput {
        let output = backbone(
            inputIDs,
            cache: cache,
            captureLayerIndices: captureLayerIndices
        )
        let finalPosition = output.hidden.dim(1) - 1
        let finalHidden = output.hidden[0..., finalPosition..., 0...]
        return NemotronHForwardOutput(
            hidden: finalHidden,
            logits: logits(from: finalHidden),
            capturedHiddenStates: output.captures
        )
    }

    func lastPositionLogits(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?]
    ) -> MLXArray {
        let output = forward(inputIDs, cache: cache)
        let finalPosition = output.logits.dim(1) - 1
        return output.logits[0..., finalPosition..., 0...]
    }

    func makeCache() -> [NemotronHLayerCache?] {
        config.layersBlockType.map { type in
            switch type {
            case "mamba": .mamba(NemotronHMambaCache())
            case "attention": .attention(Gemma4FullKVCache())
            default: nil
            }
        }
    }

    func forkCache(_ cache: [NemotronHLayerCache?]) -> [NemotronHLayerCache?] {
        cache.map { $0?.fork() }
    }
}
