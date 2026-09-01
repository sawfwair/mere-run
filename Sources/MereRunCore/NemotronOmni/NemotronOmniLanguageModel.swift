import MLX
import MLXNN

/// BF16 routed experts used by the Omni checkpoint. The source checkpoint
/// publishes one tensor per expert; the loader stacks those tensors lazily so
/// MLX can issue gather matmuls for only the six selected experts per token.
final class NemotronOmniExperts: Module {
    @ModuleInfo(key: "up_proj") var upProjection: Q35SwitchLinear
    @ModuleInfo(key: "down_proj") var downProjection: Q35SwitchLinear

    init(config: NemotronHConfig) {
        self._upProjection.wrappedValue = Q35SwitchLinear(
            weight: MLXArray.zeros(
                [config.nRoutedExperts, config.moeIntermediateSize, config.hiddenSize],
                dtype: .bfloat16
            ),
            scales: nil,
            biases: nil,
            bias: nil,
            groupSize: 64,
            bits: 4
        )
        self._downProjection.wrappedValue = Q35SwitchLinear(
            weight: MLXArray.zeros(
                [config.nRoutedExperts, config.hiddenSize, config.moeIntermediateSize],
                dtype: .bfloat16
            ),
            scales: nil,
            biases: nil,
            bias: nil,
            groupSize: 64,
            bits: 4
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
        let projected = upProjection(input, indices: indices)
        let activated = MLX.square(
            MLX.maximum(projected, MLXArray(0).asType(projected.dtype))
        )
        return downProjection(activated, indices: indices)
    }
}

final class NemotronOmniSharedExpert: Module {
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(config: NemotronHConfig) {
        self._upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.sharedExpertIntermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            config.sharedExpertIntermediateSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let projected = upProjection(input)
        let activated = MLX.square(
            MLX.maximum(projected, MLXArray(0).asType(projected.dtype))
        )
        return downProjection(activated)
    }
}

final class NemotronOmniMoE: NemotronHMixer {
    @ModuleInfo(key: "gate") var gate: NemotronHRouter
    @ModuleInfo(key: "experts") var experts: NemotronOmniExperts
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronOmniSharedExpert

    init(config: NemotronHConfig) {
        self._gate.wrappedValue = NemotronHRouter(config: config)
        self._experts.wrappedValue = NemotronOmniExperts(config: config)
        self._sharedExperts.wrappedValue = NemotronOmniSharedExpert(config: config)
        super.init()
    }

    override func callAsFunction(
        _ input: MLXArray,
        cache: NemotronHLayerCache?
    ) -> MLXArray {
        let tokenCount = input.ndim == 3 ? input.dim(1) : 1
        let ranges = Self.prefillRanges(tokenCount: tokenCount)
        guard ranges.count > 1 else {
            return mixedOutput(input)
        }

        // BF16 gather matmul selects routed matrices with `take` because
        // MLX's native gatherMM path is float32-only. Evaluating an entire
        // prompt at once can therefore page several gigabytes of selected
        // expert rows into one Metal command buffer and trip the macOS GPU
        // watchdog. Materialize small token runs independently; decode keeps
        // the original single-token path.
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(ranges.count)
        for range in ranges {
            let piece = mixedOutput(input[0..., range, 0...])
            MLX.eval(piece)
            pieces.append(piece)
        }
        return MLX.concatenated(pieces, axis: 1)
    }

    static func prefillRanges(tokenCount: Int, chunkSize: Int = 4) -> [Range<Int>] {
        precondition(tokenCount > 0 && chunkSize > 0)
        return stride(from: 0, to: tokenCount, by: chunkSize).map { start in
            start..<min(start + chunkSize, tokenCount)
        }
    }

    private func mixedOutput(_ input: MLXArray) -> MLXArray {
        let route = gate(input)
        let routed = (
            experts(input, indices: route.indices)
                * route.weights.expandedDimensions(axis: -1)
        ).sum(axis: -2)
        return routed + sharedExperts(input)
    }
}

final class NemotronOmniLanguageBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMixer

    init(config: NemotronHConfig, blockType: String) {
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.normEps
        )
        switch blockType {
        case "mamba": self._mixer.wrappedValue = NemotronHMamba(config: config)
        case "attention": self._mixer.wrappedValue = NemotronHAttention(config: config)
        case "moe": self._mixer.wrappedValue = NemotronOmniMoE(config: config)
        default: preconditionFailure("Unknown Nemotron Omni block type: \(blockType)")
        }
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: NemotronHLayerCache?
    ) -> MLXArray {
        input + mixer(norm(input), cache: cache)
    }
}

final class NemotronOmniLanguageBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronOmniLanguageBlock]
    @ModuleInfo(key: "norm_f") var finalNorm: RMSNorm

    init(config: NemotronHConfig) {
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = config.layersBlockType.map {
            NemotronOmniLanguageBlock(config: config, blockType: $0)
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
        embeddings inputEmbeddings: MLXArray,
        cache: [NemotronHLayerCache?]?,
        evaluateEachLayer: Bool = false
    ) -> MLXArray {
        precondition(cache == nil || cache?.count == layers.count)
        var hidden = inputEmbeddings
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index] ?? nil)
            if evaluateEachLayer {
                // A full 30B BF16 prefill graph can exceed the macOS Metal
                // watchdog while paging external expert weights. Layer-wise
                // barriers bound each command buffer without slowing the
                // one-token fused decode path.
                MLX.eval(hidden)
            }
        }
        return finalNorm(hidden)
    }
}

public final class NemotronOmniCausalLM: Module, @unchecked Sendable {
    @ModuleInfo(key: "backbone") var backbone: NemotronOmniLanguageBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: NemotronHConfig

    init(config: NemotronHConfig) {
        self.config = config
        self._backbone.wrappedValue = NemotronOmniLanguageBackbone(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func inputEmbeddings(_ inputIDs: MLXArray) -> MLXArray {
        backbone.inputEmbeddings(inputIDs)
    }

    func forward(
        embeddings: MLXArray,
        cache: [NemotronHLayerCache?]?,
        evaluateEachLayer: Bool = false
    ) -> (hidden: MLXArray, logits: MLXArray) {
        let hidden = backbone(
            embeddings: embeddings,
            cache: cache,
            evaluateEachLayer: evaluateEachLayer
        )
        return (hidden, lmHead(hidden))
    }

    func prefill(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?]
    ) -> MLXArray {
        prefill(embeddings: inputEmbeddings(inputIDs), cache: cache)
    }

    func prefill(
        embeddings: MLXArray,
        cache: [NemotronHLayerCache?]
    ) -> MLXArray {
        let output = forward(
            embeddings: embeddings,
            cache: cache,
            evaluateEachLayer: embeddings.dim(1) > 1
        ).logits
        let finalPosition = output.dim(1) - 1
        return output[0..., finalPosition..., 0...]
    }

    func lastPositionLogits(
        _ inputIDs: MLXArray,
        cache: [NemotronHLayerCache?]
    ) -> MLXArray {
        prefill(inputIDs, cache: cache)
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
}
