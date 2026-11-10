import MLX
import MLXNN

/// Routes of one prefill chunk grouped by expert, so each used expert runs a
/// single dense matmul over the tokens routed to it.
struct NemotronOmniExpertRoutePlan: Equatable {
    struct Group: Equatable {
        let expert: Int
        /// Flattened token row of every route assigned to this expert.
        let tokens: [Int32]
    }

    /// Used experts in ascending order.
    let groups: [Group]
    /// For each route in `[token, rank]` order, its row in the concatenated
    /// group outputs.
    let inverseOrder: [Int32]

    init(flatIndices: [Int32], topK: Int) {
        precondition(topK > 0 && flatIndices.count.isMultiple(of: topK))
        var routesByExpert: [Int: [Int32]] = [:]
        for (route, expert) in flatIndices.enumerated() {
            routesByExpert[Int(expert), default: []].append(Int32(route))
        }
        var groups: [Group] = []
        groups.reserveCapacity(routesByExpert.count)
        var inverseOrder = [Int32](repeating: 0, count: flatIndices.count)
        var row: Int32 = 0
        for expert in routesByExpert.keys.sorted() {
            let routes = routesByExpert[expert, default: []]
            for route in routes {
                inverseOrder[Int(route)] = row
                row += 1
            }
            groups.append(Group(expert: expert, tokens: routes.map { $0 / Int32(topK) }))
        }
        self.groups = groups
        self.inverseOrder = inverseOrder
    }
}

/// BF16 routed experts used by the Omni checkpoint. The source checkpoint
/// publishes one tensor per expert; the loader stacks those tensors lazily so
/// MLX can issue gather matmuls for only the six selected experts per token.
final class NemotronOmniExperts: Module {
    /// Each evaluated batch reads at most this many experts' up and down
    /// matrices, which bounds one Metal command buffer below the weight
    /// volume of the four-token gather batches this path replaced.
    static let expertsPerEvaluation = 16

    @ModuleInfo(key: "up_proj") var upProjection: Q35SwitchLinear
    @ModuleInfo(key: "down_proj") var downProjection: Q35SwitchLinear

    convenience init(config: NemotronHConfig) {
        self.init(
            upWeight: MLXArray.zeros(
                [config.nRoutedExperts, config.moeIntermediateSize, config.hiddenSize],
                dtype: .bfloat16
            ),
            downWeight: MLXArray.zeros(
                [config.nRoutedExperts, config.hiddenSize, config.moeIntermediateSize],
                dtype: .bfloat16
            )
        )
    }

    init(upWeight: MLXArray, downWeight: MLXArray) {
        self._upProjection.wrappedValue = Q35SwitchLinear(
            weight: upWeight,
            scales: nil,
            biases: nil,
            bias: nil,
            groupSize: 64,
            bits: 4
        )
        self._downProjection.wrappedValue = Q35SwitchLinear(
            weight: downWeight,
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
        return downProjection(Self.activated(projected), indices: indices)
    }

    /// Multitoken prefill. `rows` is `[tokens, hidden]`; the result is
    /// `[tokens * topK, hidden]` in `[token, rank]` route order. Expert
    /// matrices are read in place, where the gather path copies one matrix
    /// per route.
    func callAsFunction(_ rows: MLXArray, plan: NemotronOmniExpertRoutePlan) -> MLXArray {
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(plan.groups.count)
        for group in plan.groups {
            let selected = rows.take(MLXArray(group.tokens), axis: 0)
            let projected = MLX.matmul(
                selected,
                upProjection.weight[group.expert].swappedAxes(-1, -2)
            )
            outputs.append(MLX.matmul(
                Self.activated(projected),
                downProjection.weight[group.expert].swappedAxes(-1, -2)
            ))
            if outputs.count.isMultiple(of: Self.expertsPerEvaluation) {
                MLX.eval(outputs.suffix(Self.expertsPerEvaluation))
            }
        }
        return MLX.concatenated(outputs, axis: 0)
            .take(MLXArray(plan.inverseOrder), axis: 0)
    }

    private static func activated(_ projected: MLXArray) -> MLXArray {
        MLX.square(MLX.maximum(projected, MLXArray(0).asType(projected.dtype)))
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
        guard input.ndim == 3, input.dim(1) > 1 else {
            return mixedOutput(input)
        }
        return prefillOutput(input)
    }

    /// BF16 gather matmul selects routed matrices with `take` because MLX's
    /// native gatherMM path is float32-only, which copies twelve expert
    /// matrices per token. Prefill instead reads the route back and groups it
    /// by expert; decode keeps the single-token gather path.
    private func prefillOutput(_ input: MLXArray) -> MLXArray {
        let route = gate(input)
        let topK = route.indices.dim(-1)
        let plan = NemotronOmniExpertRoutePlan(
            flatIndices: route.indices.asType(.int32).reshaped([-1]).asArray(Int32.self),
            topK: topK
        )
        let routed = experts(
            input.reshaped([input.dim(0) * input.dim(1), input.dim(2)]),
            plan: plan
        ).reshaped([input.dim(0), input.dim(1), topK, input.dim(2)])
        return (routed * route.weights.expandedDimensions(axis: -1)).sum(axis: -2)
            + sharedExperts(input)
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
