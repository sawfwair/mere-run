import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

public final class NemotronBoltCache {
    let layers: [BoltLayerState]
    init(layerCount: Int) { layers = (0..<layerCount).map { _ in BoltLayerState() } }
    public var offset: Int { layers.first?.offset ?? 0 }
}

final class BoltLayerState {
    var convolution: MLXArray?
    var recurrent: MLXArray?
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0
}

enum BoltLinear {
    case dense(MLXArray)
    case packed(BoltPackedMatrix)
    case nvfp4(weight: MLXArray, scales: MLXArray, global: MLXArray)

    init(store: inout BoltWeightStore, prefix: String, input: Int, output: Int, format: BoltPackedFormat?) throws {
        if let format {
            self = .packed(try BoltPackedMatrix(
                weight: store.take(prefix + ".weight"), format: format, inputSize: input, outputSize: output
            ))
        } else if store.arrays[prefix + ".weight_scale"] != nil {
            let weight = try store.take(prefix + ".weight", shape: [output, input / 2], dtype: .uint8)
            let scales = try store.take(prefix + ".weight_scale", shape: [output, input / 16], dtype: .uint8)
            let global = try store.take(prefix + ".weight_scale_2", shape: [], dtype: .float32)
            self = .nvfp4(weight: weight.view(dtype: .uint32), scales: scales, global: global)
        } else {
            self = .dense(try store.take(prefix + ".weight", shape: [output, input], dtype: .bfloat16))
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight): return x.matmul(weight.T)
        case .packed(let matrix): return matrix(x)
        case .nvfp4(let weight, let scales, let global):
            let result = MLX.quantizedMM(
                x.asType(.float32) * global, weight, scales: scales, biases: nil, transpose: true,
                groupSize: 16, bits: 4, mode: .nvfp4
            )
            return result.asType(x.dtype)
        }
    }
}

struct BoltMamba {
    let input: BoltLinear
    let output: BoltLinear
    let convolution: MLXArray
    let convolutionBias: MLXArray
    let timeStepBias: MLXArray
    let aLog: MLXArray
    let skip: MLXArray
    let normWeight: MLXArray
    let config: NemotronBoltLanguageConfig

    init(config: NemotronBoltLanguageConfig, store: inout BoltWeightStore, prefix: String) throws {
        self.config = config
        let inner = config.mambaNumHeads * config.mambaHeadDim
        let convSize = inner + 2 * config.nGroups * config.ssmStateSize
        input = try BoltLinear(store: &store, prefix: prefix + ".in_proj", input: config.hiddenSize,
                               output: inner + convSize + config.mambaNumHeads, format: nil)
        output = try BoltLinear(store: &store, prefix: prefix + ".out_proj", input: inner,
                                output: config.hiddenSize, format: nil)
        convolution = try store.take(prefix + ".conv1d.weight", shape: [convSize, 1, config.convKernel], dtype: .bfloat16).transposed(0, 2, 1)
        convolutionBias = try store.take(prefix + ".conv1d.bias", shape: [convSize], dtype: .bfloat16)
        timeStepBias = try store.take(prefix + ".dt_bias", shape: [config.mambaNumHeads])
        aLog = try store.take(prefix + ".A_log", shape: [config.mambaNumHeads])
        skip = try store.take(prefix + ".D", shape: [config.mambaNumHeads])
        normWeight = try store.take(prefix + ".norm.weight", shape: [inner], dtype: .bfloat16)
    }

    func callAsFunction(_ x: MLXArray, state: BoltLayerState) -> MLXArray {
        let c = config
        let batch = x.dim(0)
        let length = x.dim(1)
        let inner = c.mambaNumHeads * c.mambaHeadDim
        let convSize = inner + 2 * c.nGroups * c.ssmStateSize
        let projected = input(x)
        let gate = projected[.ellipsis, ..<inner]
        let convInput = projected[.ellipsis, inner..<(inner + convSize)]
        let dt = projected[.ellipsis, (inner + convSize)...]
        let previous = state.convolution ?? MLXArray.zeros([batch, c.convKernel - 1, convSize], dtype: x.dtype)
        let padded = concatenated([previous, convInput], axis: 1)
        state.convolution = padded[0..., (padded.dim(1) - c.convKernel + 1)..., 0...]
        let convolved = MLXNN.silu(MLX.conv1d(padded, convolution, groups: convSize) + convolutionBias)
        let hidden = convolved[.ellipsis, ..<inner].reshaped(batch, length, c.mambaNumHeads, c.mambaHeadDim)
        let bEnd = inner + c.nGroups * c.ssmStateSize
        let b = convolved[.ellipsis, inner..<bEnd].reshaped(batch, length, c.nGroups, c.ssmStateSize)
        let cc = convolved[.ellipsis, bEnd...].reshaped(batch, length, c.nGroups, c.ssmStateSize)
        let result = boltSSMUpdate(
            x: hidden, b: b, c: cc, dt: dt, aLog: aLog, d: skip, dtBias: timeStepBias,
            state: state.recurrent, minimumTimeStep: c.timeStepMin, maximumTimeStep: Float.greatestFiniteMagnitude
        )
        state.recurrent = result.1
        let gated = MLXNN.silu(gate.asType(.float32)) * result.0.reshaped(batch, length, inner).asType(.float32)
        let grouped = gated.reshaped(batch, length, c.nGroups, inner / c.nGroups)
        let normalized = grouped * rsqrt(square(grouped).mean(axis: -1, keepDims: true) + c.normEps)
        return output((normalized.reshaped(batch, length, inner) * normWeight.asType(.float32)).asType(x.dtype))
    }
}

struct BoltAttention {
    let query: BoltLinear
    let key: BoltLinear
    let value: BoltLinear
    let output: BoltLinear
    let config: NemotronBoltLanguageConfig

    init(config: NemotronBoltLanguageConfig, store: inout BoltWeightStore, prefix: String) throws {
        self.config = config
        query = try BoltLinear(store: &store, prefix: prefix + ".q_proj", input: config.hiddenSize,
                               output: config.numAttentionHeads * config.headDim, format: .q8_0)
        key = try BoltLinear(store: &store, prefix: prefix + ".k_proj", input: config.hiddenSize,
                             output: config.numKeyValueHeads * config.headDim, format: .q8_0)
        value = try BoltLinear(store: &store, prefix: prefix + ".v_proj", input: config.hiddenSize,
                               output: config.numKeyValueHeads * config.headDim, format: .q8_0)
        output = try BoltLinear(store: &store, prefix: prefix + ".o_proj", input: config.numAttentionHeads * config.headDim,
                                output: config.hiddenSize, format: .q8_0)
    }

    func callAsFunction(_ x: MLXArray, state: BoltLayerState) -> MLXArray {
        let c = config
        let batch = x.dim(0)
        let length = x.dim(1)
        let q = query(x).reshaped(batch, length, c.numAttentionHeads, c.headDim).transposed(0, 2, 1, 3)
        var k = key(x).reshaped(batch, length, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        var v = value(x).reshaped(batch, length, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        if let previous = state.keys, let values = state.values {
            k = concatenated([previous, k], axis: 2)
            v = concatenated([values, v], axis: 2)
        }
        state.keys = k
        state.values = v
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length == 1 { mask = .none }
        else if state.offset == 0 { mask = .causal }
        else {
            let qp = MLXArray(Int32(state.offset)..<Int32(state.offset + length)).reshaped(length, 1)
            let kp = MLXArray(0..<Int32(k.dim(2))).reshaped(1, k.dim(2))
            let zeros = MLXArray.zeros([length, k.dim(2)], dtype: x.dtype)
            mask = .array(MLX.where(kp .<= qp, zeros, zeros - 1e9))
        }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                       scale: 1 / sqrt(Float(c.headDim)), mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, c.numAttentionHeads * c.headDim))
    }
}

public enum BoltRouter {
    /// Bias selects experts; probabilities and output scaling use unbiased scores.
    public static func route(_ x: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, scale: Float) -> (MLXArray, MLXArray) {
        let scores = sigmoid(x.asType(.float32).matmul(weight.asType(.float32).T))
        let indices = stopGradient(argPartition(-(scores + bias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        let selected = takeAlong(scores, indices, axis: -1)
        return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * scale)
    }
}

struct BoltMoE {
    let gate: MLXArray
    let correction: MLXArray
    let latentInput: BoltLinear
    let latentOutput: BoltLinear
    let sharedUp: BoltLinear
    let sharedDown: BoltLinear
    let expertUp: BoltPackedMatrix
    let expertDown: BoltPackedMatrix
    let topK: Int
    let scale: Float

    init(config: NemotronBoltLanguageConfig, block: NemotronBoltBlockConfig, store: inout BoltWeightStore, prefix: String) throws {
        guard let count = block.expertCount, let width = block.moeIntermediateSize, let k = block.topK else {
            throw NemotronBoltError.invalidCheckpoint("Missing MoE dimensions")
        }
        topK = k
        scale = config.routedScalingFactor
        gate = try store.take(prefix + ".gate.weight", shape: [count, config.hiddenSize], dtype: .bfloat16)
        correction = try store.take(prefix + ".gate.e_score_correction_bias", shape: [count], dtype: .float32)
        latentInput = try BoltLinear(store: &store, prefix: prefix + ".fc1_latent_proj", input: config.hiddenSize,
                                     output: config.moeLatentSize, format: .q8_0)
        latentOutput = try BoltLinear(store: &store, prefix: prefix + ".fc2_latent_proj", input: config.moeLatentSize,
                                      output: config.hiddenSize, format: .q8_0)
        sharedUp = try BoltLinear(store: &store, prefix: prefix + ".shared_experts.up_proj", input: config.hiddenSize,
                                  output: config.sharedExpertIntermediateSize, format: .iq2XXS)
        sharedDown = try BoltLinear(store: &store, prefix: prefix + ".shared_experts.down_proj", input: config.sharedExpertIntermediateSize,
                                    output: config.hiddenSize, format: .iq2XXS)
        func pack(_ projection: String, input: Int, output: Int) throws -> BoltPackedMatrix {
            var tensors: [MLXArray] = []
            for expert in 0..<count {
                tensors.append(try store.take(prefix + ".experts.\(expert).\(projection).weight",
                                              shape: [output, input / 256, 66], dtype: .uint8))
            }
            let packed = stacked(tensors)
            MLX.eval(packed)
            return try BoltPackedMatrix(weight: packed, format: .iq2XXS, inputSize: input, outputSize: output)
        }
        expertUp = try pack("up_proj", input: config.moeLatentSize, output: width)
        expertDown = try pack("down_proj", input: width, output: config.moeLatentSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let route = BoltRouter.route(x, weight: gate, bias: correction, topK: topK, scale: scale)
        let up = expertUp(latentInput(x), indices: route.0)
        let activated = square(maximum(up, MLXArray(0).asType(up.dtype)))
        let down = expertDown(activated, indices: route.0)
        let routed = (down.asType(.float32) * route.1.expandedDimensions(axis: -1)).sum(axis: -2).asType(x.dtype)
        let shared = sharedUp(x)
        return latentOutput(routed) + sharedDown(square(maximum(shared, MLXArray(0).asType(shared.dtype))))
    }
}

struct BoltBlock {
    enum Mixer { case mamba(BoltMamba), attention(BoltAttention), moe(BoltMoE) }
    let norm: MLXArray
    let mixer: Mixer
    let eps: Float

    init(config: NemotronBoltLanguageConfig, layer: Int, store: inout BoltWeightStore) throws {
        let prefix = "language_model.backbone.layers.\(layer)"
        norm = try store.take(prefix + ".norm.weight", shape: [config.hiddenSize], dtype: .bfloat16)
        eps = config.normEps
        let block = config.blocks[layer]
        switch block.blockType {
        case "mamba": mixer = .mamba(try BoltMamba(config: config, store: &store, prefix: prefix + ".mixer"))
        case "attention": mixer = .attention(try BoltAttention(config: config, store: &store, prefix: prefix + ".mixer"))
        case "moe": mixer = .moe(try BoltMoE(config: config, block: block, store: &store, prefix: prefix + ".mixer"))
        default: throw NemotronBoltError.invalidCheckpoint("Unknown block")
        }
    }

    func callAsFunction(_ x: MLXArray, state: BoltLayerState) -> MLXArray {
        let normalized = MLXFast.rmsNorm(x, weight: norm, eps: eps)
        let result: MLXArray
        switch mixer {
        case .mamba(let layer): result = layer(normalized, state: state)
        case .attention(let layer): result = layer(normalized, state: state)
        case .moe(let layer): result = layer(normalized)
        }
        state.offset += x.dim(1)
        return x + result
    }
}

public final class NemotronBoltModel {
    public let config: NemotronBoltConfig
    public let visionTower: NemotronBoltVisionTower
    let embeddings: MLXArray
    let norm: MLXArray
    let head: BoltLinear
    let blocks: [BoltBlock]

    public init(root: URL, progress: (Int, Int) -> Void = { _, _ in }) throws {
        config = try NemotronBoltConfig.load(at: root)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self,
            from: Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json")))
        let expected = (1...90).map { String(format: "model-%05d-of-00090.safetensors", $0) }
        guard index.shardFilenames == expected, index.metadata?.totalSize == NemotronBoltConfig.checkpointWeightBytes else {
            throw NemotronBoltError.invalidCheckpoint("Index differs from the pinned 90-shard checkpoint")
        }
        func load(_ shard: Int) throws -> BoltWeightStore {
            let filename = expected[shard]
            let arrays = try BoltCheckpoint.loadShard(at: root.appendingPathComponent(filename))
            guard Set(arrays.keys) == Set(index.weightMap.filter { $0.value == filename }.keys) else {
                throw NemotronBoltError.invalidCheckpoint("Shard keys differ from index")
            }
            progress(shard + 1, 90)
            return BoltWeightStore(arrays: arrays)
        }
        var store = try load(0)
        embeddings = try store.take("language_model.backbone.embeddings.weight", shape: [config.language.vocabSize, config.language.hiddenSize], dtype: .bfloat16)
        norm = try store.take("language_model.backbone.norm_f.weight", shape: [config.language.hiddenSize], dtype: .bfloat16)
        head = try BoltLinear(store: &store, prefix: "language_model.lm_head", input: config.language.hiddenSize,
                              output: config.language.vocabSize, format: .q8_0)
        guard store.arrays.isEmpty else { throw NemotronBoltError.invalidCheckpoint("Unexpected language tensors") }
        var loaded: [BoltBlock] = []
        for layer in config.language.blocks.indices {
            var shard = try load(layer + 1)
            loaded.append(try BoltBlock(config: config.language, layer: layer, store: &shard))
            guard shard.arrays.isEmpty else { throw NemotronBoltError.invalidCheckpoint("Unexpected layer tensors") }
            Memory.clearCache()
        }
        blocks = loaded
        visionTower = NemotronBoltVisionTower(config: config)
        let vision = try load(89)
        var updates: [(String, MLXArray)] = []
        for (key, value) in vision.arrays {
            if key.hasPrefix("vision_model.radio_model.model.") {
                let mapped = "model." + key.dropFirst("vision_model.radio_model.model.".count)
                updates.append((mapped.replacingOccurrences(of: "patch_generator.cls_token.token", with: "patch_generator.class_token"), value))
            } else if key.hasPrefix("mlp1.") {
                let mapping = ["mlp1.0.weight": "projector.norm.weight", "mlp1.1.weight": "projector.linear1.weight", "mlp1.3.weight": "projector.linear2.weight"]
                guard let mapped = mapping[key] else { throw NemotronBoltError.invalidCheckpoint("Unexpected vision projector tensor") }
                updates.append((mapped, value))
            }
        }
        let expectedKeys = Set(visionTower.parameters().flattened().map(\.0))
        guard Set(updates.map(\.0)) == expectedKeys else { throw NemotronBoltError.invalidCheckpoint("Incomplete vision weights") }
        try visionTower.update(parameters: ModuleParameters.unflattened(updates), verify: .shapeMismatch)
        visionTower.train(false)
    }

    public func makeCache() -> NemotronBoltCache { NemotronBoltCache(layerCount: blocks.count) }
    public func inputEmbeddings(_ tokens: MLXArray) -> MLXArray { embeddings[tokens] }

    public func prefill(embeddings input: MLXArray, cache: NemotronBoltCache) -> MLXArray {
        precondition(cache.layers.count == blocks.count)
        var hidden = input
        for (index, block) in blocks.enumerated() {
            let state = cache.layers[index]
            hidden = block(hidden, state: state)
            // Bound command buffers during prompt prefill and single-token decode.
            if input.dim(1) > 1 || (index + 1).isMultiple(of: 8) { MLX.eval(hidden) }
        }
        let last = MLXFast.rmsNorm(hidden[0..., (hidden.dim(1) - 1)..., 0...], weight: norm, eps: config.language.normEps)
        return head(last)
    }

    public func lastPositionLogits(_ token: MLXArray, cache: NemotronBoltCache) -> MLXArray {
        prefill(embeddings: inputEmbeddings(token.reshaped(1, 1)), cache: cache)
    }
}
