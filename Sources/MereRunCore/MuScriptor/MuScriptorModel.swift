import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

private final class MuScriptorMelConditioner: Module {
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    init(configuration: MuScriptorConfiguration) {
        self._outputProjection.wrappedValue = Linear(512, configuration.dim, bias: true)
        super.init()
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        outputProjection(mel)
    }
}

private final class MuScriptorClassConditioner: Module {
    @ModuleInfo(key: "embed") var embedding: Embedding

    init(classCount: Int, dimension: Int) {
        self._embedding.wrappedValue = Embedding(
            embeddingCount: classCount + 1,
            dimensions: dimension
        )
        super.init()
    }

    func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        embedding(tokenIDs)
    }
}

private final class MuScriptorConditioners: Module {
    @ModuleInfo(key: "self_wav") var audio: MuScriptorMelConditioner
    @ModuleInfo(key: "instrument_group") var instrument: MuScriptorClassConditioner
    @ModuleInfo(key: "dataset_name") var dataset: MuScriptorClassConditioner

    init(configuration: MuScriptorConfiguration) {
        self._audio.wrappedValue = MuScriptorMelConditioner(configuration: configuration)
        self._instrument.wrappedValue = MuScriptorClassConditioner(
            classCount: 1_000,
            dimension: configuration.dim
        )
        self._dataset.wrappedValue = MuScriptorClassConditioner(
            classCount: 4,
            dimension: configuration.dim
        )
        super.init()
    }
}

private final class MuScriptorConditionProvider: Module {
    @ModuleInfo(key: "conditioners") var conditioners: MuScriptorConditioners

    init(configuration: MuScriptorConfiguration) {
        self._conditioners.wrappedValue = MuScriptorConditioners(configuration: configuration)
        super.init()
    }
}

private final class MuScriptorAttention: Module {
    @ModuleInfo(key: "in_proj_weight") var inputProjectionWeight: MLXArray
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    private let headCount: Int
    private let headDimension: Int
    private let scale: Float

    init(configuration: MuScriptorConfiguration) {
        self.headCount = configuration.numHeads
        self.headDimension = configuration.headDimension
        self.scale = 1 / sqrt(Float(configuration.headDimension))
        self._inputProjectionWeight.wrappedValue = MLXRandom.uniform(
            low: -0.02,
            high: 0.02,
            [3 * configuration.dim, configuration.dim]
        )
        self._outputProjection.wrappedValue = Linear(
            configuration.dim,
            configuration.dim,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, cache: KVCacheSimple) -> MLXArray {
        let batch = input.dim(0)
        let sequenceLength = input.dim(1)
        let dimension = headCount * headDimension
        let projected = MLX.matmul(input, inputProjectionWeight.transposed())
        var queries = projected[.ellipsis, 0..<dimension]
            .reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var keys = projected[.ellipsis, dimension..<(2 * dimension)]
            .reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var values = projected[.ellipsis, (2 * dimension)...]
            .reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)

        (keys, values) = cache.update(keys: keys, values: values)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: sequenceLength == 1 ? .none : .causal
        )
        queries = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, sequenceLength, dimension)
        return outputProjection(queries)
    }
}

private final class MuScriptorTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuScriptorAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(configuration: MuScriptorConfiguration) {
        self._attention.wrappedValue = MuScriptorAttention(configuration: configuration)
        self._norm1.wrappedValue = LayerNorm(dimensions: configuration.dim, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: configuration.dim, eps: 1e-5)
        self._linear1.wrappedValue = Linear(
            configuration.dim,
            configuration.hiddenDimension,
            bias: false
        )
        self._linear2.wrappedValue = Linear(
            configuration.hiddenDimension,
            configuration.dim,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, cache: KVCacheSimple) -> MLXArray {
        let attended = input + attention(norm1(input), cache: cache)
        return attended + linear2(MLXNN.gelu(linear1(norm2(attended))))
    }
}

private final class MuScriptorTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MuScriptorTransformerLayer]

    init(configuration: MuScriptorConfiguration) {
        self._layers.wrappedValue = (0..<configuration.numLayers).map { _ in
            MuScriptorTransformerLayer(configuration: configuration)
        }
        super.init()
    }

    func callAsFunction(_ input: MLXArray, caches: [KVCacheSimple]) -> MLXArray {
        precondition(caches.count == layers.count)
        var hidden = input
        for (layer, cache) in zip(layers, caches) {
            hidden = layer(hidden, cache: cache)
        }
        return hidden
    }
}

public final class MuScriptorModel: Module {
    @ModuleInfo(key: "condition_provider") private var conditionProvider: MuScriptorConditionProvider
    @ModuleInfo(key: "emb") private var tokenEmbedding: Embedding
    @ModuleInfo(key: "transformer") private var transformer: MuScriptorTransformer
    @ModuleInfo(key: "out_norm") private var outputNorm: LayerNorm
    @ModuleInfo(key: "linear") private var outputProjection: Linear
    private var positionalEmbeddingTable: MLXArray?
    private var positionalEmbeddingCapacity = 0

    public let configuration: MuScriptorConfiguration

    var inferenceDType: DType {
        tokenEmbedding.weight.dtype
    }

    public init(configuration: MuScriptorConfiguration) {
        self.configuration = configuration
        self._conditionProvider.wrappedValue = MuScriptorConditionProvider(configuration: configuration)
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.card + 1,
            dimensions: configuration.dim
        )
        self._transformer.wrappedValue = MuScriptorTransformer(configuration: configuration)
        self._outputNorm.wrappedValue = LayerNorm(dimensions: configuration.dim, eps: 1e-5)
        self._outputProjection.wrappedValue = Linear(
            configuration.dim,
            configuration.card,
            bias: false
        )
        super.init()
    }

    public static func load(
        resources: MuScriptorResources,
        variant: MuScriptorVariant,
        dtype: DType = .bfloat16
    ) throws -> MuScriptorModel {
        guard FileManager.default.fileExists(atPath: resources.weightsURL.path) else {
            throw MuScriptorError.missingModelFile(resources.weightsURL)
        }
        let configuration = try resources.configuration(fallback: variant.fallbackConfiguration)
        let model = MuScriptorModel(configuration: configuration)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.weightsURL,
            to: model,
            dtype: dtype,
            verify: .none,
            mapper: { key, value in
                if key.contains("mel_spec_transform.spectrogram.window")
                    || key.contains("mel_spec_transform.mel_scale.fb") {
                    return []
                }
                if key.hasPrefix("emb.0.") {
                    return [("emb." + key.dropFirst("emb.0.".count), value)]
                }
                if key.hasPrefix("linears.0.") {
                    return [("linear." + key.dropFirst("linears.0.".count), value)]
                }
                return [(key, value)]
            }
        )
        return model
    }

    public func makeCaches() -> [KVCacheSimple] {
        (0..<configuration.numLayers).map { _ in KVCacheSimple(step: 256) }
    }

    public func conditioningPrefix(mel: MLXArray, instruments: [String]?) throws -> MLXArray {
        var audio = conditionProvider.conditioners.audio(mel)
        let frameCount = audio.dim(1)
        if frameCount > 500 {
            let maskValues = (0..<frameCount).map { $0 < 500 ? Float(1) : Float(0) }
            let mask = MLXArray(maskValues).reshaped(1, frameCount, 1).asType(audio.dtype)
            audio = audio * mask
        }

        let datasetIDs = MLXArray([Int32(1)]).reshaped(1, 1)
        let dataset = conditionProvider.conditioners.dataset(datasetIDs)
        let instrumentIDs: [Int32]
        if let instruments, !instruments.isEmpty {
            instrumentIDs = try instruments.map { instrument in
                guard let groupID = MuScriptorInstruments.groupIDs[instrument] else {
                    throw MuScriptorError.invalidInstrument(instrument)
                }
                return Int32(groupID + 2)
            }
        } else {
            instrumentIDs = [1]
        }
        let instrument = conditionProvider.conditioners.instrument(
            MLXArray(instrumentIDs).reshaped(1, instrumentIDs.count)
        )
        return MLX.concatenated([audio, dataset, instrument], axis: 1)
    }

    public func logits(
        tokenID: Int,
        prefix: MLXArray?,
        caches: [KVCacheSimple]
    ) -> MLXArray {
        batchedLogits(
            tokenIDs: MLXArray([Int32(tokenID)]).reshaped(1, 1),
            prefix: prefix,
            caches: caches
        )[0]
    }

    /// Runs one autoregressive step for several independent audio chunks.
    /// Every row owns its own cache lane while sharing one transformer graph.
    public func batchedLogits(
        tokenIDs: MLXArray,
        prefix: MLXArray?,
        caches: [KVCacheSimple]
    ) -> MLXArray {
        precondition(tokenIDs.ndim == 2 && tokenIDs.dim(1) == 1)
        var hidden = tokenEmbedding(tokenIDs)
        if let prefix {
            precondition(prefix.dim(0) == tokenIDs.dim(0))
            hidden = MLX.concatenated([prefix, hidden], axis: 1)
        }
        let offset = caches.first?.offset ?? 0
        hidden = hidden + positionalEmbedding(
            length: hidden.dim(1),
            dimension: configuration.dim,
            offset: offset,
            dtype: hidden.dtype
        )
        hidden = transformer(hidden, caches: caches)
        hidden = outputNorm(hidden)
        let lastHidden = hidden[0..., hidden.dim(1) - 1, 0...]
        return outputProjection(lastHidden)[0..., 0..<1_393].asType(.float32)
    }

    private func positionalEmbedding(
        length: Int,
        dimension: Int,
        offset: Int,
        dtype: DType
    ) -> MLXArray {
        let requiredCapacity = offset + length
        if requiredCapacity > positionalEmbeddingCapacity {
            let capacity = max(requiredCapacity, max(1_024, positionalEmbeddingCapacity * 2))
            positionalEmbeddingTable = makePositionalEmbeddingTable(
                capacity: capacity,
                dimension: dimension
            )
            positionalEmbeddingCapacity = capacity
            if let positionalEmbeddingTable {
                MLX.eval(positionalEmbeddingTable)
            }
        }
        guard let positionalEmbeddingTable else {
            preconditionFailure("MuScriptor positional embedding table was not initialized")
        }
        return positionalEmbeddingTable[0..., offset..<(offset + length), 0...].asType(dtype)
    }

    private func makePositionalEmbeddingTable(
        capacity: Int,
        dimension: Int
    ) -> MLXArray {
        let half = dimension / 2
        let positions = MLXArray(0..<capacity).asType(.float32).reshaped(capacity, 1)
        let dimensions = MLXArray(0..<half).asType(.float32).reshaped(1, half)
        let denominator = MLX.pow(MLXArray(Float(10_000)), dimensions / Float(max(half - 1, 1)))
        let phase = positions / denominator
        return MLX.concatenated([MLX.cos(phase), MLX.sin(phase)], axis: -1)
            .reshaped(1, capacity, dimension)
    }
}
