import Foundation
import MLX
import MLXNN

final class HiDreamO1TimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int

    @ModuleInfo(key: "mlp") var mlp: (Linear, SiLUModule, Linear)

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self._mlp.wrappedValue = (
            Linear(frequencyEmbeddingSize, hiddenSize, bias: true),
            SiLUModule(),
            Linear(hiddenSize, hiddenSize, bias: true)
        )
        super.init()
    }

    static func timestepEmbedding(_ timesteps: MLXArray, dim: Int, maxPeriod: Float = 10_000.0) -> MLXArray {
        let dtype = timesteps.dtype
        let halfDim = dim / 2
        var exponent = -MLX.log(MLXArray(maxPeriod))
        exponent = exponent * MLXArray(0..<halfDim).asType(dtype)
        exponent = exponent / MLXArray(Float(halfDim))
        let freqs = MLX.exp(exponent)
        let args = timesteps[.ellipsis, .newAxis] * freqs[.newAxis]
        var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)

        if dim % 2 != 0 {
            let pad = MLX.zeros([embedding.dim(0), 1], dtype: dtype)
            embedding = MLX.concatenated([embedding, pad], axis: -1)
        }
        return embedding
    }

    func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
        let tFreq = Self.timestepEmbedding(timesteps * 1_000, dim: frequencyEmbeddingSize)
        return mlp.2(mlp.1(mlp.0(tFreq)))
    }
}

final class HiDreamO1BottleneckPatchEmbed: Module {
    @ModuleInfo(key: "proj1") var proj1: Linear
    @ModuleInfo(key: "proj2") var proj2: Linear

    init(patchSize: Int = 32, inChannels: Int = 3, bottleneckDim: Int, hiddenSize: Int) {
        let patchDim = patchSize * patchSize * inChannels
        self._proj1.wrappedValue = Linear(patchDim, bottleneckDim, bias: false)
        self._proj2.wrappedValue = Linear(bottleneckDim, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        proj2(proj1(patches))
    }
}

final class HiDreamO1FinalLayer: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(hiddenSize: Int, patchSize: Int = 32, outChannels: Int = 3) {
        let patchDim = patchSize * patchSize * outChannels
        self._linear.wrappedValue = Linear(hiddenSize, patchDim, bias: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear(hiddenStates)
    }
}

final class HiDreamO1PixelHead: Module {
    @ModuleInfo(key: "t_embedder1") var timestepEmbedder: HiDreamO1TimestepEmbedder
    @ModuleInfo(key: "x_embedder") var patchEmbedder: HiDreamO1BottleneckPatchEmbed
    @ModuleInfo(key: "final_layer2") var finalLayer: HiDreamO1FinalLayer

    init(config: HiDreamO1Config) {
        let hiddenSize = config.textConfig.hiddenSize
        let bottleneckDim = hiddenSize / 4
        self._timestepEmbedder.wrappedValue = HiDreamO1TimestepEmbedder(hiddenSize: hiddenSize)
        self._patchEmbedder.wrappedValue = HiDreamO1BottleneckPatchEmbed(
            patchSize: HiDreamO1SampleBuilder.patchSize,
            inChannels: 3,
            bottleneckDim: bottleneckDim,
            hiddenSize: hiddenSize
        )
        self._finalLayer.wrappedValue = HiDreamO1FinalLayer(
            hiddenSize: hiddenSize,
            patchSize: HiDreamO1SampleBuilder.patchSize,
            outChannels: 3
        )
        super.init()
    }

    func patchEmbeddings(_ patches: MLXArray) -> MLXArray {
        patchEmbedder(patches)
    }

    func timestepEmbeddings(_ timesteps: MLXArray) -> MLXArray {
        timestepEmbedder(timesteps)
    }

    func pixelPredictions(_ hiddenStates: MLXArray) -> MLXArray {
        finalLayer(hiddenStates)
    }
}

enum HiDreamO1WeightMapper {
    static func mapPixelHeadKey(_ key: String, value: MLXArray) -> [(String, MLXArray)] {
        let prefixes = [
            "model.t_embedder1.": "t_embedder1.",
            "t_embedder1.": "t_embedder1.",
            "model.x_embedder.": "x_embedder.",
            "x_embedder.": "x_embedder.",
            "model.final_layer2.": "final_layer2.",
            "final_layer2.": "final_layer2.",
        ]
        for (source, target) in prefixes where key.hasPrefix(source) {
            let suffix = key.dropFirst(source.count)
            return [("\(target)\(suffix)", value)]
        }
        return []
    }

    static func loadPixelHead(
        from resources: HiDreamO1Resources,
        config: HiDreamO1Config,
        dtype: DType? = .bfloat16
    ) throws -> HiDreamO1PixelHead {
        let head = HiDreamO1PixelHead(config: config)
        if FileManager.default.fileExists(atPath: resources.weightsIndexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: resources.weightsIndexURL,
                to: head,
                dtype: dtype,
                verify: .none,
                mapper: mapPixelHeadKey
            )
        } else {
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.singleWeightsURL,
                to: head,
                dtype: dtype,
                verify: .none,
                mapper: mapPixelHeadKey
            )
        }
        return head
    }
}
