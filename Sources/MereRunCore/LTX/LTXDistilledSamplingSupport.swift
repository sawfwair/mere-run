import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func advanceRandomStreamForPythonParityAfterStage1(
    modelWeightsURL: URL,
    upsamplerWeightsURL: URL,
    dtype: DType
) throws {
    let dummyUpsampler = LTXLatentUpsampler(inChannels: 128, midChannels: 1024, numBlocksPerStage: 4)
    try SafetensorsStreamingLoader.applyWeightsStreaming(
        url: upsamplerWeightsURL,
        to: dummyUpsampler,
        dtype: dtype,
        verify: .none,
        include: { _ in true },
        mapper: { key, value in
            mapLTXUpsamplerWeight(key: key, value: value, dtype: dtype)
        },
        batchSize: 24
    )
    MLX.eval(dummyUpsampler)

    _ = modelWeightsURL
}

let STAGE1Sigmas: [Float] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
let STAGE2Sigmas: [Float] = [0.909375, 0.725, 0.421875, 0.0]

struct LTXLatentConditioningState {
    var latent: MLXArray
    var cleanLatent: MLXArray
    var denoiseMask: MLXArray
}
