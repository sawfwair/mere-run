import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

public actor LTXUnifiedAVGenerator {
    var textEncoder: LTXGemmaTextEncoder?
    var transformer: (any LTXUnifiedAVTransformerRuntime)?
    var audioOnlyTransformer: LTXAudioOnlyTransformerV2?
    var decoder: LTXVideoDecoder?
    var diffusionDecoder: LTXDiffusionVideoDecoder?
    var encoder: LTXVideoEncoder?
    var upsampler: LTXLatentUpsampler?
    var temporalUpsampler: LTXTemporalLatentUpsampler?
    var audioDecoder: LTXAudioDecoder?
    var vocoder: LTXAudioVocoderBase?
    var modelWeightsURL: URL?
    var videoEncoderWeightsURL: URL?
    var videoVAEWeightLayout: LTXTensorWeightLayout = .pytorch
    var videoVAEArchitecture: LTXVideoVAEArchitecture = .legacy
    var loadedDType: DType = .bfloat16
    var loadedRoot: URL?
    var audioVAEWeightsURL: URL?
    var distilledLoRAURL: URL?
    var runtimeLoRAAdapter: LTXRuntimeLoRAAdapter?
    var runtimeUserLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    var runtimeDetailingLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    var runtimeAudioLoRAAdapters: [LTXRuntimeLoRAAdapter] = []
    var loadedForAudioToVideo = false
    var loadedForFullTwoStage = false
    var loadedForReusableFullTwoStage = false
    var loadedForVideoOnlyOutput = false
    var loadedForLTX25 = false
    var loadedForDFR = false
    var twoStageGenerationConsumed = false
    var promptEmbeddingCache = LTXPromptEmbeddingCache()

    public init() {}

    public func configurePromptCache(capacity: Int) {
        promptEmbeddingCache = LTXPromptEmbeddingCache(capacity: capacity)
    }

    public func clearPromptCache() {
        promptEmbeddingCache.removeAll()
    }

    public func promptCacheStatistics() -> LTXPromptCacheStatistics {
        promptEmbeddingCache.statistics()
    }

    public func unload() async {
        runtimeLoRAAdapter?.setActive(false)
        runtimeUserLoRAAdapters.forEach { $0.setActive(false) }
        runtimeDetailingLoRAAdapters.forEach { $0.setActive(false) }
        runtimeAudioLoRAAdapters.forEach { $0.setActive(false) }
        if let textEncoder {
            await textEncoder.unload()
        }
        textEncoder = nil
        transformer = nil
        audioOnlyTransformer = nil
        decoder = nil
        diffusionDecoder = nil
        encoder = nil
        upsampler = nil
        temporalUpsampler = nil
        audioDecoder = nil
        vocoder = nil
        modelWeightsURL = nil
        videoEncoderWeightsURL = nil
        videoVAEWeightLayout = .pytorch
        videoVAEArchitecture = .legacy
        loadedRoot = nil
        audioVAEWeightsURL = nil
        distilledLoRAURL = nil
        runtimeLoRAAdapter = nil
        runtimeUserLoRAAdapters = []
        runtimeDetailingLoRAAdapters = []
        runtimeAudioLoRAAdapters = []
        loadedForAudioToVideo = false
        loadedForFullTwoStage = false
        loadedForReusableFullTwoStage = false
        loadedForVideoOnlyOutput = false
        loadedForLTX25 = false
        loadedForDFR = false
        twoStageGenerationConsumed = false
        promptEmbeddingCache.removeAll(keepingCapacity: false)
        Memory.clearCache()
    }
}
