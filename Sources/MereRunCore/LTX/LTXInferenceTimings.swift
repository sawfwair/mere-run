import Foundation

/// Wall-clock phase timings for loading a native LTX runtime.
public struct LTXLoadTimings: Codable, Hashable, Sendable {
    public let textEncoderSeconds: Double
    public let transformerSeconds: Double
    public let videoDecoderSeconds: Double
    public let upsamplerSeconds: Double
    public let audioDecoderSeconds: Double
    public let loraAdapterSeconds: Double
    public let totalSeconds: Double

    public init(
        textEncoderSeconds: Double = 0,
        transformerSeconds: Double = 0,
        videoDecoderSeconds: Double = 0,
        upsamplerSeconds: Double = 0,
        audioDecoderSeconds: Double = 0,
        loraAdapterSeconds: Double = 0,
        totalSeconds: Double = 0
    ) {
        self.textEncoderSeconds = textEncoderSeconds
        self.transformerSeconds = transformerSeconds
        self.videoDecoderSeconds = videoDecoderSeconds
        self.upsamplerSeconds = upsamplerSeconds
        self.audioDecoderSeconds = audioDecoderSeconds
        self.loraAdapterSeconds = loraAdapterSeconds
        self.totalSeconds = totalSeconds
    }
}

/// Wall-clock phase timings for one native LTX generation on an already-loaded runtime.
public struct LTXGenerationTimings: Codable, Hashable, Sendable {
    public let textEncodingSeconds: Double
    public let promptCacheHits: Int
    public let promptCacheMisses: Int
    public let guidanceProjectionCacheBuildSeconds: Double
    public let guidanceProjectionCacheBuilds: Int
    public let guidanceProjectionCacheReuses: Int
    public let guidanceProjectionCacheFallbacks: Int
    public let teaCacheDecisionSeconds: Double
    public let teaCacheComputedBlockStacks: Int
    public let teaCacheReusedBlockStacks: Int
    public let preparationSeconds: Double
    public let stage1DenoiseSeconds: Double
    public let loraFusionSeconds: Double
    public let upsampleSeconds: Double
    public let stage2DenoiseSeconds: Double
    public let videoDecodeSeconds: Double
    public let audioDecodeSeconds: Double
    public let totalSeconds: Double

    public init(
        textEncodingSeconds: Double = 0,
        promptCacheHits: Int = 0,
        promptCacheMisses: Int = 0,
        guidanceProjectionCacheBuildSeconds: Double = 0,
        guidanceProjectionCacheBuilds: Int = 0,
        guidanceProjectionCacheReuses: Int = 0,
        guidanceProjectionCacheFallbacks: Int = 0,
        teaCacheDecisionSeconds: Double = 0,
        teaCacheComputedBlockStacks: Int = 0,
        teaCacheReusedBlockStacks: Int = 0,
        preparationSeconds: Double = 0,
        stage1DenoiseSeconds: Double = 0,
        loraFusionSeconds: Double = 0,
        upsampleSeconds: Double = 0,
        stage2DenoiseSeconds: Double = 0,
        videoDecodeSeconds: Double = 0,
        audioDecodeSeconds: Double = 0,
        totalSeconds: Double = 0
    ) {
        self.textEncodingSeconds = textEncodingSeconds
        self.promptCacheHits = promptCacheHits
        self.promptCacheMisses = promptCacheMisses
        self.guidanceProjectionCacheBuildSeconds = guidanceProjectionCacheBuildSeconds
        self.guidanceProjectionCacheBuilds = guidanceProjectionCacheBuilds
        self.guidanceProjectionCacheReuses = guidanceProjectionCacheReuses
        self.guidanceProjectionCacheFallbacks = guidanceProjectionCacheFallbacks
        self.teaCacheDecisionSeconds = teaCacheDecisionSeconds
        self.teaCacheComputedBlockStacks = teaCacheComputedBlockStacks
        self.teaCacheReusedBlockStacks = teaCacheReusedBlockStacks
        self.preparationSeconds = preparationSeconds
        self.stage1DenoiseSeconds = stage1DenoiseSeconds
        self.loraFusionSeconds = loraFusionSeconds
        self.upsampleSeconds = upsampleSeconds
        self.stage2DenoiseSeconds = stage2DenoiseSeconds
        self.videoDecodeSeconds = videoDecodeSeconds
        self.audioDecodeSeconds = audioDecodeSeconds
        self.totalSeconds = totalSeconds
    }
}

@inline(__always)
func ltxMonotonicSeconds() -> Double {
    ProcessInfo.processInfo.systemUptime
}
