import Foundation

/// Wall-clock phase timings for loading a native LTX runtime.
public struct LTXLoadTimings: Codable, Hashable, Sendable {
    public let textEncoderSeconds: Double
    public let transformerSeconds: Double
    public let videoDecoderSeconds: Double
    public let upsamplerSeconds: Double
    public let audioDecoderSeconds: Double
    public let totalSeconds: Double

    public init(
        textEncoderSeconds: Double = 0,
        transformerSeconds: Double = 0,
        videoDecoderSeconds: Double = 0,
        upsamplerSeconds: Double = 0,
        audioDecoderSeconds: Double = 0,
        totalSeconds: Double = 0
    ) {
        self.textEncoderSeconds = textEncoderSeconds
        self.transformerSeconds = transformerSeconds
        self.videoDecoderSeconds = videoDecoderSeconds
        self.upsamplerSeconds = upsamplerSeconds
        self.audioDecoderSeconds = audioDecoderSeconds
        self.totalSeconds = totalSeconds
    }
}

/// Wall-clock phase timings for one native LTX generation on an already-loaded runtime.
public struct LTXGenerationTimings: Codable, Hashable, Sendable {
    public let textEncodingSeconds: Double
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
