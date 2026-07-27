import Foundation

public enum ACEStepInferenceMethod: String, CaseIterable, Codable, Sendable, Hashable {
    case ode
    case sde
}

public enum ACEStepDCWMode: String, CaseIterable, Codable, Sendable, Hashable {
    case low
    case high
    case double
    case pix
}

public enum ACEStepSamplerMode: String, CaseIterable, Codable, Sendable, Hashable {
    case euler
    case heun
}

public enum ACEStepGuidanceMode: String, CaseIterable, Codable, Sendable, Hashable {
    case apg
    case adg
    case cfg
}

public struct ACEStepInferenceConfig: Codable, Sendable, Hashable {
    /// Seconds of audio to synthesize.
    public var durationSeconds: Float

    /// Turbo "fix_nfe" (default 8).
    public var fixNFE: Int

    /// Turbo shift factor. The upstream model API defaults to 1.0; the public
    /// CLI overrides to 3.0 for its default generation path.
    public var shift: Float

    /// Optional custom timesteps in `[0, 1]` (terminal zeros are ignored).
    public var timesteps: [Float]?

    /// Cover noise initialization strength. `0` starts from pure noise; `1`
    /// starts closest to the source latents at the nearest schedule step.
    public var coverNoiseStrength: Float

    /// Optional second deterministic noise source for retakes. Variance uses
    /// upstream spherical interpolation: 0 keeps the main seed, 1 uses only
    /// the retake seed.
    public var retakeSeed: UInt64?
    public var retakeVariance: Float

    /// ODE is deterministic and matches the default turbo inference path.
    public var inferMethod: ACEStepInferenceMethod
    public var samplerMode: ACEStepSamplerMode

    /// Non-Turbo classifier-free guidance. Turbo checkpoints are distilled
    /// and force this to 1 regardless of the configured value.
    public var guidanceScale: Float
    public var guidanceMode: ACEStepGuidanceMode
    public var cfgIntervalStart: Float
    public var cfgIntervalEnd: Float
    public var velocityNormThreshold: Float
    public var velocityEMAFactor: Float

    public var useTiledVaeDecode: Bool
    public var vaeChunkSize: Int
    public var vaeOverlap: Int

    /// Differential Correction in Wavelet domain. Upstream ACE-Step enables
    /// native Haar double-band correction by default.
    public var dcwEnabled: Bool
    public var dcwMode: ACEStepDCWMode
    public var dcwScaler: Float
    public var dcwHighScaler: Float

    public var seed: UInt64?

    public init(
        durationSeconds: Float = 10.0,
        fixNFE: Int = 8,
        shift: Float = 1.0,
        timesteps: [Float]? = nil,
        coverNoiseStrength: Float = 0.0,
        retakeSeed: UInt64? = nil,
        retakeVariance: Float = 0.0,
        inferMethod: ACEStepInferenceMethod = .ode,
        samplerMode: ACEStepSamplerMode = .euler,
        guidanceScale: Float = 7.0,
        guidanceMode: ACEStepGuidanceMode = .apg,
        cfgIntervalStart: Float = 0.0,
        cfgIntervalEnd: Float = 1.0,
        velocityNormThreshold: Float = 0.0,
        velocityEMAFactor: Float = 0.0,
        useTiledVaeDecode: Bool = true,
        vaeChunkSize: Int = 512,
        vaeOverlap: Int = 64,
        dcwEnabled: Bool = true,
        dcwMode: ACEStepDCWMode = .double,
        dcwScaler: Float = 0.05,
        dcwHighScaler: Float = 0.02,
        seed: UInt64? = nil
    ) {
        self.durationSeconds = durationSeconds
        self.fixNFE = fixNFE
        self.shift = shift
        self.timesteps = timesteps
        self.coverNoiseStrength = coverNoiseStrength
        self.retakeSeed = retakeSeed
        self.retakeVariance = retakeVariance
        self.inferMethod = inferMethod
        self.samplerMode = samplerMode
        self.guidanceScale = guidanceScale
        self.guidanceMode = guidanceMode
        self.cfgIntervalStart = cfgIntervalStart
        self.cfgIntervalEnd = cfgIntervalEnd
        self.velocityNormThreshold = velocityNormThreshold
        self.velocityEMAFactor = velocityEMAFactor
        self.useTiledVaeDecode = useTiledVaeDecode
        self.vaeChunkSize = vaeChunkSize
        self.vaeOverlap = vaeOverlap
        self.dcwEnabled = dcwEnabled
        self.dcwMode = dcwMode
        self.dcwScaler = dcwScaler
        self.dcwHighScaler = dcwHighScaler
        self.seed = seed
    }
}
