import Foundation

public enum ACEStepInferenceMethod: String, Sendable, Hashable {
    case ode
    case sde
}

public enum ACEStepDCWMode: String, Sendable, Hashable {
    case low
    case high
    case double
    case pix
}

public struct ACEStepInferenceConfig: Sendable, Hashable {
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

    /// ODE is deterministic and matches the default turbo inference path.
    public var inferMethod: ACEStepInferenceMethod

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
        inferMethod: ACEStepInferenceMethod = .ode,
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
        self.inferMethod = inferMethod
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
