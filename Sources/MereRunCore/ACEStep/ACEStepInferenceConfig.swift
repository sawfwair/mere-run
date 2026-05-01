import Foundation

public enum ACEStepInferenceMethod: String, Sendable, Hashable {
    case ode
    case sde
}

public struct ACEStepInferenceConfig: Sendable, Hashable {
    /// Seconds of audio to synthesize.
    public var durationSeconds: Float

    /// Turbo "fix_nfe" (default 8).
    public var fixNFE: Int

    /// Turbo shift factor (rounded to {1,2,3} by the scheduler; default 3).
    public var shift: Float

    /// Optional custom timesteps in `[0, 1]` (terminal zeros are ignored).
    public var timesteps: [Float]?

    /// ODE is deterministic and matches the default turbo inference path.
    public var inferMethod: ACEStepInferenceMethod

    public var useTiledVaeDecode: Bool
    public var vaeChunkSize: Int
    public var vaeOverlap: Int

    public var seed: UInt64?

    public init(
        durationSeconds: Float = 10.0,
        fixNFE: Int = 8,
        shift: Float = 3.0,
        timesteps: [Float]? = nil,
        inferMethod: ACEStepInferenceMethod = .ode,
        useTiledVaeDecode: Bool = true,
        vaeChunkSize: Int = 512,
        vaeOverlap: Int = 64,
        seed: UInt64? = nil
    ) {
        self.durationSeconds = durationSeconds
        self.fixNFE = fixNFE
        self.shift = shift
        self.timesteps = timesteps
        self.inferMethod = inferMethod
        self.useTiledVaeDecode = useTiledVaeDecode
        self.vaeChunkSize = vaeChunkSize
        self.vaeOverlap = vaeOverlap
        self.seed = seed
    }
}
