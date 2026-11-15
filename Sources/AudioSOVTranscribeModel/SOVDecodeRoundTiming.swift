/// Opt-in host-stage measurements at the decoder's existing synchronization boundaries.
public struct SOVDecodeRoundTiming: Codable, Sendable {
    public var round: Int = 0
    public var proposed: Int = 0
    public var startUptime: Double = 0
    public var endUptime: Double = 0
    public var wallSeconds: Double = 0
    public var draftGraphSeconds: Double = 0
    public var draftMaterializeSeconds: Double = 0
    public var verificationGraphSeconds: Double = 0
    public var predictionMaterializeSeconds: Double = 0
    public var commitSeconds: Double = 0
    public var tailEvaluateSeconds: Double = 0
    public var accepted: Int = 0
}
