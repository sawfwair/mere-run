import Foundation

public struct LTXVideoTimingReport: Codable, Hashable, Sendable {
    public let mode: String
    public let modelRoot: String
    public let residentModelReused: Bool
    public let load: LTXLoadTimings
    public let generation: LTXGenerationTimings
    public let unloadSeconds: Double
    public let mp4WriteSeconds: Double
    public let totalSeconds: Double

    public init(
        mode: String,
        modelRoot: String,
        residentModelReused: Bool,
        load: LTXLoadTimings,
        generation: LTXGenerationTimings,
        unloadSeconds: Double,
        mp4WriteSeconds: Double,
        totalSeconds: Double
    ) {
        self.mode = mode
        self.modelRoot = modelRoot
        self.residentModelReused = residentModelReused
        self.load = load
        self.generation = generation
        self.unloadSeconds = unloadSeconds
        self.mp4WriteSeconds = mp4WriteSeconds
        self.totalSeconds = totalSeconds
    }
}
