import AudioCore
import Foundation

/// Benchmark-only expert precision, indexed by decoder layer. Original weights
/// are packed once during preparation; attention, routers, and heads are unchanged.
public struct SOVExpertRecipe: Codable, Sendable {
    public let schemaVersion: Int
    public let groupSize: Int
    public let layerBits: [Int]

    public init(groupSize: Int = 64, layerBits: [Int]) {
        schemaVersion = 1
        self.groupSize = groupSize
        self.layerBits = layerBits
    }

    public func validate(layerCount: Int) throws {
        guard schemaVersion == 1, [32, 64].contains(groupSize), layerBits.count == layerCount,
              layerBits.allSatisfy({ [4, 6, 8].contains($0) }) else {
            throw SpeechTranscriptionIssue("invalid_expert_recipe", "Expert recipes require schema 1, group size 32 or 64, and one Q4/Q6/Q8 precision per decoder layer.")
        }
    }
}
