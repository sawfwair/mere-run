import Foundation

public struct SOVTranscribeOptions: Sendable, Hashable, Codable {
    /// Natural-language instruction; nil uses the checkpoint's verbatim prompt.
    public var instruction: String?
    /// Explicit companion ID or local draft directory. Nil uses target-only decoding.
    public var draftModel: String?
    /// Opt-in affine-Q8 target pack. Created by streaming conversion if absent, reused if valid.
    public var packedWeights: String?

    public init(instruction: String? = nil, draftModel: String? = nil, packedWeights: String? = nil) {
        self.instruction = instruction
        self.draftModel = draftModel
        self.packedWeights = packedWeights
    }
}
