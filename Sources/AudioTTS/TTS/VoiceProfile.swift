import Foundation

public struct VoiceProfile: Sendable, Hashable, Codable, Identifiable {
    public let id: UUID
    public var name: String
    public var createdAt: Date
    public var updatedAt: Date
    public var transcript: String
    public var language: String?
    public var referenceAudioRelativePath: String
    public var modelFingerprint: String?

    public init(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        transcript: String,
        language: String? = nil,
        referenceAudioRelativePath: String,
        modelFingerprint: String? = nil
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.transcript = transcript
        self.language = language
        self.referenceAudioRelativePath = referenceAudioRelativePath
        self.modelFingerprint = modelFingerprint
    }
}
