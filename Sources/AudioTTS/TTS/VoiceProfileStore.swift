import Foundation
import MereRunCore

public actor VoiceProfileStore {
    public enum Error: LocalizedError {
        case profileNotFound(UUID)
        case duplicateName(String)
        case invalidName
        case invalidTranscript

        public var errorDescription: String? {
            switch self {
            case .profileNotFound(let id):
                return "Voice profile not found: \(id.uuidString)"
            case .duplicateName(let name):
                return "A voice profile named '\(name)' already exists"
            case .invalidName:
                return "Profile name cannot be empty"
            case .invalidTranscript:
                return "Reference transcript cannot be empty"
            }
        }
    }

    private let fileManager: FileManager
    private let voicesDir: URL
    private let manifestURL: URL

    public init(baseDirectory: URL? = nil, fileManager: FileManager = .default) {
        self.fileManager = fileManager
        self.voicesDir = (baseDirectory ?? MereRunModelPaths.applicationSupportBase)
            .appendingPathComponent("voices", isDirectory: true)
        self.manifestURL = voicesDir.appendingPathComponent("voice_profiles.json")
    }

    public func listProfiles() throws -> [VoiceProfile] {
        try loadManifest().sorted { lhs, rhs in
            lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
    }

    public func profile(id: UUID) throws -> VoiceProfile? {
        try loadManifest().first(where: { $0.id == id })
    }

    public func profile(matching idOrName: String) throws -> VoiceProfile? {
        let profiles = try loadManifest()
        if let id = UUID(uuidString: idOrName) {
            return profiles.first(where: { $0.id == id })
        }
        return profiles.first(where: { $0.name.compare(idOrName, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame })
    }

    @discardableResult
    public func createProfile(
        name: String,
        referenceAudioURL: URL,
        transcript: String,
        language: String? = nil,
        modelFingerprint: String? = nil
    ) throws -> VoiceProfile {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { throw Error.invalidName }

        let trimmedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTranscript.isEmpty else { throw Error.invalidTranscript }

        try ensureDirectories()

        var profiles = try loadManifest()
        if profiles.contains(where: { $0.name.compare(trimmedName, options: [.caseInsensitive, .diacriticInsensitive]) == .orderedSame }) {
            throw Error.duplicateName(trimmedName)
        }

        let id = UUID()
        let profileDir = voicesDir.appendingPathComponent(id.uuidString, isDirectory: true)
        try fileManager.createDirectory(at: profileDir, withIntermediateDirectories: true)

        let ext = referenceAudioURL.pathExtension.isEmpty ? "wav" : referenceAudioURL.pathExtension.lowercased()
        let referenceAudioFilename = "reference.\(ext)"
        let referenceAudioTargetURL = profileDir.appendingPathComponent(referenceAudioFilename)

        if fileManager.fileExists(atPath: referenceAudioTargetURL.path) {
            try fileManager.removeItem(at: referenceAudioTargetURL)
        }
        try fileManager.copyItem(at: referenceAudioURL, to: referenceAudioTargetURL)

        let referenceTextURL = profileDir.appendingPathComponent("reference.txt")
        try trimmedTranscript.write(to: referenceTextURL, atomically: true, encoding: .utf8)

        let relativePath = "\(id.uuidString)/\(referenceAudioFilename)"
        let now = Date()
        let profile = VoiceProfile(
            id: id,
            name: trimmedName,
            createdAt: now,
            updatedAt: now,
            transcript: trimmedTranscript,
            language: language,
            referenceAudioRelativePath: relativePath,
            modelFingerprint: modelFingerprint
        )
        profiles.append(profile)
        try saveManifest(profiles)
        return profile
    }

    public func deleteProfile(id: UUID) throws {
        var profiles = try loadManifest()
        guard let idx = profiles.firstIndex(where: { $0.id == id }) else {
            throw Error.profileNotFound(id)
        }

        profiles.remove(at: idx)
        try saveManifest(profiles)

        let profileDir = voicesDir.appendingPathComponent(id.uuidString, isDirectory: true)
        if fileManager.fileExists(atPath: profileDir.path) {
            try fileManager.removeItem(at: profileDir)
        }
    }

    @discardableResult
    public func invalidateCachesIfNeeded(
        profileID: UUID,
        modelFingerprint: String
    ) throws -> VoiceProfile? {
        var profiles = try loadManifest()
        guard let idx = profiles.firstIndex(where: { $0.id == profileID }) else {
            return nil
        }

        var profile = profiles[idx]
        if profile.modelFingerprint == modelFingerprint {
            return profile
        }

        let speakerURL = speakerEmbeddingCacheURL(for: profileID)
        let codesURL = referenceCodesCacheURL(for: profileID)
        if fileManager.fileExists(atPath: speakerURL.path) {
            try? fileManager.removeItem(at: speakerURL)
        }
        if fileManager.fileExists(atPath: codesURL.path) {
            try? fileManager.removeItem(at: codesURL)
        }

        profile.modelFingerprint = modelFingerprint
        profile.updatedAt = Date()
        profiles[idx] = profile
        try saveManifest(profiles)
        return profile
    }

    public func referenceAudioURL(for profile: VoiceProfile) -> URL {
        voicesDir.appendingPathComponent(profile.referenceAudioRelativePath)
    }

    public func referenceTextURL(for profileID: UUID) -> URL {
        voicesDir
            .appendingPathComponent(profileID.uuidString, isDirectory: true)
            .appendingPathComponent("reference.txt")
    }

    public func speakerEmbeddingCacheURL(for profileID: UUID) -> URL {
        voicesDir
            .appendingPathComponent(profileID.uuidString, isDirectory: true)
            .appendingPathComponent("speaker_embed.safetensors")
    }

    public func referenceCodesCacheURL(for profileID: UUID) -> URL {
        voicesDir
            .appendingPathComponent(profileID.uuidString, isDirectory: true)
            .appendingPathComponent("reference_codes.safetensors")
    }

    private func ensureDirectories() throws {
        try fileManager.createDirectory(at: voicesDir, withIntermediateDirectories: true)
    }

    private func loadManifest() throws -> [VoiceProfile] {
        try ensureDirectories()
        guard fileManager.fileExists(atPath: manifestURL.path) else { return [] }
        let data = try Data(contentsOf: manifestURL)
        if data.isEmpty { return [] }
        return try JSONDecoder().decode([VoiceProfile].self, from: data)
    }

    private func saveManifest(_ profiles: [VoiceProfile]) throws {
        try ensureDirectories()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(profiles)
        try data.write(to: manifestURL, options: [.atomic])
    }
}
