import XCTest
@testable import MereRunCore
import AudioTTS

final class VoiceProfileStoreTests: XCTestCase {
    func testCreateListDeleteProfile() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let audioURL = root.appendingPathComponent("ref.wav")
        try Data("fake-audio".utf8).write(to: audioURL)

        let store = VoiceProfileStore(baseDirectory: root)
        let created = try await store.createProfile(
            name: "Narrator",
            referenceAudioURL: audioURL,
            transcript: "Hello world",
            language: "en"
        )

        let listed = try await store.listProfiles()
        XCTAssertEqual(listed.count, 1)
        XCTAssertEqual(listed.first?.id, created.id)
        XCTAssertEqual(listed.first?.name, "Narrator")
        XCTAssertEqual(listed.first?.transcript, "Hello world")

        try await store.deleteProfile(id: created.id)
        let remaining = try await store.listProfiles()
        XCTAssertTrue(remaining.isEmpty)
    }

    func testManifestWrittenAtomically() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let audioURL = root.appendingPathComponent("ref.wav")
        try Data("fake-audio".utf8).write(to: audioURL)

        let store = VoiceProfileStore(baseDirectory: root)
        _ = try await store.createProfile(
            name: "Atomic",
            referenceAudioURL: audioURL,
            transcript: "Atomic manifest check"
        )

        let manifestURL = root
            .appendingPathComponent("voices", isDirectory: true)
            .appendingPathComponent("voice_profiles.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))

        let data = try Data(contentsOf: manifestURL)
        let decoded = try JSONDecoder().decode([VoiceProfile].self, from: data)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded.first?.name, "Atomic")
    }

    func testCacheInvalidationRemovesCachedArtifactsWhenFingerprintChanges() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let audioURL = root.appendingPathComponent("ref.wav")
        try Data("fake-audio".utf8).write(to: audioURL)

        let store = VoiceProfileStore(baseDirectory: root)
        let profile = try await store.createProfile(
            name: "Cacheable",
            referenceAudioURL: audioURL,
            transcript: "cache me",
            modelFingerprint: "model-v1"
        )

        let speakerURL = await store.speakerEmbeddingCacheURL(for: profile.id)
        let codesURL = await store.referenceCodesCacheURL(for: profile.id)
        try Data("speaker".utf8).write(to: speakerURL)
        try Data("codes".utf8).write(to: codesURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: speakerURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: codesURL.path))

        let invalidated = try await store.invalidateCachesIfNeeded(profileID: profile.id, modelFingerprint: "model-v2")
        XCTAssertNotNil(invalidated)
        XCTAssertEqual(invalidated?.modelFingerprint, "model-v2")
        XCTAssertFalse(FileManager.default.fileExists(atPath: speakerURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: codesURL.path))
    }
}
