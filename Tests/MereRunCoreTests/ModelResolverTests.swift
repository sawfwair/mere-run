import Foundation
import XCTest
@testable import MereRunCore

final class ModelResolverTests: MereRunCoreTestCase {

    func testResolvesFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot
            .appendingPathComponent("image-zimage-nano", isDirectory: true)
        try TestFileSystem.createDirectory(modelRoot)

        let manifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))
        try manifest.write(to: modelRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.zetaNano)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testMebotFallsBackToZeroNano() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let nanoRoot = modelsRoot
            .appendingPathComponent("image-klein-nano", isDirectory: true)
        try TestFileSystem.createDirectory(nanoRoot)
        try MereRunModelManifest.template(for: .kleinNano, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: nanoRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.mebot)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, nanoRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testRejectsMismatchedManifestID() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let mismatchedRoot = modelsRoot
            .appendingPathComponent("image-zimage-base", isDirectory: true)
        try TestFileSystem.createDirectory(mismatchedRoot)

        // Wrong manifest ID for the requested directory.
        let wrongManifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))
        try wrongManifest.write(to: mismatchedRoot)

        let resolver = ModelResolver()
        XCTAssertThrowsError(try resolver.resolve(.zetaBase))
    }

    func testResolvesQ35FromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("text-chat-q35", isDirectory: true)
        try TestFileSystem.createDirectory(modelRoot)
        try MereRunModelManifest.template(for: .q35, createdAt: Date(timeIntervalSince1970: 0)).write(to: modelRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.q35)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesVisionSegmentSAM31FromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("vision-segment-sam31", isDirectory: true)
        try TestFileSystem.createDirectory(modelRoot)
        try MereRunModelManifest.template(for: .visionSegmentSAM31, createdAt: Date(timeIntervalSince1970: 0)).write(to: modelRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.visionSegmentSAM31)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }
}
