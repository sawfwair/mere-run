import Foundation
import XCTest
@testable import MereRunCore

// Use XCTestCase directly: installation validation must not need MLX test setup.
final class MiniMaxH3CacheArtifactTests: XCTestCase {
    private let configuration = MiniMaxH3TransformerConfiguration(
        hiddenSize: 4,
        layerCount: 1,
        timeEmbeddingDimension: 2
    )

    func testPinnedCacheValidationUsesFileIOAndFollowsManagedSymlinks() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "cache.safetensors")
        let pin = try writeCache(to: url)
        let managed = root.appending(path: "managed", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: managed, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: managed.appending(path: pin.filename),
            withDestinationURL: url
        )

        try validate(in: managed, pin: pin)

        // Change a payload byte without changing the file size or header.
        let handle = try FileHandle(forWritingTo: url)
        try handle.seek(toOffset: UInt64(pin.byteCount - 1))
        try handle.write(contentsOf: Data([1]))
        try handle.close()
        XCTAssertThrowsError(try validate(in: managed, pin: pin)) { error in
            guard case ModelArtifactVerificationError.checksumMismatch = error else {
                return XCTFail("Expected checksum rejection, got \(error)")
            }
        }
    }

    func testPinnedCacheRejectsIncompatibleMetadataAndGeometry() throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(path: "cache.safetensors")
        for (original, replacement, reason) in [
            ("\"schema_version\": \"2\"", "\"schema_version\": \"1\"", "unsupported schema version"),
            ("test-source", "stale-source", "transformer artifact changed"),
            ("[4, 3, 2]", "[4, 3, 4]", "source tensor geometry does not match"),
            ("blocks.0.modulations", "blocks.1.modulations", "source tensor geometry does not match"),
            ("\"shape\": [5]", "\"shape\": [4]", "source tensor geometry does not match"),
        ] {
            let pin = try writeCache(to: url, replacing: original, with: replacement)
            XCTAssertThrowsError(try validate(in: root, pin: pin)) { error in
                XCTAssertTrue(error.localizedDescription.contains(reason), "\(error)")
            }
        }
    }

    private func validate(in root: URL, pin: ModelArtifactPin) throws {
        try MiniMaxH3AdaLNCache.validatePinnedArtifact(
            in: root,
            pin: pin,
            configuration: configuration,
            pointCount: 5,
            sourceIdentity: "test-source"
        )
    }

    /// Build a tiny safetensors fixture entirely with Foundation, without MLX.save.
    private func writeCache(
        to url: URL,
        replacing original: String = "",
        with replacement: String = ""
    ) throws -> ModelArtifactPin {
        var header = """
        {
          "__metadata__": {"schema_version": "2", "source_identity": "test-source"},
          "video_sigmas": {"dtype": "F32", "shape": [5], "data_offsets": [0, 20]},
          "audio_sigmas": {"dtype": "F32", "shape": [5], "data_offsets": [20, 40]},
          "time_embeddings": {"dtype": "F32", "shape": [4, 3, 2], "data_offsets": [40, 136]},
          "final_modulations": {"dtype": "F32", "shape": [4, 3, 8], "data_offsets": [136, 520]},
          "blocks.0.modulations": {"dtype": "F32", "shape": [4, 9, 24], "data_offsets": [520, 3976]}
        }
        """
        if !original.isEmpty {
            header = header.replacingOccurrences(of: original, with: replacement)
        }
        var length = UInt64(header.utf8.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(contentsOf: header.utf8)
        data.append(Data(count: 3976))
        try data.write(to: url)
        return ModelArtifactPin(
            filename: url.lastPathComponent,
            byteCount: Int64(data.count),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }
}
