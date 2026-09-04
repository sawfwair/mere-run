import Foundation
import XCTest
@testable import AudioSTT

final class ParakeetPackagingTests: XCTestCase {
    func testLoadsCoreMLHybridPackagingMarker() throws {
        let config = try loadConfig(format: "coreml-hybrid-v1")

        XCTAssertEqual(config.packaging, .coreMLHybrid)
    }

    func testDefaultsExistingCheckpointsToCompleteMLX() throws {
        let config = try loadConfig(format: nil)

        XCTAssertEqual(config.packaging, .completeMLX)
    }

    func testRejectsUnknownPackagingMarker() throws {
        XCTAssertThrowsError(try loadConfig(format: "future-format")) { error in
            guard case ParakeetConfigError.unsupportedPackaging("future-format") = error else {
                return XCTFail("Expected unsupportedPackaging, found \(error)")
            }
        }
    }

    func testFactoryCanOmitTheMLXEncoder() throws {
        let config = try loadConfig(format: "coreml-hybrid-v1")
        let model = ParakeetModelFactory.build(
            config: config,
            includeMLXEncoder: false
        )

        let baseModel = try XCTUnwrap(model as? ParakeetBaseModel)
        XCTAssertNil(baseModel.encoder)
        XCTAssertFalse(baseModel.parameters().flattened().contains { key, _ in
            key.hasPrefix("encoder.")
        })
    }

    func testCoreMLProviderFindsBundledModelRoot() throws {
        let directory = try makeModelRoot()
        for filename in ["config.json", "model.safetensors", "vocab.txt"] {
            XCTAssertTrue(
                FileManager.default.createFile(
                    atPath: directory.appendingPathComponent(filename).path,
                    contents: Data()
                )
            )
        }

        XCTAssertEqual(
            ParakeetExecutionProvider.coreML(artifactURL: directory).bundledModelURL,
            directory.standardizedFileURL
        )
    }

    func testEncoderOnlyCoreMLProviderHasNoBundledModelRoot() throws {
        let directory = try makeModelRoot()

        XCTAssertNil(
            ParakeetExecutionProvider.coreML(artifactURL: directory).bundledModelURL
        )
    }

    private func loadConfig(format: String?) throws -> ParakeetModelConfig {
        let directory = try makeModelRoot()
        let mere = format.map { #", "mere": {"format": "\#($0)"}"# } ?? ""
        let json = """
        {
          "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
          "model_defaults": {"tdt_durations": [0, 1, 2, 3, 4]},
          "preprocessor": {},
          "encoder": {}
          \(mere)
        }
        """
        let url = directory.appendingPathComponent("config.json")
        try Data(json.utf8).write(to: url)
        return try ParakeetModelConfig.load(from: url)
    }

    private func makeModelRoot() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "parakeet-packaging-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        return directory
    }
}
