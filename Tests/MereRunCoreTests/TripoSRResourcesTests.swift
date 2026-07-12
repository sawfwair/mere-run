import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class TripoSRResourcesTests: MereRunCoreTestCase {
    func testCanonicalModelAndConvertedArtifactPinsAreExact() {
        XCTAssertEqual(TripoSRResources.defaultModelID, "image-3d-triposr")
        XCTAssertEqual(
            TripoSRResources.convertedWeightsPin.byteCount,
            1_677_170_936
        )
        XCTAssertEqual(
            TripoSRResources.convertedWeightsPin.sha256,
            "f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b"
        )
        XCTAssertEqual(TripoSRResources.convertedConfigurationPin.byteCount, 378)
        XCTAssertEqual(
            TripoSRResources.convertedSourcePin.sha256,
            "5c12adbc30f80524007d946f78df11da077a0df6ba25b3409e566cda6afb902c"
        )
        XCTAssertEqual(TripoSRResources.convertedLicensePin.byteCount, 1_080)
        XCTAssertEqual(GeometryModelPins.tripoSR.license, "MIT")
    }

    func testRejectsUnknownCheckpointFileType() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("model.bin")
        try Data("not-a-checkpoint".utf8).write(to: file)

        XCTAssertThrowsError(try TripoSRResources.inspectExplicit(file)) { error in
            XCTAssertEqual(
                error as? TripoSRResourceError,
                .unsupportedCheckpointPath(file.path)
            )
        }
    }

    func testConvertedPackageRejectsConfigurationDriftBeforeDecodeOrWeights() throws {
        let root = try convertedPackageFixture(conditioningImageSize: 256)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try TripoSRResources.inspectExplicit(root)) { error in
            guard case .invalidConvertedPackage(let detail) = error as? TripoSRResourceError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("config.json checksum mismatch"), detail)
        }
    }

    func testConvertedPackageRejectsOversizedSourceManifestBeforeDecode() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("fixture".utf8).write(to: root.appendingPathComponent("model.safetensors"))
        try Data(repeating: 0x7B, count: Int(TripoSRResources.convertedSourcePin.byteCount + 1))
            .write(to: root.appendingPathComponent("SOURCE.json"))
        try Data(repeating: 0, count: Int(TripoSRResources.convertedConfigurationPin.byteCount))
            .write(to: root.appendingPathComponent("config.json"))
        try Data(repeating: 0, count: Int(TripoSRResources.convertedLicensePin.byteCount))
            .write(to: root.appendingPathComponent("LICENSE"))

        XCTAssertThrowsError(try TripoSRResources.inspectExplicit(root)) { error in
            guard case .invalidConvertedPackage(let detail) = error as? TripoSRResourceError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("SOURCE.json has 856 bytes"), detail)
        }
    }

    func testInspectsExactConvertedPackageWhenExplicitlyEnabled() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_TRIPOSR_CONVERTED"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path))

        let checkpoint = try TripoSRResources.inspectExplicit(URL(fileURLWithPath: path))
        XCTAssertEqual(checkpoint.modelID, ModelResolver.ModelID.image3DTripoSR.rawValue)
        XCTAssertEqual(checkpoint.format, .convertedSafetensors)
        XCTAssertEqual(checkpoint.weightsSHA256, TripoSRWeights.convertedSafetensorsSHA256)
        XCTAssertEqual(
            checkpoint.sourceSHA256,
            GeometryModelPins.tripoSR.artifacts.first { $0.filename == "model.ckpt" }?.sha256
        )
    }

    func testGeneratorRejectsMissingInputBeforeResolvingModel() async {
        let generator = TripoSRGenerator()
        let missing = URL(fileURLWithPath: "/tmp/mere-run-missing-\(UUID().uuidString).png")
        do {
            _ = try await generator.generate(
                imageURL: missing,
                outputDirectory: URL(fileURLWithPath: "/tmp/output")
            )
            XCTFail("Expected missing input failure")
        } catch {
            XCTAssertEqual(
                error as? TripoSRGeneratorError,
                .inputImageNotFound(missing.path)
            )
        }
    }

    func testProgressMessagesDescribeNativeProductionStages() {
        XCTAssertTrue(TripoSRProgress.loadingModel.message.contains("native MLX"))
        XCTAssertTrue(TripoSRProgress.extractingMesh.message.contains("mesh"))
        XCTAssertTrue(TripoSRProgress.exportingAssets.message.contains("GLB"))
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposr-resource-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func convertedPackageFixture(conditioningImageSize: Int) throws -> URL {
        let root = try temporaryDirectory()
        try Data("fixture".utf8).write(to: root.appendingPathComponent("model.safetensors"))
        let source = """
        {
          "conversion": {
            "converter": "convert_triposr.py",
            "converterVersion": 1,
            "environment": {
              "python": "3.11.15",
              "safetensors": "0.8.0",
              "torch": "2.13.0"
            },
            "outputByteCount": 1677170936,
            "outputFile": "model.safetensors",
            "outputSHA256": "f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b",
            "scalarCount": 419275628,
            "tensorCount": 549
          },
          "license": "MIT",
          "modelID": "image-3d-triposr",
          "source": {
            "byteCount": 1677246742,
            "filename": "model.ckpt",
            "repository": "stabilityai/TripoSR",
            "revision": "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            "sha256": "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee",
            "sourceCodeRepository": "VAST-AI-Research/TripoSR",
            "sourceCodeRevision": "107cefdc244c39106fa830359024f6a2f1c78871"
          }
        }
        """
        try Data((source + "\n").utf8).write(to: root.appendingPathComponent("SOURCE.json"))
        let configuration = """
        {
          "architecture": "triposr",
          "conditioningImageSize": \(conditioningImageSize),
          "decoderHiddenLayers": 9,
          "decoderHiddenSize": 64,
          "densityBias": -1.0,
          "densityThreshold": 25.0,
          "imageEncoder": "facebook/dino-vitb16",
          "planeChannels": 40,
          "planeSize": 64,
          "rendererRadius": 0.87,
          "transformerAttentionHeads": 16,
          "transformerLayers": 16,
          "transformerTokenChannels": 1024
        }
        """
        try Data((configuration + "\n").utf8).write(to: root.appendingPathComponent("config.json"))
        try Data(repeating: 0, count: 1_080).write(to: root.appendingPathComponent("LICENSE"))
        return root
    }
}
