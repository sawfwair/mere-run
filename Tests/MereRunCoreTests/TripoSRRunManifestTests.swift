import Foundation
@testable import MereRunCore
import XCTest

final class TripoSRRunManifestTests: XCTestCase {
    func testDurableManifestRecordsCheckpointPreprocessingExtractionAndEveryMeshHash() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposr-run-manifest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let inputURL = root.appendingPathComponent("chair.png")
        let inputData = Data("temporary API upload bytes".utf8)
        try inputData.write(to: inputURL)
        let checkpoint = TripoSRCheckpoint(
            modelID: ModelResolver.ModelID.image3DTripoSR.rawValue,
            repository: "stabilityai/TripoSR",
            revision: "5b521936b01fbe1890f6f9baed0254ab6351c04a",
            sourceRepository: "VAST-AI-Research/TripoSR",
            sourceRevision: "107cefdc244c39106fa830359024f6a2f1c78871",
            license: "MIT",
            format: .convertedSafetensors,
            rootURL: URL(fileURLWithPath: "/tmp/triposr-native"),
            weightsURL: URL(fileURLWithPath: "/tmp/triposr-native/model.safetensors"),
            configurationURL: URL(fileURLWithPath: "/tmp/triposr-native/config.json"),
            weightsByteCount: TripoSRWeights.convertedSafetensorsByteCount,
            weightsSHA256: TripoSRWeights.convertedSafetensorsSHA256,
            sourceSHA256: GeometryModelPins.tripoSR.artifacts[1].sha256,
            configurationSHA256: TripoSRResources.convertedConfigurationPin.sha256
        )
        let meshExport = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: root,
            stem: "chair",
            createdAt: Date(timeIntervalSince1970: 0)
        )

        let result = try TripoSRRunManifestExporter.export(
            meshExport: meshExport,
            checkpoint: checkpoint,
            inputURL: inputURL,
            sourceWidth: 640,
            sourceHeight: 480,
            preparedWidth: 512,
            preparedHeight: 512,
            foregroundPolicy: "automatic-transparent-alpha",
            foregroundRatio: 0.85,
            croppedTransparentForeground: true,
            extractionResolution: 256,
            densityThreshold: 25,
            includesVertexColors: true
        )

        XCTAssertEqual(result.manifest.schemaVersion, 1)
        XCTAssertEqual(result.manifest.checkpoint.format, .convertedSafetensors)
        XCTAssertEqual(result.manifest.checkpoint.sourceSHA256, checkpoint.sourceSHA256)
        XCTAssertEqual(meshExport.manifest.model.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertNotEqual(meshExport.manifest.model.weightsSHA256, checkpoint.sourceSHA256)
        XCTAssertEqual(meshExport.manifest.inputs?.first?.byteCount, Int64(inputData.count))
        XCTAssertEqual(meshExport.manifest.inputs?.first?.sha256, try ModelArtifactPin.fileSHA256(inputURL))
        XCTAssertEqual(result.manifest.input.byteCount, Int64(inputData.count))
        XCTAssertEqual(result.manifest.input.sha256, try ModelArtifactPin.fileSHA256(inputURL))
        XCTAssertEqual(result.manifest.input.foregroundPolicy, "automatic-transparent-alpha")
        XCTAssertEqual(result.manifest.input.foregroundRatio, 0.85)
        XCTAssertTrue(result.manifest.input.croppedTransparentForeground)
        XCTAssertEqual(result.manifest.extraction.resolution, 256)
        XCTAssertEqual(result.manifest.extraction.densityThreshold, 25)
        XCTAssertTrue(result.manifest.extraction.includesVertexColors)
        XCTAssertEqual(
            result.manifest.extraction.algorithm,
            TripoSRRunManifestExporter.extractionAlgorithm
        )
        XCTAssertTrue(result.manifest.extraction.topologyCompatibility.contains("torchmcubes"))
        XCTAssertTrue(result.manifest.mesh.inferredUnseenGeometry)
        XCTAssertEqual(result.manifest.mesh.vertexCount, 3)
        XCTAssertEqual(result.manifest.mesh.triangleCount, 1)
        XCTAssertEqual(
            Set(result.manifest.artifacts.map(\.kind)),
            ["obj", "ply", "glb", "mesh-manifest"]
        )
        XCTAssertTrue(result.manifest.artifacts.allSatisfy {
            $0.byteCount > 0 && $0.sha256.count == 64
        })
        XCTAssertTrue(result.manifestURL.lastPathComponent.hasSuffix("-run-manifest.json"))

        try FileManager.default.removeItem(at: inputURL)

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(
            TripoSRRunManifest.self,
            from: Data(contentsOf: result.manifestURL)
        )
        XCTAssertEqual(decoded, result.manifest)
        XCTAssertEqual(decoded.input.byteCount, Int64(inputData.count))
        XCTAssertEqual(decoded.input.sha256, result.manifest.input.sha256)
        XCTAssertEqual(try ModelArtifactPin.fileSHA256(result.manifestURL).count, 64)
    }
}
