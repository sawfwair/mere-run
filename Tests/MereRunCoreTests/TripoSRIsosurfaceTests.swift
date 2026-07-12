import Foundation
@testable import MereRunCore
import XCTest

final class TripoSRIsosurfaceTests: MereRunCoreTestCase {
    func testPublicConfigurationRetainsInvalidRuntimeValuesWithoutTrapping() {
        let configuration = TripoSRMeshExtractionConfiguration(
            resolution: Int.max,
            densityThreshold: .infinity
        )
        XCTAssertEqual(configuration.resolution, Int.max)
        XCTAssertEqual(configuration.densityThreshold, .infinity)
    }

    func testMarchingTetrahedraExtractsIndexedOutwardSphere() throws {
        let resolution = 14
        let radius: Float = 1
        let surfaceRadius: Float = 0.58
        var density = [Float]()
        density.reserveCapacity(resolution * resolution * resolution)
        for x in 0..<resolution {
            for y in 0..<resolution {
                for z in 0..<resolution {
                    let px = Float(x) / Float(resolution - 1) * 2 - 1
                    let py = Float(y) / Float(resolution - 1) * 2 - 1
                    let pz = Float(z) / Float(resolution - 1) * 2 - 1
                    density.append(surfaceRadius - sqrt(px * px + py * py + pz * pz))
                }
            }
        }
        let mesh = try TripoSRIsosurfaceExtractor.polygonize(
            density: density,
            resolution: resolution,
            radius: radius,
            threshold: 0,
            inferredUnseenGeometry: true
        )
        XCTAssertGreaterThan(mesh.vertexCount, 100)
        XCTAssertGreaterThan(mesh.triangleCount, 100)
        XCTAssertLessThan(mesh.vertexCount, mesh.indices.count)
        XCTAssertTrue(mesh.inferredUnseenGeometry)
        XCTAssertEqual(mesh.units, .normalizedObjectSpace)
        for axis in 0..<3 {
            XCTAssertEqual(abs(mesh.bounds.minimum[axis]), surfaceRadius, accuracy: 0.08)
            XCTAssertEqual(mesh.bounds.maximum[axis], surfaceRadius, accuracy: 0.08)
        }
        guard let normals = mesh.normals else {
            return XCTFail("polygonizer should generate normals")
        }
        var outward = 0
        for vertex in 0..<mesh.vertexCount {
            let base = vertex * 3
            let dot = mesh.vertices[base] * normals[base]
                + mesh.vertices[base + 1] * normals[base + 1]
                + mesh.vertices[base + 2] * normals[base + 2]
            if dot > 0 { outward += 1 }
        }
        XCTAssertGreaterThan(Float(outward) / Float(mesh.vertexCount), 0.9)
    }

    func testUniformFieldProducesExplicitEmptySurfaceError() {
        XCTAssertThrowsError(
            try TripoSRIsosurfaceExtractor.polygonize(
                density: [Float](repeating: 1, count: 4 * 4 * 4),
                resolution: 4,
                radius: 1,
                threshold: 0,
                inferredUnseenGeometry: true
            )
        ) { error in
            XCTAssertEqual(
                error as? TripoSRIsosurfaceError,
                .emptySurface(resolution: 4, threshold: 0)
            )
        }
    }

    func testAssetExporterWritesCanonicalOBJPLYAndGLBWithPinnedProvenance() throws {
        let mesh = try MeshAsset(
            vertices: [0, 0, 0, 1, 0, 0, 0, 1, 0],
            indices: [0, 1, 2],
            colorsRGBA8: [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255],
            inferredUnseenGeometry: true
        )
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("triposr-export-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: output) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let inputURL = output.appendingPathComponent("object.png")
        try Data("object pixels".utf8).write(to: inputURL)
        let pin = GeometryModelPins.tripoSR
        let checkpoint = TripoSRCheckpoint(
            modelID: pin.modelID,
            repository: pin.repository,
            revision: pin.revision,
            sourceRepository: pin.sourceCodeRepository,
            sourceRevision: pin.sourceCodeRevision,
            license: pin.license,
            format: .pinnedPyTorch,
            rootURL: output,
            weightsURL: output.appendingPathComponent("model.ckpt"),
            configurationURL: output.appendingPathComponent("config.yaml"),
            weightsByteCount: 1,
            weightsSHA256: "runtime-weights-digest",
            sourceSHA256: "source-weights-digest",
            configurationSHA256: "configuration-digest"
        )
        let result = try TripoSRAssetExporter.export(
            mesh: mesh,
            inputURL: inputURL,
            checkpoint: checkpoint,
            outputDirectory: output,
            stem: "object",
            createdAt: Date(timeIntervalSince1970: 0)
        )
        XCTAssertEqual(Set(result.manifest.artifacts.map(\.kind)), Set([.obj, .ply, .glb]))
        XCTAssertEqual(result.manifest.model.modelID, GeometryModelPins.tripoSR.modelID)
        XCTAssertEqual(result.manifest.model.license, "MIT")
        XCTAssertEqual(result.manifest.model.weightsSHA256, checkpoint.weightsSHA256)
        XCTAssertTrue(result.manifest.inferredUnseenGeometry)
        XCTAssertTrue(result.manifest.artifacts.allSatisfy { !$0.sha256.isEmpty && $0.byteCount > 0 })
    }
}
