import Foundation
import XCTest
@testable import MereRunCore

final class DepthSequenceArtifactExporterTests: XCTestCase {
    func testWritesTemporallyStableDepthBundleWithHashesAndNoInventedCamera() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("depth-sequence-export-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let frames = [
            try DepthSequenceFrame(
                index: 0,
                timeSeconds: 0,
                width: 2,
                height: 2,
                depth: [1, 2, 3, 4]
            ),
            try DepthSequenceFrame(
                index: 1,
                timeSeconds: 1.0 / 24,
                width: 2,
                height: 2,
                depth: [2, 3, 4, 5]
            ),
        ]
        let result = try DepthSequenceArtifactExporter.export(
            frames: frames,
            inputURL: URL(fileURLWithPath: "/tmp/clip.mov"),
            outputDirectory: root,
            fps: 24,
            semantics: .affineRelative,
            provenance: GeometryModelProvenance(
                modelID: "vision-depth-vda-small",
                upstreamRepository: "depth-anything/Video-Depth-Anything-Small",
                upstreamRevision: "pin",
                license: "Apache-2.0",
                weightsSHA256: String(repeating: "a", count: 64)
            ),
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(result.manifest.frameCount, 2)
        XCTAssertEqual(result.manifest.semantics, .affineRelative)
        XCTAssertFalse(result.manifest.canProjectEveryFrameToPoints)
        XCTAssertEqual(result.manifest.frames[0].artifacts.count, 2)
        XCTAssertTrue(result.manifest.frames.allSatisfy { $0.intrinsics == nil })
        XCTAssertTrue(result.manifest.frames.flatMap(\.artifacts).allSatisfy {
            $0.byteCount > 0 && $0.sha256.count == 64
        })
        XCTAssertFalse(result.manifest.frames.flatMap(\.artifacts).contains { $0.kind == .pointCloud })
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.manifestURL.path))

        let decoded = try JSONDecoder.geometryDecoder.decode(
            DepthSequenceManifest.self,
            from: Data(contentsOf: result.manifestURL)
        )
        XCTAssertEqual(decoded, result.manifest)
    }
}

private extension JSONDecoder {
    static var geometryDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
