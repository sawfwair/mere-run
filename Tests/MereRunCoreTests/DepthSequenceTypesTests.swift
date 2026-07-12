import MereRunCore
import XCTest

final class DepthSequenceTypesTests: XCTestCase {
    func testDepthOnlySequenceNeverInventsProjectionCamera() throws {
        let frame = try DepthSequenceFrame(
            index: 0,
            timeSeconds: 0,
            width: 2,
            height: 2,
            depth: [1, 2, 3, 4]
        )
        XCTAssertNil(frame.intrinsics)
        let manifest = DepthSequenceManifest(
            inputPath: "/tmp/shot.mov",
            inputByteCount: 123,
            inputSHA256: String(repeating: "a", count: 64),
            outputDirectory: "/tmp/depth",
            width: 2,
            height: 2,
            fps: 24,
            semantics: .affineRelative,
            model: GeometryModelProvenance(
                modelID: "vision-depth-vda-small",
                upstreamRepository: "depth-anything/Video-Depth-Anything-Small",
                upstreamRevision: "pin",
                license: "Apache-2.0"
            ),
            temporalWindowLength: 32,
            temporalOverlap: 10,
            frames: [DepthSequenceFrameManifest(index: 0, timeSeconds: 0, depthPath: "000000.exr")]
        )
        XCTAssertFalse(manifest.canProjectEveryFrameToPoints)
        XCTAssertEqual(manifest.semantics, .affineRelative)
    }

    func testSequenceAllowsProjectionOnlyWithEveryCamera() {
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: 2,
            imageHeight: 2,
            normalizedFX: 1,
            normalizedFY: 1
        )
        let manifest = DepthSequenceManifest(
            inputPath: "/tmp/shot.mov",
            inputByteCount: 123,
            inputSHA256: String(repeating: "a", count: 64),
            outputDirectory: "/tmp/depth",
            width: 2,
            height: 2,
            fps: 24,
            semantics: .metricMeters,
            model: GeometryModelProvenance(
                modelID: "vision-depth-vda-small-metric",
                upstreamRepository: "depth-anything/Metric-Video-Depth-Anything-Small",
                upstreamRevision: "pin",
                license: "Apache-2.0"
            ),
            temporalWindowLength: 32,
            temporalOverlap: 10,
            frames: [
                DepthSequenceFrameManifest(index: 0, timeSeconds: 0, depthPath: "000000.exr", intrinsics: intrinsics),
            ]
        )
        XCTAssertTrue(manifest.canProjectEveryFrameToPoints)
    }
}
