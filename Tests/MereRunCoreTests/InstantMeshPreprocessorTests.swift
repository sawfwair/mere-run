import Foundation
import MediaIO
import MLX
@testable import MereRunCore
import XCTest

final class InstantMeshPreprocessorTests: MereRunCoreTestCase {
    func testOfficialSixViewRigMatchesReleasedCameraConvention() throws {
        let cameras = try InstantMeshCameraRig.official(viewCount: 6)
        XCTAssertEqual(cameras.count, 6)
        XCTAssertTrue(cameras.allSatisfy { $0.count == 16 && $0.allSatisfy(\.isFinite) })
        XCTAssertEqual(cameras[0][12], 1.866_025_4, accuracy: 1e-6)
        XCTAssertEqual(cameras[0][13], 1.866_025_4, accuracy: 1e-6)
        XCTAssertEqual(cameras[0][14], 0.5, accuracy: 1e-7)
        XCTAssertEqual(cameras[0][15], 0.5, accuracy: 1e-7)

        let four = try InstantMeshCameraRig.official(viewCount: 4)
        XCTAssertEqual(four, [cameras[0], cameras[2], cameras[4], cameras[5]])
    }

    func testOfficialCameraRigRejectsUnsafeRadiusAndFieldOfView() {
        XCTAssertThrowsError(try InstantMeshCameraRig.official(viewCount: 4, radius: 0)) {
            XCTAssertEqual($0 as? InstantMeshPreprocessingError, .invalidCameraRadius(0))
        }
        XCTAssertThrowsError(try InstantMeshCameraRig.official(viewCount: 4, radius: .nan)) {
            guard case .invalidCameraRadius(let radius) = $0 as? InstantMeshPreprocessingError else {
                return XCTFail("Unexpected error: \($0)")
            }
            XCTAssertTrue(radius.isNaN)
        }
        for fieldOfView: Float in [0, 180, .infinity, .leastNonzeroMagnitude] {
            XCTAssertThrowsError(try InstantMeshCameraRig.official(
                viewCount: 4,
                fovDegrees: fieldOfView
            )) {
                guard case .invalidCameraFieldOfView(let actual) =
                    $0 as? InstantMeshPreprocessingError else {
                    return XCTFail("Unexpected error: \($0)")
                }
                XCTAssertEqual(actual, fieldOfView)
            }
        }
    }

    func testOfficialCameraRigNormalizesVerySmallFiniteRadius() throws {
        let cameras = try InstantMeshCameraRig.official(
            viewCount: 4,
            radius: Float.leastNormalMagnitude
        )
        XCTAssertTrue(cameras.flatMap { $0 }.allSatisfy(\.isFinite))
    }

    func testPrepareAcceptsOnlyFourOrSixUserViewsAndCompositesWhite() throws {
        let transparent = try MediaImage(
            width: 1,
            height: 1,
            rgba8: [255, 0, 0, 128]
        )
        let result = try InstantMeshPreprocessor.prepare(
            sourceImages: [transparent, transparent, transparent, transparent],
            size: 2
        )
        MLX.eval(result.images, result.cameras)
        XCTAssertEqual(result.images.shape, [1, 4, 2, 2, 3])
        XCTAssertEqual(result.cameras.shape, [1, 4, 16])
        XCTAssertTrue(result.usedOfficialCameraRig)
        let values = result.images.asArray(Float.self)
        XCTAssertEqual(values[0], 1, accuracy: 1e-7)
        XCTAssertEqual(values[1], 127 / 255, accuracy: 1e-7)
        XCTAssertEqual(values[2], 127 / 255, accuracy: 1e-7)

        XCTAssertThrowsError(try InstantMeshPreprocessor.prepare(sourceImages: [transparent])) {
            XCTAssertEqual($0 as? InstantMeshPreprocessingError, .invalidViewCount(1))
        }
    }

    func testSuppliedCameraValidationIsStrict() throws {
        let image = try MediaImage(width: 1, height: 1, rgba8: [0, 0, 0, 255])
        XCTAssertThrowsError(try InstantMeshPreprocessor.prepare(
            sourceImages: [image, image, image, image],
            cameras: [[Float](repeating: 0, count: 16)]
        )) {
            XCTAssertEqual(
                $0 as? InstantMeshPreprocessingError,
                .cameraCountMismatch(expected: 4, actual: 1)
            )
        }
    }

    func testRejectsNonPositiveConditioningSize() throws {
        let image = try MediaImage(width: 1, height: 1, rgba8: [0, 0, 0, 255])
        XCTAssertThrowsError(try InstantMeshPreprocessor.prepare(
            sourceImages: [image, image, image, image],
            size: 0
        )) {
            XCTAssertEqual($0 as? InstantMeshPreprocessingError, .invalidImageSize(0))
        }
    }

    func testNon320ResizeMatchesPinnedTorchvisionFixtureWhenAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_INSTANTMESH_PREPROCESS"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_INSTANTMESH_PREPROCESS to the frozen Torchvision fixture."
        )
        let root = URL(fileURLWithPath: path)
        let manifest = try JSONDecoder().decode(
            PreprocessManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("manifest.json"))
        )
        XCTAssertEqual(
            manifest.sourceRevision,
            GeometryModelPins.instantMeshBase.sourceCodeRevision
        )
        XCTAssertNotEqual(manifest.sourceSize, manifest.targetSize)
        XCTAssertEqual(manifest.sourceShape, [
            manifest.viewCount, manifest.sourceSize, manifest.sourceSize, 3,
        ])
        XCTAssertEqual(manifest.outputShape, [
            manifest.viewCount, manifest.targetSize, manifest.targetSize, 3,
        ])

        let source = [UInt8](try Data(contentsOf: root.appendingPathComponent("source-rgb.u8")))
        let pixelsPerView = manifest.sourceSize * manifest.sourceSize
        XCTAssertEqual(source.count, manifest.viewCount * pixelsPerView * 3)
        let images = try (0..<manifest.viewCount).map { view -> MediaImage in
            var rgba = [UInt8](repeating: 255, count: pixelsPerView * 4)
            let viewOffset = view * pixelsPerView * 3
            for pixel in 0..<pixelsPerView {
                rgba[pixel * 4] = source[viewOffset + pixel * 3]
                rgba[pixel * 4 + 1] = source[viewOffset + pixel * 3 + 1]
                rgba[pixel * 4 + 2] = source[viewOffset + pixel * 3 + 2]
            }
            return try MediaImage(
                width: manifest.sourceSize,
                height: manifest.sourceSize,
                rgba8: rgba
            )
        }

        let batch = try InstantMeshPreprocessor.prepare(
            sourceImages: images,
            size: manifest.targetSize
        )
        MLX.eval(batch.images)
        XCTAssertEqual(Array(batch.images.shape.dropFirst()), manifest.outputShape)
        let actual = batch.images.asArray(Float.self)
        let expected = try readFloat32(root.appendingPathComponent("resized.f32"))
        XCTAssertEqual(actual.count, expected.count)
        guard actual.count == expected.count else { return }
        let differences = zip(actual, expected).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(max(1, differences.count))
        let maximum = differences.max() ?? 0
        print("InstantMesh preprocess parity: MAE=\(mean), max=\(maximum)")
        XCTAssertLessThanOrEqual(mean, 2e-6, "InstantMesh preprocess MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, 2e-5, "InstantMesh preprocess MAE \(mean), max \(maximum)")
    }

    private struct PreprocessManifest: Decodable {
        let sourceRevision: String
        let sourceSize: Int
        let targetSize: Int
        let viewCount: Int
        let sourceShape: [Int]
        let outputShape: [Int]
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        precondition(data.count.isMultiple(of: 4))
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }
}
