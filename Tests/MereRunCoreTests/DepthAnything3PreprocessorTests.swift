import Foundation
import MediaIO
import MLX
@testable import MereRunCore
import XCTest

final class DepthAnything3PreprocessorTests: MereRunCoreTestCase {
    func testUpperBoundResizeNearestFourteenAndBatchCenterCrop() throws {
        let landscape = try image(width: 67, height: 37, value: 32)
        let portrait = try image(width: 37, height: 67, value: 224)
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: [landscape, portrait],
            processResolution: 28
        )
        MLX.eval(batch.normalizedImages)

        XCTAssertEqual(batch.normalizedImages.shape, [1, 2, 14, 14, 3])
        XCTAssertEqual(batch.processedImages.map(\.width), [14, 14])
        XCTAssertEqual(batch.processedImages.map(\.height), [14, 14])
        XCTAssertEqual(batch.plans[0].boundaryWidth, 28)
        XCTAssertEqual(batch.plans[0].boundaryHeight, 15)
        XCTAssertEqual(batch.plans[0].divisibleWidth, 28)
        XCTAssertEqual(batch.plans[0].divisibleHeight, 14)
        XCTAssertEqual(batch.plans[0].batchCropLeft, 7)
        XCTAssertEqual(batch.plans[0].batchCropTop, 0)
        XCTAssertEqual(batch.plans[1].boundaryWidth, 15)
        XCTAssertEqual(batch.plans[1].boundaryHeight, 28)
        XCTAssertEqual(batch.plans[1].divisibleWidth, 14)
        XCTAssertEqual(batch.plans[1].divisibleHeight, 28)
        XCTAssertEqual(batch.plans[1].batchCropLeft, 0)
        XCTAssertEqual(batch.plans[1].batchCropTop, 7)
    }

    func testConstantImageMatchesImageNetNormalizationAfterTwoResizes() throws {
        let source = try image(width: 5, height: 3, value: 128)
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: [source],
            processResolution: 14
        )
        MLX.eval(batch.normalizedImages)
        XCTAssertEqual(batch.normalizedImages.shape, [1, 1, 14, 14, 3])
        let values = batch.normalizedImages.asArray(Float.self)
        let expected: [Float] = [
            (Float(128) / 255 - 0.485) / 0.229,
            (Float(128) / 255 - 0.456) / 0.224,
            (Float(128) / 255 - 0.406) / 0.225,
        ]
        for pixel in 0..<(14 * 14) {
            for channel in 0..<3 {
                XCTAssertEqual(values[pixel * 3 + channel], expected[channel], accuracy: 1e-6)
            }
        }
    }

    func testIntrinsicsFollowResizeAndBatchCrop() throws {
        let landscape = try image(width: 67, height: 37, value: 64)
        let portrait = try image(width: 37, height: 67, value: 192)
        let cameras = try [
            knownCamera(width: 67, height: 37, fx: 60, fy: 40, cx: 33.5, cy: 18.5),
            knownCamera(width: 37, height: 67, fx: 40, fy: 60, cx: 18.5, cy: 33.5),
        ]
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: [landscape, portrait],
            knownCameras: cameras,
            processResolution: 28
        )
        let processed = try XCTUnwrap(batch.processedKnownCameras)
        XCTAssertEqual(processed[0].intrinsics.imageWidth, 14)
        XCTAssertEqual(processed[0].intrinsics.imageHeight, 14)
        XCTAssertEqual(processed[0].intrinsics.pixelFX, 60 * 28 / 67, accuracy: 1e-9)
        XCTAssertEqual(processed[0].intrinsics.pixelFY, 40 * 14 / 37, accuracy: 1e-9)
        XCTAssertEqual(processed[0].intrinsics.pixelCX, 7, accuracy: 1e-9)
        XCTAssertEqual(processed[0].intrinsics.pixelCY, 7, accuracy: 1e-9)
        XCTAssertEqual(processed[1].intrinsics.pixelCX, 7, accuracy: 1e-9)
        XCTAssertEqual(processed[1].intrinsics.pixelCY, 7, accuracy: 1e-9)
    }

    func testConditioningMakesFirstCameraIdentityAndUsesLowerMedianBaseline() throws {
        let source = try image(width: 28, height: 14, value: 128)
        let cameras = try [0.0, -2.0, -4.0].map { translation in
            try knownCamera(
                width: 28,
                height: 14,
                fx: 28,
                fy: 14,
                cx: 14,
                cy: 7,
                translationX: translation
            )
        }
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: [source, source, source],
            knownCameras: cameras,
            processResolution: 28
        )
        let normalized = try XCTUnwrap(batch.conditioningCameras)
        XCTAssertEqual(normalized[0].extrinsics, .identity)
        XCTAssertEqual(normalized[1].extrinsics.translation[0], -1, accuracy: 1e-9)
        XCTAssertEqual(normalized[2].extrinsics.translation[0], -2, accuracy: 1e-9)
        let conditioning = try XCTUnwrap(batch.conditioning)
        MLX.eval(conditioning.extrinsics, conditioning.intrinsics)
        XCTAssertEqual(conditioning.extrinsics.shape, [1, 3, 4, 4])
        XCTAssertEqual(conditioning.intrinsics.shape, [1, 3, 3, 3])
    }

    func testKnownCameraIsCodableAndCameraDimensionsAreValidated() throws {
        let camera = try knownCamera(width: 20, height: 10, fx: 12, fy: 13, cx: 10, cy: 5)
        XCTAssertEqual(
            try JSONDecoder().decode(
                DepthAnything3KnownCamera.self,
                from: JSONEncoder().encode(camera)
            ),
            camera
        )
        let wrongImage = try image(width: 21, height: 10, value: 0)
        XCTAssertThrowsError(
            try DepthAnything3Preprocessor.prepare(
                sourceImages: [wrongImage],
                knownCameras: [camera],
                processResolution: 28
            )
        ) { error in
            XCTAssertEqual(
                error as? DepthAnything3PreprocessingError,
                .cameraImageDimensionMismatch(
                    index: 0,
                    expectedWidth: 21,
                    expectedHeight: 10,
                    actualWidth: 20,
                    actualHeight: 10
                )
            )
        }
    }

    func testPreprocessorRejectsNonRigidSuppliedCameraBeforeConditioning() throws {
        let source = try image(width: 28, height: 14, value: 128)
        let camera = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 28,
                imageHeight: 14,
                normalizedFX: 1,
                normalizedFY: 1
            ),
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0.1, 0, 0, 1, 0, 0, 0, 1],
                translation: [0, 0, 0]
            )
        )

        XCTAssertThrowsError(try DepthAnything3Preprocessor.prepare(
            sourceImages: [source],
            knownCameras: [camera],
            processResolution: 28
        )) { error in
            guard let validation = error as? DepthAnything3CameraValidationError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(validation.index, 0)
            XCTAssertTrue(validation.reason.contains("rotation"))
        }
    }

    func testMatchesPinnedOfficialPreprocessingFixtureWhenAvailable() throws {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_DA3_PREPROCESS"] ?? ""
        try XCTSkipIf(path.isEmpty || !FileManager.default.fileExists(atPath: path))
        let root = URL(fileURLWithPath: path)
        let manifest = try JSONDecoder().decode(
            PreprocessManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("manifest.json"))
        )
        XCTAssertEqual(
            manifest.sourceRevision,
            DepthAnything3SmallCheckpoint.upstreamSourceRevision
        )
        let sourceImages = try manifest.sourceSpecs.map { spec in
            try gradientImage(width: spec[0], height: spec[1], seed: spec[2])
        }
        let cameras = try manifest.sourceSpecs.enumerated().map { index, spec in
            try knownCamera(
                width: spec[0],
                height: spec[1],
                fx: Double(spec[0]) * 0.8,
                fy: Double(spec[1]) * 0.9,
                cx: Double(spec[0]) * 0.47,
                cy: Double(spec[1]) * 0.53,
                translationX: -0.2 * Double(index)
            )
        }
        let batch = try DepthAnything3Preprocessor.prepare(
            sourceImages: sourceImages,
            knownCameras: cameras,
            processResolution: manifest.processResolution
        )
        MLX.eval(batch.normalizedImages)
        // The official preprocessor fixture is unbatched NHWC after
        // permutation; the native model contract adds a singleton B axis.
        XCTAssertEqual(Array(batch.normalizedImages.shape.dropFirst()), manifest.normalizedShape)
        let expected = try readFloat32(root.appendingPathComponent("normalized.f32"))
        let actual = batch.normalizedImages.asArray(Float.self)
        let differences = zip(actual, expected).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(max(1, differences.count))
        let maximum = differences.max() ?? 0
        print("DA3 preprocess parity: MAE=\(mean), max=\(maximum)")
        XCTAssertLessThanOrEqual(mean, 2e-5, "DA3 preprocess MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, 2e-4, "DA3 preprocess MAE \(mean), max \(maximum)")

        let expectedRGB = [UInt8](try Data(contentsOf: root.appendingPathComponent("processed-rgb.u8")))
        let actualRGB = batch.processedImages.flatMap { image in
            (0..<(image.width * image.height)).flatMap { pixel in
                Array(image.rgba8[(pixel * 4)..<(pixel * 4 + 3)])
            }
        }
        XCTAssertEqual(actualRGB, expectedRGB)
        let expectedIntrinsics = try readFloat32(root.appendingPathComponent("intrinsics.f32"))
        let actualIntrinsics = try XCTUnwrap(batch.processedKnownCameras).flatMap {
            $0.intrinsics.pixelMatrixRowMajor.map(Float.init)
        }
        for (actualValue, expectedValue) in zip(actualIntrinsics, expectedIntrinsics) {
            XCTAssertEqual(actualValue, expectedValue, accuracy: 1e-5)
        }
    }

    private func image(width: Int, height: Int, value: UInt8) throws -> MediaImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            rgba[pixel * 4] = value
            rgba[pixel * 4 + 1] = value
            rgba[pixel * 4 + 2] = value
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    private func gradientImage(width: Int, height: Int, seed: Int) throws -> MediaImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            for channel in 0..<3 {
                let scalar = pixel * 3 + channel
                rgba[pixel * 4 + channel] = UInt8((scalar * (seed * 2 + 1) + seed) % 251)
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }

    private func knownCamera(
        width: Int,
        height: Int,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        translationX: Double = 0
    ) throws -> DepthAnything3KnownCamera {
        DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: width,
                imageHeight: height,
                normalizedFX: fx / Double(width),
                normalizedFY: fy / Double(height),
                normalizedCX: cx / Double(width),
                normalizedCY: cy / Double(height)
            ),
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [translationX, 0, 0]
            )
        )
    }
}

private struct PreprocessManifest: Decodable {
    let sourceRevision: String
    let processResolution: Int
    let sourceSpecs: [[Int]]
    let normalizedShape: [Int]
}
