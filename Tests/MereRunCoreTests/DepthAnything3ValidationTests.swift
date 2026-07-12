import MereRunCore
import XCTest

final class DepthAnything3ValidationTests: XCTestCase {
    func testDA3ImageAdmissionLimitsBoundEncodedAndDecodedInputs() {
        let limits = DepthAnything3Limits.imageInputLimits
        XCTAssertEqual(limits.maximumDimension, 8_192)
        XCTAssertEqual(
            limits.maximumPixelsPerImage,
            DepthAnything3Limits.maximumSourcePixelCountPerView
        )
        XCTAssertEqual(
            limits.maximumAggregatePixels,
            DepthAnything3Limits.maximumTotalSourcePixelCount
        )
        XCTAssertEqual(
            limits.maximumEncodedBytesPerImage,
            DepthAnything3Limits.maximumEncodedBytesPerView
        )
        XCTAssertEqual(
            limits.maximumAggregateEncodedBytes,
            DepthAnything3Limits.maximumTotalEncodedBytes
        )
    }

    func testRequestLimitsAcceptDocumentedBoundaryWorkloads() throws {
        XCTAssertNoThrow(try DepthAnything3Limits.validateRequest(
            viewCount: 8,
            processResolution: 504
        ))
        XCTAssertNoThrow(try DepthAnything3Limits.validateRequest(
            viewCount: 2,
            processResolution: 1_008
        ))
        XCTAssertNoThrow(try DepthAnything3Limits.validateRequest(
            viewCount: 16,
            processResolution: 252
        ))
    }

    func testEncodedByteLimitsUseOverflowSafePerViewAndTotalAccounting() throws {
        XCTAssertNoThrow(try DepthAnything3Limits.validateEncodedByteCounts([
            DepthAnything3Limits.maximumEncodedBytesPerView,
            DepthAnything3Limits.maximumEncodedBytesPerView,
        ]))
        XCTAssertThrowsError(try DepthAnything3Limits.validateEncodedByteCounts([
            DepthAnything3Limits.maximumEncodedBytesPerView + 1,
        ])) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .encodedByteBudgetExceeded(
                    index: 0,
                    actual: DepthAnything3Limits.maximumEncodedBytesPerView + 1,
                    maximum: DepthAnything3Limits.maximumEncodedBytesPerView
                )
            )
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateEncodedByteCounts([
            DepthAnything3Limits.maximumEncodedBytesPerView,
            DepthAnything3Limits.maximumEncodedBytesPerView,
            1,
        ])) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .totalEncodedByteBudgetExceeded(
                    actual: DepthAnything3Limits.maximumTotalEncodedBytes + 1,
                    maximum: DepthAnything3Limits.maximumTotalEncodedBytes
                )
            )
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateEncodedByteCounts([Int64.max, 1]))
    }

    func testRequestLimitsRejectViewResolutionAndActivationAbuse() throws {
        XCTAssertThrowsError(try DepthAnything3Limits.validateRequest(
            viewCount: 17,
            processResolution: 14
        )) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .viewCountOutOfRange(actual: 17, minimum: 1, maximum: 16)
            )
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateRequest(
            viewCount: 1,
            processResolution: 1_009
        )) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .processResolutionOutOfRange(actual: 1_009, minimum: 14, maximum: 1_008)
            )
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateRequest(
            viewCount: 9,
            processResolution: 504
        )) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .processedPixelBudgetExceeded(
                    actual: 9 * 504 * 504,
                    maximum: 8 * 504 * 504
                )
            )
        }
    }

    func testSourcePixelLimitsUseOverflowSafePerViewAndTotalAccounting() throws {
        XCTAssertNoThrow(try DepthAnything3Limits.validateSourceDimensions([
            (width: 8_192, height: 8_192),
            (width: 8_192, height: 8_192),
        ]))
        XCTAssertThrowsError(try DepthAnything3Limits.validateSourceDimensions([
            (width: 8_193, height: 8_192),
        ])) { error in
            guard case .sourcePixelBudgetExceeded(let index, _, _, _, let maximum) =
                error as? DepthAnything3LimitError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(index, 0)
            XCTAssertEqual(maximum, 8_192 * 8_192)
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateSourceDimensions([
            (width: 8_192, height: 8_192),
            (width: 8_192, height: 8_192),
            (width: 1, height: 1),
        ])) { error in
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .totalSourcePixelBudgetExceeded(
                    actual: 2 * 8_192 * 8_192 + 1,
                    maximum: 2 * 8_192 * 8_192
                )
            )
        }
        XCTAssertThrowsError(try DepthAnything3Limits.validateSourceDimensions([
            (width: Int.max, height: Int.max),
        ])) { error in
            guard case .sourcePixelBudgetExceeded(_, _, _, let actual, _) =
                error as? DepthAnything3LimitError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(actual, Int.max)
        }
    }

    func testCameraValidationRequiresFloatSafeProperRigidTransform() throws {
        let valid = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 64,
                imageHeight: 48,
                normalizedFX: 0.8,
                normalizedFY: 0.9
            ),
            extrinsics: .identity
        )
        XCTAssertNil(DepthAnything3CameraValidation.issue(for: valid))

        let hugeFocal = DepthAnything3KnownCamera(
            intrinsics: GeometryCameraIntrinsics(
                imageWidth: 64,
                imageHeight: 48,
                normalizedFX: Double(Float.greatestFiniteMagnitude) * 2,
                normalizedFY: 1
            ),
            extrinsics: .identity
        )
        XCTAssertTrue(
            DepthAnything3CameraValidation.issue(for: hugeFocal)?.contains("Float-representable") == true
        )

        let scaled = DepthAnything3KnownCamera(
            intrinsics: valid.intrinsics,
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [2, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [0, 0, 0]
            )
        )
        XCTAssertTrue(
            DepthAnything3CameraValidation.issue(for: scaled)?.contains("not unit length") == true
        )

        let reflected = DepthAnything3KnownCamera(
            intrinsics: valid.intrinsics,
            extrinsics: try GeometryCameraExtrinsics(
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, -1],
                translation: [0, 0, 0]
            )
        )
        XCTAssertTrue(
            DepthAnything3CameraValidation.issue(for: reflected)?.contains("determinant +1") == true
        )
    }

    func testGeneratorRejectsUnsafeRequestBeforeFilesystemOrCheckpointWork() async {
        let generator = DepthAnything3Generator()
        let missingURLs = (0..<17).map {
            URL(fileURLWithPath: "/definitely-missing/da3-view-\($0).png")
        }
        do {
            _ = try await generator.generate(
                imageURLs: missingURLs,
                processResolution: 14
            )
            XCTFail("Expected view-count validation to fail")
        } catch {
            XCTAssertEqual(
                error as? DepthAnything3LimitError,
                .viewCountOutOfRange(actual: 17, minimum: 1, maximum: 16)
            )
        }
    }
}
