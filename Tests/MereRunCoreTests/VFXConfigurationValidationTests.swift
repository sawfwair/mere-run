import MLX
@testable import MereRunCore
import XCTest

final class VFXConfigurationValidationTests: XCTestCase {
    func testDepthAnything3ConfigurationRejectsInvalidCallerValues() {
        XCTAssertThrowsError(try DepthAnything3Configuration(headCount: 0)) {
            XCTAssertEqual(
                $0 as? DepthAnything3ConfigurationError,
                .invalidParameter("headCount")
            )
        }
        XCTAssertThrowsError(try DepthAnything3Configuration(outputLayers: [5, 5, 9, 11])) {
            XCTAssertEqual(
                $0 as? DepthAnything3ConfigurationError,
                .invalidParameter("outputLayers")
            )
        }
        XCTAssertThrowsError(try DepthAnything3Configuration(rotaryBaseFrequency: .nan)) {
            XCTAssertEqual(
                $0 as? DepthAnything3ConfigurationError,
                .invalidParameter("rotaryBaseFrequency")
            )
        }
        XCTAssertThrowsError(try DepthAnything3Configuration(cameraEncoderHeadCount: 0)) {
            XCTAssertEqual(
                $0 as? DepthAnything3ConfigurationError,
                .invalidParameter("cameraEncoderHeadCount")
            )
        }
    }

    func testDepthAnything3CameraConditioningRejectsInvalidShapes() {
        XCTAssertThrowsError(try DepthAnything3CameraConditioning(
            extrinsics: MLX.zeros([1, 4, 4]),
            intrinsics: MLX.zeros([1, 1, 3, 3])
        )) {
            XCTAssertEqual(
                $0 as? DepthAnything3CameraConditioningError,
                .invalidExtrinsicsShape([1, 4, 4])
            )
        }
        XCTAssertThrowsError(try DepthAnything3CameraConditioning(
            extrinsics: MLX.zeros([1, 2, 4, 4]),
            intrinsics: MLX.zeros([1, 1, 3, 3])
        )) {
            XCTAssertEqual(
                $0 as? DepthAnything3CameraConditioningError,
                .batchViewShapeMismatch(extrinsics: [1, 2], intrinsics: [1, 1])
            )
        }
    }

    func testTripoSRConfigurationRejectsInvalidAndOverflowingValues() {
        XCTAssertThrowsError(try TripoSRConfiguration(imagePatchSize: 0)) {
            XCTAssertEqual(
                $0 as? TripoSRConfigurationError,
                .invalidParameter("imagePatchSize")
            )
        }
        XCTAssertThrowsError(try TripoSRConfiguration(
            tokenChannels: 1,
            transformerHeadCount: Int.max,
            transformerHeadDimension: 2
        )) {
            XCTAssertEqual(
                $0 as? TripoSRConfigurationError,
                .invalidParameter("transformerHeadDimension")
            )
        }
        XCTAssertThrowsError(try TripoSRConfiguration(scenePlaneChannels: Int.max)) {
            XCTAssertEqual(
                $0 as? TripoSRConfigurationError,
                .invalidParameter("scenePlaneChannels")
            )
        }
        XCTAssertThrowsError(try TripoSRConfiguration(rendererRadius: .nan)) {
            XCTAssertEqual(
                $0 as? TripoSRConfigurationError,
                .invalidParameter("rendererRadius")
            )
        }
        XCTAssertThrowsError(try TripoSRMemoryConfiguration(queryChunkSize: 0)) {
            XCTAssertEqual(
                $0 as? TripoSRConfigurationError,
                .invalidParameter("queryChunkSize")
            )
        }
    }

    func testInstantMeshConfigurationRejectsInvalidAndOverflowingValues() {
        XCTAssertThrowsError(try InstantMeshConfiguration(imagePatchSize: 0)) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("imagePatchSize")
            )
        }
        XCTAssertThrowsError(try InstantMeshConfiguration(
            triplaneLowResolution: Int.max,
            triplaneHighResolution: 64
        )) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("triplaneHighResolution")
            )
        }
        XCTAssertThrowsError(try InstantMeshConfiguration(transformerHeadCount: 0)) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("transformerHeadCount")
            )
        }
        XCTAssertThrowsError(try InstantMeshConfiguration(triplaneChannels: Int.max)) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("triplaneChannels")
            )
        }
        XCTAssertThrowsError(try InstantMeshConfiguration(gridScale: .nan)) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("gridScale")
            )
        }
        XCTAssertThrowsError(try InstantMeshMemoryConfiguration(fieldQueryChunkSize: 0)) {
            XCTAssertEqual(
                $0 as? InstantMeshConfigurationError,
                .invalidParameter("fieldQueryChunkSize")
            )
        }
    }

    func testPublishedDefaultsRemainValid() {
        XCTAssertEqual(DepthAnything3Configuration.small.hiddenSize, 384)
        XCTAssertEqual(TripoSRConfiguration.production.conditioningImageSize, 512)
        XCTAssertEqual(TripoSRMemoryConfiguration.appleSilicon.queryChunkSize, 8_192)
        XCTAssertEqual(InstantMeshConfiguration.production.conditioningImageSize, 320)
        XCTAssertEqual(InstantMeshMemoryConfiguration.appleSilicon.imageViewBatchSize, 1)
    }
}
