import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class Wan2DreamXCameraParityTests: MereRunCoreTestCase {
    func testSourceResizeMatchesReleasedDreamXPillowContract() throws {
        let source = try patternedImage(width: 7, height: 5)
        let resized = try Wan2DreamXImagePreprocessor.resized(source, width: 4, height: 3)

        // Exported with Pillow 11.3.0's RGBA/BILINEAR path. DreamX passes a PIL
        // image through torchvision.transforms.Resize before ToTensor, which
        // dispatches to this Pillow implementation for the released pipeline.
        XCTAssertEqual(resized.rgba8, [
            24, 37, 54, 255, 82, 63, 165, 255, 150, 95, 111, 255, 208, 121, 150, 255,
            41, 120, 84, 255, 99, 146, 139, 255, 167, 178, 85, 255, 225, 174, 180, 255,
            58, 203, 114, 255, 116, 189, 153, 255, 184, 115, 99, 255, 166, 84, 210, 255,
        ])

        let smallSource = try patternedImage(width: 3, height: 2)
        let upsampled = try Wan2DreamXImagePreprocessor.resized(smallSource, width: 5, height: 4)
        XCTAssertEqual(upsampled.rgba8, [
            0, 5, 9, 255, 15, 12, 37, 255, 37, 22, 80, 255, 59, 32, 123, 255, 74, 39, 151, 255,
            3, 18, 14, 255, 18, 25, 42, 255, 40, 35, 85, 255, 62, 45, 128, 255, 77, 52, 156, 255,
            8, 45, 23, 255, 23, 52, 51, 255, 45, 62, 94, 255, 67, 72, 137, 255, 82, 79, 165, 255,
            11, 58, 28, 255, 26, 65, 56, 255, 48, 75, 99, 255, 70, 85, 142, 255, 85, 92, 170, 255,
        ])
    }

    func testARTrajectoryMatchesDreamXChunkRelativeFixture() throws {
        let fixtureURL = ProcessInfo.processInfo.environment["MERERUN_DREAMX_AR_CAMERA_FIXTURE"]
            .map { URL(fileURLWithPath: $0) }
            ?? Bundle.module.url(
                forResource: "dreamx-ar-camera",
                withExtension: "json",
                subdirectory: "Fixtures/DreamX"
            )!
        let fixture = try JSONDecoder().decode(
            DreamXARCameraFixture.self,
            from: Data(contentsOf: fixtureURL)
        )
        let compiled = Wan2DreamXARTrajectory.compile(
            segments: fixture.trajectory.segments.map {
                Wan2DreamXARTrajectorySegment(action: $0.action, weight: $0.value)
            },
            pixelFrameCount: fixture.trajectory.pixelFrameCount,
            speed: fixture.trajectory.speed,
            chunkRelative: fixture.trajectory.chunkRelative
        )
        assertClose(compiled.viewMatrices, fixture.viewMatrices.values, tolerance: 3e-5)
        assertClose(compiled.intrinsics, fixture.intrinsics.values, tolerance: 1e-6)
    }

    func testTrajectoryAndProjectiveTransformsMatchDreamX() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_DREAMX_CAMERA_FIXTURE"] else {
            throw XCTSkip("Set MERERUN_DREAMX_CAMERA_FIXTURE to a DreamX camera fixture JSON file.")
        }
        let fixture = try JSONDecoder().decode(
            DreamXCameraFixture.self,
            from: Data(contentsOf: URL(fileURLWithPath: path))
        )
        let compiled = Wan2DreamXCameraTrajectory.compile(
            segments: fixture.trajectory.segments.map {
                Wan2DreamXTrajectorySegment(action: $0.action, speed: $0.speed)
            },
            pixelFrameCount: fixture.trajectory.pixelFrameCount
        )
        assertClose(compiled.viewMatrices, fixture.viewMatrices.values, tolerance: 2e-5)
        assertClose(compiled.intrinsics, fixture.intrinsics.values, tolerance: 1e-6)

        let conditioning = Wan2ProjectiveCameraConditioning(
            frameCount: fixture.viewMatrices.shape[1],
            viewMatrices: fixture.viewMatrices.values,
            intrinsics: fixture.intrinsics.values
        )
        let transforms = Wan2ProjectivePositionEncoding.prepare(
            conditioning: conditioning,
            batchSize: 1,
            dtype: .float32
        )
        try assertTransform(
            input: fixture.queryInput,
            expected: fixture.queryOutput,
            matrices: transforms.query,
            frameCount: conditioning.frameCount
        )
        try assertTransform(
            input: fixture.keyInput,
            expected: fixture.keyOutput,
            matrices: transforms.keyValue,
            frameCount: conditioning.frameCount
        )
        try assertTransform(
            input: fixture.valueInput,
            expected: fixture.valueOutput,
            matrices: transforms.keyValue,
            frameCount: conditioning.frameCount
        )
        try assertTransform(
            input: fixture.outputInput,
            expected: fixture.outputOutput,
            matrices: transforms.output,
            frameCount: conditioning.frameCount
        )
    }

    private func assertTransform(
        input: TensorFixture,
        expected: TensorFixture,
        matrices: MLXArray,
        frameCount: Int
    ) throws {
        let source = MLXArray(input.values, input.shape)
        let result = Wan2ProjectivePositionEncoding.apply(
            source,
            matrices: matrices,
            cameraFrames: frameCount
        )
        eval(result)
        XCTAssertEqual(result.shape, expected.shape)
        assertClose(result.asArray(Float.self), expected.values, tolerance: 2e-5)
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float) {
        XCTAssertEqual(actual.count, expected.count)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: tolerance, "Mismatch at index \(index)")
        }
    }

    private func patternedImage(width: Int, height: Int) throws -> MediaImage {
        var rgba8: [UInt8] = []
        rgba8.reserveCapacity(width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                rgba8.append(UInt8((x * 37 + y * 11) % 256))
                rgba8.append(UInt8((x * 17 + y * 53 + 5) % 256))
                rgba8.append(UInt8((x * 71 + y * 19 + 9) % 256))
                rgba8.append(255)
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba8)
    }
}

private struct DreamXARCameraFixture: Decodable {
    let trajectory: ARTrajectoryFixture
    let viewMatrices: TensorFixture
    let intrinsics: TensorFixture

    enum CodingKeys: String, CodingKey {
        case trajectory
        case viewMatrices = "view_matrices"
        case intrinsics
    }
}

private struct ARTrajectoryFixture: Decodable {
    let segments: [ARTrajectorySegmentFixture]
    let pixelFrameCount: Int
    let speed: Float
    let chunkRelative: Bool

    enum CodingKeys: String, CodingKey {
        case segments
        case pixelFrameCount = "pixel_frame_count"
        case speed
        case chunkRelative = "chunk_relative"
    }
}

private struct ARTrajectorySegmentFixture: Decodable {
    let action: String
    let value: Float

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        action = try container.decode(String.self)
        value = try container.decode(Float.self)
    }
}

private struct DreamXCameraFixture: Decodable {
    let trajectory: TrajectoryFixture
    let viewMatrices: TensorFixture
    let intrinsics: TensorFixture
    let queryInput: TensorFixture
    let queryOutput: TensorFixture
    let keyInput: TensorFixture
    let keyOutput: TensorFixture
    let valueInput: TensorFixture
    let valueOutput: TensorFixture
    let outputInput: TensorFixture
    let outputOutput: TensorFixture

    enum CodingKeys: String, CodingKey {
        case trajectory
        case viewMatrices = "view_matrices"
        case intrinsics
        case queryInput = "query_input"
        case queryOutput = "query_output"
        case keyInput = "key_input"
        case keyOutput = "key_output"
        case valueInput = "value_input"
        case valueOutput = "value_output"
        case outputInput = "output_input"
        case outputOutput = "output_output"
    }
}

private struct TrajectoryFixture: Decodable {
    let segments: [TrajectorySegmentFixture]
    let pixelFrameCount: Int

    enum CodingKeys: String, CodingKey {
        case segments
        case pixelFrameCount = "pixel_frame_count"
    }
}

private struct TrajectorySegmentFixture: Decodable {
    let action: String
    let speed: Float

    init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        action = try container.decode(String.self)
        speed = try container.decode(Float.self)
    }
}

private struct TensorFixture: Decodable {
    let shape: [Int]
    let values: [Float]
}
