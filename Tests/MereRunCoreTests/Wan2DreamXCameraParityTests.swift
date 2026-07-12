import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Wan2DreamXCameraParityTests: MereRunCoreTestCase {
    func testARTrajectoryMatchesDreamXChunkRelativeFixture() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_DREAMX_AR_CAMERA_FIXTURE"] else {
            throw XCTSkip("Set MERERUN_DREAMX_AR_CAMERA_FIXTURE to a DreamX AR camera fixture JSON file.")
        }
        let fixture = try JSONDecoder().decode(
            DreamXARCameraFixture.self,
            from: Data(contentsOf: URL(fileURLWithPath: path))
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
