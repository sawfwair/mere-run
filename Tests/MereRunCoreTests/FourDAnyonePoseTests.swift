import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class FourDAnyonePoseTests: MereRunCoreTestCase {
    func testFullPoseEncoderAndLearnedNullMatchUpstream() throws {
        let fixture = try FourDAnyoneFixture.manifest()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("4danyone-pose-\(UUID()).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let weights = Dictionary(uniqueKeysWithValues: fixture.poseWeights.map { specification in
            let values = (0..<specification.shape.reduce(1, *)).map { index -> Float in
                let pattern = (index * 17 + specification.index * 13) % 101 - 50
                return Float(pattern) / 50 * specification.amplitude + specification.offset
            }
            return ("pose_encoder." + specification.key, MLXArray(values, specification.shape))
        })
        try MLX.save(arrays: weights, metadata: [:], url: url)
        let encoder = try FourDAnyoneModelLoader.loadPoseEncoder(from: url, dtype: .float32)
        let tensors = try FourDAnyoneFixture.tensors()
        let video = tensors["pose.video"]!
        let output = try encoder(video)
        FourDAnyoneFixture.assertClose(output, tensors["pose.output"]!, tolerance: 2e-4)
        let null = try encoder(MLX.full(video.shape, values: MLXArray(Float(-1))))
        FourDAnyoneFixture.assertClose(null, tensors["pose.nullOutput"]!, tolerance: 2e-4)
        XCTAssertGreaterThan(MLX.max(MLX.abs(null)).item(Float.self), 1e-3)
        XCTAssertGreaterThan(MLX.max(MLX.abs(output - null)).item(Float.self), 1e-3)
        XCTAssertThrowsError(try encoder(MLX.zeros([1, 3, 4, 32, 32])))
        XCTAssertThrowsError(try encoder(video, checkCancellation: { throw CancellationError() })) {
            XCTAssertTrue($0 is CancellationError)
        }
    }
}
