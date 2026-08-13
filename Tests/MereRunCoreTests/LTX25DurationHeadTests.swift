import MLX
import XCTest
@testable import MereRunCore

final class LTX25DurationHeadTests: MereRunCoreTestCase {
    func testDurationHeadMapsOfficialCheckpointKeysWithoutUntypedDecoding() {
        let value = MLX.ones([256, 4_096], dtype: .float32)
        let mapped = mapLTX25DurationHeadWeight(
            key: "duration_head.video_input_proj.weight",
            value: value,
            dtype: .bfloat16
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "video_input_proj.weight")
        XCTAssertEqual(mapped[0].1.dtype, .bfloat16)
        XCTAssertTrue(
            mapLTX25DurationHeadWeight(
                key: "model.diffusion_model.patchify_proj.weight",
                value: value,
                dtype: .bfloat16
            ).isEmpty
        )
    }

    func testDurationHeadAcceptsEitherOrBothConnectorModalities() throws {
        let head = LTX25DurationHead()
        let video = MLX.zeros([1, 4, 4_096], dtype: .float32)
        let audio = MLX.zeros([1, 3, 2_048], dtype: .float32)

        let videoOnly = try head(videoTokens: video, audioTokens: nil)
        let audioOnly = try head(videoTokens: nil, audioTokens: audio)
        let both = try head(videoTokens: video, audioTokens: audio)
        MLX.eval(videoOnly, audioOnly, both)

        XCTAssertEqual(videoOnly.shape, [1])
        XCTAssertEqual(audioOnly.shape, [1])
        XCTAssertEqual(both.shape, [1])
        XCTAssertThrowsError(try head(videoTokens: nil, audioTokens: nil))
    }

    func testAutoDurationClampsAndSnapsToCausalGrid() {
        XCTAssertEqual(ltx25FrameCount(predictedSeconds: 0.1, frameRate: 24), 25)
        XCTAssertEqual(ltx25FrameCount(predictedSeconds: 4, frameRate: 24), 89)
        XCTAssertEqual(ltx25FrameCount(predictedSeconds: 100, frameRate: 24), 473)
        XCTAssertEqual(
            ltx25FrameCount(
                predictedSeconds: 7,
                frameRate: 24,
                range: LTX25AutoDuration(minimumSeconds: 5, maximumSeconds: 8)
            ),
            161
        )
    }
}
