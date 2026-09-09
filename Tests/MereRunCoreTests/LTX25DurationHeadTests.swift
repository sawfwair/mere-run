import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunLTXModel

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
}
