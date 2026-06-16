import MLX
@testable import MereRunCore
import XCTest

final class LTXUnifiedAVTransformerV2Tests: XCTestCase {
    func testLTX23TransformerMapperKeepsPromptAndGateKeys() {
        let promptTable = mapLTX23UnifiedTransformerWeight(
            key: "transformer.transformer_blocks.0.prompt_scale_shift_table",
            value: MLXArray.zeros([2, 4096]),
            dtype: .float32
        )
        let gate = mapLTX23UnifiedTransformerWeight(
            key: "transformer.transformer_blocks.0.attn1.to_gate_logits.weight",
            value: MLXArray.zeros([32, 4096]),
            dtype: .float32
        )

        XCTAssertEqual(promptTable.map(\.0), ["transformer_blocks.0.prompt_scale_shift_table"])
        XCTAssertEqual(gate.map(\.0), ["transformer_blocks.0.attn1.to_gate_logits.weight"])
    }

    func testLTX23TransformerMapperNormalizesFeedForwardKeys() {
        let mapped = mapLTX23UnifiedTransformerWeight(
            key: "transformer.transformer_blocks.47.audio_ff.net.2.bias",
            value: MLXArray.zeros([2048]),
            dtype: .float32
        )

        XCTAssertEqual(mapped.map(\.0), ["transformer_blocks.47.audio_ff.proj_out.bias"])
    }

    func testLTX23TransformerMapperIgnoresNonTransformerConnectorKeys() {
        XCTAssertTrue(mapLTX23UnifiedTransformerWeight(
            key: "connector.text_embedding_projection.video_aggregate_embed.weight",
            value: MLXArray.zeros([4096, 188160]),
            dtype: .float32
        ).isEmpty)
    }
}
