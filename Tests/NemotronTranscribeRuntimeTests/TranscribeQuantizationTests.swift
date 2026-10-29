import Foundation
import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import AudioNemotronTranscribeModel

final class NemotronTranscribeQuantizationTests: MLXTestCase {
    private let key = "llm.model.layers.1.mixer.experts.gate_and_up_projs"

    func testPackedExpertGatherMatchesExplicitDequantizedProjections() {
        // Repeated and out-of-order expert indices exercise token and top-k broadcasting.
        for dtype in [DType.float32, .bfloat16] {
            for tokens in [1, 5] {
                // FP32 isolates packing/routing error. Metal accumulates BF16 QMM in FP32;
                // MLX's scalar CPU BF16 kernel rounds its accumulator after each of 64 products.
                let bf16Tolerance: Float = Device.defaultDevice().deviceType == .gpu ? 0.005 : 0.02
                let tolerance: Float = dtype == .bfloat16 ? bf16Tolerance : 1e-5
                let original = (sin(MLXArray(0..<(3 * 128 * 64)).asType(.float32)) * Float(0.02))
                    .reshaped(3, 128, 64).asType(dtype)
                let packed = quantized(original, groupSize: 64, bits: 4)
                let weights = TranscribeWeights(arrays: [key + ".qweight": packed.wq,
                    key + ".scales": packed.scales, key + ".biases": packed.biases!], quantization: .supported)
                let dense = dequantized(packed.wq, scales: packed.scales.asType(.float32),
                    biases: packed.biases!.asType(.float32), groupSize: 64, bits: 4)
                let indices = MLXArray((0..<(tokens * 2)).map { Int32(($0 * 2 + 2) % 3) }).reshaped(1, tokens, 2)
                let input = cos(MLXArray(0..<(tokens * 64)).asType(.float32)).reshaped(1, tokens, 1, 1, 64).asType(dtype)
                let actual = weights.routedLinear(input, indices: indices, key: key)
                let expected = matmul(input.asType(.float32), dense.take(indices, axis: 0).swappedAxes(-1, -2))
                XCTAssertEqual(actual.shape, [1, tokens, 2, 1, 128])
                XCTAssertLessThan(MLX.max(abs(actual.asType(.float32) - expected.asType(.float32))).item(Float.self), tolerance)
                // The down projection has a distinct left-hand matrix for every selected expert.
                let selectedInput = square(relu(actual))
                let repeated = weights.routedLinear(selectedInput[.ellipsis, ..<64], indices: indices, key: key)
                let repeatedExpected = matmul(selectedInput[.ellipsis, ..<64].asType(.float32),
                    dense.take(indices, axis: 0).swappedAxes(-1, -2))
                XCTAssertLessThan(MLX.max(abs(repeated.asType(.float32) - repeatedExpected.asType(.float32))).item(Float.self), tolerance)
            }
        }
    }

    func testQuantizedInventoryKeepsNonExpertsAtPublishedPrecision() throws {
        let original = MLXArray.ones([3, 64, 128], dtype: .bfloat16)
        let packed = quantized(original.swappedAxes(-1, -2), groupSize: 64, bits: 4)
        let inventory = [key: TranscribeTensorSpec(shape: original.shape, dtype: "BF16"),
                         "router.weight": TranscribeTensorSpec(shape: [3, 64], dtype: "BF16")]
        let arrays = [key + ".qweight": packed.wq, key + ".scales": packed.scales,
                      key + ".biases": packed.biases!, "router.weight": MLXArray.ones([3, 64], dtype: .bfloat16)]
        let weights = try TranscribeWeights.validated(arrays: arrays, inventory: inventory, quantization: .supported)
        XCTAssertEqual(weights.arrays.count, 4)
        XCTAssertEqual(weights.arrays["router.weight"]?.dtype, .bfloat16)
        for suffix in [".qweight", ".scales", ".biases"] {
            var missing = arrays
            missing.removeValue(forKey: key + suffix)
            XCTAssertThrowsError(try TranscribeWeights.validated(arrays: missing, inventory: inventory, quantization: .supported))
        }
        for invalid in ["router.weight", key + ".scales"] {
            var wrongDType = arrays
            wrongDType[invalid] = wrongDType[invalid]!.asType(.float32)
            XCTAssertThrowsError(try TranscribeWeights.validated(arrays: wrongDType, inventory: inventory, quantization: .supported))
        }
        var wrongLayout = arrays
        wrongLayout[key + ".qweight"] = packed.wq.swappedAxes(-1, -2)
        XCTAssertThrowsError(try TranscribeWeights.validated(arrays: wrongLayout, inventory: inventory, quantization: .supported))
        var mixed = arrays
        mixed[key] = original
        XCTAssertThrowsError(try TranscribeWeights.validated(arrays: mixed, inventory: inventory, quantization: .supported))
        XCTAssertThrowsError(try TranscribeWeights.validated(arrays: arrays, inventory: inventory, quantization: nil))
    }

    func testUnsupportedQuantizationMetadataIsRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertNil(try TranscribeExpertQuantization.load(at: root))
        let url = root.appendingPathComponent(TranscribeExpertQuantization.filename)
        try JSONEncoder().encode(TranscribeExpertQuantization.supported).write(to: url)
        XCTAssertEqual(try TranscribeExpertQuantization.load(at: root), .supported)
        for (field, replacement) in [("bits", "8"), ("groupSize", "32"), ("version", "2"),
                                      ("scope", "\"all-linears\""), ("mode", "\"nvfp4\""), ("layout", "\"input-output\"")] {
            let format = TranscribeExpertQuantization.supported
            let encoded = try JSONEncoder().encode(format)
            let text = String(decoding: encoded, as: UTF8.self)
            let pattern = "\"\(field)\":(?:\"[^\"]*\"|[0-9]+)"
            let regex = try NSRegularExpression(pattern: pattern)
            let changed = regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text),
                                                        withTemplate: "\"\(field)\":\(replacement)")
            try Data(changed.utf8).write(to: url)
            XCTAssertThrowsError(try TranscribeExpertQuantization.load(at: root), field)
        }
    }
}
