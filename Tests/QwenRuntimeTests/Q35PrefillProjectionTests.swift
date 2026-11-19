import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor

final class Q35PrefillProjectionTests: MLXTestCase {
    func testFP32PrefillProjectionExperiment() throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["MERERUN_TEST_Q35_PREFILL_COMPONENT"] == "1"
                          && (env["MLX_ENABLE_TF32"] == "0" || env["MERERUN_TEST_Q35_FP32_TF32"] == "1"),
                          "The FP32 component experiment requires an explicit precision policy.")
        MLXRandom.seed(765)
        for (k, n) in [(5_120, 17_408), (17_408, 5_120)] {
            let (codes, scales, biases) = MLX.quantized(MLXRandom.normal([n, k]).asType(.bfloat16), groupSize: 64, bits: 4)
            let bias = try XCTUnwrap(biases)
            MLX.eval(codes, scales, bias)
            let x = MLXRandom.normal([1, 1024, k]).asType(.bfloat16)
            MLX.eval(x)
            func apply(_ dense: Bool) -> MLXArray {
                if dense {
                    let decoded = MLX.dequantized(
                        codes, scales: scales.asType(.float32), biases: bias.asType(.float32),
                        groupSize: 64, bits: 4
                    ).asType(.bfloat16).asType(.float32)
                    return MLX.matmul(x.asType(.float32), decoded.T).asType(.bfloat16)
                }
                return MLX.quantizedMM(x, codes, scales: scales, biases: bias, groupSize: 64, bits: 4)
            }
            let reference = apply(false)
            let delta = apply(true).asType(.float32) - reference.asType(.float32)
            let rms = (delta.square().mean() / reference.asType(.float32).square().mean()).sqrt().item(Float.self)
            let maximum = delta.abs().max().item(Float.self)
            var native: [Double] = [], dense: [Double] = []
            for iteration in 0..<9 {
                for flag in iteration.isMultiple(of: 2) ? [false, true] : [true, false] {
                    let start = DispatchTime.now().uptimeNanoseconds
                    MLX.eval(apply(flag))
                    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    if iteration >= 2 {
                        if flag { dense.append(elapsed) } else { native.append(elapsed) }
                    }
                }
            }
            let row = Timing(k: k, n: n, rows: 1024, normalizedRMS: rms, maximum: maximum,
                             nativeMilliseconds: native, denseMilliseconds: dense)
            print("[q35-fp32-prefill] \(String(decoding: try JSONEncoder().encode(row), as: UTF8.self))")
        }
    }

    func testMaterializationRequiresItsScopeAndLargeBF16Rows() throws {
        let x = MLXArray.ones([1, 8, 64], dtype: .bfloat16)
        let (codes, scales, biases) = MLX.quantized(MLXArray.ones([64, 64], dtype: .bfloat16), groupSize: 64, bits: 4)
        func apply(_ input: MLXArray) -> MLXArray? {
            QuantizedPrefillContext.projection(input, weight: codes, scales: scales, biases: biases,
                                               groupSize: 64, bits: 4, mode: .affine)
        }
        XCTAssertNil(apply(x))
        QuantizedPrefillContext.$mode.withValue(.all) {
            XCTAssertNil(apply(x), "Decode rows keep their existing arithmetic")
            XCTAssertNil(apply(MLXArray.ones([512, 64], dtype: .float32)))
        }
        QuantizedPrefillContext.$mode.withValue(.gateUp) {
            XCTAssertNil(apply(MLXArray.ones([512, 64], dtype: .bfloat16)), "Only the qualified gate/up geometry is eligible")
        }
        for mode in [QuantizedPrefillMode.packedSIMD, .packedTensor] {
            XCTAssertFalse(mode.materializesWeights)
            QuantizedPrefillContext.$mode.withValue(mode) {
                XCTAssertNil(apply(x), "Decode retains its current projection path")
                XCTAssertNil(apply(MLXArray.ones([512, 64], dtype: .bfloat16)), "Small output projections retain native split-K")
            }
        }
        XCTAssertEqual(QuantizedPrefillContext.mode, .reference)
    }

    func testPrefillProjectionComponentExperiment() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_Q35_PREFILL_COMPONENT"] == "1",
                          "Prefill projection experiments are diagnostic and opt-in.")
        MLXRandom.seed(764)
        for (k, n) in [(5_120, 17_408), (17_408, 5_120)] {
            let weights = MLXRandom.normal([n, k]).asType(.bfloat16)
            let (codes, scales, biases) = MLX.quantized(weights, groupSize: 64, bits: 4)
            let bias = try XCTUnwrap(biases)
            MLX.eval(codes, scales, bias)
            for count in [128, 512, 1024] {
                let x = MLXRandom.normal([1, count, k]).asType(.bfloat16)
                MLX.eval(x)
                func apply(_ dense: Bool) -> MLXArray {
                    if dense {
                        let decoded = MLX.dequantized(
                            codes, scales: scales.asType(.float32), biases: bias.asType(.float32),
                            groupSize: 64, bits: 4
                        ).asType(.bfloat16)
                        return MLX.matmul(x, decoded.T)
                    }
                    return MLX.quantizedMM(x, codes, scales: scales, biases: bias, groupSize: 64, bits: 4)
                }
                let reference = apply(false)
                let candidate = apply(true)
                let delta = candidate.asType(.float32) - reference.asType(.float32)
                let rms = (delta.square().mean() / reference.asType(.float32).square().mean()).sqrt().item(Float.self)
                let maximum = delta.abs().max().item(Float.self)
                var nativeTimes: [Double] = []
                var denseTimes: [Double] = []
                for iteration in 0..<11 {
                    for dense in iteration.isMultiple(of: 2) ? [false, true] : [true, false] {
                        let started = DispatchTime.now().uptimeNanoseconds
                        MLX.eval(apply(dense))
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6
                        if iteration >= 2 {
                            if dense { denseTimes.append(ms) } else { nativeTimes.append(ms) }
                        }
                    }
                }
                let row = Timing(k: k, n: n, rows: count, normalizedRMS: rms, maximum: maximum,
                                 nativeMilliseconds: nativeTimes, denseMilliseconds: denseTimes)
                print("[q35-prefill-projection] \(String(decoding: try JSONEncoder().encode(row), as: UTF8.self))")
            }
        }
    }

    private struct Timing: Encodable {
        let tf32Enabled = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] == "1"
        let k: Int
        let n: Int
        let rows: Int
        let normalizedRMS: Float
        let maximum: Float
        let nativeMilliseconds: [Double]
        let denseMilliseconds: [Double]
    }
}
