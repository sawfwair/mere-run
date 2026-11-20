import Foundation
import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor

#if os(macOS)
final class PackedAffineQ4PrefillTests: MLXTestCase {
    func testPackedArithmeticAndTails() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Packed prefill requires Metal.")
        MLXRandom.seed(1923)
        for (m, n, k) in [(519, 1093, 128), (512, 1024, 512), (1024, 1024, 1024)] {
            let x = MLXRandom.normal([1, m, k]).asType(.bfloat16)
            let (w, s, b) = MLX.quantized(MLXRandom.normal([n, k]).asType(.bfloat16), groupSize: 64, bits: 4)
            let biases = try XCTUnwrap(b)
            let reference = MLX.quantizedMM(x, w, scales: s, biases: biases, groupSize: 64, bits: 4)
            MLX.eval(reference)
            for engine in [PackedAffineQ4Prefill.Engine.simd, .tensor, .gdnTensor] {
                if engine != .simd, !M5AffineQ4Verification.isSupported { continue }
                for bm in [32, 64] {
                    let value = try XCTUnwrap(PackedAffineQ4Prefill.matmul(
                        x, weight: w, scales: s, biases: biases, engine: engine, tileM: bm
                    ))
                    let delta = (value.asType(.float32) - reference.asType(.float32)).abs().max().item(Float.self)
                    print("[packed-q4-check] engine=\(engine.rawValue) m=\(m) n=\(n) k=\(k) bm=\(bm) max=\(delta)")
                    if engine == .simd { XCTAssertEqual(delta, 0) }
                    else {
                        let difference = value.asType(.float32) - reference.asType(.float32)
                        let rms = (difference.square().mean() / reference.asType(.float32).square().mean()).sqrt().item(Float.self)
                        XCTAssertLessThan(rms, 0.0002, "Component tolerance; full-model qualification has its own frozen gates.")
                    }
                }
            }
        }
    }

    func testRepairedGDNProjectionMatchesEarlyReference() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu && M5AffineQ4Verification.isSupported,
                          "The GDN repair test requires M5 Metal.")
        MLXRandom.seed(1937)
        let (weight, scales, optionalBias) = MLX.quantized(MLXRandom.normal([16_384, 5_120]).asType(.bfloat16),
                                                           groupSize: 64, bits: 4)
        let biases = try XCTUnwrap(optionalBias)
        for rows in [526, 1024] {
            let x = MLXRandom.normal([1, rows, 5_120]).asType(.bfloat16)
            let reference = try XCTUnwrap(PackedAffineQ4Prefill.matmul(
                x, weight: weight, scales: scales, biases: biases, engine: .simd))
            let repaired = try XCTUnwrap(PackedAffineQ4Prefill.matmul(
                x, weight: weight, scales: scales, biases: biases, engine: .gdnRepaired))
            XCTAssertEqual(repaired.shape, reference.shape)
            XCTAssertEqual((repaired .!= reference).sum().item(Int.self), 0, "rows=\(rows)")
        }
    }

    func testComponentSweep() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["MERERUN_TEST_PACKED_PREFILL_COMPONENT"] == "1"
                          && Device.defaultDevice().deviceType == .gpu, "Packed prefill component sweep is opt-in.")
        MLXRandom.seed(1924)
        for (k, n) in [(5_120, 17_408), (17_408, 5_120)] {
            let x = MLXRandom.normal([1, 1024, k]).asType(.bfloat16)
            let (w, s, b) = MLX.quantized(MLXRandom.normal([n, k]).asType(.bfloat16), groupSize: 64, bits: 4)
            let biases = try XCTUnwrap(b)
            MLX.eval(x, w, s, biases)
            let labels = ["native", "simd32", "simd64", "tensor32", "tensor64"]
            func apply(_ label: String) throws -> MLXArray {
                if label == "native" { return MLX.quantizedMM(x, w, scales: s, biases: biases, groupSize: 64, bits: 4) }
                return try XCTUnwrap(PackedAffineQ4Prefill.matmul(x, weight: w, scales: s, biases: biases,
                    engine: label.hasPrefix("simd") ? .simd : .tensor, tileM: label.hasSuffix("32") ? 32 : 64))
            }
            let reference = try apply("native")
            MLX.eval(reference)
            for label in labels {
                let value = try apply(label)
                let difference = value.asType(.float32) - reference.asType(.float32)
                let maximum = difference.abs().max().item(Float.self)
                let rms = (difference.square().mean() / reference.asType(.float32).square().mean()).sqrt().item(Float.self)
                print("[packed-q4-difference] k=\(k) n=\(n) engine=\(label) max=\(maximum) relative_rms=\(rms)")
            }
            var times = Dictionary(uniqueKeysWithValues: labels.map { ($0, [Double]()) })
            for iteration in 0..<9 {
                let order = iteration.isMultiple(of: 2) ? labels : Array(labels.reversed())
                for label in order {
                    let started = DispatchTime.now().uptimeNanoseconds
                    MLX.eval(try apply(label))
                    if iteration >= 2 { times[label, default: []].append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6) }
                }
            }
            print("[packed-q4-times] k=\(k) n=\(n) \(String(decoding: try JSONEncoder().encode(times), as: UTF8.self))")
        }
    }
}
#endif
