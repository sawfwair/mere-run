import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeGDNCoefficientsTests: MLXTestCase {
    func testDecayAndBetaMatchCompiledMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(937)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeGDNCoefficients(device: device)
        let aLog = MLXRandom.uniform(-2..<2, [48]), dtBias = MLXRandom.uniform(-4..<4, [48])
        let aLogView = try M5NativeArrayBuffer(aLog, device: device), dtView = try M5NativeArrayBuffer(dtBias, device: device)
        for limit in [Float(4), 80] {
            let ba = MLXRandom.uniform(-limit..<limit, [1, 16, 96]).asType(.bfloat16)
            let baBuffer = try XCTUnwrap(ba.asMTLBuffer(device: device))
            for rows in [1, 3, 7, 8, 15, 16] {
                let a = ba[0..., 0..<rows, 48...], b = ba[0..., 0..<rows, 0..<48]
                let expectedG = Q35CompiledOperations.current.computeG(aLog, a, dtBias)
                let expectedBeta = MLX.sigmoid(b).asType(.float32)
                MLX.eval(expectedG, expectedBeta)
                let g = try XCTUnwrap(device.makeBuffer(length: 16 * 48 * 4, options: .storageModeShared))
                let beta = try XCTUnwrap(device.makeBuffer(length: 16 * 48 * 4, options: .storageModeShared))
                let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                program.encode(into: encoder, ba: baBuffer, aLog: aLogView, dtBias: dtView, g: g, beta: beta, rows: rows)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                let actualG = MLXArray(rawPointer: g.contents(), [1, rows, 48], dtype: .float32) { _ = g }
                let actualBeta = MLXArray(rawPointer: beta.contents(), [1, rows, 48], dtype: .float32) { _ = beta }
                XCTAssertEqual((actualG - expectedG).abs().max().item(Float.self), 0, "g rows=\(rows), range=\(limit)")
                XCTAssertEqual((actualBeta - expectedBeta).abs().max().item(Float.self), 0, "beta rows=\(rows), range=\(limit)")
            }
        }
    }
}
#endif
