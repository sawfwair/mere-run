import MLX
import MLXRandom
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
@testable import MereRunTensor
#if os(macOS)
import Metal

final class Q35NativeGDNPreworkTests: MLXTestCase {
    func testStridedTreeConvolutionAndNormalizationMatchMLX() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        MLXRandom.seed(938)
        try QuantizedVerificationContext.$maximumM5Rows.withValue(16) {
            let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
            let program = try Q35NativeGDNPrework(device: device)
            let full = MLXRandom.uniform(-1..<1, [1, 16, 16_384]).asType(.bfloat16)
            let base = MLXRandom.uniform(-1..<1, [1, 5, 10_240]).asType(.bfloat16)
            let convolution = base[0..., 1..<4, 0...]
            let weight = MLXRandom.uniform(-0.25..<0.25, [10_240, 4, 1]).asType(.bfloat16)
            let inputView = try M5NativeArrayBuffer(full, device: device)
            let convolutionView = try M5NativeArrayBuffer(convolution, device: device)
            let weightView = try M5NativeArrayBuffer(weight, device: device)
            func buffer(_ count: Int) throws -> any MTLBuffer {
                try XCTUnwrap(device.makeBuffer(length: count * 2, options: .storageModeShared))
            }
            let output = try Q35NativeGDNPrework.Output(q: buffer(16 * 16 * 128), k: buffer(16 * 16 * 128),
                v: buffer(16 * 48 * 128), convolution: buffer(3 * 10_240))
            for rows in [1, 3, 7, 8, 15, 16] {
                for balanced in [false, true] {
                    let tree = try Q35VerificationTree(parents: [-1] + (1..<rows).map { balanced ? ($0 - 1) / 2 : $0 - 1 })
                    let qkv = full[0..., 0..<rows, 0..<10_240]
                    let expected = try XCTUnwrap(q35GDNPreworkMetal(qkv: qkv, convState: convolution, convWeight: weight,
                        numKeyHeads: 16, numValueHeads: 48, keyHeadDim: 128, valueHeadDim: 128, tree: tree))
                    MLX.eval(expected.q, expected.k, expected.v, expected.convState)
                    let ancestry = tree.convolutionRows(keep: 3).flatMap { $0 }.map(Int32.init)
                    let plan = try ancestry.withUnsafeBytes {
                        try XCTUnwrap(device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
                    }
                    let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
                    program.encode(into: encoder, qkv: inputView.buffer, qkvStride: 16_384,
                        convolution: convolutionView, weight: weightView, ancestry: plan, output: output,
                        rows: rows, qkvOffset: inputView.offset)
                    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                    XCTAssertNil(command.error)
                    for (buffer, value) in [(output.q, expected.q), (output.k, expected.k), (output.v, expected.v),
                                            (output.convolution, expected.convState)] {
                        let actual = MLXArray(rawPointer: buffer.contents(), value.shape, dtype: .bfloat16) { _ = buffer }
                        XCTAssertEqual((actual.asType(.float32) - value.asType(.float32)).abs().max().item(Float.self),
                            0, "rows=\(rows), balanced=\(balanced)")
                    }
                }
            }
        }
    }
}
#endif
