#if os(macOS)
import MereRunMLXTestSupport
import MereRunTensor
import Metal
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MereRunQwenModel

final class Q35NativeDraftBlockTests: MLXTestCase {
    func testNativeDraftAttentionUsesFullBlockAndSlidingPrefix() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        Device.setDefault(device: .gpu)
        MLXRandom.seed(1234)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let program = try Q35NativeDraftKernels(device: device), rows = 3
        typealias Binding = Q35NativeDraftKernels.Binding
        for prefix in [5, 2048] {
            let total = prefix + rows
            let q = MLXRandom.uniform(-0.5..<0.5, [1, rows, 32, 128]).asType(.bfloat16)
            let k = MLXRandom.uniform(-0.5..<0.5, [1, total, 8, 128]).asType(.bfloat16)
            let v = MLXRandom.uniform(-0.5..<0.5, [1, total, 8, 128]).asType(.bfloat16)
            let mask = try XCTUnwrap(device.makeBuffer(length: rows * total, options: .storageModePrivate))
            let output = try XCTUnwrap(device.makeBuffer(length: rows * 4096 * 2, options: .storageModeShared))
            let parts = try XCTUnwrap(device.makeBuffer(length: rows * 32 * 9 * 130 * 4, options: .storageModePrivate))
            let buffers = try [q, k, v].map { try Binding(M5NativeArrayBuffer($0, device: device)) } + [.init(mask)]
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
            program.linear(.mask, encoder, [.init(mask)], [rows, total, prefix], count: rows * total)
            let scale = 1 / Float(128).squareRoot()
            let args = [UInt32(rows), UInt32(total), UInt32(total), 0, scale.bitPattern]
            if prefix < 1024 {
                program.encode(.attention, encoder, buffers + [.init(output)], args,
                    groups: MTLSize(width: 32, height: rows, depth: 1), threads: 32)
            } else {
                program.encode(.partial, encoder, buffers + [.init(parts)], args,
                    groups: MTLSize(width: 8, height: rows, depth: (total + 255) / 256), threads: 32)
                program.encode(.merge, encoder, [.init(parts), .init(output)], args,
                    groups: MTLSize(width: 32, height: rows, depth: 1), threads: 32)
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            XCTAssertNil(command.error)
            let actual = MLXArray(rawPointer: output.contents(), [1, rows, 32, 128], dtype: .bfloat16) { _ = output }
            let expected = MLXFast.scaledDotProductAttention(queries: q.transposed(0, 2, 1, 3),
                keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3), scale: scale,
                mask: .array(Q35DFlash2Attention.blockMask(offset: prefix, length: rows, contextLength: prefix,
                                                         window: 2048, dtype: .bfloat16))).transposed(0, 2, 1, 3)
            let difference = (actual.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
            XCTAssertLessThan(difference, 0.003, "prefix=\(prefix)")
        }
    }

    func testNativeDraftRingWrapsAndKeepsTheNewestAcceptedRows() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let kernels = try Q35NativeDraftKernels(device: device)
        func buffer(_ words: Int) throws -> any MTLBuffer {
            try XCTUnwrap(device.makeBuffer(length: words * 2, options: .storageModeShared))
        }
        let prefixK = try buffer(4 * 1024), prefixV = try buffer(4 * 1024)
        let ringK = try buffer(2048 * 1024), ringV = try buffer(2048 * 1024)
        let blockK = try buffer(1024), blockQKV = try buffer(6144)
        let keys = try buffer(5 * 1024), values = try buffer(5 * 1024)
        func initialize(_ buffer: any MTLBuffer, _ values: (Int) -> UInt16) {
            let data = buffer.contents().assumingMemoryBound(to: UInt16.self)
            for index in 0..<(buffer.length / 2) { data[index] = values(index) }
        }
        initialize(prefixK) { UInt16(10 + $0 / 128) }
        initialize(prefixV) { UInt16(20 + $0 / 128) }
        initialize(blockK) { UInt16(100 + $0 / 128) }
        initialize(blockQKV) { UInt16(200 + $0 / 128) }
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(command.makeComputeCommandEncoder(dispatchType: .concurrent))
        kernels.linear(.seed, encoder, [.init(prefixK), .init(prefixV), .init(ringK), .init(ringV)], [4, 2046], count: 4096)
        kernels.linear(.gather, encoder, [.init(ringK), .init(ringV), .init(blockK), .init(blockQKV), .init(keys), .init(values)],
                       [5, 4, 2046], count: 5120)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertNil(command.error)
        let actualK = keys.contents().assumingMemoryBound(to: UInt16.self)
        let actualV = values.contents().assumingMemoryBound(to: UInt16.self)
        for head in 0..<8 {
            for row in 0..<4 {
                XCTAssertEqual(actualK[row * 1024 + head * 128], UInt16(10 + head * 4 + row))
                XCTAssertEqual(actualV[row * 1024 + head * 128], UInt16(20 + head * 4 + row))
            }
            XCTAssertEqual(actualK[4096 + head * 128], UInt16(100 + head))
            XCTAssertEqual(actualV[4096 + head * 128], UInt16(240 + head))
        }
    }
}
#endif
