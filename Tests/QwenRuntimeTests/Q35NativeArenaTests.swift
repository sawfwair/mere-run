import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunQwenModel
#if os(macOS)
import Metal

final class Q35NativeArenaTests: MLXTestCase {
    func testOffsetViewsRetainStorageWithoutRetainingArena() throws {
        guard #available(macOS 26.0, *) else { throw XCTSkip("Requires Metal 4.") }
        Device.setDefault(device: .gpu)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        for dtype in [DType.bfloat16, .float32] {
            weak var released: Q35NativeArena?
            func produce() throws -> (MLXArray, MLXArray) {
                let arena = try Q35NativeArena(device: device, bytes: 2048, dtype: dtype)
                released = arena
                let first = arena.allocate(48), second = arena.allocate(64)
                XCTAssertGreaterThan(second.offset, first.offset)
                let x = MLXArray(Array(0..<48).map(Float.init)).asType(dtype)
                let y = MLXArray(Array(0..<64).map { Float($0 + 64) }).asType(dtype)
                let sourceX = try XCTUnwrap(x.asMTLBuffer(device: device)), sourceY = try XCTUnwrap(y.asMTLBuffer(device: device))
                let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeBlitCommandEncoder())
                encoder.copy(from: sourceX, sourceOffset: 0, to: first.buffer, destinationOffset: first.offset, size: 48 * dtype.size)
                encoder.copy(from: sourceY, sourceOffset: 0, to: second.buffer, destinationOffset: second.offset, size: 64 * dtype.size)
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                XCTAssertNil(command.error)
                return (first.array([1, 3, 16]), second.array([1, 4, 16]))
            }
            let (first, second) = try produce()
            XCTAssertNil(released, "Returned MLX views retain storage without an ownership cycle.")
            XCTAssertEqual(first.asType(.float32).asArray(Float.self), Array(0..<48).map(Float.init))
            XCTAssertEqual(second.asType(.float32).asArray(Float.self), Array(0..<64).map { Float($0 + 64) })
            XCTAssertEqual(MLX.square(first.asType(.float32)).sum().item(Float.self), (0..<48).map { Float($0 * $0) }.reduce(0, +))
        }
    }
}
#endif
