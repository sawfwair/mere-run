import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunTensor
#if os(macOS)
import Metal

final class M5NativeArrayBufferTests: MLXTestCase {
    func testSlicedStridedBuffersRetainBackingThroughQueuedGPURead() throws {
        try XCTSkipUnless(M5AffineQ4Verification.isSupported, "Requires M5 tensor support.")
        Device.setDefault(device: .gpu)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice()), queue = try XCTUnwrap(device.makeCommandQueue())
        let library = try device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            kernel void copy_view(const device uint* x [[buffer(0)]], device uint* y [[buffer(1)]],
                constant int* shape [[buffer(2)]], constant int* strides [[buffer(3)]],
                uint at [[thread_position_in_grid]]) {
                if (at >= uint(shape[0] * shape[1] * shape[2])) return;
                uint i = at / (shape[1] * shape[2]), j = (at / shape[2]) % shape[1], k = at % shape[2];
                y[at] = x[i * strides[0] + j * strides[1] + k * strides[2]];
            }
            """, options: nil)
        let pipeline = try device.makeComputePipelineState(function: try XCTUnwrap(library.makeFunction(name: "copy_view")))
        for strided in [false, true] {
            weak var owner: MLXArray?
            var expected: [UInt32] = []
            var lease: M5NativeArrayBuffer? = try {
                let base = MLXArray.arange(3 * 17 * 37).asType(.uint32).reshaped(3, 17, 37)
                let selected = base[1..<3, 1..<16, 1..<36]
                let array = strided ? selected.transposed(1, 0, 2) : selected.contiguous()
                expected = array.asArray(UInt32.self)
                owner = array
                return try M5NativeArrayBuffer(array, device: device)
            }()
            XCTAssertNotNil(owner)
            let shape = try XCTUnwrap(lease).shape.map(Int32.init)
            let strides = try XCTUnwrap(lease).strides.map(Int32.init)
            let output = try XCTUnwrap(device.makeBuffer(length: expected.count * 4, options: .storageModeShared))
            let command = try XCTUnwrap(queue.makeCommandBuffer()), encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(try XCTUnwrap(lease).buffer, offset: try XCTUnwrap(lease).offset, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            strides.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
            encoder.dispatchThreads(MTLSize(width: expected.count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
            lease = nil
            XCTAssertNotNil(owner, "The command's Metal resource must retain the MLX backing owner.")
            command.commit(); command.waitUntilCompleted()
            XCTAssertNil(command.error)
            let values = output.contents().bindMemory(to: UInt32.self, capacity: expected.count)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: values, count: expected.count)), expected)
        }
    }
}
#endif
