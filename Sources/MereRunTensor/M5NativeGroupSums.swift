#if os(macOS)
import Metal

@available(macOS 26.0, *)
package final class M5NativeGroupSums {
    enum Failure: Error { case missingFunction }
    private let inputSize: Int
    private let pipeline: any MTLComputePipelineState

    package init(device: any MTLDevice, inputSize: Int) throws {
        precondition(inputSize > 0 && inputSize.isMultiple(of: 64))
        self.inputSize = inputSize
        let name = "mere_native_group_sums_\(inputSize)"
        let library = try M5NativeFFN.makeLibrary(device: device, source: M5NativeFFN.shaderHeader + "\n"
            + M5NativeFFN.sums(name: name, width: inputSize))
        guard let function = library.makeFunction(name: name) else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                        sums: any MTLBuffer, rows: Int, inputOffset: Int = 0) {
        precondition((1...16).contains(rows))
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        encoder.setBuffer(sums, offset: 0, index: 1)
        var count = Int32(rows)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 2)
        encoder.dispatchThreads(MTLSize(width: inputSize / 64 * 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }
}
#endif
