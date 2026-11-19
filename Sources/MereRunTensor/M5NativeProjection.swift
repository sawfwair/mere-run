#if os(macOS)
import Metal

/// Group-affine verification projection with caller-provided input sums.
@available(macOS 26.0, *)
package final class M5NativeProjection {
    enum Failure: Error { case missingFunction }
    package let inputSize: Int
    package let outputSize: Int
    package let projectedSize: Int
    private let splits: Int
    private let pipeline: any MTLComputePipelineState

    package init(device: any MTLDevice, inputSize: Int, outputSize: Int, arithmetic: GroupAffineArithmetic = .standard, partitionCount: Int? = nil,
                 skippingOutputRows: Range<Int>? = nil) throws {
        precondition(inputSize >= 512 && inputSize.isMultiple(of: 512) && outputSize.isMultiple(of: 64))
        if let gap = skippingOutputRows {
            precondition(!gap.isEmpty && gap.lowerBound >= 0 && gap.upperBound <= outputSize && gap.count < outputSize)
            precondition(gap.lowerBound.isMultiple(of: 32) && gap.upperBound.isMultiple(of: 32))
        }
        projectedSize = outputSize - (skippingOutputRows?.count ?? 0)
        self.inputSize = inputSize
        self.outputSize = outputSize
        splits = partitionCount ?? M5TiledAffineQ4.groupAffinePartitionCount(inputSize: inputSize, outputSize: outputSize, rows: 16)
        precondition([1, 2, 4, 8].contains(splits) && inputSize.isMultiple(of: 64 * splits) && inputSize / splits >= 512)
        let name = "mere_native_affine_\(inputSize)_\(outputSize)"
        let source = M5NativeFFN.shaderHeader + "\n" + M5NativeFFN.projection(
            name: name, width: inputSize, columns: projectedSize, splits: splits, stacked: false,
            arithmetic: arithmetic, skippingOutputRows: skippingOutputRows)
        let library = try M5NativeFFN.makeLibrary(device: device, source: source)
        guard let function = library.makeFunction(name: name) else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                        sums: any MTLBuffer, output: any MTLBuffer, weights: M5NativeFFN.Projection,
                        rows: Int, inputOffset: Int = 0, outputOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0 && weights.owner.originalWeight.shape == [outputSize, inputSize / 8])
        hazards?.prepare(encoder: encoder,
            reads: [.init(input, offset: inputOffset, count: rows * inputSize * 2), .init(sums)],
            writes: [.init(output, offset: outputOffset, count: ((rows + 15) / 16) * 16 * projectedSize * 2)])
        encoder.setComputePipelineState(pipeline)
        let buffers = [(weights.codes, weights.codesOffset), (weights.scales, weights.scalesOffset),
                       (weights.biases, weights.biasesOffset)]
        for (index, item) in buffers.enumerated() {
            encoder.setBuffer(item.0, offset: item.1, index: index)
            encoder.setBuffer(item.0, offset: item.1, index: index + 3)
        }
        encoder.setBuffer(input, offset: inputOffset, index: 6)
        encoder.setBuffer(sums, offset: 0, index: 7)
        encoder.setBuffer(output, offset: outputOffset, index: 8)
        var count = Int32(rows)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 9)
        encoder.dispatchThreadgroups(MTLSize(width: projectedSize / 32, height: (rows + 15) / 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: splits * 32, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }
}
#endif
