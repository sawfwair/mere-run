#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeFullOutput {
    enum Failure: Error { case missingFunction }
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let source = M5NativeFFN.shaderHeader + "\n" + Self.source
        let library = try M5NativeFFN.makeLibrary(device: device, source: source)
        guard let function = library.makeFunction(name: "q35_native_full_gate_sums") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, attention: any MTLBuffer,
                qkv: any MTLBuffer, output: any MTLBuffer, sums: any MTLBuffer, rows: Int, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0)
        hazards?.prepare(encoder: encoder,
            reads: [.init(attention, count: rows * 6144 * 2), .init(qkv, count: rows * 14336 * 2)],
            writes: [.init(output, count: rows * 6144 * 2), .init(sums)])
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in [attention, qkv, output, sums].enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        var rows = Int32(rows)
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: 96, height: ((Int(rows) + 15) / 16) * 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    static let source = """
        kernel void q35_native_full_gate_sums(const device bfloat* attention [[buffer(0)]],
            const device bfloat* qkv [[buffer(1)]], device bfloat* output [[buffer(2)]], device float* sums [[buffer(3)]],
            constant int& rows [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
            uint lane [[thread_index_in_threadgroup]]) {
            const uint row = group.y, column = group.x * 64 + lane;
            if (row >= uint(rows)) {
                if (lane == 0) sums[group.x * (((rows + 15) / 16) * 16) + row] = 0.0f;
                return;
            }
            const uint head = column / 256, dim = column % 256;
            const bfloat result = attention[row * 6144 + column] * mere_tiled_sigmoid(qkv[row * 14336 + head * 512 + 256 + dim]);
            threadgroup bfloat values[64];
            values[lane] = result;
            output[row * 6144 + column] = result;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                float total = 0.0f;
                for (uint i = 0; i < 64; ++i) total += float(values[i]);
                sums[group.x * (((rows + 15) / 16) * 16) + row] = total;
            }
        }
        """
}
#endif
