#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeGDNOutput {
    enum Failure: Error { case missingFunction }
    private let pipeline: any MTLComputePipelineState

    private let gateStride: Int

    init(device: any MTLDevice, separateGate: Bool = false) throws {
        gateStride = separateGate ? 6144 : 16384
        let geometry = "#define Q35_GATE_STRIDE " + String(gateStride) + "\n#define Q35_GATE_OFFSET " + (separateGate ? "0" : "10240") + "\n"
        let library = try M5NativeFFN.makeLibrary(device: device, source: M5NativeFFN.shaderHeader + "\n" + geometry + Self.source)
        guard let function = library.makeFunction(name: "q35_native_gdn_norm_gate_sums") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, updated: any MTLBuffer, qkvz: any MTLBuffer,
                weight: M5NativeArrayBuffer, output: any MTLBuffer, sums: any MTLBuffer, rows: Int, epsilon: Float, qkvzOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0 && weight.shape == [128] && weight.dtype == .bfloat16)
        hazards?.prepare(encoder: encoder,
            reads: [.init(updated, count: rows * 6144 * 2), .init(qkvz, offset: qkvzOffset, count: rows * gateStride * 2)],
            writes: [.init(output, count: rows * 6144 * 2), .init(sums)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(updated, offset: 0, index: 0)
        encoder.setBuffer(qkvz, offset: qkvzOffset, index: 1)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(sums, offset: 0, index: 4)
        var count = Int32(rows), epsilon = epsilon
        encoder.setBytes(&count, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 48, height: ((Int(rows) + 15) / 16) * 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    static let source = """
        kernel void q35_native_gdn_norm_gate_sums(const device bfloat* updated [[buffer(0)]],
            const device bfloat* qkvz [[buffer(1)]], const device bfloat* weight [[buffer(2)]],
            device bfloat* output [[buffer(3)]], device float* sums [[buffer(4)]],
            constant int& rows [[buffer(5)]], constant float& eps [[buffer(6)]],
            uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            const uint head = group.x, row = group.y;
            if (row >= uint(rows)) {
                if (lane < 2) sums[(head * 2 + lane) * (((rows + 15) / 16) * 16) + row] = 0.0f;
                return;
            }
            float x[4], acc = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                x[i] = float(updated[(row * 48 + head) * 128 + lane * 4 + i]);
                acc += x[i] * x[i];
            }
            acc = simd_sum(acc);
            const float inverse = metal::precise::rsqrt(acc / 128.0f + eps);
            threadgroup bfloat values[128];
            for (uint i = 0; i < 4; ++i) {
                const uint column = lane * 4 + i;
                const bfloat norm = weight[column] * bfloat(x[i] * inverse);
                const float gate = float(qkvz[row * Q35_GATE_STRIDE + Q35_GATE_OFFSET + head * 128 + column]);
                const float y = 1 / (1 + metal::precise::exp(metal::abs(gate)));
                const float activated = gate * ((gate < 0) ? y : 1 - y);
                const bfloat result = bfloat(activated * float(norm));
                values[column] = result;
                output[(row * 48 + head) * 128 + column] = result;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 2) {
                float total = 0.0f;
                for (uint i = 0; i < 64; ++i) total += float(values[lane * 64 + i]);
                sums[(head * 2 + lane) * (((rows + 15) / 16) * 16) + row] = total;
            }
        }
        """
}
#endif
