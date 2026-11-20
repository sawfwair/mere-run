#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeGDNPrework {
    enum Failure: Error { case missingFunction }
    struct Output {
        let q: any MTLBuffer
        let k: any MTLBuffer
        let v: any MTLBuffer
        let convolution: any MTLBuffer
        let qOffset: Int
        let kOffset: Int
        let vOffset: Int
        let convolutionOffset: Int
        init(q: any MTLBuffer, k: any MTLBuffer, v: any MTLBuffer, convolution: any MTLBuffer,
             qOffset: Int = 0, kOffset: Int = 0, vOffset: Int = 0, convolutionOffset: Int = 0) {
            self.q = q; self.k = k; self.v = v; self.convolution = convolution
            self.qOffset = qOffset; self.kOffset = kOffset; self.vOffset = vOffset
            self.convolutionOffset = convolutionOffset
        }
    }
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let source = Q35GDNPreworkMetalKernel.source
            .replacingOccurrences(of: "qkv[uint(source) * uint(C) + channel]", with: "qkv[uint(source) * uint(qkvStride) + channel]")
            .replacingOccurrences(of: "conv_state[uint(int(NKEEP) + source) * uint(C) + channel]",
                with: "conv_state[uint(int(NKEEP) + source) * uint(convStride) + channel]")
            .replacingOccurrences(of: "conv_state[state_base + i]",
                with: "conv_state[state_row * uint(convStride) + channel_base + lane * 4 + i]")
            .replacingOccurrences(of: "uint(S) + state_row", with: "uint(rows) + state_row")
            .replacingOccurrences(of: "conv_out[state_base + i] = Tree ?", with: "conv_out[state_base + i] = !advance ?")
            .replacingOccurrences(of: "conv_state[input_row * uint(C) + channel]",
                                  with: "conv_state[input_row * uint(convStride) + channel]")
            .replacingOccurrences(of: "qkv[(input_row - uint(NKEEP)) * uint(C) + channel]",
                                  with: "qkv[(input_row - uint(NKEEP)) * uint(qkvStride) + channel]")
        let shader = M5NativeFFN.shaderHeader + "\n" + """
            kernel void q35_native_gdn_prework(const device bfloat* qkv [[buffer(0)]],
                const device bfloat* conv_state [[buffer(1)]], const device bfloat* conv_w [[buffer(2)]],
                const device int* ancestry [[buffer(3)]], device bfloat* q_out [[buffer(4)]],
                device bfloat* k_out [[buffer(5)]], device bfloat* v_out [[buffer(6)]],
                device bfloat* conv_out [[buffer(7)]], constant int& qkvStride [[buffer(8)]],
                constant int& convStride [[buffer(9)]], constant int& rows [[buffer(10)]],
                constant uint& advance [[buffer(11)]], uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
                typedef bfloat T;
                constexpr bool Tree = true;
                constexpr int HK = 16, HV = 48, DK = 128, DV = 128, NKEEP = 3, C = 10240;
                const bfloat q_scale = bfloat(0.0078125f), k_scale = bfloat(0.0883883476f);
                \(source)
            }
            """
        let library = try M5NativeFFN.makeLibrary(device: device, source: shader)
        guard let function = library.makeFunction(name: "q35_native_gdn_prework") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, qkv: any MTLBuffer, qkvStride: Int,
                convolution: M5NativeArrayBuffer, weight: M5NativeArrayBuffer, ancestry: any MTLBuffer,
                output: Output, rows: Int, qkvOffset: Int = 0, advance: Bool = false, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0 && qkvStride >= 10_240)
        precondition(convolution.shape == [1, 3, 10_240] && convolution.strides.last == 1)
        precondition(weight.shape == [10_240, 4, 1] && weight.strides[0] == 4 && weight.strides[1] == 1)
        precondition(convolution.dtype == .bfloat16 && weight.dtype == .bfloat16)
        hazards?.prepare(encoder: encoder,
            reads: [.init(qkv, offset: qkvOffset, count: ((rows - 1) * qkvStride + 10240) * 2)],
            writes: [.init(output.q, offset: output.qOffset, count: rows * 16 * 128 * 2),
                     .init(output.k, offset: output.kOffset, count: rows * 16 * 128 * 2),
                     .init(output.v, offset: output.vOffset, count: rows * 48 * 128 * 2), .init(output.convolution, offset: output.convolutionOffset, count: 3 * 10240 * 2)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qkv, offset: qkvOffset, index: 0)
        encoder.setBuffer(convolution.buffer, offset: convolution.offset, index: 1)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 2)
        encoder.setBuffer(ancestry, offset: 0, index: 3)
        let offsets = [output.qOffset, output.kOffset, output.vOffset, output.convolutionOffset]
        for (index, buffer) in [output.q, output.k, output.v, output.convolution].enumerated() {
            encoder.setBuffer(buffer, offset: offsets[index], index: index + 4)
        }
        var qkvStride = Int32(qkvStride), convStride = Int32(convolution.strides[1])
        encoder.setBytes(&qkvStride, length: 4, index: 8)
        encoder.setBytes(&convStride, length: 4, index: 9)
        var count = Int32(rows), retainFinal = UInt32(advance ? 1 : 0)
        encoder.setBytes(&count, length: 4, index: 10)
        encoder.setBytes(&retainFinal, length: 4, index: 11)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 80),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }
}
#endif
