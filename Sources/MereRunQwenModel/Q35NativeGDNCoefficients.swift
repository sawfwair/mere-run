#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeGDNCoefficients {
    enum Failure: Error { case missingFunction }
    private let pipeline: any MTLComputePipelineState

    init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: M5NativeFFN.shaderHeader + "\n" + Self.source)
        guard let function = library.makeFunction(name: "q35_native_gdn_coefficients") else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, ba: any MTLBuffer,
                aLog: M5NativeArrayBuffer, dtBias: M5NativeArrayBuffer,
                g: any MTLBuffer, beta: any MTLBuffer, rows: Int,
                baOffset: Int = 0, gOffset: Int = 0, betaOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0)
        precondition(aLog.shape == [48] && dtBias.shape == [48] && aLog.dtype == .float32 && dtBias.dtype == .float32)
        hazards?.prepare(encoder: encoder, reads: [.init(ba, offset: baOffset, count: rows * 96 * 2)],
            writes: [.init(g, offset: gOffset, count: rows * 48 * 4), .init(beta, offset: betaOffset, count: rows * 48 * 4)])
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(ba, offset: baOffset, index: 0)
        encoder.setBuffer(aLog.buffer, offset: aLog.offset, index: 1)
        encoder.setBuffer(dtBias.buffer, offset: dtBias.offset, index: 2)
        encoder.setBuffer(g, offset: gOffset, index: 3)
        encoder.setBuffer(beta, offset: betaOffset, index: 4)
        var count = Int32(rows)
        encoder.setBytes(&count, length: MemoryLayout<Int32>.size, index: 5)
        encoder.dispatchThreads(MTLSize(width: rows * 48, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    // log1p and logaddexp follow Apple's MIT-licensed MLX utils.h and
    // binary_ops.h. Exp and sigmoid retain the existing precise boundaries.
    static let source = """
        inline float q35_native_log1p(float x) {
            const float xp1 = 1.0f + x;
            if (xp1 == metal::numeric_limits<float>::max()) return metal::numeric_limits<float>::max();
            if (xp1 == 1.0f) return x;
            return x * (metal::log(xp1) / (xp1 - 1.0f));
        }
        inline float q35_native_softplus(float x) {
            if (metal::isnan(x)) return metal::numeric_limits<float>::quiet_NaN();
            const float high = metal::max(x, 0.0f), low = metal::min(x, 0.0f);
            return (low == -INFINITY || high == INFINITY) ? high
                : high + q35_native_log1p(metal::exp(low - high));
        }
        kernel void q35_native_gdn_coefficients(const device bfloat* ba [[buffer(0)]],
            const device float* a_log [[buffer(1)]], const device float* dt_bias [[buffer(2)]],
            device float* g [[buffer(3)]], device float* beta [[buffer(4)]],
            constant int& rows [[buffer(5)]], uint at [[thread_position_in_grid]]) {
            if (at >= uint(rows * 48)) return;
            const uint row = at / 48, head = at % 48;
            const float activation = float(ba[row * 96 + 48 + head]);
            const float delta = q35_native_softplus(activation + dt_bias[head]);
            const float decay = metal::precise::exp(a_log[head]);
            g[at] = metal::precise::exp(-decay * delta);
            beta[at] = float(mere_tiled_sigmoid(ba[row * 96 + head]));
        }
        """
}
#endif
