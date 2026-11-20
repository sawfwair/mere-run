#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeDraftKernels {
    enum Operation: String, CaseIterable {
        case embeddings = "q35_draft_embeddings"
        case taps = "q35_draft_taps"
        case convolution = "q35_draft_convolution"
        case rotary = "q35_draft_rotary"
        case seed = "q35_draft_seed"
        case append = "q35_draft_append"
        case gather = "q35_draft_gather"
        case mask = "q35_draft_mask"
        case attention = "tf_shared_lane_attention"
        case partial = "tf_shared_lane_attention_split"
        case merge = "tf_shared_lane_attention_merge"
    }

    enum Failure: Error { case allocation, geometry, command(Error?) }
    struct Binding {
        let buffer: any MTLBuffer
        let offset: Int
        init(_ buffer: any MTLBuffer, offset: Int = 0) { self.buffer = buffer; self.offset = offset }
        init(_ view: M5NativeArrayBuffer) { self.init(view.buffer, offset: view.offset) }
    }

    private let pipelines: [Operation: any MTLComputePipelineState]

    init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source + "\n" + Self.attentionSource)
        pipelines = try Dictionary(uniqueKeysWithValues: Operation.allCases.map { operation in
            guard let function = library.makeFunction(name: operation.rawValue) else { throw Failure.geometry }
            return try (operation, device.makeComputePipelineState(function: function))
        })
    }

    func encode(_ operation: Operation, _ encoder: any MTLComputeCommandEncoder, _ bindings: [Binding],
                _ arguments: [UInt32], groups: MTLSize, threads: Int) {
        encoder.setComputePipelineState(pipelines[operation]!)
        let argumentIndex = operation == .partial || operation == .attention ? 4 : operation == .merge ? 1 : bindings.count
        for (index, buffer) in bindings.enumerated() {
            let slot = index >= argumentIndex ? index + 1 : index
            encoder.setBuffer(buffer.buffer, offset: buffer.offset, index: slot)
        }
        arguments.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: argumentIndex) }
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }

    func linear(_ operation: Operation, _ encoder: any MTLComputeCommandEncoder, _ buffers: [Binding],
                _ arguments: [Int], count: Int) {
        encode(operation, encoder, buffers, arguments.map(UInt32.init),
               groups: MTLSize(width: (count + 255) / 256, height: 1, depth: 1), threads: 256)
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;
        constant uint H = 5120, KV = 1024, Window = 2048;

        kernel void q35_draft_embeddings(const device uint* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
            const device bfloat* biases [[buffer(2)]], device bfloat* y [[buffer(3)]],
            constant uint* a [[buffer(4)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * H) return;
            const uint c = index % H, t = index < H ? a[1] : a[2];
            const uint word = w[t * (H / 8) + c / 8], group = t * (H / 64) + c / 64;
            const float code = float((word >> ((c % 8) * 4)) & 15);
            y[index] = bfloat(fma(float(scales[group]), code, float(biases[group])));
        }

        kernel void q35_draft_taps(const device bfloat* p0 [[buffer(0)]], const device bfloat* p1 [[buffer(1)]],
            const device bfloat* p2 [[buffer(2)]], const device bfloat* p3 [[buffer(3)]], const device bfloat* p4 [[buffer(4)]],
            device bfloat* output [[buffer(5)]], constant uint* a [[buffer(6)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * H * 5) return;
            const uint column = index % H, row = index / (H * 5), layer = (index / H) % 5;
            const device bfloat* taps[] = {p0, p1, p2, p3, p4};
            output[index] = taps[layer][row * H + column];
        }

        kernel void q35_draft_convolution(const device bfloat* x [[buffer(0)]], const device bfloat* dynamic [[buffer(1)]],
            const device bfloat* base [[buffer(2)]], const device bfloat* residual [[buffer(3)]],
            device bfloat* output [[buffer(4)]], constant uint* a [[buffer(5)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * H) return;
            const uint row = index / H, col = index % H, branch = a[1];
            const uint d = row * 1280 + branch * 640 + col / 16, b = branch * H * 2 + col;
            const bfloat first = bfloat(float(bfloat(float(base[b]) + float(dynamic[d]))) * float(x[index]));
            const bfloat second = row == 0 ? bfloat(0) : bfloat(float(bfloat(float(base[b + H])
                                                   + float(dynamic[d + 320]))) * float(x[index - H]));
            const bfloat result = bfloat(float(first) + float(second));
            output[index] = branch == 0 ? result : bfloat(float(residual[index]) + float(result));
        }

        kernel void q35_draft_rotary(const device bfloat* x [[buffer(0)]], const device bfloat* weight [[buffer(1)]],
            device bfloat* output [[buffer(2)]], constant uint* a [[buffer(3)]],
            uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            const uint row = g.x, head = g.y;
            const device bfloat* input = x + row * a[0] + a[1] + head * 128;
            device bfloat* y = output + (row * a[2] + head) * 128;
            float square = 0;
            for (uint i = 0; i < 4; ++i) { const float value = float(input[lane * 4 + i]); square += value * value; }
            const float inverse = rsqrt(simd_sum(square) / 128.0f + as_type<float>(a[4]));
            for (uint i = 0; i < 2; ++i) {
                const uint column = lane + i * 32;
                const float freq = exp2(-float(column) / 64.0f * as_type<float>(a[5]));
                const float angle = float(a[3] + row) * freq;
                const float left = float(bfloat(float(bfloat(float(input[column]) * inverse)) * float(weight[column])));
                const float right = float(bfloat(float(bfloat(float(input[column + 64]) * inverse)) * float(weight[column + 64])));
                const float cosine = fast::cos(angle), sine = fast::sin(angle);
                y[column] = bfloat(left * cosine - right * sine);
                y[column + 64] = bfloat(left * sine + right * cosine);
            }
        }

        kernel void q35_draft_seed(const device bfloat* k [[buffer(0)]], const device bfloat* v [[buffer(1)]],
            device bfloat* ringK [[buffer(2)]], device bfloat* ringV [[buffer(3)]],
            constant uint* a [[buffer(4)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * KV) return;
            const uint t = index / KV, lane = index % KV, dst = ((a[1] + t) % Window) * KV + lane;
            const uint src = ((lane / 128) * a[0] + t) * 128 + lane % 128;
            ringK[dst] = k[src]; ringV[dst] = v[src];
        }

        kernel void q35_draft_append(const device bfloat* keys [[buffer(0)]], const device bfloat* kv [[buffer(1)]],
            device bfloat* ringK [[buffer(2)]], device bfloat* ringV [[buffer(3)]],
            constant uint* a [[buffer(4)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * KV) return;
            const uint row = index / KV, lane = index % KV, dst = ((a[1] + row) % Window) * KV + lane;
            ringK[dst] = keys[index]; ringV[dst] = kv[row * (KV * 2) + KV + lane];
        }

        kernel void q35_draft_gather(const device bfloat* ringK [[buffer(0)]], const device bfloat* ringV [[buffer(1)]],
            const device bfloat* blockK [[buffer(2)]], const device bfloat* qkv [[buffer(3)]],
            device bfloat* keys [[buffer(4)]], device bfloat* values [[buffer(5)]],
            constant uint* a [[buffer(6)]], uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * KV) return;
            const uint row = index / KV, lane = index % KV, prefix = a[1];
            if (row < prefix) {
                const uint src = ((a[2] + row) % Window) * KV + lane;
                keys[index] = ringK[src]; values[index] = ringV[src];
            } else {
                keys[index] = blockK[(row - prefix) * KV + lane];
                values[index] = qkv[(row - prefix) * 6144 + 5120 + lane];
            }
        }

        kernel void q35_draft_mask(device uchar* mask [[buffer(0)]], constant uint* a [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            if (index >= a[0] * a[1]) return;
            const uint column = index % a[1], row = index / a[1], prefix = a[2];
            const uint hidden = prefix + row + 1 > Window ? prefix + row + 1 - Window : 0;
            mask[index] = uchar(column >= prefix || column >= hidden);
        }
        """
}
#endif
