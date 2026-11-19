import MereRunKVCache
import MLX
import MLXNN

struct Q35GreedyTreeRequest {
    let head: Linear
    let tokens: [Int]
    let budget: Int
    let eos: Set<Int>
}

package struct Q35GreedyTreeSelection {
    package let path: [Int]
    package let following: Int
    package let picks: [Int]
}

struct Q35GreedyTreeKV {
    let keys: MLXArray
    let values: MLXArray
}

package struct Q35GreedyTreeOutput {
    package let selection: Q35GreedyTreeSelection
    package let taps: [Int: MLXArray]
    let attention: [Int: Q35GreedyTreeKV]
    #if os(macOS)
    var nativeDraftTaps: [Int: Q35NativeDraftTap] = [:]
    #endif

    package func commit(candidates: [Q35LayerCache?], to live: [Q35LayerCache?]) {
        for index in live.indices {
            if let selected = attention[index], case .full(let cache as KVCacheSimple)? = live[index] {
                _ = cache.update(keys: selected.keys, values: selected.values)
            } else {
                live[index]!.commitVerificationTree(candidate: candidates[index]!, path: selection.path)
            }
        }
    }
}

#if os(macOS)
import Metal
import MereRunTensor

struct Q35NativeDraftTap {
    let buffer: any MTLBuffer
    let offset: Int
}

/// The native body supplies normalized BF16 rows and the exact group-affine
/// input sums. This tail keeps the head, reduction, and accepted taps on its
/// last command buffer. Only the small selection record is read by the host.
@available(macOS 26.0, *)
final class Q35NativeGreedyTail {
    private static let vocabulary = 248_320
    private static let chunk = 2_048
    private static let chunks = (vocabulary + chunk - 1) / chunk
    private static let metadataCount = 34
    static let pathOffset = 2 * MemoryLayout<Int32>.stride

    let headIdentity: ObjectIdentifier
    private let weights: M5NativeFFN.Projection
    private let head: M5NativeProjection
    private let rowSums: M5NativeGroupSums
    private let reduce: any MTLComputePipelineState
    private let match: any MTLComputePipelineState
    private let gather: any MTLComputePipelineState
    private let gatherKV: any MTLComputePipelineState
    let logits: any MTLBuffer
    private let partials: any MTLBuffer
    let metadata: any MTLBuffer

    init?(layer: Linear, device: any MTLDevice) throws {
        guard let weights = try PortableQuantizedLinear.nativeProjectionWeights(layer, device: device),
              weights.inputSize == 5_120, weights.outputSize == Self.vocabulary else { return nil }
        headIdentity = ObjectIdentifier(layer)
        self.weights = weights
        head = try M5NativeProjection(device: device, inputSize: weights.inputSize, outputSize: weights.outputSize)
        rowSums = try M5NativeGroupSums(device: device, inputSize: weights.inputSize)
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        func pipeline(_ name: String) throws -> any MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw Failure.function }
            return try device.makeComputePipelineState(function: function)
        }
        reduce = try pipeline("q35_head_partials")
        match = try pipeline("q35_head_match")
        gather = try pipeline("q35_gather_taps")
        gatherKV = try pipeline("q35_gather_kv")
        let privateOptions: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        guard let logits = device.makeBuffer(length: 16 * Self.vocabulary * 2, options: privateOptions),
              let partials = device.makeBuffer(length: 16 * Self.chunks * 8, options: privateOptions),
              let metadata = Self.makeMetadata(device: device)
        else { throw Failure.allocation }
        self.logits = logits
        self.partials = partials
        self.metadata = metadata
    }

    private enum Failure: Error { case function, allocation }

    static func makeMetadata(device: any MTLDevice) -> (any MTLBuffer)? {
        device.makeBuffer(length: metadataCount * MemoryLayout<Int32>.stride,
                          options: [.storageModeShared, .hazardTrackingModeUntracked])
    }

    func encode(into encoder: any MTLComputeCommandEncoder, normalized: Q35NativeArena.Slice, sums: any MTLBuffer,
                request: Q35GreedyTreeRequest, tree: Q35VerificationTree, metadata: any MTLBuffer, hazards: M5NativeHazards) {
        encodeHead(into: encoder, normalized: normalized, inputOffset: normalized.offset, sums: sums,
                   tokens: request.tokens, parents: tree.parents, budget: request.budget, eos: request.eos,
                   metadata: metadata, hazards: hazards)
    }

    func encodeLast(
        into encoder: any MTLComputeCommandEncoder, normalized: Q35NativeArena.Slice, sums: any MTLBuffer,
        rows: Int, token: Int, metadata: any MTLBuffer, hazards: M5NativeHazards
    ) {
        let inputOffset = normalized.offset + (rows - 1) * 5_120 * 2
        hazards.prepare(encoder: encoder, reads: [.init(normalized.buffer, offset: inputOffset, count: 5_120 * 2)],
                        writes: [.init(sums)])
        rowSums.encode(into: encoder, input: normalized.buffer, sums: sums, rows: 1, inputOffset: inputOffset)
        encodeHead(into: encoder, normalized: normalized, inputOffset: inputOffset, sums: sums,
                   tokens: [token], parents: [-1], budget: 1, eos: [],
                   metadata: metadata, hazards: hazards)
    }

    func project(into encoder: any MTLComputeCommandEncoder, normalized: Q35NativeArena.Slice,
                 sums: any MTLBuffer, rows: Int, output: any MTLBuffer, hazards: M5NativeHazards) {
        head.encode(into: encoder, input: normalized.buffer, sums: sums, output: output, weights: weights,
                    rows: rows, inputOffset: normalized.offset, hazards: hazards)
    }

    private func encodeHead(
        into encoder: any MTLComputeCommandEncoder, normalized: Q35NativeArena.Slice,
        inputOffset: Int, sums: any MTLBuffer, tokens: [Int], parents: [Int], budget: Int, eos: Set<Int>,
        metadata: any MTLBuffer, hazards: M5NativeHazards
    ) {
        let rows = tokens.count
        head.encode(into: encoder, input: normalized.buffer, sums: sums, output: logits, weights: weights,
            rows: rows, inputOffset: inputOffset, hazards: hazards)
        hazards.prepare(encoder: encoder, reads: [.init(logits)], writes: [.init(partials)])
        encoder.setComputePipelineState(reduce)
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(partials, offset: 0, index: 1)
        encoder.dispatchThreadgroups(MTLSize(width: Self.chunks, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        hazards.prepare(encoder: encoder, reads: [.init(partials)], writes: [.init(metadata)])
        encoder.setComputePipelineState(match)
        encoder.setBuffer(partials, offset: 0, index: 0)
        encoder.setBuffer(metadata, offset: 0, index: 1)
        let arguments = [rows, min(budget, rows), eos.count] + tokens + parents + eos.sorted()
        arguments.map(Int32.init).withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        let scalar = MTLSize(width: 1, height: 1, depth: 1)
        encoder.dispatchThreadgroups(scalar, threadsPerThreadgroup: scalar)
    }

    func gatherTap(_ source: Q35NativeArena.Slice, into output: Q35NativeArena.Slice,
                   metadata: any MTLBuffer, encoder: any MTLComputeCommandEncoder, hazards: M5NativeHazards) {
        hazards.prepare(encoder: encoder, reads: [.init(metadata), .init(source.buffer, offset: source.offset, count: source.byteCount)],
                        writes: [.init(output.buffer, offset: output.offset, count: output.byteCount)])
        encoder.setComputePipelineState(gather)
        encoder.setBuffer(source.buffer, offset: source.offset, index: 0)
        encoder.setBuffer(output.buffer, offset: output.offset, index: 1)
        encoder.setBuffer(metadata, offset: 0, index: 2)
        encoder.dispatchThreads(MTLSize(width: 5_120 / 4, height: 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func gatherAttention(_ inputs: [Q35NativeArena.Slice], into outputs: [Q35NativeArena.Slice],
                         metadata: any MTLBuffer, encoder: any MTLComputeCommandEncoder, hazards: M5NativeHazards) {
        let input = inputs[0].buffer, output = outputs[0].buffer
        hazards.prepare(encoder: encoder, reads: [.init(metadata), .init(input)], writes: [.init(output)])
        encoder.setComputePipelineState(gatherKV)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(metadata, offset: 0, index: 2)
        let offsets = zip(inputs, outputs).flatMap { [UInt32($0.offset / 8), UInt32($1.offset / 8)] }
        offsets.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        encoder.dispatchThreads(MTLSize(width: 64, height: 16, depth: inputs.count * 4),
            threadsPerThreadgroup: MTLSize(width: 64, height: 2, depth: 1))
    }

    func selection(from metadata: any MTLBuffer, rows: Int) -> Q35GreedyTreeSelection? {
        let values = metadata.contents().assumingMemoryBound(to: Int32.self)
        let count = Int(values[0])
        guard (1...rows).contains(count) else { return nil }
        return Q35GreedyTreeSelection(path: (2..<(2 + count)).map { Int(values[$0]) },
            following: Int(values[1]), picks: (18..<(18 + rows)).map { Int(values[$0]) })
    }

    private static let source = """
        #include <metal_stdlib>
        using namespace metal;
        struct Pick { float score; uint id; };
        constant uint Vocab = \(vocabulary), Chunk = \(chunk), Chunks = \(chunks);
        constant uint NaNFlag = 0x80000000u, IDMask = 0x7fffffffu;

        kernel void q35_head_partials(const device bfloat* logits [[buffer(0)]],
            device Pick* partial [[buffer(1)]], uint2 group [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
            uint simd [[simdgroup_index_in_threadgroup]]) {
            threadgroup Pick subgroup[8];
            bool bad = false;
            Pick local = {-INFINITY, IDMask};
            for (uint i = tid; i < Chunk; i += 256) {
                uint id = group.x * Chunk + i;
                if (id >= Vocab) break;
                float value = float(logits[group.y * Vocab + id]);
                bad |= isnan(value);
                if (value > local.score || (value == local.score && id < local.id)) local = {value, id};
            }
            float score = simd_max(local.score);
            uint id = simd_min(local.score == score ? local.id : IDMask);
            bool hasNaN = simd_any(bad);
            if (lane == 0) subgroup[simd] = {score, id | (hasNaN ? NaNFlag : 0)};
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                Pick best = {-INFINITY, IDMask};
                uint flag = 0;
                for (uint i = 0; i < 8; ++i) {
                    Pick candidate = subgroup[i];
                    flag |= candidate.id & NaNFlag;
                    candidate.id &= IDMask;
                    if (candidate.score > best.score || (candidate.score == best.score && candidate.id < best.id)) best = candidate;
                }
                partial[group.y * Chunks + group.x] = {best.score, best.id | flag};
            }
        }

        kernel void q35_head_match(const device Pick* partial [[buffer(0)]],
            device int* result [[buffer(1)]], constant int* config [[buffer(2)]]) {
            int rows = config[0], budget = config[1], eosCount = config[2];
            constant int* tokens = config + 3;
            constant int* parents = tokens + rows;
            constant int* eos = parents + rows;
            result[0] = 0;
            for (int row = 0; row < rows; ++row) {
                Pick best = {-INFINITY, IDMask};
                for (uint i = 0; i < Chunks; ++i) {
                    Pick candidate = partial[uint(row) * Chunks + i];
                    if ((candidate.id & NaNFlag) != 0) return;
                    if (candidate.score > best.score || (candidate.score == best.score && candidate.id < best.id)) best = candidate;
                }
                result[18 + row] = int(best.id);
            }
            int current = 0, count = 1;
            result[2] = 0;
            while (count < budget) {
                int token = result[18 + current];
                bool stop = false;
                for (int i = 0; i < eosCount; ++i) stop |= token == eos[i];
                if (stop) break;
                int child = -1;
                for (int i = 1; i < rows; ++i)
                    if (parents[i] == current && tokens[i] == token) { child = i; break; }
                if (child < 0) break;
                result[2 + count++] = child;
                current = child;
            }
            result[1] = result[18 + current];
            result[0] = count;
        }

        kernel void q35_gather_taps(const device ushort4* taps [[buffer(0)]],
            device ushort4* accepted [[buffer(1)]], const device int* result [[buffer(2)]],
            uint2 point [[thread_position_in_grid]]) {
            constexpr uint Width = 5120 / 4;
            if (point.y < uint(result[0]))
                accepted[point.y * Width + point.x] = taps[uint(result[2 + point.y]) * Width + point.x];
        }

        kernel void q35_gather_kv(const device ushort4* source [[buffer(0)]],
            device ushort4* accepted [[buffer(1)]], const device int* result [[buffer(2)]],
            constant uint2* offsets [[buffer(3)]], uint3 point [[thread_position_in_grid]]) {
            const uint count = uint(result[0]);
            if (point.y >= count || uint(result[1 + count]) == count - 1) return;
            const uint2 base = offsets[point.z / 4];
            const uint head = point.z % 4;
            const uint row = uint(result[2 + point.y]);
            accepted[base.y + (head * count + point.y) * 64 + point.x] =
                source[base.x + (head * 16 + row) * 64 + point.x];
        }
        """
}
#endif
