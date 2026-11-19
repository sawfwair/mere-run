#if os(macOS)
import Metal
import MereRunTensor

struct Q35RejectionDraws {
    let proposal: Q35SampledProposal
    let uniforms: [Float]
    let keys: [UInt32]
}

/// The kth largest tile maximum bounds the kth largest vocabulary score from below.
/// Compacting above that bound preserves cutoff ties. Overflow is resolved by the host
/// from the same head output before the verifier publishes any state.
@available(macOS 26.0, *)
final class Q35NativeSampledHead {
    private let kernels: [any MTLComputePipelineState]
    private let scratch: any MTLBuffer
    private let records: any MTLBuffer
    private enum Failure: Error { case allocation, kernel }

    init(device: any MTLDevice) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: Self.source)
        kernels = try ["tile_max", "lower_bound", "compact", "sample_row", "match_path"].map {
            guard let fn = library.makeFunction(name: $0) else { throw Failure.kernel }
            return try device.makeComputePipelineState(function: fn)
        }
        guard let scratch = device.makeBuffer(length: 16 * 128 * 4, options: .storageModePrivate),
              let records = device.makeBuffer(length: 16 * (2 + 512 * 2) * 4, options: .storageModePrivate)
        else { throw Failure.allocation }
        self.scratch = scratch
        self.records = records
    }

    func encode(into encoder: any MTLComputeCommandEncoder, logits: any MTLBuffer, sampler: Q35PositionSampler,
                position: Int, tree: Q35VerificationTree, request: Q35GreedyTreeRequest,
                metadata: any MTLBuffer, hazards: M5NativeHazards, rejection: Q35RejectionDraws? = nil) {
        let rows = tree.count
        let mode: Int
        switch sampler.serialKeys?.order {
        case .sorted: mode = 2
        case .vocabulary: mode = 3
        case nil: mode = rejection == nil ? 0 : 1
        }
        let args = [rows, min(request.budget, rows), request.eos.count, sampler.topK] + request.tokens
            + tree.parents + tree.depths.map { position + $0 } + request.eos.sorted() + [mode]
        let params = [Float(sampler.temperature), Float(sampler.topP)]
        var seed = sampler.seed
        hazards.prepare(encoder: encoder, reads: [.init(logits)], writes: [.init(scratch), .init(records), .init(metadata)])
        encoder.setBuffer(logits, offset: 0, index: 0)
        encoder.setBuffer(scratch, offset: 0, index: 1)
        encoder.setBuffer(records, offset: 0, index: 2)
        encoder.setBuffer(metadata, offset: 0, index: 3)
        args.map(Int32.init).withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        params.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 5) }
        encoder.setBytes(&seed, length: 8, index: 6)
        func bind<T>(_ values: [T], _ index: Int) {
            values.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
        }
        bind(rejection?.proposal.candidates ?? [0], 7)
        bind(rejection?.proposal.probabilities ?? [0], 8)
        bind(rejection?.uniforms ?? [0], 9)
        if let serialKeys = sampler.serialKeys {
            bind(tree.depths.map { serialKeys[position + $0] }, 10)
        } else {
            bind(rejection?.keys ?? [0], 10)
        }
        for (index, shape) in [(122, rows, 256), (1, rows, 128), (122, rows, 256), (1, rows, 512), (1, 1, 1)].enumerated() {
            if index > 0 { encoder.memoryBarrier(scope: .buffers) }
            encoder.setComputePipelineState(kernels[index])
            encoder.dispatchThreadgroups(.init(width: shape.0, height: shape.1, depth: 1),
                                         threadsPerThreadgroup: .init(width: shape.2, height: 1, depth: 1))
        }
    }

    static let source = """
    #include <metal_stdlib>
    using namespace metal;
    constant uint V = 248320, Tiles = 122, Tile = 2048, Capacity = 512;
    struct Entry { float score; uint id; };
    struct Record { atomic_uint count; float bound; Entry entries[512]; };
    kernel void tile_max(device const bfloat* logits [[buffer(0)]], device float* maxima [[buffer(1)]],
        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup float part[8];
        float best = -INFINITY;
        for (uint i = g.x * Tile + t; i < min(V, (g.x + 1) * Tile); i += 256)
            best = max(best, float(logits[g.y * V + i]));
        best = simd_max(best);
        if (lane == 0) part[sg] = best;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float peak = simd_max(lane < 8 ? part[lane] : -INFINITY);
            if (!lane) maxima[g.y * 128 + g.x] = peak;
        }
    }

    kernel void lower_bound(device const float* maxima [[buffer(1)]], device Record* records [[buffer(2)]],
        constant int* args [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
        threadgroup float a[128];
        a[t] = t < Tiles ? maxima[g.y * 128 + t] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 2; k <= 128; k *= 2) for (uint j = k / 2; j; j /= 2) {
            uint peer = t ^ j;
            if (peer > t) {
                float x = a[t], y = a[peer];
                bool up = (t & k) == 0;
                a[t] = up ? min(x,y) : max(x,y); a[peer] = up ? max(x,y) : min(x,y);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!t) {
            records[g.y].bound = a[128 - args[3]];
            atomic_store_explicit(&records[g.y].count, 0, memory_order_relaxed);
        }
    }

    kernel void compact(device const bfloat* logits [[buffer(0)]], device Record* records [[buffer(2)]],
        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
        device Record& record = records[g.y];
        for (uint offset = t; offset < Tile; offset += 256) {
            uint id = g.x * Tile + offset;
            float score = id < V ? float(logits[g.y * V + id]) : -INFINITY;
            uint keep = uint(id < V && score >= record.bound);
            uint n = simd_sum(keep);
            uint at = 0;
            if (n && lane == 0) at = atomic_fetch_add_explicit(&record.count, n, memory_order_relaxed);
            at = simd_broadcast_first(at) + simd_prefix_exclusive_sum(keep);
            if (keep && at < Capacity) record.entries[at] = {score, id};
        }
    }

    inline ulong mix_key(ulong x) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ul;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebul;
        return x ^ (x >> 31);
    }
    inline bool before(Entry a, Entry b, bool rejection) {
        return a.score < b.score || (a.score == b.score && (rejection ? a.id < b.id : a.id > b.id));
    }

    // MLX fills the two halves of an even-sized random_bits array with a Threefry pair.
    inline float mlx_gumbel(uint2 key, uint id) {
        uint i = id % (V/2);
        uint x = i + key.x, y = i + V/2 + key.y;
        const uint k[3] = {key.x, key.y, key.x ^ key.y ^ 0x1bd11bdau};
        const uint rotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};
        for (uint group=0; group<5; ++group) {
            for (uint step=0; step<4; ++step) {
                x += y;
                uint r = rotations[(group%2)*4+step];
                y = ((y << r) | (y >> (32-r))) ^ x;
            }
            x += k[(group+1)%3]; y += k[(group+2)%3] + group + 1;
        }
        float u = min(float(id < V/2 ? x : y) * 0x1p-32f, 0x1.fffffep-1f);
        return -log(-log(u));
    }

    kernel void sample_row(device Record* records [[buffer(2)]], device int* metadata [[buffer(3)]],
        constant int* args [[buffer(4)]], constant float* params [[buffer(5)]], constant ulong& seed [[buffer(6)]],
        constant int* candidates [[buffer(7)]], constant float* q [[buffer(8)]],
        constant float* acceptDraw [[buffer(9)]], constant uint2* keys [[buffer(10)]],
        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
        threadgroup Entry a[512];
        threadgroup float sums[16];
        threadgroup uint ids[16];
        device Record& record = records[g.y];
        uint n = atomic_load_explicit(&record.count, memory_order_relaxed);
        if (n > Capacity || n < uint(args[3])) { if (!t) metadata[18+g.y] = -1; return; }
        int mode = args[4 + 3*args[0] + args[2]];
        bool rejection = mode == 1;
        uint size = 32; while (size < n) size *= 2;
        a[t] = t < n ? record.entries[t] : Entry{-INFINITY, 0};
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 2; k <= size; k *= 2) for (uint j = k / 2; j; j /= 2) {
            uint peer = t ^ j;
            if (t < size && peer > t) {
                Entry x = a[t], y = a[peer];
                if ((t & k) == 0 ? before(y,x,mode != 0) : before(x,y,mode != 0)) { a[t]=y; a[peer]=x; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        Entry item = a[t];
        float weight = t < size && (mode != 0 ? item.score >= a[size - args[3]].score : t >= size - args[3]) ?
            exp((item.score - a[size-1].score) / params[0]) : 0.0f;
        float cumulative = simd_prefix_inclusive_sum(weight), sub = simd_sum(weight);
        if (!lane) sums[sg] = sub;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        for (uint i = 0; i < 16; ++i) { if (i < sg) cumulative += sums[i]; total += sums[i]; }
        float race = -INFINITY;
        bool kept = weight > 0 && cumulative > (1.0f-params[1])*total;
        if (rejection) {
            // Renormalize the nucleus before computing p/q. Non-candidate tokens have q=0.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float partial = simd_sum(kept ? weight : 0.0f);
            if (!lane) sums[sg] = partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float keptTotal = 0; for (uint j = 0; j < 16; ++j) keptTotal += sums[j];
            float p = kept ? weight / keptTotal : 0;
            float draft = 0; bool accepted = false;
            if (g.y + 1 < uint(args[0])) {
                for (uint j=0;j<16;++j) if (uint(candidates[g.y*16+j])==item.id) draft += q[g.y*16+j];
                accepted = kept && item.id == uint(args[4+g.y+1]) && acceptDraw[g.y] < min(1.0f,p/max(draft,FLT_MIN));
            }
            bool anyAccepted = simd_any(accepted);
            if (!lane) ids[sg] = uint(anyAccepted);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            bool matched = false; for (uint j=0;j<16;++j) matched |= ids[j] != 0;
            if (matched) { if (!t) metadata[18+g.y] = args[4+g.y+1] | 0x40000000; return; }
            float mass = max(0.0f,p-draft);
            if (mass > 0) race = log(mass) + mlx_gumbel(keys[g.y], item.id);
        } else if (kept && mode >= 2) {
            race = log(weight / total) + mlx_gumbel(keys[g.y], mode == 3 ? item.id : V - size + t);
        } else if (kept) {
            uint pos = uint(args[4+2*args[0]+g.y]);
            ulong base = mix_key(mix_key(seed + 0x9e3779b97f4a7c15ul) ^ (ulong(pos) * 0xd1b54a32d192ed03ul));
            float u = min((float(uint(mix_key(base ^ ulong(item.id)) >> 40)) + 0.5f) * 0x1p-24f, 0x1.fffffep-1f);
            race = item.score / params[0] - log(-log(u));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = simd_max(race);
        uint id = simd_min(race == best ? item.id : 0xffffffffu);
        if (!lane) { sums[sg] = best; ids[sg] = id; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!t) {
            Entry winner = {-INFINITY, 0xffffffffu};
            for (uint i = 0; i < 16; ++i) if (sums[i] > winner.score || (sums[i] == winner.score && ids[i] < winner.id))
                winner = {sums[i], ids[i]};
            metadata[18+g.y] = int(winner.id);
        }
    }

    kernel void match_path(device int* result [[buffer(3)]], constant int* args [[buffer(4)]]) {
        int rows = args[0], budget = args[1], count = 1, row = 0;
        constant int* tokens = args + 4;
        constant int* parents = tokens + rows;
        constant int* eos = parents + 2 * rows;
        result[0] = 0; result[2] = 0;
        while (true) {
            int next = result[18 + row];
            if (next < 0) return;
            bool rejection = eos[args[2]] == 1;
            bool stop = count >= budget || (rejection && !(next & 0x40000000));
            if (rejection) next &= 0x3fffffff;
            for (int i = 0; i < args[2]; ++i) stop |= next == eos[i];
            int child = -1;
            if (!stop) for (int i = 1; i < rows; ++i)
                if (parents[i] == row && tokens[i] == next) { child = i; break; }
            if (child < 0) { result[0] = count; result[1] = next; return; }
            result[2 + count++] = child; row = child;
        }
    }
    """
}
#endif
