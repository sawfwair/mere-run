#if os(macOS)
// Wide BF16 top-k kernels from TensorFold 1.2.7 under Apache-2.0.
// See THIRD_PARTY_NOTICES.md for the upstream source and license.
@available(macOS 26.0, *)
extension M5NativeTopK {
    static let source = """
    // One input scan builds local min-heaps, sorted once before exact score/lowest-token-ID merges.
    #include <metal_stdlib>
    using namespace metal;
    // Each is also a constant in bf16_topk_gpu.zig, which does not compile when its value differs.
    #define TF_TOPK_LANES 256
    #define TF_TOPK_MAX_K 16
    #define TF_TOPK_SPANS 16
    struct Params { uint rows, vocab, k, stride, gap_start, gap_count; };
    struct Entry { uint token, score_bits; };
    struct Row { uint nonfinite, count; Entry entries[TF_TOPK_MAX_K]; };
    inline uint tf_bf16_key(ushort raw) {
      const uint bits = (raw & 0x7fffu) == 0 ? 0u : uint(raw);
      return (bits & 0x8000u) != 0 ? (~bits & 0xffffu) : (bits | 0x8000u);
    }
    inline bool tf_topk_before(uint key, uint id, uint other_key, uint other_id) {
      return key > other_key || (key == other_key && id < other_id);
    }
    inline void tf_topk_swap(thread uint* key, thread uint* id, thread uint* bits, uint a, uint b) {
      const uint k = key[a], i = id[a], s = bits[a]; key[a] = key[b]; id[a] = id[b]; bits[a] = bits[b];
      key[b] = k; id[b] = i; bits[b] = s;
    }
    inline void tf_topk_sift(thread uint* key, thread uint* id, thread uint* bits, uint count) {
      uint at = 0;
      while (2 * at + 1 < count) {
        uint child = 2 * at + 1;
        if (child + 1 < count && tf_topk_before(key[child], id[child], key[child + 1], id[child + 1])) child++;
        if (!tf_topk_before(key[at], id[at], key[child], id[child])) break;
        tf_topk_swap(key, id, bits, at, child); at = child;
      }
    }
    // The scan, local heaps and tournament over tokens [begin, end) of one row, into one Row.
    inline void tf_topk_span(const device ushort* logits, device Row& out, ulong begin, ulong end, constant Params& p,
                             uint lane, threadgroup uint* keys, threadgroup uint* ids, threadgroup uint* score_bits,
                             threadgroup uint* masks) {
      uint local_key[TF_TOPK_MAX_K], local_id[TF_TOPK_MAX_K], local_bits[TF_TOPK_MAX_K];
      for (uint i = 0; i < TF_TOPK_MAX_K; i++) { local_key[i] = 0; local_id[i] = 0xffffffffu; local_bits[i] = 0; }
      if (lane == 0) {
        out.nonfinite = 0; out.count = 0;
        for (uint i = 0; i < TF_TOPK_MAX_K; i++) out.entries[i] = { 0xffffffffu, 0u };
      }
      uint mask = 0, count = 0;
      for (ulong token = begin + lane; token < end; token += TF_TOPK_LANES) {
        const ushort raw = logits[token];
        if ((raw & 0x7f80u) == 0x7f80u) {
          mask |= (raw & 0x7fu) != 0 ? 1u : (raw & 0x8000u) == 0 ? 2u : 4u;
          continue;
        }
        const uint key = tf_bf16_key(raw), id = uint(token), bits = uint(raw) << 16;
        if (count < p.k) {
          uint at = count++;
          local_key[at] = key; local_id[at] = id; local_bits[at] = bits;
          while (at > 0) {
            const uint parent = (at - 1) / 2;
            if (!tf_topk_before(local_key[parent], local_id[parent], local_key[at], local_id[at])) break;
            tf_topk_swap(local_key, local_id, local_bits, at, parent); at = parent;
          }
        } else if (tf_topk_before(key, id, local_key[0], local_id[0])) {
          local_key[0] = key; local_id[0] = id; local_bits[0] = bits;
          tf_topk_sift(local_key, local_id, local_bits, count);
        }
      }
      for (uint last = count; last > 1; last--) {
        tf_topk_swap(local_key, local_id, local_bits, 0, last - 1);
        tf_topk_sift(local_key, local_id, local_bits, last - 1);
      }
      uint cursor = 0;
      for (uint rank = 0; rank < p.k; rank++) {
        keys[lane] = cursor < p.k ? local_key[cursor] : 0;
        ids[lane] = cursor < p.k ? local_id[cursor] : 0xffffffffu;
        score_bits[lane] = cursor < p.k ? local_bits[cursor] : 0;
        masks[lane] = mask;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = TF_TOPK_LANES / 2; step > 0; step >>= 1) {
          if (lane < step) {
            if (tf_topk_before(keys[lane + step], ids[lane + step], keys[lane], ids[lane])) {
              keys[lane] = keys[lane + step]; ids[lane] = ids[lane + step]; score_bits[lane] = score_bits[lane + step];
            }
            masks[lane] |= masks[lane + step];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (masks[0] != 0) { if (lane == 0) out.nonfinite = masks[0]; return; }
        const uint winner = ids[0];
        if (lane == 0) { out.entries[rank] = { winner, score_bits[0] }; out.count = rank + 1; }
        if (cursor < p.k && local_id[cursor] == winner) cursor++;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // A wide row's spans each leave their top k and non-finite mask in part (the span's Row); the merge picks the row's.
    kernel void tf_bf16_topk_span(const device ushort* logits [[buffer(0)]],
                                 device Row* part [[buffer(1)]], constant Params& p [[buffer(2)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
      threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
      const ulong width = (ulong(p.vocab) + TF_TOPK_SPANS - 1) / TF_TOPK_SPANS;
      const ulong begin = min(ulong(tg.x) * width, ulong(p.vocab)), end = min(begin + width, ulong(p.vocab));
      tf_topk_span(logits + ulong(tg.y) * p.stride, part[tg.y * TF_TOPK_SPANS + tg.x], begin, end, p, lane, keys, ids, score_bits, masks);
    }

    kernel void tf_bf16_topk_merge(const device Row* part [[buffer(0)]],
                                  device Row* output [[buffer(1)]], constant Params& p [[buffer(2)]],
                                  uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
      threadgroup uint keys[TF_TOPK_LANES], ids[TF_TOPK_LANES], score_bits[TF_TOPK_LANES], masks[TF_TOPK_LANES];
      const device Row* spans = part + row * TF_TOPK_SPANS;
      uint key = 0, id = 0xffffffffu, bits = 0, mask = 0;
      if (lane < TF_TOPK_SPANS) mask = spans[lane].nonfinite;
      const uint span = lane / TF_TOPK_MAX_K, at = lane % TF_TOPK_MAX_K;
      if (span < TF_TOPK_SPANS && at < p.k && spans[span].entries[at].token != 0xffffffffu) {
        bits = spans[span].entries[at].score_bits;
        key = tf_bf16_key(ushort(bits >> 16)); id = spans[span].entries[at].token;
      }
      if (lane == 0) {
        output[row].nonfinite = 0; output[row].count = 0;
        for (uint i = 0; i < TF_TOPK_MAX_K; i++) output[row].entries[i] = { 0xffffffffu, 0u };
      }
      for (uint rank = 0; rank < p.k; rank++) {
        keys[lane] = key; ids[lane] = id; score_bits[lane] = bits; masks[lane] = mask;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint step = TF_TOPK_LANES / 2; step > 0; step >>= 1) {
          if (lane < step) {
            if (tf_topk_before(keys[lane + step], ids[lane + step], keys[lane], ids[lane])) {
              keys[lane] = keys[lane + step]; ids[lane] = ids[lane + step]; score_bits[lane] = score_bits[lane + step];
            }
            masks[lane] |= masks[lane + step];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (masks[0] != 0) { if (lane == 0) output[row].nonfinite = masks[0]; return; }
        const uint winner = ids[0];
        if (lane == 0) {
          const uint token = winner >= p.gap_start ? winner + p.gap_count : winner;
          output[row].entries[rank] = { token, score_bits[0] };
          output[row].count = rank + 1;
        }
        if (id == winner) { key = 0; id = 0xffffffffu; bits = 0; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    """
}
#endif
