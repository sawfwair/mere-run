import MLX
import MLXFast

// Indexed reads adapt Apple MLX sdpa_vector.h (MIT); see THIRD_PARTY_NOTICES.md.
enum Q35IndexedTreeKernels {

    static let header = """
        #include <metal_simdgroup>
        using namespace metal;
        template <typename T, int D>
        struct MereTreeKV {
            const device T* base;
            const device T* current;
            const device int* path;
            int offset, prefix, sequence_stride, dimension_stride;
            MereTreeKV(const device T* b, const device int* p, int at, int pre, int ss, int ds)
                : base(b), path(p), offset(at), prefix(pre), sequence_stride(ss), dimension_stride(ds) { refresh(); }
            inline void refresh() {
                const int logical = offset / D, dim = offset % D;
                // Native SDPA can advance an inactive lane past the last row.
                // Its masked loads are skipped; keep that lane's cursor valid.
                const int physical = logical < prefix ? logical : prefix + path[min(logical - prefix, 15)];
                current = base + physical * sequence_stride + dim * dimension_stride;
            }
            // Native SDPA loads only a thread's contiguous within-row fragment.
            // Translate the token once at pointer advance, rather than per value.
            inline T operator[](int delta) const { return current[delta * dimension_stride]; }
            inline MereTreeKV operator+(int delta) const { auto p = *this; p += delta; return p; }
            inline void operator+=(int delta) { offset += delta; refresh(); }
        };
        struct MereTreeMask {
            int offset, length;
            inline bool operator[](int delta) const { return offset + delta < length; }
            inline void operator+=(int delta) { offset += delta; }
        };
        """

    static let single = MLXFast.metalKernel(
        name: "q35_indexed_tree_single_v2",
        inputNames: ["q_input", "k_input", "v_input", "paths", "rows", "lengths", "info", "scale_input"], outputNames: ["y"],
        source: singleSource, header: header, ensureRowContiguous: false
    )

    static let singleSource = """
            const uint3 tid = threadgroup_position_in_grid, tpg = threadgroups_per_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
            constexpr bool has_mask = true, bool_mask = true, float_mask = false;
            constexpr bool has_sinks = false, query_transposed = false, do_causal = false;
            const device T* queries = q_input;
            const device T* fmask = nullptr;
            const device T* sinks = nullptr;
            const int N = info[0], num_q_heads = info[2];
            const int gqa_factor = G;
            const int k_head_stride = 0, v_head_stride = 0, k_seq_stride = D, v_seq_stride = D;
            const int mask_kv_seq_stride = 1, mask_q_seq_stride = 0, mask_head_stride = 0;
            const float scale = scale_input[0];
            const int batch_row = tid.x / info[2], physical_head = (tid.x % info[2]) / G;
            device T* out = y;
            const int original_row = rows[batch_row];
            MereTreeKV<T, D> keys{k_input + physical_head * k_input_strides[1], paths + original_row * 16, 0, info[1], int(k_input_strides[2]), int(k_input_strides[3])};
            MereTreeKV<T, V> values{v_input + physical_head * v_input_strides[1], paths + original_row * 16, 0, info[1], int(v_input_strides[2]), int(v_input_strides[3])};
            MereTreeMask bmask{0, info[1] + lengths[original_row]};
            constexpr int BN = 32;
              constexpr int BD = 32;
              constexpr int qk_per_thread = D / BD;
              constexpr int v_per_thread = V / BD;
              int inner_k_stride = BN * int(k_seq_stride);
              int inner_v_stride = BN * int(v_seq_stride);

              typedef float U;

              thread U q[qk_per_thread];
              thread U k[qk_per_thread];
              thread U o[v_per_thread];

              threadgroup U outputs[BN * BD];
              threadgroup U max_scores[BN];
              threadgroup U sum_exp_scores[BN];

              // Adjust positions
              const int q_batch_head_idx = tid.x;
              const int q_seq_idx = tid.y;
              const int kv_head_idx = q_batch_head_idx / gqa_factor;
              const int kv_seq_idx = simd_gid;
              const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
              const int q_offset =
                  query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
              queries += q_offset * D + simd_lid * qk_per_thread;
              keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
                  simd_lid * qk_per_thread;
              values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
                  simd_lid * v_per_thread;
              if (bool_mask) {
                bmask += q_batch_head_idx * mask_head_stride +
                    kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
              }
              if (float_mask) {
                fmask += q_batch_head_idx * mask_head_stride +
                    kv_seq_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
              }

              out += o_offset * V + simd_gid * v_per_thread;

              // Read the query and 0 the output accumulator
              for (int i = 0; i < qk_per_thread; i++) {
                q[i] = static_cast<U>(scale) * queries[i];
              }
              for (int i = 0; i < v_per_thread; i++) {
                o[i] = 0;
              }

              U max_score = (-MAXFLOAT);
              U sum_exp_score = 0;
              if (has_sinks && simd_gid == 0) {
                max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
                sum_exp_score = 1;
              }

              // For each key
              for (int i = simd_gid; i < N; i += BN) {
                bool use_key = true;
                if (do_causal) {
                  use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
                } else if (bool_mask) {
                  use_key = bmask[0];
                } else if (float_mask) {
                  use_key = (fmask[0] >= (-MAXFLOAT));
                }
                if (use_key) {
                  // Read the key
                  for (int j = 0; j < qk_per_thread; j++) {
                    k[j] = keys[j];
                  }

                  // Compute the i-th score
                  U score = 0;
                  for (int j = 0; j < qk_per_thread; j++) {
                    score += q[j] * k[j];
                  }
                  score = simd_sum(score);
                  if (float_mask) {
                    score += static_cast<U>(fmask[0]);
                  }

                  // Update the accumulators
                  U new_max = max(max_score, score);
                  U factor = fast::exp(max_score - new_max);
                  U exp_score = fast::exp(score - new_max);

                  max_score = new_max;
                  sum_exp_score = sum_exp_score * factor + exp_score;

                  // Update the output accumulator
                  for (int j = 0; j < v_per_thread; j++) {
                    o[j] = o[j] * factor + exp_score * values[j];
                  }
                }

                // Move the pointers to the next kv
                keys += inner_k_stride;
                values += inner_v_stride;
                if (bool_mask) {
                  bmask += BN * mask_kv_seq_stride;
                }
                if (float_mask) {
                  fmask += BN * mask_kv_seq_stride;
                }
              }

              // Each thread has a partial part of the output so we need to combine them.

              // First let's communicate the max and sum_exp
              if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              max_score = max_scores[simd_lid];
              U new_max = simd_max(max_score);
              U factor = fast::exp(max_score - new_max);
              sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

              // Now we need to aggregate all the outputs
              for (int i = 0; i < v_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }

              // And write the output
              if (simd_lid == 0) {
                for (int i = 0; i < v_per_thread; i++) {
                  out[i] = static_cast<T>(o[i]);
                }
              }
            """

    static let partial = MLXFast.metalKernel(
        name: "q35_indexed_tree_partial_v2",
        inputNames: ["q_input", "k_input", "v_input", "paths", "rows", "lengths", "info", "scale_input"], outputNames: ["partials", "sums", "maxs"],
        source: partialSource, header: header, ensureRowContiguous: false
    )

    static let partialSource = """
            const uint3 tid = threadgroup_position_in_grid, tpg = threadgroups_per_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
            constexpr bool has_mask = true, bool_mask = true, float_mask = false;
            constexpr bool has_sinks = false, query_transposed = false, do_causal = false;
            const device T* queries = q_input;
            const device T* fmask = nullptr;
            const device T* sinks = nullptr;
            const int N = info[0];

            const int k_head_stride = 0, v_head_stride = 0, k_seq_stride = D, v_seq_stride = D;
            const int mask_kv_seq_stride = 1, mask_q_seq_stride = 0, mask_head_stride = 0;
            const float scale = scale_input[0];
            const uint3 tptg = threads_per_threadgroup, tidtg = thread_position_in_threadgroup;
            constexpr int blocks = Blocks, TK = 4;
            const int batch_row = tid.y, physical_head = tid.x;
            device T* out = partials;
            const int original_row = rows[batch_row];
            MereTreeKV<T, D> keys{k_input + physical_head * k_input_strides[1], paths + original_row * 16, 0, info[1], int(k_input_strides[2]), int(k_input_strides[3])};
            MereTreeKV<T, V> values{v_input + physical_head * v_input_strides[1], paths + original_row * 16, 0, info[1], int(v_input_strides[2]), int(v_input_strides[3])};
            MereTreeMask bmask{0, info[1] + lengths[original_row]};
            constexpr int BD = 32;
              constexpr int qk_per_thread = D / BD;
              constexpr int v_per_thread = V / BD;

              typedef float U;

              thread U q[qk_per_thread];
              thread U o[v_per_thread] = {0};

              // Adjust positions
              const int kv_head_idx = tid.x;
              const int batch_idx = tid.y;
              const int block_idx = tid.z;
              const int gqa_factor = tptg.y;
              const int q_seq_len = tptg.z;
              const int q_seq_idx = tidtg.z;
              const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
              const int num_kv_heads = tpg.x;
              const int num_q_heads = num_kv_heads * gqa_factor;
              const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
              const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;
              const int q_offset =
                  query_transposed ? num_q_heads * q_seq_idx + q_batch_head_idx : o_offset;

              queries += q_offset * D + simd_lid * qk_per_thread;

              const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
              keys += kv_batch_head_idx * k_head_stride + block_idx * k_seq_stride +
                  simd_lid * qk_per_thread;
              values += kv_batch_head_idx * v_head_stride + block_idx * v_seq_stride +
                  simd_lid * v_per_thread;
              out += o_offset * blocks * V + block_idx * V + simd_lid * v_per_thread;
              if (bool_mask) {
                bmask += q_batch_head_idx * mask_head_stride +
                    block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
              }
              if (float_mask) {
                fmask += q_batch_head_idx * mask_head_stride +
                    block_idx * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
              }
              sums += o_offset * blocks + block_idx;
              maxs += o_offset * blocks + block_idx;

              // Read the query
              for (int i = 0; i < qk_per_thread; i++) {
                q[i] = static_cast<U>(scale) * queries[i];
              }

              U max_score = (-MAXFLOAT);
              U sum_exp_score = 0;
              if (has_sinks && block_idx == 0) {
                max_score = static_cast<U>(sinks[q_head_idx]);
                sum_exp_score = 1;
              }

              const int stride_k_tok = blocks * int(k_seq_stride);
              const int stride_v_tok = blocks * int(v_seq_stride);
              const int stride_mask_tok =
                  (bool_mask || float_mask) ? blocks * mask_kv_seq_stride : 0;

              // For each key
              int i = block_idx;
              for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
                // Prefetch: issue all TK K/V loads first so the HW can overlap them.
                T k_reg[TK][qk_per_thread];
                T v_reg[TK][v_per_thread];
                for (int t = 0; t < TK; t++) {
                  auto kp = keys + t * stride_k_tok;
                  for (int j = 0; j < qk_per_thread; j++) {
                    k_reg[t][j] = kp[j];
                  }
                }
                for (int t = 0; t < TK; t++) {
                  auto vp = values + t * stride_v_tok;
                  for (int j = 0; j < v_per_thread; j++) {
                    v_reg[t][j] = vp[j];
                  }
                }

                for (int t = 0; t < TK; t++) {
                  int ii = i + t * blocks;
                  bool use_key = true;
                  if (do_causal) {
                    use_key = ii <= (N - q_seq_len + int(q_seq_idx));
                  } else if (bool_mask) {
                    use_key = bmask[t * stride_mask_tok];
                  } else if (float_mask) {
                    use_key = (fmask[t * stride_mask_tok] >= (-MAXFLOAT));
                  }
                  if (use_key) {
                    U score = 0;
                    for (int j = 0; j < qk_per_thread; j++) {
                      score += q[j] * static_cast<U>(k_reg[t][j]);
                    }
                    score = simd_sum(score);
                    if (float_mask) {
                      score += static_cast<U>(fmask[t * stride_mask_tok]);
                    }

                    U new_max = max(max_score, score);
                    U factor = fast::exp(max_score - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;
                    for (int j = 0; j < v_per_thread; j++) {
                      o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
                    }
                  }
                }

                keys += TK * stride_k_tok;
                values += TK * stride_v_tok;
                if (bool_mask) {
                  bmask += TK * stride_mask_tok;
                }
                if (float_mask) {
                  fmask += TK * stride_mask_tok;
                }
              }

              // Tail (fewer than TK remaining keys)
              for (; i < N; i += blocks) {
                bool use_key = true;
                if (do_causal) {
                  use_key = i <= (N - q_seq_len + int(q_seq_idx));
                } else if (bool_mask) {
                  use_key = bmask[0];
                } else if (float_mask) {
                  use_key = (fmask[0] >= (-MAXFLOAT));
                }
                if (use_key) {
                  // Compute the i-th score
                  U score = 0;
                  for (int j = 0; j < qk_per_thread; j++) {
                    score += q[j] * keys[j];
                  }
                  score = simd_sum(score);

                  if (float_mask) {
                    score += fmask[0];
                  }

                  // Update the accumulators
                  U new_max = max(max_score, score);
                  U factor = fast::exp(max_score - new_max);
                  U exp_score = fast::exp(score - new_max);

                  max_score = new_max;
                  sum_exp_score = sum_exp_score * factor + exp_score;

                  // Update the output accumulator
                  for (int j = 0; j < v_per_thread; j++) {
                    o[j] = o[j] * factor + exp_score * values[j];
                  }
                }

                // Move the pointers to the next kv
                keys += stride_k_tok;
                values += stride_v_tok;
                if (bool_mask) {
                  bmask += stride_mask_tok;
                }
                if (float_mask) {
                  fmask += stride_mask_tok;
                }
              }

              // Write the sum and max and outputs
              if (simd_lid == 0) {
                sums[0] = sum_exp_score;
                maxs[0] = max_score;
              }

              for (int i = 0; i < v_per_thread; i++) {
                out[i] = static_cast<T>(o[i]);
              }
            """

    static let merge = MLXFast.metalKernel(
        name: "q35_indexed_tree_merge_v2",
        inputNames: ["partials", "sums", "maxs"], outputNames: ["y"],
        source: mergeSource, ensureRowContiguous: true
    )

    static let mergeSource = """
            const uint3 tid = threadgroup_position_in_grid, tpg = threadgroups_per_grid;
            const uint simd_gid = simdgroup_index_in_threadgroup, simd_lid = thread_index_in_simdgroup;
            constexpr int blocks = Blocks;
            device T* out = y;
            constexpr int BN = 32;
              constexpr int BD = 32;
              constexpr int elem_per_thread = D / BD;

              typedef float U;

              thread U o[elem_per_thread] = {0};
              threadgroup U outputs[BN * BD];

              // Adjust positions
              const int head_idx = tid.x;
              const int q_seq_idx = tid.y;
              const int q_offset = head_idx * tpg.y + q_seq_idx;
              partials += q_offset * blocks * D + simd_gid * D + simd_lid * elem_per_thread;
              sums += q_offset * blocks;
              maxs += q_offset * blocks;
              out += q_offset * D + simd_gid * elem_per_thread;

              // Set defaults
              U sum_exp_score = 0.0;
              U max_score = (-MAXFLOAT);

              // Reduce the max
              for (int b = 0; b < blocks / BN; ++b) {
                max_score = max(max_score, maxs[simd_lid + BN * b]);
              }
              max_score = simd_max(max_score);

              // Reduce the d
              for (int b = 0; b < blocks / BN; ++b) {
                U factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
                sum_exp_score += factor * sums[simd_lid + BN * b];
              }
              sum_exp_score = simd_sum(sum_exp_score);

              // Reduce the sum exp and partials
              for (int b = 0; b < blocks / BN; ++b) {
                U factor = fast::exp(maxs[simd_gid] - max_score);

                // Update the output accumulator
                for (int i = 0; i < elem_per_thread; i++) {
                  o[i] += factor * static_cast<U>(partials[i]);
                }
                maxs += BN;
                sums += BN;
                partials += BN * D;
              }

              // Use shared memory to transpose and reduce the final block
              for (int i = 0; i < elem_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }

              // And write the output
              if (simd_lid == 0) {
                for (int i = 0; i < elem_per_thread; i++) {
                  out[i] = static_cast<T>(o[i]);
                }
              }
            """
}
