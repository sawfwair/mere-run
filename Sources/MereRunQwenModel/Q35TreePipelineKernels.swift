import MLX
import MLXFast

enum Q35TreePipelineKernels {
    static let match = MLXFast.metalKernel(
        name: "q35_authoritative_tree_match_v1", inputNames: ["picks", "tokens", "parents", "eos", "bounds"],
        outputNames: ["path", "absolute", "info"], source: """
            for (int i = 0; i < Rows; ++i) { path[i] = 0; absolute[i] = bounds[1]; }
            int count = 1, current = 0;
            while (count < bounds[0] && count < Rows) {
                const int token = picks[current];
                bool stop = false;
                for (int i = 0; i < EOSCount; ++i) stop |= token == eos[i];
                if (stop) break;
                int child = -1;
                for (int i = 1; i < Rows; ++i)
                    if (parents[i] == current && tokens[i] == token) { child = i; break; }
                if (child < 0) break;
                path[count] = child; absolute[count] = bounds[1] + child;
                current = child; ++count;
            }
            info[0] = count; info[1] = picks[current];
            """
    )

    static let recurrent = MLXFast.metalKernel(
        name: "q35_selected_tree_recurrent_commit_v1",
        inputNames: ["k", "v", "g", "beta", "initial", "path", "info"], outputNames: ["next"],
        source: recurrentSource
    )

    static let recurrentSource = """
            const uint hv = thread_position_in_grid.z, hk = hv / (Hv / Hk);
            const uint lane = thread_position_in_threadgroup.x, dv = thread_position_in_grid.y;
            constexpr int PerThread = Dk / 32;
            float state[PerThread];
            for (int i = 0; i < PerThread; ++i)
                state[i] = initial[(hv * Dv + dv) * Dk + lane * PerThread + i];
            for (int depth = 0; depth < info[0]; ++depth) {
                const int row = path[depth];
                const auto key = k + (row * Hk + hk) * Dk;
                float memory = 0.0f;
                for (int i = 0; i < PerThread; ++i) {
                    state[i] = state[i] * g[row * Hv + hv];
                    memory += state[i] * key[lane * PerThread + i];
                }
                memory = simd_sum(memory);
                const float delta = (v[(row * Hv + hv) * Dv + dv] - memory) * beta[row * Hv + hv];
                for (int i = 0; i < PerThread; ++i)
                    state[i] = state[i] + key[lane * PerThread + i] * delta;
            }
            for (int i = 0; i < PerThread; ++i)
                next[(hv * Dv + dv) * Dk + lane * PerThread + i] = state[i];
            """

    static let convolution = MLXFast.metalKernel(
        name: "q35_selected_tree_convolution_commit_v1",
        inputNames: ["qkv", "initial", "path", "info"], outputNames: ["next"],
        source: convolutionSource
    )

    static let convolutionSource = """
            const uint i = thread_position_in_grid.x;
            if (i >= Channels * Keep) return;
            const int time = info[0] - Keep + int(i / Channels), channel = i % Channels;
            next[i] = time < 0 ? initial[(Keep + time) * Channels + channel]
                               : qkv[path[time] * Channels + channel];
            """
}
