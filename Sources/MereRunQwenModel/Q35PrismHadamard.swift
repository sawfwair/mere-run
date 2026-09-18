import Foundation
import MLX
import MLXFast

/// One launch for Prism's FP32 signed butterfly and activation conversion.
package enum Q35PrismHadamard {
    private static let enabled = ProcessInfo.processInfo.environment["MERERUN_Q35_PRISM_METAL"] != "0"

    #if os(macOS)
    private static let kernel = MLXFast.metalKernel(
        name: "q35_prism_signed_hadamard_1024",
        inputNames: ["x", "signs"],
        outputNames: ["y"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint base = threadgroup_position_in_grid.x * 1024;
            threadgroup float values[1024];
            for (uint i = lane; i < 1024; i += 256) {
                float value = float(x[base + i]);
                if (!INVERSE) value *= float(signs[(base + i) % WIDTH]);
                values[i] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < 1024; stride *= 2) {
                for (uint pair = lane; pair < 512; pair += 256) {
                    uint low = (pair / stride) * (2 * stride) + pair % stride;
                    float a = values[low];
                    float b = values[low + stride];
                    values[low] = a + b;
                    values[low + stride] = a - b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint i = lane; i < 1024; i += 256) {
                float value = values[i] * (1.0f / 32.0f);
                if (INVERSE) value *= float(signs[(base + i) % WIDTH]);
                y[base + i] = T(value);
            }
            """,
        ensureRowContiguous: true
    )
    #endif

    package static func apply(_ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool) -> MLXArray? {
        #if os(macOS)
        guard enabled, block == 1024, Device.defaultDevice().deviceType == .gpu else { return nil }
        return kernel(
            [input, signs],
            template: [("T", input.dtype), ("WIDTH", input.dim(-1)), ("INVERSE", inverse)],
            grid: (input.size / 1024 * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }
}
