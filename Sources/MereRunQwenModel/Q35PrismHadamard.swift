import Foundation
import MLX
import MLXFast

/// One launch for Prism's FP32 signed butterfly and activation conversion.
package enum Q35PrismHadamard {
    private static let enabled = ProcessInfo.processInfo.environment["MERERUN_Q35_PRISM_METAL"] != "0"

    #if os(macOS)
    private static let kernel = MLXFast.metalKernel(
        name: "q35_prism_signed_hadamard",
        inputNames: ["x", "signs"],
        outputNames: ["y"],
        source: """
            uint lane = thread_position_in_threadgroup.x;
            uint base = threadgroup_position_in_grid.x * BLOCK;
            threadgroup float values[BLOCK];
            for (uint i = lane; i < BLOCK; i += 256) {
                float value = float(x[base + i]);
                if (!INVERSE) value *= float(signs[(base + i) % WIDTH]);
                values[i] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 1; stride < BLOCK; stride *= 2) {
                for (uint pair = lane; pair < BLOCK / 2; pair += 256) {
                    uint low = (pair / stride) * (2 * stride) + pair % stride;
                    float a = values[low];
                    float b = values[low + stride];
                    values[low] = a + b;
                    values[low + stride] = a - b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint i = lane; i < BLOCK; i += 256) {
                float value = values[i] * as_type<float>(uint(SCALE_BITS));
                if (INVERSE) value *= float(signs[(base + i) % WIDTH]);
                y[base + i] = T(value);
            }
            """,
        ensureRowContiguous: true
    )
    #endif

    package static func apply(_ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool) -> MLXArray? {
        #if os(macOS)
        // Whole-model controls retain the reference path for 4096-element blocks.
        guard enabled, [512, 1024].contains(block), Device.defaultDevice().deviceType == .gpu else { return nil }
        // Preserve the qualified prefill path for Small packs; short decode rows
        // benefit from combining the transform with signs and activation casts.
        guard block == 1024 || input.size / input.dim(-1) <= 8 else { return nil }
        let scale = 1 / Float(block).squareRoot()
        return kernel(
            [input, signs],
            template: [("T", input.dtype), ("WIDTH", input.dim(-1)), ("INVERSE", inverse),
                       ("BLOCK", block), ("SCALE_BITS", Int(scale.bitPattern))],
            grid: (input.size / block * 256, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [input.shape],
            outputDTypes: [input.dtype]
        )[0]
        #else
        return nil
        #endif
    }
}
