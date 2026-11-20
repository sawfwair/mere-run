#if os(macOS)
import Metal

/// Experimental full-width Qwen GDN projection. The matrix pass marks outputs
/// near BF16 rounding boundaries. A second dispatch recomputes those cells with
/// serial QMV arithmetic. The margin is empirical and requires qualification.
@available(macOS 26.0, *)
package final class M5NativeGDNRepair {
    private let projection: any MTLComputePipelineState
    private let repair: any MTLComputePipelineState
    private let counts: any MTLBuffer
    private let cells: any MTLBuffer
    private static let tiles = 16_384 / 32

    package init(device: any MTLDevice, narrowMargin: Bool = false) throws {
        let library = try M5NativeFFN.makeLibrary(device: device, source: M5NativeFFN.shaderHeader + "\n" + Self.source(narrowMargin: narrowMargin))
        guard let first = library.makeFunction(name: "mere_gdn_candidates"),
              let second = library.makeFunction(name: "mere_gdn_repair") else { throw M5NativeProjection.Failure.missingFunction }
        projection = try device.makeComputePipelineState(function: first)
        repair = try device.makeComputePipelineState(function: second)
        let options: MTLResourceOptions = [.storageModePrivate, .hazardTrackingModeUntracked]
        guard let counts = device.makeBuffer(length: Self.tiles * 4, options: options),
              let cells = device.makeBuffer(length: 16 * 16_384 * 2, options: options) else { throw M5NativeFFN.Failure.allocation }
        self.counts = counts
        self.cells = cells
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                        sums: any MTLBuffer, output: any MTLBuffer, offset: Int, serial: M5NativeSmallProjection.Weights,
                        packed: M5NativeFFN.Projection, hazards: M5NativeHazards) {
        let result = M5NativeHazards.Access(output, offset: offset, count: 16 * 16_384 * 2)
        let inputAccess = M5NativeHazards.Access(input, count: 16 * 5_120 * 2)
        let grid = MTLSize(width: Self.tiles, height: 1, depth: 1)
        hazards.prepare(encoder: encoder, reads: [inputAccess, .init(sums)], writes: [result, .init(counts), .init(cells)])
        encoder.setComputePipelineState(projection)
        encoder.setBuffer(packed.codes, offset: packed.codesOffset, index: 0)
        encoder.setBuffer(packed.scales, offset: packed.scalesOffset, index: 1)
        encoder.setBuffer(packed.biases, offset: packed.biasesOffset, index: 2)
        encoder.setBuffer(input, offset: 0, index: 3)
        encoder.setBuffer(output, offset: offset, index: 4)
        encoder.setBuffer(counts, offset: 0, index: 5)
        encoder.setBuffer(cells, offset: 0, index: 6)
        encoder.setBuffer(sums, offset: 0, index: 7)
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        hazards.prepare(encoder: encoder, reads: [inputAccess, .init(counts), .init(cells)], writes: [result])
        encoder.setComputePipelineState(repair)
        for (index, weight) in [serial.codes, serial.scales, serial.biases].enumerated() {
            encoder.setBuffer(weight.buffer, offset: weight.offset, index: index)
        }
        encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private static func source(narrowMargin: Bool) -> String {
        """
        kernel void mere_gdn_candidates(
            const device uint* w0 [[buffer(0)]], const device bfloat* s0 [[buffer(1)]], const device bfloat* b0 [[buffer(2)]],
            const device bfloat* x [[buffer(3)]], device bfloat* y [[buffer(4)]], device uint* counts [[buffer(5)]],
            device ushort* cells [[buffer(6)]], const device float* sums [[buffer(7)]],
            uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
            uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
            constexpr int K = 5120, N = 16384, Splits = 2, Stacked = 0, Arithmetic = 0;
            const int x_shape[] = {16, K}, sums_shape[] = {80, 16};
            const device uint* w1 = w0;
            const device bfloat* s1 = s0;
            const device bfloat* b1 = b0;
            \(M5TiledAffineQ4.groupProjectionAccumulator)
            if (split != 0) return;
            uint flags = 0;
            for (int i = 0; i < 16; ++i) {
                const uint row = mb + 8 * ((i >> 2) & 1), col = nb + 16 * (i >> 3) + (i & 3);
                const float value = values[i], magnitude = abs(value);
                y[row * N + outColumn + col] = bfloat(value);
                const float midpoint = as_type<float>((as_type<uint>(magnitude) & 0xffff0000u) | 0x8000u);
                if (abs(magnitude - midpoint) <= \(narrowMargin ? "4e-6f" : "1e-5f") * (1.0f + magnitude)) flags |= 1u << i;
            }
            const uint count = popcount(flags), total = simd_sum(count);
            uint at = simd_prefix_exclusive_sum(count);
            if (lane == 0) counts[block] = total;
            while (flags) {
                const uint i = ctz(flags);
                const uint row = mb + 8 * ((i >> 2) & 1), col = nb + 16 * (i >> 3) + (i & 3);
                cells[block * 512 + at++] = ushort(row * 32 + col);
                flags &= flags - 1;
            }
        }

        kernel void mere_gdn_repair(
            const device uint* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]], const device bfloat* biases [[buffer(2)]],
            const device bfloat* x [[buffer(3)]], device bfloat* y [[buffer(4)]], const device uint* counts [[buffer(5)]],
            const device ushort* cells [[buffer(6)]], uint block [[threadgroup_position_in_grid]],
            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
            for (uint at = sg; at < counts[block]; at += 8) {
                const uint cell = cells[block * 512 + at], row = cell / 32, column = block * 32 + cell % 32;
                float total = 0;
                for (uint k = 0; k < 5120; k += 512) {
                    const uint start = k + lane * 16, group = column * 80 + start / 64;
                    const ushort4 words = *reinterpret_cast<const device ushort4*>(w + column * 640 + start / 8);
                    float partial = 0, sum = 0;
                    for (int i = 0; i < 4; ++i) {
                        const float4 a = float4(*reinterpret_cast<const device bfloat4*>(x + row * 5120 + start + 4 * i));
                        const uint q = words[i];
                        sum += a.x + a.y + a.z + a.w;
                        partial += a.x * (q & 15) + a.y * ((q >> 4) & 15) + a.z * ((q >> 8) & 15) + a.w * ((q >> 12) & 15);
                    }
                    total += float(scales[group]) * partial + sum * float(biases[group]);
                }
                const float value = simd_sum(total);
                if (lane == 0) y[row * 16384 + column] = bfloat(value);
            }
        }
        """
    }
}
#endif
