#if os(macOS)
import Foundation
import Metal
import MLX

/// Serial-QMV arithmetic for small affine-Q4 gates. Inputs and outputs use
/// physical sixteen-row buffers; a final partial group writes unused padded rows.
@available(macOS 26.0, *)
package final class M5NativeSmallProjection {
    enum Failure: Error { case incompatibleWeights, missingFunction, allocation }

    package struct Weights {
        let codes: M5NativeArrayBuffer
        let scales: M5NativeArrayBuffer
        let biases: M5NativeArrayBuffer
        package let inputSize: Int
        package let outputSize: Int

        init(weight: MLXArray, scales: MLXArray, biases: MLXArray, device: any MTLDevice) throws {
            guard weight.dtype == .uint32, scales.dtype == .bfloat16, biases.dtype == .bfloat16,
                  weight.ndim == 2, weight.dim(1).isMultiple(of: 64), weight.dim(0).isMultiple(of: 8),
                  scales.shape == [weight.dim(0), weight.dim(1) / 8], biases.shape == scales.shape else {
                throw Failure.incompatibleWeights
            }
            inputSize = weight.dim(1) * 8; outputSize = weight.dim(0)
            codes = try M5NativeArrayBuffer(weight, device: device)
            self.scales = try M5NativeArrayBuffer(scales, device: device)
            self.biases = try M5NativeArrayBuffer(biases, device: device)
        }
    }

    private let inputSize: Int
    private let outputSize: Int
    private let rowsPerGroup: Int
    private let simdgroups: Int
    private let pipeline: any MTLComputePipelineState
    private let biasSums: (pipeline: any MTLComputePipelineState, buffer: any MTLBuffer)?

    package var biasSumsBuffer: (any MTLBuffer)? { biasSums?.buffer }

    package init(device: any MTLDevice, inputSize: Int, outputSize: Int, simdgroups: Int = 2,
                 precomputeBiasSums: Bool = false) throws {
        precondition(inputSize > 0 && inputSize.isMultiple(of: 512) && outputSize > 0
            && [2, 4].contains(simdgroups) && outputSize.isMultiple(of: simdgroups * 4))
        self.inputSize = inputSize; self.outputSize = outputSize
        self.simdgroups = simdgroups
        rowsPerGroup = outputSize >= 512 ? 4 : 2
        precondition(!precomputeBiasSums || rowsPerGroup == 4)
        let name = "mere_native_scalar_\(inputSize)_\(outputSize)"
        // A native BF16 vector replaces MLX's vector utility wrapper while
        // retaining the same FP32 lane arithmetic and aligned eight-byte load.
        let header = SmallBatchAffineQMV.header.replacingOccurrences(
            of: #"const vec<bfloat16_t, 4> xv =\s*\*reinterpret_cast<const device vec<bfloat16_t, 4>\*>\(xm\);"#,
            with: "const bfloat4 xv = *reinterpret_cast<const device bfloat4*>(xm);", options: .regularExpression)
        let sumArgument = precomputeBiasSums ? "const device float4* sums [[buffer(5)]]," : ""
        let sumParameter = precomputeBiasSums ? ", sums" : ""
        let sumSource = precomputeBiasSums ? Self.biasSumSource(width: inputSize) : ""
        let source = M5NativeFFN.shaderHeader + "\ntypedef bfloat bfloat16_t;\n" + header + "\n" + """
            kernel void \(name)(const device uint* w [[buffer(0)]], const device bfloat* scales [[buffer(1)]],
                const device bfloat* biases [[buffer(2)]], const device bfloat* x [[buffer(3)]],
                device bfloat* y [[buffer(4)]], \(sumArgument) uint3 tid [[threadgroup_position_in_grid]],
                uint sgi [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
                mere_affine4_qmv_wide<\(rowsPerGroup), 4, \(precomputeBiasSums)>(w, scales, biases, x, y, \(inputSize), \(outputSize),
                    int(tid.x) * \(rowsPerGroup), (int(tid.y) * \(simdgroups) + int(sgi)) * 4, lane\(sumParameter));
            }
            """ + "\n" + sumSource
        let library = try M5NativeFFN.makeLibrary(device: device, source: source)
        guard let function = library.makeFunction(name: name) else { throw Failure.missingFunction }
        pipeline = try device.makeComputePipelineState(function: function)
        if precomputeBiasSums {
            guard let kernel = library.makeFunction(name: "mere_native_qmv_bias_sums") else { throw Failure.missingFunction }
            guard let buffer = device.makeBuffer(length: 16 * (inputSize / 16) * MemoryLayout<Float>.stride,
                options: [.storageModePrivate, .hazardTrackingModeUntracked]) else { throw Failure.allocation }
            biasSums = (try device.makeComputePipelineState(function: kernel), buffer)
        } else {
            biasSums = nil
        }
    }

    package func encode(into encoder: any MTLComputeCommandEncoder, input: any MTLBuffer,
                        output: any MTLBuffer, weights: Weights, rows: Int, inputOffset: Int = 0, outputOffset: Int = 0, sumsPrepared: Bool = false, outputColumns: Int? = nil, hazards: M5NativeHazards? = nil) {
        precondition(rows > 0 && weights.inputSize == inputSize && weights.outputSize == outputSize)
        let columns = outputColumns ?? outputSize
        precondition(columns > 0 && columns <= outputSize && columns.isMultiple(of: simdgroups * 4))
        let paddedRows = ((rows + rowsPerGroup - 1) / rowsPerGroup) * rowsPerGroup
        precondition(inputOffset.isMultiple(of: 8))
        precondition(input.length >= inputOffset + paddedRows * inputSize * 2
            && output.length >= outputOffset + paddedRows * outputSize * 2)
        let inputAccess = M5NativeHazards.Access(input, offset: inputOffset, count: paddedRows * inputSize * 2)
        let result = M5NativeHazards.Access(output, offset: outputOffset, count: paddedRows * outputSize * 2)
        precondition(!sumsPrepared || biasSums != nil)
        if let biasSums {
            precondition(paddedRows <= 16)
            let count = paddedRows / 4 * (inputSize / 16)
            let scratch = M5NativeHazards.Access(biasSums.buffer, count: count * MemoryLayout<SIMD4<Float>>.stride)
            if !sumsPrepared {
                hazards?.prepare(encoder: encoder, reads: [inputAccess], writes: [scratch])
                encoder.setComputePipelineState(biasSums.pipeline)
                encoder.setBuffer(input, offset: inputOffset, index: 0)
                encoder.setBuffer(biasSums.buffer, offset: 0, index: 1)
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
            }
            hazards?.prepare(encoder: encoder, reads: [inputAccess, scratch], writes: [result])
            encoder.setBuffer(biasSums.buffer, offset: 0, index: 5)
        } else {
            hazards?.prepare(encoder: encoder, reads: [inputAccess], writes: [result])
        }
        encoder.setComputePipelineState(pipeline)
        for (index, view) in [weights.codes, weights.scales, weights.biases].enumerated() {
            encoder.setBuffer(view.buffer, offset: view.offset, index: index)
        }
        encoder.setBuffer(input, offset: inputOffset, index: 3)
        encoder.setBuffer(output, offset: outputOffset, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: (rows + rowsPerGroup - 1) / rowsPerGroup,
            height: columns / (simdgroups * 4), depth: 1),
            threadsPerThreadgroup: MTLSize(width: simdgroups * 32, height: 1, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }

    private static func biasSumSource(width: Int) -> String {
        """
        kernel void mere_native_qmv_bias_sums(const device bfloat* x [[buffer(0)]], device float4* sums [[buffer(1)]],
                                            uint index [[thread_position_in_grid]]) {
            constexpr int width = \(width), laneCount = width / 16;
            const int firstRow = (index / laneCount) * 4;
            const int laneStart = (index % laneCount) * 16;
            float4 total = 0;
            for (int step = 0; step < 4; ++step) {
                for (int row = 0; row < 4; ++row) {
                    const device bfloat* input = x + (firstRow + row) * width + laneStart + 4 * step;
                    const bfloat4 a = *reinterpret_cast<const device bfloat4*>(input);
                    total[row] += float(a[0]) + float(a[1]) + float(a[2]) + float(a[3]);
                }
            }
            sums[index] = total;
        }
        """
    }
}
#endif
