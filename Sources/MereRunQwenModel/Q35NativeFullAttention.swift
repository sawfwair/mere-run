#if os(macOS)
import MLX
import Metal
import MereRunTensor

/// Native vector SDPA with shared committed KV and separately owned tree rows.
@available(macOS 26.0, *)
final class Q35NativeFullAttention {
    enum Failure: Error { case missingFunction, allocation }
    struct Bucket {
        fileprivate let rows: any MTLBuffer
        fileprivate let info: any MTLBuffer
        fileprivate let count: Int
        fileprivate let blocks: Int
        fileprivate let partial: any MTLComputePipelineState
        fileprivate let merge: (any MTLComputePipelineState)?
    }
    struct Prepared {
        fileprivate let plan: Q35NativeTreePlan
        fileprivate let buckets: [Bucket]
        fileprivate let scratch: Scratch
    }
    struct Scratch {
        fileprivate let partials: any MTLBuffer
        fileprivate let sums: any MTLBuffer
        fileprivate let maxs: any MTLBuffer
        fileprivate let slots: Int

        fileprivate init(device: any MTLDevice, slots: Int) throws {
            func allocate(_ bytes: Int) throws -> any MTLBuffer {
                guard let buffer = device.makeBuffer(length: bytes, options: [.storageModePrivate, .hazardTrackingModeUntracked])
                else { throw Failure.allocation }
                return buffer
            }
            partials = try allocate(max(4, slots * 256 * 2))
            sums = try allocate(max(4, slots * 4))
            maxs = try allocate(max(4, slots * 4))
            self.slots = slots
        }
    }
    private let device: any MTLDevice
    private let architecture: Character?
    private var pipelines: [Int: (any MTLComputePipelineState, (any MTLComputePipelineState)?)] = [:]

    init(device: any MTLDevice) {
        self.device = device
        architecture = device.architecture.name.last
    }

    func prepare(plan: Q35NativeTreePlan, prefix: Int, scratch: inout Scratch?) throws -> Prepared {
        precondition(prefix >= 0 && prefix <= Int(Int32.max) - plan.count)
        let groups = Dictionary(grouping: 0..<plan.count) {
            Q35IndexedTreeAttention.blocks(length: prefix + plan.pathLengths[$0], gqa: 6, architecture: architecture)
        }.sorted { $0.key < $1.key }
        func ints(_ values: [Int]) throws -> any MTLBuffer {
            try values.map(Int32.init).withUnsafeBytes {
                guard let value = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                else { throw Failure.allocation }
                return value
            }
        }
        let buckets = try groups.map { blocks, rows in
            let (partial, merge) = try pipeline(blocks: blocks)
            return try Bucket(rows: ints(rows), info: ints([prefix + rows.map { plan.pathLengths[$0] }.max()!, prefix, 24, 4, plan.branchRows, plan.pathStride, plan.count]),
                count: rows.count, blocks: blocks, partial: partial, merge: merge)
        }
        let slots = plan.count * 24 * (groups.last?.key ?? 0)
        let workspace: Scratch
        if let scratch, scratch.slots >= slots {
            workspace = scratch
        } else {
            workspace = try Scratch(device: device, slots: slots)
            scratch = workspace
        }
        return Prepared(plan: plan, buckets: buckets, scratch: workspace)
    }

    private func pipeline(blocks: Int) throws -> (any MTLComputePipelineState, (any MTLComputePipelineState)?) {
        if let value = pipelines[blocks] { return value }
        let main = blocks == 0 ? Self.singleSource : Self.partialSource
        let source = "#include <metal_stdlib>\n" + Self.header + "\n" + main + (blocks == 0 ? "" : Self.mergeSource)
        let expanded = source.replacingOccurrences(of: "NATIVE_BLOCKS", with: String(blocks))
        let library = try M5NativeFFN.makeLibrary(device: device, source: expanded)
        guard let function = library.makeFunction(name: "q35_native_tree_attention") else { throw Failure.missingFunction }
        let first = try device.makeComputePipelineState(function: function)
        let second: (any MTLComputePipelineState)?
        if blocks > 0 {
            guard let merge = library.makeFunction(name: "q35_native_tree_attention_merge") else { throw Failure.missingFunction }
            second = try device.makeComputePipelineState(function: merge)
        } else { second = nil }
        pipelines[blocks] = (first, second)
        return (first, second)
    }

    func encode(into encoder: any MTLComputeCommandEncoder, queries: any MTLBuffer,
                prefixKeys: M5NativeArrayBuffer?, prefixValues: M5NativeArrayBuffer?,
                branchKeys: any MTLBuffer, branchValues: any MTLBuffer, output: any MTLBuffer, prepared: Prepared,
                branchKeysOffset: Int = 0, branchValuesOffset: Int = 0, hazards: M5NativeHazards? = nil) {
        precondition((prefixKeys == nil) == (prefixValues == nil))
        for view in [prefixKeys, prefixValues].compactMap({ $0 }) {
            precondition(view.shape.count == 4 && view.shape[0] == 1 && view.shape[1] == 4
                && view.shape[3] == 256 && view.dtype == .bfloat16)
        }
        var strides = [prefixKeys?.strides[1] ?? 4096, prefixKeys?.strides[2] ?? 256, prefixKeys?.strides[3] ?? 1,
                       prefixValues?.strides[1] ?? 4096, prefixValues?.strides[2] ?? 256, prefixValues?.strides[3] ?? 1].map(Int32.init)
        var scale = Float(1.0 / 16.0)
        let scratch = prepared.scratch
        for bucket in prepared.buckets {
            hazards?.prepare(encoder: encoder,
                reads: [.init(queries, count: prepared.plan.count * 6144 * 2),
                        .init(branchKeys, offset: branchKeysOffset, count: 4 * prepared.plan.branchRows * 256 * 2),
                        .init(branchValues, offset: branchValuesOffset, count: 4 * prepared.plan.branchRows * 256 * 2)],
                writes: bucket.blocks == 0 ? [.init(output)] : [.init(scratch.partials), .init(scratch.sums), .init(scratch.maxs)])
            encoder.setComputePipelineState(bucket.partial)
            encoder.setBuffer(queries, offset: 0, index: 0)
            encoder.setBuffer(prefixKeys?.buffer ?? branchKeys, offset: prefixKeys?.offset ?? branchKeysOffset, index: 1)
            encoder.setBuffer(prefixValues?.buffer ?? branchValues, offset: prefixValues?.offset ?? branchValuesOffset, index: 2)
            for (index, buffer) in [prepared.plan.paths, bucket.rows, prepared.plan.lengths, bucket.info].enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index + 3)
            }
            encoder.setBytes(&scale, length: 4, index: 7)
            encoder.setBuffer(bucket.blocks == 0 ? output : scratch.partials, offset: 0, index: 8)
            encoder.setBuffer(scratch.sums, offset: 0, index: 9)
            encoder.setBuffer(scratch.maxs, offset: 0, index: 10)
            encoder.setBuffer(branchKeys, offset: branchKeysOffset, index: 11)
            encoder.setBuffer(branchValues, offset: branchValuesOffset, index: 12)
            strides.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 13) }
            if bucket.blocks == 0 {
                encoder.dispatchThreadgroups(MTLSize(width: bucket.count * 24, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
            } else {
                encoder.dispatchThreadgroups(MTLSize(width: 4, height: bucket.count, depth: bucket.blocks),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 6, depth: 1))
            }
            if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
            if let merge = bucket.merge {
                hazards?.prepare(encoder: encoder,
                    reads: [.init(scratch.partials), .init(scratch.sums), .init(scratch.maxs)], writes: [.init(output)])
                encoder.setComputePipelineState(merge)
                for (index, buffer) in [scratch.partials, scratch.sums, scratch.maxs, output, bucket.rows].enumerated() {
                    encoder.setBuffer(buffer, offset: 0, index: index)
                }
                encoder.dispatchThreadgroups(MTLSize(width: bucket.count * 24, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
            }
        }
    }

    private static let header = Q35IndexedTreeKernels.header
        .replacingOccurrences(of: "const device T* base;", with: "const device T* base; const device T* branch;")
        .replacingOccurrences(of: "const device T* b, const device int* p", with: "const device T* b, const device T* c, const device int* p")
        .replacingOccurrences(of: "int ss, int ds)", with: "int ss, int ds, int br)")
        .replacingOccurrences(of: "int offset, prefix,", with: "int branch_rows, offset, prefix,")
        .replacingOccurrences(of: ": base(b), path(p)", with: ": base(b), branch(c), path(p), branch_rows(br)")
        .replacingOccurrences(of: "current = base + physical * sequence_stride + dim * dimension_stride;",
            with: "current = logical < prefix ? base + physical * sequence_stride + dim * dimension_stride : branch + path[min(logical - prefix, branch_rows - 1)] * D + dim;")
        .replacingOccurrences(of: "path[min(logical - prefix, 15)]", with: "path[min(logical - prefix, branch_rows - 1)]")
        .replacingOccurrences(of: "return current[delta * dimension_stride];", with: "return current[delta * (offset / D < prefix ? dimension_stride : 1)];")

    private static let parameters = """
        const device bfloat* q_input [[buffer(0)]], const device bfloat* k_input [[buffer(1)]],
        const device bfloat* v_input [[buffer(2)]], const device int* paths [[buffer(3)]],
        const device int* rows [[buffer(4)]], const device int* lengths [[buffer(5)]], const device int* info [[buffer(6)]],
        constant float* scale_input [[buffer(7)]], device bfloat* y [[buffer(8)]], device float* sums [[buffer(9)]],
        device float* maxs [[buffer(10)]], const device bfloat* branch_k [[buffer(11)]], const device bfloat* branch_v [[buffer(12)]],
        constant int* strides [[buffer(13)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
        uint3 threadgroups_per_grid [[threadgroups_per_grid]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
        uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint3 threads_per_threadgroup [[threads_per_threadgroup]],
        uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
        """

    private static func adapted(_ source: String) -> String {
        source.replacingOccurrences(of: "MereTreeKV<T, D> keys{k_input + physical_head * k_input_strides[1],",
            with: "MereTreeKV<T, D> keys{k_input + physical_head * k_input_strides[1], branch_k + physical_head * info[4] * 256,")
            .replacingOccurrences(of: "MereTreeKV<T, V> values{v_input + physical_head * v_input_strides[1],",
                with: "MereTreeKV<T, V> values{v_input + physical_head * v_input_strides[1], branch_v + physical_head * info[4] * 256,")
            .replacingOccurrences(of: "paths + original_row * 16", with: "paths + original_row * info[5]")
            .replacingOccurrences(of: "int(k_input_strides[3])};", with: "int(k_input_strides[3]), info[5] == 0 ? info[6] : 16};")
            .replacingOccurrences(of: "int(v_input_strides[3])};", with: "int(v_input_strides[3]), info[5] == 0 ? info[6] : 16};")
    }

    private static let declarations = """
        typedef bfloat T;
        constexpr int D = 256, V = 256, G = 6, Blocks = NATIVE_BLOCKS;
        const int k_input_strides[4] = {0, strides[0], strides[1], strides[2]};
        const int v_input_strides[4] = {0, strides[3], strides[4], strides[5]};
        """
    private static let singleSource = """
        kernel void q35_native_tree_attention(\(parameters)) {
            \(declarations)
            \(adapted(Q35IndexedTreeKernels.singleSource).replacingOccurrences(
                of: "const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;",
                with: "const int o_offset = original_row * 24 + (tid.x % 24);"))
        }
        """
    private static let partialSource = """
        kernel void q35_native_tree_attention(\(parameters)) {
            \(declarations)
            device bfloat* partials = y;
            \(adapted(Q35IndexedTreeKernels.partialSource).replacingOccurrences(
                of: "queries += q_offset * D", with: "queries += (original_row * 24 + q_head_idx) * D"))
        }
        """
    private static let mergeSource = """
        kernel void q35_native_tree_attention_merge(device bfloat* partials [[buffer(0)]],
            device float* sums [[buffer(1)]], device float* maxs [[buffer(2)]], device bfloat* y [[buffer(3)]],
            const device int* rows [[buffer(4)]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
            uint3 threadgroups_per_grid [[threadgroups_per_grid]], uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
            uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
            typedef bfloat T;
            constexpr int D = 256, Blocks = NATIVE_BLOCKS;
            \(Q35IndexedTreeKernels.mergeSource.replacingOccurrences(of: "out += q_offset * D + simd_gid * elem_per_thread;",
                with: "out += (rows[head_idx / 24] * 24 + head_idx % 24) * D + simd_gid * elem_per_thread;"))
        }
        """
}
#endif
