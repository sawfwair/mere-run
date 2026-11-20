#if os(macOS)
import Metal
import MereRunTensor

@available(macOS 26.0, *)
final class Q35NativeGDNDelta {
    enum Failure: Error { case missingFunction }
    private struct Shape: Hashable { let rows: Int; let slots: Int; let advance: Bool }
    private let device: any MTLDevice
    private var pipelines: [Shape: any MTLComputePipelineState] = [:]
    struct Prepared {
        fileprivate let pipeline: any MTLComputePipelineState
        fileprivate let plan: Q35NativeTreePlan
    }

    init(device: any MTLDevice) { self.device = device }

    func prepare(_ plan: Q35NativeTreePlan) throws -> Prepared {
        try Prepared(pipeline: pipeline(for: plan), plan: plan)
    }

    private func pipeline(for plan: Q35NativeTreePlan) throws -> any MTLComputePipelineState {
        let shape = Shape(rows: plan.count, slots: plan.slotCount, advance: plan.pathStride == 0)
        precondition(shape.slots <= 8)
        if let pipeline = pipelines[shape] { return pipeline }
        let name = "q35_native_tree_delta_\(shape.rows)_\(shape.slots)"
        let advance = """
            if (step + 1 == S) {
                for (int i = 0; i < n_per_t; ++i)
                    next[(hv * Dv + dv) * Dk + n_per_t * lane + i] = state[i];
            }
            """
        let body = shape.advance ? Q35TreeCompactDeltaMetal.source.replacingOccurrences(
            of: "switch (write_slots[row])", with: advance + "\n switch (write_slots[row])"
        ) : Q35TreeCompactDeltaMetal.source
        let source = M5NativeFFN.shaderHeader + "\n" + """
            kernel void \(name)(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]],
                const device bfloat* v [[buffer(2)]], const device float* g [[buffer(3)]], const device float* beta [[buffer(4)]],
                const device float* state_in [[buffer(5)]], const device int* read_slots [[buffer(6)]],
                const device int* write_slots [[buffer(7)]], const device int* rows [[buffer(8)]], device bfloat* y [[buffer(9)]],
                device float* next [[buffer(10)]],
                uint3 thread_position_in_grid [[thread_position_in_grid]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
                typedef bfloat InT;
                constexpr int S = \(shape.rows), Slots = \(shape.slots), Hk = 16, Hv = 48, Dk = 128, Dv = 128;
                \(body)
            }
            """
        let library = try M5NativeFFN.makeLibrary(device: device, source: source)
        guard let function = library.makeFunction(name: name) else { throw Failure.missingFunction }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[shape] = pipeline
        return pipeline
    }

    func encode(into encoder: any MTLComputeCommandEncoder, q: any MTLBuffer, k: any MTLBuffer,
                v: any MTLBuffer, g: any MTLBuffer, beta: any MTLBuffer, state: M5NativeArrayBuffer,
                output: any MTLBuffer, prepared: Prepared, qOffset: Int = 0, kOffset: Int = 0,
                vOffset: Int = 0, gOffset: Int = 0, betaOffset: Int = 0,
                destination: Q35NativeArena.Slice? = nil, hazards: M5NativeHazards? = nil) {
        precondition(state.shape == [1, 48, 128, 128] && state.dtype == .float32
            && Array(state.strides.dropFirst()) == [16_384, 128, 1])
        let plan = prepared.plan
        var writes: [M5NativeHazards.Access] = [.init(output, count: plan.count * 48 * 128 * 2)]
        if let destination { writes.append(.init(destination.buffer, offset: destination.offset, count: destination.byteCount)) }
        hazards?.prepare(encoder: encoder,
            reads: [.init(q, offset: qOffset, count: plan.count * 16 * 128 * 2),
                    .init(k, offset: kOffset, count: plan.count * 16 * 128 * 2),
                    .init(v, offset: vOffset, count: plan.count * 48 * 128 * 2),
                    .init(g, offset: gOffset, count: plan.count * 48 * 4),
                    .init(beta, offset: betaOffset, count: plan.count * 48 * 4)],
            writes: writes)
        encoder.setComputePipelineState(prepared.pipeline)
        let offsets = [qOffset, kOffset, vOffset, gOffset, betaOffset]
        for (index, buffer) in [q, k, v, g, beta].enumerated() { encoder.setBuffer(buffer, offset: offsets[index], index: index) }
        encoder.setBuffer(state.buffer, offset: state.offset, index: 5)
        for (index, buffer) in [plan.reads, plan.writes, plan.order, output].enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index + 6)
        }
        encoder.setBuffer(destination?.buffer ?? state.buffer, offset: destination?.offset ?? state.offset, index: 10)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 32, depth: 48),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        if hazards == nil { encoder.memoryBarrier(scope: .buffers) }
    }
}
#endif
