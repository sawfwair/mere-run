#if os(macOS)
import Foundation
import Metal
import MereRunTensor
import MLX

@available(macOS 26.0, *)
private final class Q35NativeStateCompletion {
    private let lock = NSLock()
    private var pending: (any MTLCommandBuffer)?
    private var failure: Error?
    init(_ command: any MTLCommandBuffer) { pending = command }

    func wait() throws {
        if let command = lock.withLock({ pending }) {
            command.waitUntilCompleted()
            lock.withLock {
                if pending != nil {
                    if command.status != .completed { failure = Q35NativeStateCommit.Failure.command(command.error) }
                    pending = nil
                }
            }
        }
        if let failure = lock.withLock({ failure }) { throw failure }
    }
}

@available(macOS 26.0, *)
final class Q35NativeStoredLinearState: Q35NativeLinearState {
    private enum Backing {
        case empty
        case mlx(MLXArray, MLXArray)
        case native(Q35NativeArena.Slice, Q35NativeArena.Slice, Q35NativeStateCompletion)
    }
    private let backing: Backing
    let convolutionBuffer: M5NativeArrayBuffer
    let recurrentBuffer: M5NativeArrayBuffer

    init(zeroedOn device: any MTLDevice) throws {
        let conv = try Q35NativeArena(device: device, bytes: 3 * 10_240 * 2, dtype: .bfloat16).allocate(3 * 10_240)
        let state = try Q35NativeArena(device: device, bytes: 48 * 128 * 128 * 4, dtype: .float32).allocate(48 * 128 * 128)
        conv.buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: conv.byteCount)
        state.buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: state.byteCount)
        backing = .empty
        convolutionBuffer = M5NativeArrayBuffer(buffer: conv.buffer, offset: conv.offset, shape: [1, 3, 10_240], dtype: .bfloat16)
        recurrentBuffer = M5NativeArrayBuffer(buffer: state.buffer, offset: state.offset, shape: [1, 48, 128, 128], dtype: .float32)
    }

    init(convolution: MLXArray, recurrent: MLXArray, device: any MTLDevice) throws {
        backing = .mlx(convolution, recurrent)
        convolutionBuffer = try M5NativeArrayBuffer(convolution, device: device)
        recurrentBuffer = try M5NativeArrayBuffer(recurrent, device: device)
    }

    fileprivate init(convolution: Q35NativeArena.Slice, recurrent: Q35NativeArena.Slice, completion: Q35NativeStateCompletion) {
        backing = .native(convolution, recurrent, completion)
        convolutionBuffer = M5NativeArrayBuffer(buffer: convolution.buffer, offset: convolution.offset,
            shape: [1, 3, 10240], dtype: .bfloat16)
        recurrentBuffer = M5NativeArrayBuffer(buffer: recurrent.buffer, offset: recurrent.offset,
            shape: [1, 48, 128, 128], dtype: .float32)
    }

    func waitUntilReady() throws {
        if case .native(_, _, let completion) = backing { try completion.wait() }
    }

    func materialized() throws -> (convolution: MLXArray, recurrent: MLXArray) {
        try waitUntilReady()
        switch backing {
        case .empty:
            return (MLXArray.zeros([1, 3, 10_240], dtype: .bfloat16), MLXArray.zeros([1, 48, 128, 128], dtype: .float32))
        case .mlx(let conv, let state): return (conv.reshaped(conv.shape), state.reshaped(state.shape))
        case .native(let conv, let state, _): return (conv.array([1, 3, 10240]), state.array([1, 48, 128, 128]))
        }
    }
}

@available(macOS 26.0, *)
struct Q35NativeReplayInput {
    let initial: any Q35NativeLinearState
    let qkvz: Q35NativeArena.Slice
    let ba: Q35NativeArena.Slice
    let q: Q35NativeArena.Slice
    let k: Q35NativeArena.Slice
    let v: Q35NativeArena.Slice
    let g: Q35NativeArena.Slice
    let beta: Q35NativeArena.Slice
    let aLog: MLXArray
    let dtBias: MLXArray
}

/// One native submission commits every linear layer. Enqueueing does not wait;
/// the next draft can run while these immutable next-state buffers are filled.
@available(macOS 26.0, *)
final class Q35NativeStateCommit {
    enum Failure: Error { case allocation, missingFunction, command(Error?) }
    struct Prepared {
        fileprivate let command: any MTLCommandBuffer
        fileprivate let outputs: [any Q35NativeLinearState]
    }
    struct Destination {
        let convolution: Q35NativeArena.Slice
        let recurrent: Q35NativeArena.Slice

        static func bank(device: any MTLDevice, layers: Int) throws -> [Destination] {
            let convolution = try Q35NativeArena(device: device, bytes: layers * 3 * 10_240 * 2, dtype: .bfloat16)
            let recurrent = try Q35NativeArena(device: device, bytes: layers * 48 * 128 * 128 * 4, dtype: .float32)
            return (0..<layers).map { _ in
                Destination(convolution: convolution.allocate(3 * 10_240), recurrent: recurrent.allocate(48 * 128 * 128))
            }
        }
    }
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let recurrent: any MTLComputePipelineState
    private let convolution: any MTLComputePipelineState
    private let countLock = NSLock()
    private var count = 0
    var batchCount: Int { countLock.withLock { count } }

    init(device: any MTLDevice, queue: any MTLCommandQueue) throws {
        self.device = device; self.queue = queue
        let source = M5NativeFFN.shaderHeader + "\n" + """
            kernel void q35_native_recurrent_commit(const device bfloat* k [[buffer(0)]],
                const device bfloat* v [[buffer(1)]], const device float* g [[buffer(2)]], const device float* beta [[buffer(3)]],
                const device float* initial [[buffer(4)]], const device int* path [[buffer(5)]], const device int* info [[buffer(6)]],
                device float* next [[buffer(7)]], uint3 thread_position_in_grid [[thread_position_in_grid]],
                uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {
                constexpr int Dk = 128, Dv = 128, Hk = 16, Hv = 48;
                \(Q35TreePipelineKernels.recurrentSource)
            }
            kernel void q35_native_convolution_commit(const device bfloat* qkv [[buffer(0)]],
                const device bfloat* initial [[buffer(1)]], const device int* path [[buffer(2)]], const device int* info [[buffer(3)]],
                device bfloat* next [[buffer(4)]], constant int& initialStride [[buffer(5)]],
                uint3 thread_position_in_grid [[thread_position_in_grid]]) {
                constexpr int Channels = 10240, Keep = 3;
                \(Q35TreePipelineKernels.convolutionSource
                    .replacingOccurrences(of: "initial[(Keep + time) * Channels + channel]", with: "initial[(Keep + time) * initialStride + channel]")
                    .replacingOccurrences(of: "qkv[path[time] * Channels + channel]", with: "qkv[path[time] * 16384 + channel]"))
            }
            """
        let library = try M5NativeFFN.makeLibrary(device: device, source: source)
        guard let recurrent = library.makeFunction(name: "q35_native_recurrent_commit"),
              let convolution = library.makeFunction(name: "q35_native_convolution_commit") else { throw Failure.missingFunction }
        self.recurrent = try device.makeComputePipelineState(function: recurrent)
        self.convolution = try device.makeComputePipelineState(function: convolution)
    }

    func enqueue(inputs: [Q35NativeReplayInput], path: [Int]) throws -> [any Q35NativeLinearState] {
        precondition(!inputs.isEmpty && (1...16).contains(path.count))
        func integers(_ values: [Int32]) throws -> any MTLBuffer {
            try values.withUnsafeBytes {
                guard let value = device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                    options: [.storageModeShared, .hazardTrackingModeUntracked]) else { throw Failure.allocation }
                return value
            }
        }
        let rows = try integers(path.map(Int32.init)), info = try integers([Int32(path.count)])
        return submit(try prepare(inputs: inputs, rows: rows, pathOffset: 0, info: info))
    }

    func prepare(
        inputs: [Q35NativeReplayInput], selection: any MTLBuffer, destinations: [Destination]? = nil,
        appendKV: ((any MTLComputeCommandEncoder) -> Void)? = nil
    ) throws -> Prepared {
        try prepare(inputs: inputs, rows: selection, pathOffset: Q35NativeGreedyTail.pathOffset, info: selection,
                    destinations: destinations, appendKV: appendKV)
    }

    private func prepare(
        inputs: [Q35NativeReplayInput], rows: any MTLBuffer, pathOffset: Int, info: any MTLBuffer,
        destinations: [Destination]? = nil, appendKV: ((any MTLComputeCommandEncoder) -> Void)? = nil
    ) throws -> Prepared {
        let targets = try destinations ?? Destination.bank(device: device, layers: inputs.count)
        precondition(targets.count == inputs.count)
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent)
        else { throw Failure.allocation }
        command.label = "Qwen native accepted-state commit"
        let completion = Q35NativeStateCompletion(command)
        var outputs: [any Q35NativeLinearState] = []
        for (input, destination) in zip(inputs, targets) {
            let conv = destination.convolution, state = destination.recurrent
            let initial = input.initial.recurrentBuffer
            precondition(initial.shape == [1, 48, 128, 128] && initial.dtype == .float32
                && Array(initial.strides.dropFirst()) == [16384, 128, 1])
            encoder.setComputePipelineState(recurrent)
            for (index, slice) in [input.k, input.v, input.g, input.beta].enumerated() {
                encoder.setBuffer(slice.buffer, offset: slice.offset, index: index)
            }
            encoder.setBuffer(initial.buffer, offset: initial.offset, index: 4)
            encoder.setBuffer(rows, offset: pathOffset, index: 5)
            encoder.setBuffer(info, offset: 0, index: 6)
            encoder.setBuffer(state.buffer, offset: state.offset, index: 7)
            encoder.dispatchThreadgroups(MTLSize(width: 1, height: 32, depth: 48),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
            let oldConv = input.initial.convolutionBuffer
            precondition(oldConv.shape == [1, 3, 10240] && oldConv.dtype == .bfloat16 && oldConv.strides.last == 1)
            encoder.setComputePipelineState(convolution)
            encoder.setBuffer(input.qkvz.buffer, offset: input.qkvz.offset, index: 0)
            encoder.setBuffer(oldConv.buffer, offset: oldConv.offset, index: 1)
            encoder.setBuffer(rows, offset: pathOffset, index: 2)
            encoder.setBuffer(info, offset: 0, index: 3)
            encoder.setBuffer(conv.buffer, offset: conv.offset, index: 4)
            var stride = Int32(oldConv.strides[1])
            encoder.setBytes(&stride, length: 4, index: 5)
            encoder.dispatchThreads(MTLSize(width: 3 * 10240, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            outputs.append(Q35NativeStoredLinearState(convolution: conv, recurrent: state, completion: completion))
        }
        appendKV?(encoder)
        // Every output interval is distinct; no kernel reads another commit's output.
        encoder.endEncoding()
        return Prepared(command: command, outputs: outputs)
    }

    func completed(_ targets: [Destination], command: any MTLCommandBuffer) -> [any Q35NativeLinearState] {
        let completion = Q35NativeStateCompletion(command)
        return targets.map {
            Q35NativeStoredLinearState(convolution: $0.convolution, recurrent: $0.recurrent, completion: completion)
        }
    }

    func submit(_ prepared: Prepared) -> [any Q35NativeLinearState] {
        prepared.command.commit()
        countLock.withLock { count += 1 }
        return prepared.outputs
    }
}

@available(macOS 26.0, *)
final class Q35NativeReplayBatch {
    let tree: Q35VerificationTree
    let inputs: [Q35NativeReplayInput]
    private let program: Q35NativeStateCommit
    private let lock = NSLock()
    private var prepared: (path: [Int], batch: Q35NativeStateCommit.Prepared)?
    private var results: [[Int]: [any Q35NativeLinearState]] = [:]

    init(tree: Q35VerificationTree, inputs: [Q35NativeReplayInput], program: Q35NativeStateCommit,
         prepared: (path: [Int], batch: Q35NativeStateCommit.Prepared)? = nil) {
        self.tree = tree; self.inputs = inputs; self.program = program
        self.prepared = prepared
    }

    func state(path: [Int], index: Int) throws -> any Q35NativeLinearState {
        precondition(!path.isEmpty && path == tree.paths[path.last!])
        return try lock.withLock {
            if let outputs = results[path] { return outputs[index] }
            let outputs: [any Q35NativeLinearState]
            if let prepared, prepared.path == path {
                outputs = program.submit(prepared.batch)
                self.prepared = nil
            } else {
                outputs = try program.enqueue(inputs: inputs, path: path)
            }
            results[path] = outputs
            return outputs[index]
        }
    }
}

@available(macOS 26.0, *)
final class Q35NativeReplayVertex: Q35NativeLinearReplay {
    private let batch: Q35NativeReplayBatch
    private let index: Int
    private let lock = NSLock()
    private var cachedLegacy: Q35LinearVerificationReplay?
    init(batch: Q35NativeReplayBatch, index: Int) { self.batch = batch; self.index = index }

    func acceptedState(path: [Int]) throws -> any Q35NativeLinearState { try batch.state(path: path, index: index) }
    var gates: (g: MLXArray, beta: MLXArray) {
        let input = batch.inputs[index], count = batch.tree.count
        return (input.g.array([1, count, 48]), input.beta.array([1, count, 48]))
    }
    var legacy: Q35LinearVerificationReplay {
        lock.withLock {
            if let cachedLegacy { return cachedLegacy }
            let input = batch.inputs[index], count = batch.tree.count, base = input.initial.exported()
            let projected = input.qkvz.array([1, 16, 16384]), ba = input.ba.array([1, 16, 96])
            let replay = Q35LinearVerificationReplay(baseConvState: base.convolution, baseRecurrentState: base.recurrent,
                qkv: projected[0..., 0..<count, 0..<10240], q: input.q.array([1, count, 16, 128]),
                k: input.k.array([1, count, 16, 128]), v: input.v.array([1, count, 48, 128]),
                a: ba[0..., 0..<count, 48...], b: ba[0..., 0..<count, 0..<48], aLog: input.aLog, dtBias: input.dtBias,
                numKeyHeads: 16, numValueHeads: 48, valueHeadDim: 128, convKeep: 3)
            cachedLegacy = replay
            return replay
        }
    }
}
#endif
