#if os(macOS)
import Foundation
import Metal
import MereRunTensor
import MLX
import MLXNN

@available(macOS 26.0, *)
final class Q35NativeDraftProgram {
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_NATIVE_BLOCK")
    static var enabled: Bool { enabledFlag.enabled }
    typealias Binding = Q35NativeDraftKernels.Binding
    typealias Failure = Q35NativeDraftKernels.Failure

    struct Projection {
        let weights: M5NativeSmallProjection.Weights
        let operation: M5NativeSmallProjection

        func encode(_ encoder: any MTLComputeCommandEncoder, _ input: any MTLBuffer, _ output: any MTLBuffer, _ rows: Int) {
            operation.encode(into: encoder, input: input, output: output, weights: weights, rows: rows)
        }
    }

    struct Convolution {
        let projection: Projection
        let base: M5NativeArrayBuffer
    }

    struct Layer {
        let inputNorm: M5NativeArrayBuffer
        let attentionConv: Convolution
        let qkv: Projection
        let kv: Projection
        let queryNorm: M5NativeArrayBuffer
        let keyNorm: M5NativeArrayBuffer
        let output: Projection
        let postNorm: M5NativeArrayBuffer
        let mlpConv: Convolution
        let ffn: M5NativeFFN.Weights
    }

    struct Ring {
        let keys: any MTLBuffer
        let values: any MTLBuffer
    }

    final class Session: Q35NativeDraftSession {
        fileprivate let program: Q35NativeDraftProgram
        fileprivate let rings: [Ring]
        fileprivate var offset: Int
        fileprivate var count: Int
        fileprivate var pending: [(any MTLBuffer, Int)] = []
        fileprivate var pendingCount = 0
        #if DEBUG || MERERUN_RELEASE_REPLAY
        var observeCandidates: (([Int32]) -> Void)?
        #endif

        fileprivate init(program: Q35NativeDraftProgram, rings: [Ring], offset: Int, count: Int) {
            self.program = program
            self.rings = rings
            self.offset = offset
            self.count = count
        }

        func propose(anchor: Int, depth: Int, nodes: Int) throws -> DFlash2TreeProposal {
            try program.propose(session: self, anchor: anchor, depth: depth, nodes: nodes)
        }

        func propose(anchor: Int, depth: Int, nodes: Int, sampling: Q35PositionSampler, position: Int) throws -> DFlash2TreeProposal {
            try program.propose(session: self, anchor: anchor, depth: depth, nodes: nodes,
                                sampling: sampling, position: position)
        }

        func sampledChain(anchor: Int, depth: Int, noise: MLXArray) throws -> Q35SampledProposal {
            let result = try program.candidates(session: self, anchor: anchor, depth: depth, referenceScores: true)
            return Q35SampledProposal(candidates: result.tokens, scores: result.scores!, depth: depth, noise: noise.asArray(Float.self))
        }

        func stage(_ output: Q35GreedyTreeOutput) throws {
            precondition(pending.isEmpty)
            let accepted = output.selection.path.count
            pending = try program.tapIDs.map { id in
                if let tap = output.nativeDraftTaps[id] { return (tap.buffer, tap.offset) }
                guard let tensor = output.taps[id] else { throw Failure.geometry }
                let view = try M5NativeArrayBuffer(contiguous(tensor), device: program.device)
                return (view.buffer, view.offset)
            }
            pendingCount = accepted
        }

        func stageNative(_ taps: [Int: Q35NativeDraftTap], count: Int, skipping: Int = 0) throws {
            precondition(pending.isEmpty && count > 0 && skipping >= 0)
            pending = try program.tapIDs.map { id in
                guard let tap = taps[id] else { throw Failure.geometry }
                return (tap.buffer, tap.offset + skipping * 5_120 * 2)
            }
            pendingCount = count
        }

        func absorbNative() throws {
            if pendingCount > 0 { try program.flush(session: self) }
        }

        func finish(_ caches: [Q35DFlash2Cache]) throws {
            if !pending.isEmpty { try program.flush(session: self) }
            publish(caches)
        }

        private func publish(_ caches: [Q35DFlash2Cache]) {
            for (cache, ring) in zip(caches, rings) {
                func view(_ buffer: any MTLBuffer) -> MLXArray {
                    let physical = MLXArray(rawPointer: buffer.contents(), [Self.window, 8, 128], dtype: .bfloat16) { _ = buffer }
                    let start = (offset - count) % Self.window
                    let first = min(count, Self.window - start)
                    let a = physical[start..<(start + first), 0..., 0...]
                    let ordered = first == count ? a : concatenated([a, physical[0..<(count - first), 0..., 0...]], axis: 0)
                    return ordered.transposed(1, 0, 2).reshaped(1, 8, count, 128)
                }
                cache.replaceCommittedContext(keys: count == 0 ? nil : view(ring.keys),
                                              values: count == 0 ? nil : view(ring.values), offset: offset)
            }
        }

        private static let window = 2048
    }

    private static let width = 5120
    private static let window = 2048
    private static let kvWidth = 1024

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let lock = NSLock()
    private let kernels: Q35NativeDraftKernels
    private let norm: M5NativeRMSNorm
    private let ffn: M5NativeFFN
    private let ffnWorkspace: M5NativeFFN.Workspace
    private let head: Q35DFlash2NativeDraftHead.MetalProgram
    private let tapIDs: [Int]
    private let maskToken: Int
    private let epsilon: Float
    private let ropeBase: Float
    private let embedding: [M5NativeArrayBuffer]
    private let layers: [Layer]
    private let contextProjection: Projection
    private let contextNorm: M5NativeArrayBuffer
    private let finalNorm: M5NativeArrayBuffer
    private let scratch: Scratch
    private var reported = false
    private let targetIdentity: ObjectIdentifier
    private let compactHead = Q35DFlash2DraftHead.enabled

    func matches(_ target: Q35Model) -> Bool {
        targetIdentity == ObjectIdentifier(target) && compactHead == Q35DFlash2DraftHead.enabled
    }

    struct Scratch {
        let hidden, combined, normalized, convolution, attended, projected, dynamic, taps: any MTLBuffer
        let query, blockKeys, keys, values, mask, partials: any MTLBuffer

        init(device: any MTLDevice) throws {
            func allocate(_ bytes: Int) throws -> any MTLBuffer {
                guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else { throw Failure.allocation }
                return buffer
            }
            hidden = try allocate(16 * 5120 * 2)
            combined = try allocate(16 * 5120 * 2)
            normalized = try allocate(16 * 5120 * 2)
            convolution = try allocate(16 * 5120 * 2)
            attended = try allocate(16 * 5120 * 2)
            projected = try allocate(16 * 6144 * 2)
            dynamic = try allocate(16 * 1280 * 2)
            taps = try allocate(16 * 5120 * 5 * 2)
            query = try allocate(16 * 4096 * 2)
            blockKeys = try allocate(16 * 1024 * 2)
            keys = try allocate((2048 + 16) * 1024 * 2)
            values = try allocate((2048 + 16) * 1024 * 2)
            mask = try allocate(16 * (2048 + 16))
            partials = try allocate(16 * 32 * 9 * (128 + 2) * 4)
        }
    }

    init?(draft: Q35DFlash2Model, target: Q35Model) throws {
        let config = draft.config
        guard config.hiddenSize == 5120, config.intermediateSize == 17_408,
              config.numHiddenLayers == 5, config.numAttentionHeads == 32, config.numKeyValueHeads == 8,
              config.headDim == 128, config.slidingWindow == 2048, config.dflash.targetLayerIDs.count == 5,
              config.dflash.convKernelSize == 2, config.dflash.convGroupSize == 16, config.dflash.selectorTopK == 16,
              let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let words = target.model.embedTokens as? PreQuantizedEmbedding,
              words.groupSize == 64, words.bits == 4, words.mode == .affine,
              words.weight.shape == [config.vocabSize, 640], words.scales.dtype == .bfloat16,
              let biases = words.biases, biases.dtype == .bfloat16, let lmHead = target.lmHead,
              let head = try Q35DFlash2NativeDraftHead.MetalProgram(lmHead, selector: draft.selector, compact: compactHead),
              head.supportsScoring else { return nil }
        self.device = device
        self.queue = queue
        self.head = head
        targetIdentity = ObjectIdentifier(target)
        tapIDs = config.dflash.targetLayerIDs
        maskToken = config.dflash.maskTokenID
        epsilon = config.rmsNormEps
        ropeBase = log2(config.rope.theta)
        kernels = try Q35NativeDraftKernels(device: device)
        norm = try M5NativeRMSNorm(device: device)
        ffn = try M5NativeFFN(device: device)
        ffnWorkspace = try M5NativeFFN.Workspace(device: device)
        scratch = try Scratch(device: device)

        func view(_ array: MLXArray) throws -> M5NativeArrayBuffer { try M5NativeArrayBuffer(array, device: device) }
        var projectionPrograms: [[Int]: M5NativeSmallProjection] = [:]
        func projection(_ weights: M5NativeSmallProjection.Weights?) throws -> Projection {
            guard let weights else { throw Failure.geometry }
            let shape = [weights.inputSize, weights.outputSize]
            let operation = try projectionPrograms[shape]
                ?? M5NativeSmallProjection(device: device, inputSize: weights.inputSize, outputSize: weights.outputSize)
            projectionPrograms[shape] = operation
            return Projection(weights: weights, operation: operation)
        }
        func linear(_ linear: Linear) throws -> Projection {
            try projection(PortableQuantizedLinear.nativeSmallProjectionWeights(linear, device: device))
        }
        func convolution(_ module: DFlash2GroupedDynamicCausalConv) throws -> Convolution {
            let weights = module.convolutionWeights
            return try Convolution(projection: linear(weights.projection), base: view(weights.base))
        }
        contextProjection = try linear(draft.fc)
        contextNorm = try view(draft.hiddenNorm.weight)
        finalNorm = try view(draft.norm.weight)
        embedding = try [words.weight, words.scales, biases].map(view)
        layers = try draft.layers.map { layer in
            guard let gate = try PortableQuantizedLinear.nativeProjectionWeights(layer.mlp.gate, device: device),
                  let up = try PortableQuantizedLinear.nativeProjectionWeights(layer.mlp.up, device: device),
                  let down = try PortableQuantizedLinear.nativeProjectionWeights(layer.mlp.down, device: device)
            else { throw Failure.geometry }
            let attention = layer.attention
            return try Layer(inputNorm: view(layer.inputNorm.weight), attentionConv: convolution(layer.attentionConv),
                qkv: projection(attention.draftProjections.qkv?.nativeSerialWeights(device: device)),
                kv: projection(attention.draftProjections.kv?.nativeSerialWeights(device: device)),
                queryNorm: view(attention.qNorm.weight), keyNorm: view(attention.kNorm.weight), output: linear(attention.oProj),
                postNorm: view(layer.postNorm.weight), mlpConv: convolution(layer.mlpConv),
                ffn: M5NativeFFN.Weights(gate: gate, up: up, down: down))
        }
    }

    func session(_ caches: [Q35DFlash2Cache]) throws -> Session {
        guard caches.count == layers.count else { throw Failure.geometry }
        let offset = caches[0].offset, count = caches[0].keys?.dim(2) ?? 0
        guard count <= Self.window, count <= offset,
              caches.allSatisfy({ $0.offset == offset && ($0.keys?.dim(2) ?? 0) == count
                  && ($0.values?.dim(2) ?? 0) == count })
        else { throw Failure.geometry }
        let rings = try makeRings()
        if count > 0 {
            let sources = try caches.map { cache in
                try [M5NativeArrayBuffer(contiguous(cache.keys!), device: device),
                     M5NativeArrayBuffer(contiguous(cache.values!), device: device)]
            }
            try submit("Qwen draft cache import") { encoder in
                for (source, ring) in zip(sources, rings) {
                    kernels.linear(.seed, encoder, source.map(Binding.init) + [.init(ring.keys), .init(ring.values)],
                                   [count, offset - count], count: count * Self.kvWidth)
                }
            }
        }
        return Session(program: self, rings: rings, offset: offset, count: count)
    }

    func freshSession(offset: Int) throws -> Session {
        try Session(program: self, rings: makeRings(), offset: offset, count: 0)
    }

    private func makeRings() throws -> [Ring] {
        let bytes = Self.window * Self.kvWidth * 2
        return try layers.map { _ in
            guard let keys = device.makeBuffer(length: bytes, options: .storageModeShared),
                  let values = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw Failure.allocation }
            return Ring(keys: keys, values: values)
        }
    }

    private func submit(_ label: String, _ encode: (any MTLComputeCommandEncoder) -> Void) throws {
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else { throw Failure.allocation }
        command.label = label
        encode(encoder)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else { throw Failure.command(command.error) }
    }

    private func flush(session: Session) throws {
        try lock.withLock {
            try submit("Qwen draft context flush") { encodeAbsorption($0, session) }
            advance(session)
        }
    }

    private func advance(_ session: Session) {
        session.offset += session.pendingCount
        session.count = min(Self.window, session.count + session.pendingCount)
        session.pending.removeAll(keepingCapacity: true)
        session.pendingCount = 0
    }

    private func propose(session: Session, anchor: Int, depth: Int, nodes: Int,
                         sampling: Q35PositionSampler? = nil, position: Int = 0) throws -> DFlash2TreeProposal {
        let result = try candidates(session: session, anchor: anchor, depth: depth, referenceScores: false)
        return DFlash2CandidateSelector.buildTree(anchor: anchor, candidates: result.tokens.map(Int.init),
                scores: sampling?.draftScores(result.scores!, tokens: result.tokens, depth: depth,
                    width: M5NativeTopK.count, position: position) ?? result.scores!,
                depthCount: depth, candidateCount: M5NativeTopK.count, nodeBudget: nodes)
    }

    private func candidates(session: Session, anchor: Int, depth: Int, referenceScores: Bool) throws -> Q35DFlash2NativeDraftHead.Output {
        try lock.withLock {
            let rows = depth + 1, offset = session.offset + session.pendingCount
            let prefix = min(Self.window, session.count + session.pendingCount)
            try submit("Qwen native draft block through tree scores") { encoder in
                if session.pendingCount > 0 { encodeAbsorption(encoder, session) }
                kernels.linear(.embeddings, encoder, embedding.map(Binding.init) + [.init(scratch.hidden)],
                               [rows, anchor, maskToken], count: rows * Self.width)
                kernels.linear(.mask, encoder, [.init(scratch.mask)], [rows, prefix + rows, prefix], count: rows * (prefix + rows))
                for (layer, ring) in zip(layers, session.rings) {
                    encodeLayer(encoder, layer, ring, rows: rows, prefix: prefix, offset: offset)
                }
                normalize(encoder, scratch.hidden, finalNorm, rows: depth, inputOffset: Self.width * 2)
                head.encode(into: encoder, hidden: scratch.normalized, sums: ffnWorkspace.inputSums,
                            rows: depth, anchor: anchor, useScorer: true, alreadyPadded: true, referenceScores: referenceScores)
            }
            advance(session)
            guard let result = head.read(rows: depth, useScorer: true), result.scores != nil else { throw Failure.geometry }
            if !reported {
                FileHandle.standardError.write(Data("[q35] dflash2_native_draft_block=true\n".utf8))
                reported = true
            }
            #if DEBUG || MERERUN_RELEASE_REPLAY
            session.observeCandidates?(result.tokens)
            #endif
            return result
        }
    }

    private func normalize(_ e: any MTLComputeCommandEncoder, _ input: any MTLBuffer, _ weights: M5NativeArrayBuffer,
                           rows: Int, inputOffset: Int = 0) {
        norm.encode(into: e, input: input, residual: nil, weight: weights.buffer, combined: scratch.combined,
                    normalized: scratch.normalized, sums: ffnWorkspace.inputSums, rows: rows, epsilon: epsilon,
                    inputOffset: inputOffset, weightOffset: weights.offset)
    }

    private func rotate(_ e: any MTLComputeCommandEncoder, weight: M5NativeArrayBuffer, output: any MTLBuffer,
                        heads: Int, stride: Int, start: Int, rows: Int, offset: Int) {
        kernels.encode(.rotary, e, [.init(scratch.projected), .init(weight), .init(output)],
            [UInt32(stride), UInt32(start), UInt32(heads), UInt32(offset), epsilon.bitPattern, ropeBase.bitPattern],
            groups: MTLSize(width: rows, height: heads, depth: 1), threads: 32)
    }

    private func encodeAbsorption(_ e: any MTLComputeCommandEncoder, _ session: Session) {
        let rows = session.pendingCount
        kernels.linear(.taps, e, session.pending.map { .init($0.0, offset: $0.1) } + [.init(scratch.taps)],
                       [rows], count: rows * Self.width * 5)
        contextProjection.encode(e, scratch.taps, scratch.convolution, rows)
        normalize(e, scratch.convolution, contextNorm, rows: rows)
        for (layer, ring) in zip(layers, session.rings) {
            layer.kv.encode(e, scratch.normalized, scratch.projected, rows)
            rotate(e, weight: layer.keyNorm, output: scratch.blockKeys, heads: 8, stride: 2048,
                   start: 0, rows: rows, offset: session.offset)
            kernels.linear(.append, e, [.init(scratch.blockKeys), .init(scratch.projected), .init(ring.keys), .init(ring.values)],
                           [rows, session.offset], count: rows * Self.kvWidth)
        }
    }

    private func convolve(_ e: any MTLComputeCommandEncoder, _ weights: Convolution, input: any MTLBuffer,
                          output: any MTLBuffer, residual: any MTLBuffer, branch: Int, rows: Int) {
        kernels.linear(.convolution, e, [.init(input), .init(scratch.dynamic), .init(weights.base), .init(residual), .init(output)],
                       [rows, branch], count: rows * Self.width)
    }

    private func encodeLayer(_ e: any MTLComputeCommandEncoder, _ layer: Layer, _ ring: Ring, rows: Int, prefix: Int, offset: Int) {
        let s = scratch
        normalize(e, s.hidden, layer.inputNorm, rows: rows)
        layer.attentionConv.projection.encode(e, s.normalized, s.dynamic, rows)
        convolve(e, layer.attentionConv, input: s.normalized, output: s.convolution, residual: s.hidden, branch: 0, rows: rows)
        layer.qkv.encode(e, s.convolution, s.projected, rows)
        rotate(e, weight: layer.queryNorm, output: s.query, heads: 32, stride: 6144, start: 0, rows: rows, offset: offset)
        rotate(e, weight: layer.keyNorm, output: s.blockKeys, heads: 8, stride: 6144, start: 4096, rows: rows, offset: offset)
        let keys = prefix + rows
        kernels.linear(.gather, e, [.init(ring.keys), .init(ring.values), .init(s.blockKeys),
                                   .init(s.projected), .init(s.keys), .init(s.values)],
                       [keys, prefix, offset - prefix], count: keys * Self.kvWidth)
        let args: [UInt32] = [UInt32(rows), UInt32(keys), UInt32(keys), 0, (1 / sqrt(Float(128))).bitPattern]
        let inputs: [Binding] = [.init(s.query), .init(s.keys), .init(s.values), .init(s.mask)]
        if keys > 1024 {
            kernels.encode(.partial, e, inputs + [.init(s.partials)], args,
                groups: MTLSize(width: 8, height: rows, depth: (keys + 255) / 256), threads: 32)
            kernels.encode(.merge, e, [.init(s.partials), .init(s.query)], args,
                groups: MTLSize(width: 32, height: rows, depth: 1), threads: 32)
        } else {
            kernels.encode(.attention, e, inputs + [.init(s.convolution)], args,
                groups: MTLSize(width: 32, height: rows, depth: 1), threads: 32)
        }
        layer.output.encode(e, keys > 1024 ? s.query : s.convolution, s.projected, rows)
        convolve(e, layer.attentionConv, input: s.projected, output: s.attended, residual: s.hidden, branch: 1, rows: rows)
        normalize(e, s.attended, layer.postNorm, rows: rows)
        layer.mlpConv.projection.encode(e, s.normalized, s.dynamic, rows)
        convolve(e, layer.mlpConv, input: s.normalized, output: s.convolution, residual: s.attended, branch: 0, rows: rows)
        ffn.encode(into: e, input: s.convolution, output: s.projected, weights: layer.ffn, workspace: ffnWorkspace, rows: rows)
        convolve(e, layer.mlpConv, input: s.projected, output: s.hidden, residual: s.attended, branch: 1, rows: rows)
    }
}
#endif
