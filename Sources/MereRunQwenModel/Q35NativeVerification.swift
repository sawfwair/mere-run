#if os(macOS)
import Foundation
import Metal
import MereRunKVCache
import MereRunTensor
import MLX
import MLXNN

package enum Q35NativeVerificationPolicy {
    @TaskLocal package static var timingObserver: (@Sendable ([String: Double]) -> Void)?
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_VERIFY")
    @TaskLocal package static var enabledOverride: Bool?
    package static var enabled: Bool { enabledOverride ?? enabledFlag.enabled }
    private static let directEmbeddingFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_TOKEN_EMBEDDING")
    package static var directEmbedding: Bool { directEmbeddingFlag.enabled }
    private static let wideQKVZFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_VERIFY_WIDE_QKVZ")
    package static var wideQKVZ: Bool { wideQKVZFlag.enabled }
    static let precomputeQKVZBiasSums = ProcessInfo.processInfo.environment["MERERUN_Q38_NATIVE_VERIFY_QKVZ_BIAS_SUMS"] == "1"
        && !Q35NativeExecutionPolicy.enabled
    private static let fuseProjectionSumsFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_FUSE_SUMS")
    static var fuseProjectionSums: Bool { fuseProjectionSumsFlag.enabled && !Q35NativeExecutionPolicy.enabled }
    private static let repairGDNFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_GDN_REPAIR")
    static var repairGDN: Bool { repairGDNFlag.enabled }
    private static let greedyTailFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_VERIFY_GREEDY_TAIL")
    package static var greedyTail: Bool { greedyTailFlag.enabled }
    private static let prepareCommitFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_PREPARE_COMMIT")
    @TaskLocal package static var prepareCommitOverride: Bool?
    package static var prepareCommit: Bool { prepareCommitOverride ?? prepareCommitFlag.enabled }
    private static let deferDraftContextFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_DEFER_DRAFT_CONTEXT")
    static var deferDraftContext: Bool { deferDraftContextFlag.enabled }
    private static let narrowRepairFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_NATIVE_GDN_NARROW_REPAIR")
    static var narrowRepair: Bool { narrowRepairFlag.enabled }
    private static let configuredSubmissionLayers: Int? = {
        let layers = Int(ProcessInfo.processInfo.environment["MERERUN_Q38_NATIVE_VERIFY_SUBMIT_LAYERS"] ?? "")
        return layers.map { [4, 8, 16].contains($0) ? $0 : 0 }
    }()
    @TaskLocal package static var submissionLayersOverride: Int?
    package static var submissionLayers: Int { submissionLayersOverride ?? configuredSubmissionLayers ?? (QwenDFlashDefaults.active ? 4 : 0) }
}

/// Availability-erased interface keeps the existing macOS 15 model usable.
protocol Q35NativeVerificationProgram: AnyObject {
    var roundCount: Int { get }
    var commitBatchCount: Int { get }
    var nativeStateHitLayers: Int { get }
    var greedyRoundCount: Int { get }
    func matches(layers: [Q35DecoderLayer], norm: Q35RMSNorm, embedding: Embedding) -> Bool
    func forward(hidden: MLXArray, tokens: [Int]?, caches: [Q35LayerCache?], tree: Q35VerificationTree,
                 captureLayerIndices: Set<Int>, greedy: Q35GreedyTreeRequest?) throws -> Q35TransformerOutput
}

/// Experimental Qwen 27B target execution. Entry joins MLX once; native layer
/// batches share a Metal queue. Only completed immutable outputs are published
/// to the existing accepted-path cache machinery.
@available(macOS 26.0, *)
final class Q35NativeVerification: Q35NativeVerificationProgram {
    enum Failure: LocalizedError {
        case allocation, cacheFamily, command(Error?), geometry, invalidSelection

        var errorDescription: String? {
            switch self {
            case .allocation: "Metal could not allocate the Qwen buffers."
            case .cacheFamily: "The Qwen cache layout does not match the target layers."
            case .command(let error): error?.localizedDescription ?? "Metal did not complete the Qwen command buffer."
            case .geometry: "The Qwen tensors do not match the native kernel dimensions."
            case .invalidSelection: "The Qwen target head returned an invalid greedy selection."
            }
        }
    }
    private enum LinearInput {
        case matrix(M5NativeFFN.Projection), serial(M5NativeSmallProjection.Weights)
        case splitGate(M5NativeSmallProjection.Weights, M5NativeFFN.Projection)
        case repaired(M5NativeSmallProjection.Weights, M5NativeFFN.Projection)

        func usesSerialInput(rows: Int) -> Bool {
            switch self {
            case .serial: true
            case .repaired: rows != 16
            case .splitGate, .matrix: false
            }
        }
    }
    private struct LinearWeights {
        let input: LinearInput
        let ba: M5NativeSmallProjection.Weights
        let output: M5NativeFFN.Projection
        let conv: M5NativeArrayBuffer
        let aLog: M5NativeArrayBuffer
        let dtBias: M5NativeArrayBuffer
        let norm: M5NativeArrayBuffer
        let aLogArray: MLXArray
        let dtBiasArray: MLXArray
    }
    private struct FullWeights {
        let input: M5NativeFFN.Projection
        let output: M5NativeFFN.Projection
        let qNorm: M5NativeArrayBuffer
        let kNorm: M5NativeArrayBuffer
    }
    private enum Mixer { case linear(LinearWeights), full(FullWeights) }
    private struct Layer {
        let inputNorm: M5NativeArrayBuffer
        let postNorm: M5NativeArrayBuffer
        let ffn: M5NativeFFN.Weights
        let mixer: Mixer
    }
    private enum LinearBase {
        case mlx(MLXArray, MLXArray)
        case native(any Q35NativeLinearState)
    }
    private struct LinearFrame {
        let cache: Q35LinearCache?
        let base: LinearBase
        let qkvz: Q35NativeArena.Slice
        let ba: Q35NativeArena.Slice
        let q: Q35NativeArena.Slice
        let k: Q35NativeArena.Slice
        let v: Q35NativeArena.Slice
        let g: Q35NativeArena.Slice
        let beta: Q35NativeArena.Slice
    }
    private struct FullFrame {
        let cache: KVCacheSimple?
        let prefix: (MLXArray, MLXArray)?
        let keys: Q35NativeArena.Slice
        let values: Q35NativeArena.Slice
    }
    private enum Frame { case linear(LinearFrame), full(FullFrame) }
    private enum PreparedState {
        case linear(any Q35NativeLinearState)
        case full(M5NativeArrayBuffer?, M5NativeArrayBuffer?)
    }
    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let lock = NSLock()
    private(set) var roundCount = 0
    private(set) var nativeStateHitLayers = 0
    private(set) var greedyRoundCount = 0
    private var greedyTail: Q35NativeGreedyTail?
    private var positionHead: Q35NativeSampledHead?
    private var nativeKVCommit: Q35NativeKVCommit?
    var commitBatchCount: Int { stateCommit.batchCount }
    private let stateCommit: Q35NativeStateCommit
    private let identities: [ObjectIdentifier]
    private let embeddingIdentity: ObjectIdentifier
    private let embedding: Q35NativeTargetEmbedding?
    private let embeddingBuffer: any MTLBuffer
    private let layers: [Layer]
    private let finalNorm: M5NativeArrayBuffer
    private let epsilon: Float
    private let norm: M5NativeRMSNorm
    private let smallProjectionNorm: M5NativeRMSNorm?
    private let ffn: M5NativeFFN
    private let qkvz: M5NativeProjection
    private let serialQKVZ: M5NativeSmallProjection
    private let repairProjection: M5NativeGDNRepair?
    private let fullQKV: M5NativeProjection
    private let outputProjection: M5NativeProjection
    private let splitGateLayerCount: Int
    private let splitGate: (projection: M5NativeProjection, output: Q35NativeGDNOutput)?
    private let ba: M5NativeSmallProjection
    private let coefficients: Q35NativeGDNCoefficients
    private let prework: Q35NativeGDNPrework
    private let delta: Q35NativeGDNDelta
    private let gdnOutput: Q35NativeGDNOutput
    private let fullPrework: Q35NativeFullPrework
    private let attention: Q35NativeFullAttention
    private let fullOutput: Q35NativeFullOutput
    private let ffnWorkspace: M5NativeFFN.Workspace
    private let hiddenBuffer: any MTLBuffer
    private let normalized: any MTLBuffer
    private let pending: any MTLBuffer
    private let projected: any MTLBuffer
    private let queries: any MTLBuffer
    private let attentionOutput: any MTLBuffer
    private let gated: any MTLBuffer
    private let gatedSums: any MTLBuffer
    private let convolutionScratch: any MTLBuffer

    init?(layers source: [Q35DecoderLayer], norm final: Q35RMSNorm, embedding: Embedding,
          epsilon: Float, device: any MTLDevice) throws {
        var prepared: [Layer] = []
        var gateLayers = 0
        var needsRepair = false
        for layer in source {
            guard !layer.isMLPOnly, layer.ple == nil, layer.attentionHyperConnection == nil,
                  layer.mlpHyperConnection == nil,
                  let gate = layer.mlp.gateProj, let up = layer.mlp.upProj, let down = layer.mlp.downProj,
                  let gateWeight = try PortableQuantizedLinear.nativeProjectionWeights(gate, device: device),
                  let upWeight = try PortableQuantizedLinear.nativeProjectionWeights(up, device: device),
                  let downWeight = try PortableQuantizedLinear.nativeProjectionWeights(down, device: device) else { return nil }
            let mixer: Mixer
            switch layer.layerType {
            case .linear:
                let linear = layer.linearAttention, fused = linear.resolvedFusedInputs()
                guard let input = fused.qkvz, let ba = fused.ba,
                      let portable = input as? PortableQuantizedLinear,
                      let baWeight = try PortableQuantizedLinear.nativeSmallProjectionWeights(ba, device: device),
                      let output = try PortableQuantizedLinear.nativeProjectionWeights(linear.outProj, device: device),
                      baWeight.inputSize == 5120, baWeight.outputSize == 96,
                      output.inputSize == 6144, output.outputSize == 5120 else { return nil }
                let inputWeight: LinearInput
                if let matrix = try PortableQuantizedLinear.nativeProjectionWeights(input, device: device),
                   matrix.inputSize == 5120, matrix.outputSize == 16384 {
                    inputWeight = .matrix(matrix)
                } else if portable.effectiveVerificationBackend == .reference,
                          let serial = try PortableQuantizedLinear.nativeSmallProjectionWeights(input, device: device),
                          serial.inputSize == 5120, serial.outputSize == 16384 {
                    if let gate = linear.matrixOutputGate,
                       let packed = try PortableQuantizedLinear.nativeProjectionWeights(gate, device: device) {
                        inputWeight = .splitGate(serial, packed)
                        gateLayers += 1
                    } else if Q35NativeVerificationPolicy.repairGDN,
                              let packed = try PortableQuantizedLinear.nativeProjectionWeights(input, device: device, backend: .reference) {
                        inputWeight = .repaired(serial, packed)
                        needsRepair = true
                    } else {
                        inputWeight = .serial(serial)
                    }
                } else { return nil }
                mixer = try .linear(LinearWeights(input: inputWeight, ba: baWeight, output: output,
                    conv: M5NativeArrayBuffer(linear.conv1d.weight, device: device),
                    aLog: M5NativeArrayBuffer(linear.aLog.asType(.float32), device: device),
                    dtBias: M5NativeArrayBuffer(linear.dtBias.asType(.float32), device: device),
                    norm: M5NativeArrayBuffer(linear.norm.weight, device: device),
                    aLogArray: linear.aLog, dtBiasArray: linear.dtBias))
            case .full:
                let full = layer.selfAttention
                guard let input = try full.resolvedFusedQKV()?.nativeWeights(device: device),
                      let output = try PortableQuantizedLinear.nativeProjectionWeights(full.oProj, device: device),
                      input.inputSize == 5120, input.outputSize == 14336,
                      output.inputSize == 6144, output.outputSize == 5120 else { return nil }
                mixer = try .full(FullWeights(input: input, output: output,
                    qNorm: M5NativeArrayBuffer(full.qNorm.resolvedWeight(), device: device),
                    kNorm: M5NativeArrayBuffer(full.kNorm.resolvedWeight(), device: device)))
            }
            prepared.append(try Layer(inputNorm: M5NativeArrayBuffer(layer.inputLayerNorm.resolvedWeight(), device: device),
                postNorm: M5NativeArrayBuffer(layer.postAttentionLayerNorm.resolvedWeight(), device: device),
                ffn: M5NativeFFN.Weights(gate: gateWeight, up: upWeight, down: downWeight), mixer: mixer))
        }
        guard prepared.allSatisfy({
            $0.inputNorm.shape == [5120] && $0.postNorm.shape == [5120]
                && $0.inputNorm.dtype == .bfloat16 && $0.postNorm.dtype == .bfloat16
        }) else { return nil }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw Failure.allocation }
        self.queue = queue
        self.epsilon = epsilon
        layers = prepared
        identities = Self.sourceIdentities(layers: source, norm: final)
        embeddingIdentity = ObjectIdentifier(embedding)
        self.embedding = Q35NativeVerificationPolicy.directEmbedding || Q35NativeExecutionPolicy.enabled
            ? try Q35NativeTargetEmbedding(embedding: embedding, device: device) : nil
        finalNorm = try M5NativeArrayBuffer(final.resolvedWeight(), device: device)
        guard finalNorm.shape == [5120] && finalNorm.dtype == .bfloat16 else { throw Failure.geometry }
        norm = try M5NativeRMSNorm(device: device)
        let fuseSums = Q35NativeVerificationPolicy.fuseProjectionSums
        smallProjectionNorm = fuseSums ? try M5NativeRMSNorm(device: device, smallProjectionSums: true) : nil
        ffn = try M5NativeFFN(device: device, fuseActivationSums: fuseSums)
        qkvz = try M5NativeProjection(device: device, inputSize: 5120, outputSize: 16384,
                                       arithmetic: Q35GDNProjectionPolicy.matrixArithmetic)
        serialQKVZ = try M5NativeSmallProjection(device: device, inputSize: 5120, outputSize: 16384,
            simdgroups: Q35NativeVerificationPolicy.wideQKVZ ? 4 : 2,
            precomputeBiasSums: fuseSums || Q35NativeVerificationPolicy.precomputeQKVZBiasSums)
        let narrowRepair = Q35NativeVerificationPolicy.narrowRepair
        repairProjection = needsRepair ? try M5NativeGDNRepair(device: device, narrowMargin: narrowRepair) : nil
        splitGateLayerCount = gateLayers
        splitGate = gateLayers > 0 ? try (
            M5NativeProjection(device: device, inputSize: 5120, outputSize: 6144, partitionCount: 4),
            Q35NativeGDNOutput(device: device, separateGate: true)
        ) : nil
        fullQKV = try M5NativeProjection(device: device, inputSize: 5120, outputSize: 14336)
        outputProjection = try M5NativeProjection(device: device, inputSize: 6144, outputSize: 5120)
        ba = try M5NativeSmallProjection(device: device, inputSize: 5120, outputSize: 96)
        coefficients = try Q35NativeGDNCoefficients(device: device)
        prework = try Q35NativeGDNPrework(device: device)
        delta = Q35NativeGDNDelta(device: device)
        gdnOutput = try Q35NativeGDNOutput(device: device)
        fullPrework = try Q35NativeFullPrework(device: device)
        attention = Q35NativeFullAttention(device: device)
        fullOutput = try Q35NativeFullOutput(device: device)
        let rows = Q35NativeExecutionPolicy.enabled ? Q35NativeExecutionPolicy.promptRows : 16
        ffnWorkspace = try M5NativeFFN.Workspace(device: device, untracked: true, rows: rows)
        func scratch(_ count: Int, bytes: Int = 2) throws -> any MTLBuffer {
            guard let value = device.makeBuffer(length: count * bytes, options: [.storageModePrivate, .hazardTrackingModeUntracked]) else { throw Failure.allocation }
            return value
        }
        embeddingBuffer = try scratch(rows * 5120)
        hiddenBuffer = try scratch(rows * 5120); normalized = try scratch(rows * 5120); pending = try scratch(rows * 5120)
        projected = try scratch(rows * 14336); queries = try scratch(rows * 6144)
        attentionOutput = try scratch(rows * 6144); gated = try scratch(rows * 6144)
        gatedSums = try scratch(96 * rows, bytes: 4); convolutionScratch = try scratch(3 * 10240)
        stateCommit = try Q35NativeStateCommit(device: device, queue: queue)
    }

    private static func sourceIdentities(layers: [Q35DecoderLayer], norm: Q35RMSNorm) -> [ObjectIdentifier] {
        layers.flatMap { layer -> [ObjectIdentifier] in
            var modules: [Module] = [layer, layer.inputLayerNorm, layer.postAttentionLayerNorm]
            modules += [layer.mlp.gateProj, layer.mlp.upProj, layer.mlp.downProj].compactMap { $0 }
            switch layer.layerType {
            case .linear:
                let a = layer.linearAttention
                modules += [a.inProjQKV, a.inProjZ, a.inProjA, a.inProjB, a.outProj, a.conv1d, a.norm]
            case .full:
                let a = layer.selfAttention
                modules += [a.qProj, a.kProj, a.vProj, a.oProj, a.qNorm, a.kNorm]
            }
            return modules.map(ObjectIdentifier.init)
        } + [ObjectIdentifier(norm)]
    }

    func matches(layers: [Q35DecoderLayer], norm: Q35RMSNorm, embedding: Embedding) -> Bool {
        embeddingIdentity == ObjectIdentifier(embedding) && identities == Self.sourceIdentities(layers: layers, norm: norm)
    }

    private func openSubmission() throws -> (any MTLCommandBuffer, any MTLComputeCommandEncoder) {
        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
            throw Failure.allocation
        }
        command.label = "Qwen native tree verification"
        return (command, encoder)
    }

    private struct Body {
        let input: M5NativeArrayBuffer
        let frames: [Frame]
        let states: [PreparedState]
        let captures: [Int: Q35NativeArena.Slice]
        let plan: Q35NativeTreePlan
        let recurrent: Q35NativeGDNDelta.Prepared
        let attention: Q35NativeFullAttention.Prepared
        let prefix: Int
        var advancing: [Int: Q35NativeStateCommit.Destination] = [:]
    }

    private func encodeBody(
        _ body: Body, command: inout any MTLCommandBuffer, encoder: inout any MTLComputeCommandEncoder,
        hazards: inout M5NativeHazards, submitted: inout [any MTLCommandBuffer], submissionLayers: Int
    ) throws {
        let plan = body.plan, input = body.input
        let captures = body.captures, frames = body.frames, states = body.states
        let recurrentPlan = body.recurrent, attentionPlan = body.attention
        let fence = submissionLayers > 0 ? device.makeFence() : nil
        if submissionLayers > 0, fence == nil { throw Failure.allocation }
        for (index, layer) in layers.enumerated() {
            let combined = captures[index - 1]?.buffer ?? hiddenBuffer
            let combinedOffset = captures[index - 1]?.offset ?? 0
            let inputNorm: M5NativeRMSNorm
            let inputSums: any MTLBuffer
            if case .linear(let mixer) = layer.mixer, mixer.input.usesSerialInput(rows: plan.count),
               let smallProjectionNorm, let scratch = serialQKVZ.biasSumsBuffer {
                inputNorm = smallProjectionNorm
                inputSums = scratch
            } else {
                inputNorm = norm
                inputSums = ffnWorkspace.inputSums
            }
            inputNorm.encode(into: encoder, input: index == 0 ? input.buffer : hiddenBuffer, residual: index == 0 ? nil : pending,
                weight: layer.inputNorm.buffer, combined: combined, normalized: normalized, sums: inputSums,
                rows: plan.count, epsilon: epsilon, inputOffset: index == 0 ? input.offset : 0, weightOffset: layer.inputNorm.offset, combinedOffset: combinedOffset, hazards: hazards)
            switch (layer.mixer, frames[index], states[index]) {
            case let (.linear(weights), .linear(frame), .linear(base)):
                var gateProgram = gdnOutput
                var gateBuffer = frame.qkvz.buffer
                var gateOffset = frame.qkvz.offset
                switch weights.input {
                case let .repaired(serial, packed):
                    if plan.count == 16 {
                        guard let repairProjection else { throw Failure.geometry }
                        repairProjection.encode(into: encoder, input: normalized, sums: inputSums,
                            output: frame.qkvz.buffer, offset: frame.qkvz.offset, serial: serial, packed: packed, hazards: hazards)
                    } else {
                        serialQKVZ.encode(into: encoder, input: normalized, output: frame.qkvz.buffer, weights: serial,
                            rows: plan.count, outputOffset: frame.qkvz.offset, sumsPrepared: smallProjectionNorm != nil, hazards: hazards)
                    }
                case let .splitGate(state, gate):
                    guard let splitGate else { throw Failure.geometry }
                    serialQKVZ.encode(into: encoder, input: normalized, output: frame.qkvz.buffer, weights: state, rows: plan.count,
                        outputOffset: frame.qkvz.offset, outputColumns: 10240, hazards: hazards)
                    splitGate.projection.encode(into: encoder, input: normalized, sums: inputSums,
                        output: projected, weights: gate, rows: plan.count, hazards: hazards)
                    gateProgram = splitGate.output
                    gateBuffer = projected
                    gateOffset = 0
                case .matrix(let weights):
                    qkvz.encode(into: encoder, input: normalized, sums: ffnWorkspace.inputSums,
                        output: frame.qkvz.buffer, weights: weights, rows: plan.count, outputOffset: frame.qkvz.offset, hazards: hazards)
                case .serial(let weights):
                    serialQKVZ.encode(into: encoder, input: normalized, output: frame.qkvz.buffer, weights: weights, rows: plan.count,
                        outputOffset: frame.qkvz.offset, sumsPrepared: smallProjectionNorm != nil, hazards: hazards)
                }
                ba.encode(into: encoder, input: normalized, output: frame.ba.buffer, weights: weights.ba, rows: plan.count, outputOffset: frame.ba.offset, hazards: hazards)
                let destination = body.advancing[index]
                prework.encode(into: encoder, qkv: frame.qkvz.buffer, qkvStride: 16384, convolution: base.convolutionBuffer, weight: weights.conv,
                    ancestry: plan.ancestry, output: .init(q: frame.q.buffer, k: frame.k.buffer, v: frame.v.buffer,
                        convolution: destination?.convolution.buffer ?? convolutionScratch,
                        qOffset: frame.q.offset, kOffset: frame.k.offset, vOffset: frame.v.offset,
                        convolutionOffset: destination?.convolution.offset ?? 0), rows: plan.count,
                    qkvOffset: frame.qkvz.offset, advance: destination != nil, hazards: hazards)
                coefficients.encode(into: encoder, ba: frame.ba.buffer, aLog: weights.aLog, dtBias: weights.dtBias,
                    g: frame.g.buffer, beta: frame.beta.buffer, rows: plan.count,
                    baOffset: frame.ba.offset, gOffset: frame.g.offset, betaOffset: frame.beta.offset, hazards: hazards)
                delta.encode(into: encoder, q: frame.q.buffer, k: frame.k.buffer, v: frame.v.buffer, g: frame.g.buffer, beta: frame.beta.buffer,
                    state: base.recurrentBuffer, output: attentionOutput, prepared: recurrentPlan,
                    qOffset: frame.q.offset, kOffset: frame.k.offset, vOffset: frame.v.offset, gOffset: frame.g.offset, betaOffset: frame.beta.offset,
                    destination: destination?.recurrent, hazards: hazards)
                gateProgram.encode(into: encoder, updated: attentionOutput, qkvz: gateBuffer, weight: weights.norm,
                    output: gated, sums: gatedSums, rows: plan.count, epsilon: epsilon, qkvzOffset: gateOffset, hazards: hazards)
                outputProjection.encode(into: encoder, input: gated, sums: gatedSums, output: pending, weights: weights.output, rows: plan.count, hazards: hazards)
            case let (.full(weights), .full(frame), .full(prefixKeys, prefixValues)):
                fullQKV.encode(into: encoder, input: normalized, sums: ffnWorkspace.inputSums, output: projected, weights: weights.input, rows: plan.count, hazards: hazards)
                fullPrework.encode(into: encoder, qkv: projected, queryWeight: weights.qNorm, keyWeight: weights.kNorm,
                    queries: queries, keys: frame.keys.buffer, values: frame.values.buffer, plan: plan, prefix: body.prefix, epsilon: epsilon,
                    keysOffset: frame.keys.offset, valuesOffset: frame.values.offset, hazards: hazards)
                attention.encode(into: encoder, queries: queries, prefixKeys: prefixKeys, prefixValues: prefixValues,
                    branchKeys: frame.keys.buffer, branchValues: frame.values.buffer, output: attentionOutput, prepared: attentionPlan,
                    branchKeysOffset: frame.keys.offset, branchValuesOffset: frame.values.offset, hazards: hazards)
                fullOutput.encode(into: encoder, attention: attentionOutput, qkv: projected, output: gated, sums: gatedSums, rows: plan.count, hazards: hazards)
                outputProjection.encode(into: encoder, input: gated, sums: gatedSums, output: pending, weights: weights.output, rows: plan.count, hazards: hazards)
            default: preconditionFailure("Prepared native layer families must match")
            }
            norm.encode(into: encoder, input: combined, residual: pending, weight: layer.postNorm.buffer,
                combined: hiddenBuffer, normalized: normalized, sums: ffnWorkspace.inputSums,
                rows: plan.count, epsilon: epsilon, inputOffset: combinedOffset, weightOffset: layer.postNorm.offset, hazards: hazards)
            ffn.encode(into: encoder, input: normalized, output: pending, weights: layer.ffn,
                workspace: ffnWorkspace, rows: plan.count, inputSums: .provided, hazards: hazards)
            if let fence, index + 1 < layers.count, (index + 1).isMultiple(of: submissionLayers) {
                // The next encoder reuses untracked scratch and consumes this
                // batch's residual. The fence orders all those accesses.
                encoder.updateFence(fence)
                encoder.endEncoding()
                command.commit()
                submitted.append(command)
                (command, encoder) = try openSubmission()
                encoder.waitForFence(fence)
                hazards = M5NativeHazards()
            }
        }
    }

    func forward(hidden: MLXArray, tokens: [Int]?, caches: [Q35LayerCache?], tree: Q35VerificationTree,
                 captureLayerIndices: Set<Int>, greedy: Q35GreedyTreeRequest?) throws -> Q35TransformerOutput {
        lock.lock()
        defer { lock.unlock() }
        let observer = Q35NativeVerificationPolicy.timingObserver
        var timings: [String: Double] = [:]
        var last = observer == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        func mark(_ name: String) {
            guard observer != nil else { return }
            let now = DispatchTime.now().uptimeNanoseconds
            timings[name] = Double(now - last) / 1e9
            last = now
        }
        precondition(hidden.shape == [1, tree.count, 5120] && hidden.dtype == .bfloat16 && caches.count == layers.count)
        let plan = try Q35NativeTreePlan(tree: tree, device: device), recurrentPlan = try delta.prepare(plan)
        let linearCount = layers.filter { if case .linear = $0.mixer { return true }; return false }.count
        let replayArena = try Q35NativeArena(device: device,
            bytes: linearCount * 855040 + (layers.count - linearCount) * 65536, dtype: .bfloat16)
        let gateArena = try Q35NativeArena(device: device, bytes: linearCount * 6144, dtype: .float32)
        let captureIndices = captureLayerIndices.filter { layers.indices.contains($0) }
        let tail: Q35NativeGreedyTail?
        if let greedy, greedy.eos.count <= 128 {
            if greedyTail?.headIdentity != ObjectIdentifier(greedy.head) {
                greedyTail = try Q35NativeGreedyTail(layer: greedy.head, device: device)
            }
            tail = greedyTail
        } else {
            tail = nil
        }
        let commitMetadata = tail != nil && Q35NativeVerificationPolicy.prepareCommit
            ? Q35NativeGreedyTail.makeMetadata(device: device) : nil
        let metadata = commitMetadata ?? tail?.metadata
        let tapCount = captureIndices.count * (tail == nil ? 1 : 2)
        let attentionBytes = tail == nil ? 0 : (layers.count - linearCount) * 65536
        let outputArena = try Q35NativeArena(device: device,
            bytes: (tapCount + 1) * 163840 + attentionBytes, dtype: .bfloat16)
        func owned(_ count: Int, dtype: DType = .bfloat16) -> Q35NativeArena.Slice {
            precondition(dtype == .bfloat16 || dtype == .float32)
            return (dtype == .float32 ? gateArena : replayArena).allocate(count)
        }
        let nativeTokens = tokens.flatMap { ids in
            Q35NativeVerificationPolicy.directEmbedding && ids.count == tree.count
                && embedding?.supports(ids) == true ? ids : nil
        }
        var frames: [Frame] = [], inputs = nativeTokens == nil ? [hidden] : [], prefix: Int?
        for (layer, entry) in zip(layers, caches) {
            switch (layer.mixer, entry) {
            case (.linear, .linear(let cache)?):
                let base: LinearBase
                if let state = cache.nativeState {
                    base = .native(state)
                } else {
                    let conv = cache.convState ?? MLXArray.zeros([1, 3, 10240], dtype: .bfloat16)
                    let state = cache.recurrentState ?? MLXArray.zeros([1, 48, 128, 128], dtype: .float32)
                    inputs += [conv, state]
                    base = .mlx(conv, state)
                }
                frames.append(.linear(LinearFrame(cache: cache, base: base,
                    qkvz: owned(16 * 16384), ba: owned(16 * 96), q: owned(16 * 16 * 128), k: owned(16 * 16 * 128),
                    v: owned(16 * 48 * 128), g: owned(16 * 48, dtype: .float32), beta: owned(16 * 48, dtype: .float32))))
            case (.full, .full(let cache as KVCacheSimple)?):
                guard type(of: cache) == KVCacheSimple.self, prefix == nil || prefix == cache.offset else { throw Failure.cacheFamily }
                prefix = cache.offset
                let state = cache.committedVerificationState
                if let state { inputs += [state.0, state.1] }
                frames.append(.full(FullFrame(cache: cache, prefix: state, keys: owned(4 * 16 * 256), values: owned(4 * 16 * 256))))
            default: throw Failure.cacheFamily
            }
        }
        var attentionScratch: Q35NativeFullAttention.Scratch?
        let attentionPlan = try attention.prepare(plan: plan, prefix: prefix ?? 0, scratch: &attentionScratch)
        let finalOutput = outputArena.allocate(16 * 5120)
        var captures: [Int: Q35NativeArena.Slice] = [:]
        for index in captureIndices.sorted() { captures[index] = outputArena.allocate(16 * 5120) }
        let acceptedTaps = tail == nil ? [:] : captures.mapValues { _ in outputArena.allocate(16 * 5120) }
        var attentionInputs: [Q35NativeArena.Slice] = []
        var attentionOutputs: [Q35NativeArena.Slice] = []
        var attentionLayers: [Int] = []
        if tail != nil {
            for (index, frame) in frames.enumerated() {
                if case .full(let entry) = frame {
                    attentionInputs += [entry.keys, entry.values]
                    attentionOutputs += [outputArena.allocate(4 * 16 * 256), outputArena.allocate(4 * 16 * 256)]
                    attentionLayers.append(index)
                }
            }
        }
        // Join all pending MLX state/embedding work before opening the encoder.
        mark("prepare")
        if !inputs.isEmpty { MLX.eval(inputs) }
        mark("joinMLX")
        let input = nativeTokens == nil ? try M5NativeArrayBuffer(hidden, device: device)
            : M5NativeArrayBuffer(buffer: embeddingBuffer, offset: 0, shape: hidden.shape, dtype: .bfloat16)
        let states: [PreparedState] = try frames.map { frame in
            switch frame {
            case .linear(let f):
                switch f.base {
                case .mlx(let conv, let recurrent):
                    return try .linear(Q35NativeStoredLinearState(convolution: conv, recurrent: recurrent, device: device))
                case .native(let state):
                    try state.waitUntilReady()
                    nativeStateHitLayers += 1
                    return .linear(state)
                }
            case .full(let f):
                guard let p = f.prefix else { return .full(nil, nil) }
                return try .full(M5NativeArrayBuffer(p.0, device: device), M5NativeArrayBuffer(p.1, device: device))
            }
        }
        mark("bridgeInputs")
        var (command, encoder) = try openSubmission()
        var hazards = M5NativeHazards()
        if let embedding, let nativeTokens {
            embedding.encode(nativeTokens, into: embeddingBuffer, encoder: encoder, hazards: hazards)
        }
        var submitted: [any MTLCommandBuffer] = []
        defer { submitted.last?.waitUntilCompleted() }
        let body = Body(input: input, frames: frames, states: states, captures: captures, plan: plan,
                        recurrent: recurrentPlan, attention: attentionPlan, prefix: prefix ?? 0)
        try encodeBody(body, command: &command, encoder: &encoder, hazards: &hazards, submitted: &submitted,
                       submissionLayers: Q35NativeVerificationPolicy.submissionLayers)
        norm.encode(into: encoder, input: hiddenBuffer, residual: pending, weight: finalNorm.buffer,
            combined: captures[layers.count - 1]?.buffer ?? hiddenBuffer, normalized: finalOutput.buffer,
            sums: ffnWorkspace.inputSums, rows: tree.count, epsilon: epsilon, weightOffset: finalNorm.offset,
            combinedOffset: captures[layers.count - 1]?.offset ?? 0, normalizedOffset: finalOutput.offset, hazards: hazards)
        if let tail, let greedy, let metadata {
            tail.encode(into: encoder, normalized: finalOutput, sums: ffnWorkspace.inputSums,
                request: greedy, tree: tree, metadata: metadata, hazards: hazards)
            for (index, output) in acceptedTaps {
                tail.gatherTap(captures[index]!, into: output, metadata: metadata, encoder: encoder, hazards: hazards)
            }
            if !attentionInputs.isEmpty {
                tail.gatherAttention(attentionInputs, into: attentionOutputs, metadata: metadata, encoder: encoder, hazards: hazards)
            }
        }
        encoder.endEncoding()
        mark("encode")
        command.commit()
        submitted.append(command)
        var replayInputs: [Q35NativeReplayInput] = []
        for index in layers.indices {
            if case .linear(let weights) = layers[index].mixer,
               case .linear(let frame) = frames[index], case .linear(let initial) = states[index] {
                replayInputs.append(Q35NativeReplayInput(initial: initial, qkvz: frame.qkvz, ba: frame.ba,
                    q: frame.q, k: frame.k, v: frame.v, g: frame.g, beta: frame.beta,
                    aLog: weights.aLogArray, dtBias: weights.dtBiasArray))
            }
        }
        // Encode against this round's immutable selection buffer while the GPU
        // verifies it. Submit only once the caller accepts the verified path.
        let prepared = commitMetadata.flatMap { try? stateCommit.prepare(inputs: replayInputs, selection: $0) }
        if commitMetadata != nil { mark("prepareCommit") }
        command.waitUntilCompleted()
        mark("submitAndWait")
        if observer != nil {
            timings["gpuSpan"] = command.gpuEndTime - submitted[0].gpuStartTime
            timings["commandBufferCount"] = Double(submitted.count)
            timings["nativeTokenEmbedding"] = nativeTokens == nil ? 0 : 1
            timings["nativeQKVZBiasSums"] = serialQKVZ.biasSumsBuffer == nil ? 0 : 1
            timings["nativeGDNRepair"] = repairProjection != nil && tree.count == 16 ? 1 : 0
            timings["nativeSplitGateLayers"] = Double(splitGateLayerCount)
            timings["nativeFusedProjectionSums"] = Q35NativeVerificationPolicy.fuseProjectionSums ? 1 : 0
        }
        for batch in submitted where batch.status != .completed { throw Failure.command(batch.error) }
        submitted.removeAll(keepingCapacity: false)
        roundCount += 1
        let selection = metadata.flatMap { tail?.selection(from: $0, rows: tree.count) }
        let commit = prepared.flatMap { batch in selection.map { (path: $0.path, batch: batch) } }
        let batch = Q35NativeReplayBatch(tree: tree, inputs: replayInputs, program: stateCommit, prepared: commit)
        if selection != nil { greedyRoundCount += 1 }
        let gatheredAttention = selection.map { $0.path.last! != $0.path.count - 1 } ?? false
        var linearIndex = 0
        for frame in frames {
            switch frame {
            case .linear(let f):
                let cache = f.cache!
                cache.verificationReplay = nil
                cache.verificationGates = nil
                cache.verificationTree = tree
                cache.nativeVerificationReplay = Q35NativeReplayVertex(batch: batch, index: linearIndex)
                linearIndex += 1
            case .full(let f):
                if !gatheredAttention {
                    _ = f.cache!.update(keys: f.keys.array([1, 4, 16, 256])[0..., 0..., 0..<tree.count, 0...],
                        values: f.values.array([1, 4, 16, 256])[0..., 0..., 0..<tree.count, 0...])
                }
            }
        }
        let publishedTaps = selection.map { selected in
            acceptedTaps.mapValues { $0.array([1, selected.path.count, 5120]) }
        } ?? captures.mapValues { $0.array([1, tree.count, 5120]) }
        var acceptedAttention: [Int: Q35GreedyTreeKV] = [:]
        if let path = selection?.path, gatheredAttention {
            let shape = [1, 4, path.count, 256]
            for (index, layer) in attentionLayers.enumerated() {
                acceptedAttention[layer] = Q35GreedyTreeKV(keys: attentionOutputs[index * 2].array(shape),
                    values: attentionOutputs[index * 2 + 1].array(shape))
            }
        }
        let nativeDraftTaps = selection == nil ? [:] : acceptedTaps.mapValues { Q35NativeDraftTap(buffer: $0.buffer, offset: $0.offset) }
        let output = Q35TransformerOutput(hidden: finalOutput.array([1, tree.count, 5120]), mtpHidden: nil,
            capturedHiddenStates: publishedTaps, greedySelection: selection, acceptedAttention: acceptedAttention,
            nativeDraftTaps: nativeDraftTaps)
        mark("publishMLX")
        observer?(timings)
        return output
    }
}


@available(macOS 26.0, *)
extension Q35NativeVerification {
    private func nativeTailAndCommit(head: Linear) throws -> (Q35NativeGreedyTail, Q35NativeKVCommit) {
        try lock.withLock {
            if greedyTail?.headIdentity != ObjectIdentifier(head) {
                greedyTail = try Q35NativeGreedyTail(layer: head, device: device)
            }
            guard let tail = greedyTail else { throw Failure.geometry }
            if let nativeKVCommit { return (tail, nativeKVCommit) }
            let commit = try Q35NativeKVCommit(device: device)
            nativeKVCommit = commit
            return (tail, commit)
        }
    }

    struct NativeRound {
        let count: Int
        let following: Int?
        let path: [Int]
        let taps: [Int: Q35NativeDraftTap]
        var loss: Double? = nil
    }

    /// A generation request owns the target’s KV bank, recurrent buffers, and
    /// replay scratch. The model's queue and operators are shared across requests.
    final class Session {
        private let program: Q35NativeVerification
        private let head: Q35NativeGreedyTail
        private let headWeights: Linear
        private let kvCommit: Q35NativeKVCommit
        private let frames: [Frame]
        private let kv: [Int: Q35NativeKVBank]
        private let banks: [[Q35NativeStateCommit.Destination]]
        private let captures: [Int: Q35NativeArena.Slice]
        private let gathered: [Int: Q35NativeArena.Slice]
        private let finalOutput: Q35NativeArena.Slice
        private let selection: any MTLBuffer
        private var linear: [any Q35NativeLinearState]
        private var bank = 0
        private var chainPlans: [Int: Q35NativeTreePlan] = [:]
        private struct SampledVerification {
            let logits: any MTLBuffer
            let sampler: Q35PositionSampler?
            let position: Int
            var rejection: Q35RejectionDraws? = nil
            let reference: (any MTLBuffer, Int) -> (path: [Int], following: Int)
        }
        private var sampledHead: Q35NativeSampledHead?
        private var sampledLogits: (any MTLBuffer)?
        private var attentionScratch: Q35NativeFullAttention.Scratch?
        private(set) var offset = 0

        init(program: Q35NativeVerification, head: Linear, capacity: Int, tapIDs: Set<Int>, importingPrompt: Bool) throws {
            guard program.embedding != nil, capacity > 0,
                  tapIDs.allSatisfy(program.layers.indices.contains) else { throw Failure.geometry }
            let device = program.device
            self.program = program
            self.headWeights = head
            (self.head, kvCommit) = try program.nativeTailAndCommit(head: head)
            let zero = try Q35NativeStoredLinearState(zeroedOn: device)
            let rows = importingPrompt ? 16 : Q35NativeExecutionPolicy.promptRows
            guard let selection = device.makeBuffer(length: max(34, rows + 2) * 4, options: .storageModeShared)
            else { throw Failure.allocation }
            self.selection = selection
            let count = program.layers.count(where: { if case .linear = $0.mixer { return true }; return false })
            let replay = try Q35NativeArena(device: device,
                bytes: rows / 16 * (count * 855_040 + (program.layers.count - count) * 65_536), dtype: .bfloat16)
            let gates = try Q35NativeArena(device: device, bytes: rows / 16 * count * 6_144, dtype: .float32)
            var storage: [Int: Q35NativeKVBank] = [:]
            var frames: [Frame] = []
            for (index, layer) in program.layers.enumerated() {
                switch layer.mixer {
                case .linear:
                    frames.append(.linear(LinearFrame(cache: nil, base: .native(zero),
                        qkvz: replay.allocate(rows * 16_384), ba: replay.allocate(rows * 96),
                        q: replay.allocate(rows * 16 * 128), k: replay.allocate(rows * 16 * 128),
                        v: replay.allocate(rows * 48 * 128), g: gates.allocate(rows * 48), beta: gates.allocate(rows * 48))))
                case .full:
                    storage[index] = try Q35NativeKVBank(device: device, capacity: capacity, exportable: importingPrompt)
                    frames.append(.full(FullFrame(cache: nil, prefix: nil,
                        keys: replay.allocate(4 * rows * 256), values: replay.allocate(4 * rows * 256))))
                }
            }
            kv = storage
            self.frames = frames
            linear = Array(repeating: zero, count: count)
            banks = try (0..<2).map { _ in try Q35NativeStateCommit.Destination.bank(device: device, layers: count) }
            let output = try Q35NativeArena(device: device, bytes: ((tapIDs.count + 1) * rows + tapIDs.count * 16) * 5_120 * 2, dtype: .bfloat16)
            captures = Dictionary(uniqueKeysWithValues: tapIDs.sorted().map { ($0, output.allocate(rows * 5_120)) })
            gathered = Dictionary(uniqueKeysWithValues: tapIDs.sorted().map { ($0, output.allocate(16 * 5_120)) })
            finalOutput = output.allocate(rows * 5_120)
        }

        func importPrompt(from caches: [Q35LayerCache?], count: Int) throws {
            guard let command = program.queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder() else {
                throw Failure.allocation
            }
            command.label = "Qwen import committed prompt"
            var states: [any Q35NativeLinearState] = []
            do {
                for (index, entry) in caches.enumerated() {
                    switch entry {
                    case .linear(let cache):
                        if let state = cache.nativeState {
                            try state.waitUntilReady()
                            states.append(state)
                        } else if let conv = cache.convState, let recurrent = cache.recurrentState {
                            states.append(try Q35NativeStoredLinearState(convolution: conv, recurrent: recurrent, device: program.device))
                        } else { throw Failure.cacheFamily }
                    case .full(let cache as KVCacheSimple):
                        guard let bank = kv[index], cache.offset == count, let prefix = cache.committedVerificationState
                        else { throw Failure.cacheFamily }
                        try bank.importPrefix(prefix, count: count, device: program.device, into: blit)
                    default: throw Failure.cacheFamily
                    }
                }
            } catch {
                blit.endEncoding()
                throw error
            }
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw Failure.command(command.error) }
            linear = states
            offset = count
        }

        func publish(to caches: [Q35LayerCache?]) throws {
            try finish()
            var stateIndex = 0
            for (index, entry) in caches.enumerated() {
                switch entry {
                case .linear(let cache):
                    cache.installNativeState(linear[stateIndex])
                    stateIndex += 1
                case .full(let cache as KVCacheSimple) where cache.offset < offset:
                    let suffix = kv[index]!.suffix(cache.offset..<offset)
                    _ = cache.update(keys: suffix.0, values: suffix.1)
                default: break
                }
            }
        }

        #if DEBUG || MERERUN_RELEASE_REPLAY
        /// The qualification fixture owns this model exclusively. Copy its last verified rows before issuing another round.
        func replayLogits(rows: Int) throws -> MLXArray {
            let vocabulary = 248_320
            let bytes = rows * vocabulary * 2
            guard (1...16).contains(rows),
                  let buffer = program.device.makeBuffer(length: bytes, options: .storageModeShared),
                  let command = program.queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
            else { throw Failure.allocation }
            blit.copy(from: head.logits, sourceOffset: 0, to: buffer, destinationOffset: 0, size: bytes)
            blit.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { throw Failure.command(command.error) }
            return MLXArray(rawPointer: buffer.contents(), [rows, vocabulary], dtype: .bfloat16) { _ = buffer }
        }
        #endif

        func makeTeacher() throws -> Q35NativeTeacherHead {
            try Q35NativeTeacherHead(device: program.device, vocabulary: 248_320)
        }

        func prefill(
            _ tokens: [Int], capture: Bool, score: Bool, teacher: (Q35NativeTeacherHead, Int)? = nil
        ) throws -> NativeRound {
            let plan: Q35NativeTreePlan
            if let cached = chainPlans[tokens.count] {
                plan = cached
            } else {
                plan = try Q35NativeTreePlan(causalRows: tokens.count, device: program.device)
                chainPlans[tokens.count] = plan
            }
            return try run(tokens, plan: plan, verification: nil, capture: capture, score: score, teacher: teacher)
        }

        func verify(_ proposal: DFlash2TreeProposal, budget: Int, eos: Set<Int>) throws -> NativeRound {
            let tree = try Q35VerificationTree(parents: proposal.parents)
            let plan = try Q35NativeTreePlan(tree: tree, device: program.device)
            let request = Q35GreedyTreeRequest(head: headWeights, tokens: proposal.tokens, budget: budget, eos: eos)
            return try run(proposal.tokens, plan: plan, verification: (request, tree), capture: true, score: true)
        }

        func verifySampledTree(_ proposal: DFlash2TreeProposal, budget: Int, eos: Set<Int>,
                               sampler: Q35PositionSampler, position: Int) throws -> NativeRound {
            let tree = try Q35VerificationTree(parents: proposal.parents)
            let plan = try Q35NativeTreePlan(tree: tree, device: program.device)
            let request = Q35GreedyTreeRequest(head: headWeights, tokens: proposal.tokens, budget: budget, eos: eos)
            if Q35SampledTreeMode.gpuHeadOverride ?? (sampler.treeMode == .vocabulary) {
                try prepareSampledHead()
            }
            let sampled = SampledVerification(logits: try sampledBuffer(), sampler: sampler, position: position) { buffer, _ in
                var path = [0]
                let words = buffer.contents().assumingMemoryBound(to: UInt16.self)
                while let row = path.last {
                    let token = sampler.choose(UnsafeBufferPointer(start: words + row * 248_320, count: 248_320),
                                               position: position + tree.depths[row])
                    guard path.count < budget, !eos.contains(token),
                          let child = tree.parents.indices.dropFirst().first(where: {
                              tree.parents[$0] == row && proposal.tokens[$0] == token
                          }) else { return (path, token) }
                    path.append(child)
                }
                preconditionFailure("Missing tree root")
            }
            return try run(proposal.tokens, plan: plan, verification: (request, tree), capture: true, score: true, sampled: sampled)
        }

        func selectedArrays(count: Int) -> [Int: MLXArray] {
            gathered.mapValues { $0.array([1, count, 5_120]) }
        }

        func verifyReferenceChain(_ tokens: [Int], eos: Set<Int>, rule: Q35PositionSampler, rejection: Q35RejectionDraws?, decide: @escaping (UnsafePointer<UInt16>) -> (path: [Int], following: Int)) throws -> NativeRound {
            let tree = try Q35VerificationTree(parents: Array(-1..<(tokens.count - 1)))
            let plan: Q35NativeTreePlan
            if let cached = chainPlans[tokens.count] { plan = cached } else {
                plan = try Q35NativeTreePlan(tree: tree, device: program.device)
                chainPlans[tokens.count] = plan
            }
            let request = Q35GreedyTreeRequest(head: headWeights, tokens: tokens, budget: tokens.count, eos: eos)
            if rejection != nil { try prepareSampledHead() }
            let sampled = SampledVerification(logits: try sampledBuffer(), sampler: rejection == nil ? nil : rule,
                                              position: 0, rejection: rejection) { buffer, _ in
                decide(UnsafePointer(buffer.contents().assumingMemoryBound(to: UInt16.self)))
            }
            return try run(tokens, plan: plan, verification: (request, tree), capture: true, score: true, sampled: sampled)
        }

        private func prepareSampledHead() throws {
            if sampledHead == nil {
                sampledHead = try program.lock.withLock {
                    if program.positionHead == nil { program.positionHead = try Q35NativeSampledHead(device: program.device) }
                    return program.positionHead
                }
            }
        }

        private func sampledBuffer() throws -> any MTLBuffer {
            if sampledLogits == nil {
                sampledLogits = program.device.makeBuffer(length: 16 * 248_320 * 2, options: .storageModeShared)
            }
            guard let sampledLogits else { throw Failure.allocation }
            return sampledLogits
        }

        func finish() throws {
            try linear.first?.waitUntilReady()
        }

        private func run(
            _ tokens: [Int], plan: Q35NativeTreePlan, verification: (Q35GreedyTreeRequest, Q35VerificationTree)?,
            capture: Bool, score: Bool, teacher: (Q35NativeTeacherHead, Int)? = nil,
            sampled: SampledVerification? = nil
        ) throws -> NativeRound {
            try program.lock.withLock {
                guard let embedding = program.embedding, embedding.supports(tokens),
                      tokens.count == plan.count, (verification?.0.eos.count ?? 0) <= 128
                else { throw Failure.geometry }
                try finish()
                let rows = tokens.count
                let prefix = offset
                let activeCaptures = capture ? captures : [:]
                let states = preparedStates()
                let advancing = verification == nil ? advancingDestinations() : [:]
                let body = try Body(input: .init(buffer: program.embeddingBuffer, offset: 0,
                                                  shape: [1, rows, 5_120], dtype: .bfloat16),
                    frames: frames, states: states, captures: activeCaptures, plan: plan,
                    recurrent: program.delta.prepare(plan),
                    attention: program.attention.prepare(plan: plan, prefix: prefix, scratch: &attentionScratch),
                    prefix: prefix, advancing: advancing)
                var (command, encoder) = try program.openSubmission()
                var hazards = M5NativeHazards()
                var submitted: [any MTLCommandBuffer] = []
                defer { submitted.last?.waitUntilCompleted() }
                embedding.encode(tokens, into: program.embeddingBuffer, encoder: encoder, hazards: hazards)
                try program.encodeBody(body, command: &command, encoder: &encoder, hazards: &hazards,
                                       submitted: &submitted, submissionLayers: 4)
                if score || activeCaptures[frames.count - 1] != nil {
                    program.norm.encode(into: encoder, input: program.hiddenBuffer, residual: program.pending,
                        weight: program.finalNorm.buffer,
                        combined: activeCaptures[frames.count - 1]?.buffer ?? program.hiddenBuffer,
                        normalized: finalOutput.buffer, sums: program.ffnWorkspace.inputSums, rows: rows,
                        epsilon: program.epsilon, weightOffset: program.finalNorm.offset,
                        combinedOffset: activeCaptures[frames.count - 1]?.offset ?? 0,
                        normalizedOffset: finalOutput.offset, hazards: hazards)
                }
                let picks: (any MTLBuffer)? = score ? head.metadata : nil
                if let sampled {
                    head.project(into: encoder, normalized: finalOutput, sums: program.ffnWorkspace.inputSums,
                                 rows: rows, output: sampled.logits, hazards: hazards)
                    if let sampledHead, let sampler = sampled.sampler, let (request, tree) = verification {
                        sampledHead.encode(into: encoder, logits: sampled.logits, sampler: sampler,
                            position: sampled.position, tree: tree, request: request, metadata: selection, hazards: hazards, rejection: sampled.rejection)
                        for (index, source) in activeCaptures {
                            head.gatherTap(source, into: gathered[index]!, metadata: selection, encoder: encoder, hazards: hazards)
                        }
                    }
                } else if let (request, tree) = verification {
                    head.encode(into: encoder, normalized: finalOutput, sums: program.ffnWorkspace.inputSums,
                                request: request, tree: tree, metadata: selection, hazards: hazards)
                    for (index, source) in activeCaptures {
                        head.gatherTap(source, into: gathered[index]!, metadata: selection, encoder: encoder, hazards: hazards)
                    }
                } else {
                    if let picks {
                        head.encodeLast(into: encoder, normalized: finalOutput, sums: program.ffnWorkspace.inputSums,
                                        rows: rows, token: tokens[rows - 1], metadata: picks, hazards: hazards)
                        if let (observer, label) = teacher {
                            observer.encode(logits: head.logits, label: label, encoder: encoder, hazards: hazards)
                        }
                    }
                    let info = selection.contents().assumingMemoryBound(to: Int32.self)
                    info[0] = Int32(rows)
                    for row in 0..<rows { info[2 + row] = Int32(row) }
                    encoder.memoryBarrier(scope: .buffers)
                    appendKV(encoder, selection: selection, plan: plan)
                }
                encoder.endEncoding()
                command.commit()
                submitted.append(command)
                let prepared = try verification.map { _ in try prepareCommit(states, metadata: selection, plan: plan) }
                command.waitUntilCompleted()
                for batch in submitted where batch.status != .completed { throw Failure.command(batch.error) }
                submitted.removeAll(keepingCapacity: true)

                let path: [Int]
                let following: Int?
                if sampled?.sampler != nil, sampledHead != nil, let selected = head.selection(from: selection, rows: rows) {
                    path = selected.path
                    following = selected.following
                } else if let sampled {
                    let accepted = sampled.reference(sampled.logits, rows)
                    path = accepted.path
                    guard (1...rows).contains(path.count) else { throw Failure.invalidSelection }
                    following = accepted.following
                    let info = selection.contents().assumingMemoryBound(to: Int32.self)
                    info[0] = Int32(path.count)
                    info[1] = Int32(accepted.following)
                    for (i, row) in path.enumerated() { info[2 + i] = Int32(row) }
                    for (index, source) in activeCaptures {
                        let destination = gathered[index]!
                        for (i, row) in path.enumerated() {
                            memcpy(destination.buffer.contents() + destination.offset + i * 10_240,
                                   source.buffer.contents() + source.offset + row * 10_240, 10_240)
                        }
                    }
                } else if verification != nil {
                    guard let accepted = head.selection(from: selection, rows: rows) else { throw Failure.invalidSelection }
                    path = accepted.path
                    following = accepted.following
                } else {
                    path = Array(0..<rows)
                    if let picks {
                        guard let accepted = head.selection(from: picks, rows: 1) else { throw Failure.invalidSelection }
                        following = accepted.following
                    } else {
                        following = nil
                    }
                }
                linear = prepared.map(program.stateCommit.submit)
                    ?? program.stateCommit.completed(banks[bank], command: command)
                if verification != nil {
                    program.roundCount += 1
                    if sampled == nil { program.greedyRoundCount += 1 }
                }
                bank ^= 1
                offset += path.count
                let tapOutputs = verification == nil ? activeCaptures : gathered
                return NativeRound(count: path.count, following: following, path: path,
                    taps: tapOutputs.mapValues { Q35NativeDraftTap(buffer: $0.buffer, offset: $0.offset) },
                    loss: teacher?.0.loss())
            }
        }

        private func preparedStates() -> [PreparedState] {
            var index = 0
            return frames.enumerated().map { position, frame in
                switch frame {
                case .linear:
                    defer { index += 1 }
                    return .linear(linear[index])
                case .full:
                    return offset == 0 ? .full(nil, nil) : .full(kv[position]!.keys, kv[position]!.values)
                }
            }
        }

        private func advancingDestinations() -> [Int: Q35NativeStateCommit.Destination] {
            let layers = frames.indices.filter { if case .linear = frames[$0] { return true }; return false }
            return Dictionary(uniqueKeysWithValues: zip(layers, banks[bank]))
        }

        private func appendKV(
            _ encoder: any MTLComputeCommandEncoder, selection: any MTLBuffer, plan: Q35NativeTreePlan
        ) {
            for index in frames.indices {
                if case .full(let frame) = frames[index] {
                    kvCommit.encode(into: encoder, keys: frame.keys, values: frame.values,
                                    bank: kv[index]!, selection: selection, offset: offset, rows: plan.count,
                                    branchRows: plan.branchRows)
                }
            }
        }

        private func prepareCommit(
            _ states: [PreparedState], metadata: any MTLBuffer, plan: Q35NativeTreePlan
        ) throws -> Q35NativeStateCommit.Prepared {
            let inputs: [Q35NativeReplayInput] = frames.indices.compactMap { index in
                guard case .linear(let weights) = program.layers[index].mixer,
                      case .linear(let frame) = frames[index], case .linear(let initial) = states[index] else { return nil }
                return Q35NativeReplayInput(initial: initial, qkvz: frame.qkvz, ba: frame.ba,
                    q: frame.q, k: frame.k, v: frame.v, g: frame.g, beta: frame.beta,
                    aLog: weights.aLogArray, dtBias: weights.dtBiasArray)
            }
            return try program.stateCommit.prepare(inputs: inputs, selection: metadata, destinations: banks[bank]) { [self] encoder in
                appendKV(encoder, selection: metadata, plan: plan)
            }
        }
    }
}
#endif
