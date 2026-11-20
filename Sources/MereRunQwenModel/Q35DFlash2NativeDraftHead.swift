#if os(macOS)
import Foundation
import MereRunTensor
import Metal
import MLX
import MLXNN

final class Q35DFlash2NativeDraftHead {
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_NATIVE_HEAD")
    static var enabled: Bool { enabledFlag.enabled }
    private static let nativeScorerFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_NATIVE_SCORER")
    private static var nativeScorer: Bool { nativeScorerFlag.enabled }

    struct Output {
        let tokens: [Int32]
        let unary: [Float]
        let scores: [Float]?
    }

    protocol Program {
        func run(_ hidden: MLXArray, anchor: Int, score: Bool) -> Output?
    }

    private var source: ObjectIdentifier?
    private var program: (any Program)?
    private let lock = NSLock()

    func tree(_ hidden: MLXArray, target: Q35Model, selector: DFlash2CandidateSelector,
              anchor: Int, nodeBudget: Int, calibration: DFlash2TreeCalibration) -> DFlash2TreeProposal? {
        guard #available(macOS 26.0, *), Device.defaultDevice().deviceType == .gpu,
              hidden.dtype == .bfloat16, hidden.dim(-1) == 5120,
              target.config.textConfig.vocabSize == Q35DFlash2DraftHead.vocabularySize,
              let head = target.lmHead else { return nil }
        let score = Self.nativeScorer && calibration.edgeWeight == 1 && calibration.temperature == 1
        let output = lock.withLock {
            if source != ObjectIdentifier(head) {
                program = try? MetalProgram(head, selector: Self.nativeScorer ? selector : nil)
                source = ObjectIdentifier(head)
            }
            return program?.run(hidden, anchor: anchor, score: score)
        }
        guard let output else { return nil }
        let rows = hidden.dim(1)
        if let scores = output.scores {
            return DFlash2CandidateSelector.buildTree(anchor: anchor, candidates: output.tokens.map(Int.init),
                scores: scores, depthCount: rows, candidateCount: M5NativeTopK.count, nodeBudget: nodeBudget)
        }
        let shape = [1, rows, M5NativeTopK.count]
        return selector.tree(hidden: hidden, candidates: MLXArray(output.tokens, shape),
            unary: MLXArray(output.unary, shape).asType(.bfloat16), anchor: anchor, nodeBudget: nodeBudget, calibration: calibration)
    }

    @available(macOS 26.0, *)
    final class MetalProgram: Program {
        private let device: any MTLDevice
        private let queue: any MTLCommandQueue
        private let weights: M5NativeFFN.Projection
        private let projection: M5NativeProjection
        private let sum: M5NativeGroupSums
        private let skippedTokenIDs: Range<Int>?
        private let topK: M5NativeTopK
        private let sums: any MTLBuffer
        private let logits: any MTLBuffer
        private let scorer: Q35NativeDraftScorer?
        private var reported = false
        private var reportedScorer = false
        var supportsScoring: Bool { scorer != nil }

        init?(_ head: Linear, selector: DFlash2CandidateSelector?, compact: Bool = false) throws {
            skippedTokenIDs = compact ? Q35DFlash2DraftHead.prefixCount..<Q35DFlash2DraftHead.controlStart : nil
            let columns = compact ? Q35DFlash2DraftHead.realCount : Q35DFlash2DraftHead.vocabularySize
            guard let device = MTLCreateSystemDefaultDevice(),
                  let weights = try PortableQuantizedLinear.nativeProjectionWeights(head, device: device),
                  weights.inputSize == 5120, weights.outputSize == Q35DFlash2DraftHead.vocabularySize,
                  let queue = device.makeCommandQueue(),
                  let sums = device.makeBuffer(length: 16 * weights.inputSize / 64 * 4, options: .storageModePrivate),
                  let logits = device.makeBuffer(length: 16 * columns * 2, options: .storageModePrivate)
            else { return nil }
            self.device = device
            self.weights = weights
            self.queue = queue
            self.sums = sums
            self.logits = logits
            projection = try M5NativeProjection(device: device, inputSize: weights.inputSize, outputSize: weights.outputSize,
                                                skippingOutputRows: skippedTokenIDs)
            sum = try M5NativeGroupSums(device: device, inputSize: weights.inputSize)
            topK = try M5NativeTopK(device: device)
            if let scoring = selector?.scoringWeights, scoring.predecessors.dim(0) == weights.outputSize {
                scorer = try? Q35NativeDraftScorer(projection: scoring.projection, predecessors: scoring.predecessors,
                                                  successors: scoring.successors, device: device)
            } else {
                scorer = nil
            }
        }

        func run(_ hidden: MLXArray, anchor: Int, score: Bool) -> Output? {
            let rows = hidden.dim(1)
            guard let input = try? M5NativeArrayBuffer(contiguous(hidden), device: device),
                  let command = queue.makeCommandBuffer() else { return nil }
            let scorer = score ? self.scorer : nil
            if let scorer, !scorer.stageInput(input, on: command, rows: rows) { return nil }
            guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else { return nil }
            command.label = "Qwen native draft projection and top-k"
            sum.encode(into: encoder, input: input.buffer, sums: sums, rows: rows, inputOffset: input.offset)
            encode(into: encoder, hidden: input.buffer, sums: sums, rows: rows, anchor: anchor, useScorer: score,
                   inputOffset: input.offset)
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else { return nil }
            return read(rows: rows, useScorer: score)
        }

        func encode(into encoder: any MTLComputeCommandEncoder, hidden: any MTLBuffer, sums: any MTLBuffer,
                    rows: Int, anchor: Int, useScorer: Bool, inputOffset: Int = 0, alreadyPadded: Bool = false, referenceScores: Bool = false) {
            projection.encode(into: encoder, input: hidden, sums: sums, output: logits, weights: weights,
                              rows: rows, inputOffset: inputOffset)
            topK.encode(into: encoder, logits: logits, rows: rows, vocabulary: projection.projectedSize,
                        skippingTokenIDs: skippedTokenIDs)
            if useScorer {
                scorer?.encode(into: encoder, topK: topK, anchor: anchor, rows: rows,
                               nativeInput: alreadyPadded ? hidden : nil,
                               calibration: !referenceScores && alreadyPadded && Q35DFlash2TreePolicy.calibratedScoresEnabled ? .candidate : .reference)
            }
        }

        func read(rows: Int, useScorer: Bool) -> Output? {
            guard let result = topK.read(rows: rows) else { return nil }
            if !reported {
                FileHandle.standardError.write(Data("[q35] dflash2_native_draft_head=true\n".utf8))
                reported = true
            }
            let scores = useScorer ? scorer?.read(rows: rows) : nil
            if scores != nil, !reportedScorer {
                FileHandle.standardError.write(Data("[q35] dflash2_native_draft_scorer=true\n".utf8))
                reportedScorer = true
            }
            return Output(tokens: result.tokens, unary: result.scores, scores: scores)
        }
    }
}
#endif
