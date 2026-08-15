import Foundation
import MLX

/// Input to a shared autoregressive decode: everything the loop needs that
/// is not the model itself.
public struct AutoregressiveDecodeRequest {
    public let initialLogits: MLXArray
    public let generationConfig: GenerationConfig
    public let eosTokens: Set<Int>
    public let tokenBudget: Int
    /// Seeds the on-GPU repetition window (typically the prompt tokens).
    public let historySeedTokens: [Int]
    /// Optional additive logit bias applied every step (e.g. token bans).
    public let banMask: MLXArray?

    public init(
        initialLogits: MLXArray,
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int] = [],
        banMask: MLXArray? = nil
    ) {
        self.initialLogits = initialLogits
        self.generationConfig = generationConfig
        self.eosTokens = eosTokens
        self.tokenBudget = tokenBudget
        self.historySeedTokens = historySeedTokens
        self.banMask = banMask
    }
}

public struct AutoregressiveDecodeResult {
    public let generatedTokens: [Int]
    public let decodeSeconds: Double
    /// Seconds from decode start to the first confirmed token (time to
    /// first token, excluding prefill). Nil when nothing was generated.
    public let firstTokenSeconds: Double?
    /// Host time spent building/scheduling each step's graph (sample +
    /// forward + asyncEval). With the wait time, callers can emit the same
    /// build/wait decode-trace lines the hand-rolled loops printed.
    public let buildSeconds: Double
    /// Host time spent blocked on step confirmation readbacks.
    public let waitSeconds: Double

    public init(
        generatedTokens: [Int],
        decodeSeconds: Double,
        firstTokenSeconds: Double? = nil,
        buildSeconds: Double = 0,
        waitSeconds: Double = 0
    ) {
        self.generatedTokens = generatedTokens
        self.decodeSeconds = decodeSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.buildSeconds = buildSeconds
        self.waitSeconds = waitSeconds
    }
}

/// The platform's one serial decode loop. Every autoregressive engine that
/// adopted this shape independently (Gemma4, Q35, GLM-4.7 Flash, Qwen3 TTS,
/// Qwen3 ASR, OCR) measured the same wins over the naive loop, so the shape
/// lives here once:
///
/// - sampling stays on GPU (`sampledTokenArray`, argPartition top-p
///   prefilter, on-GPU repetition window) — no host readback per sample;
/// - the sampled token feeds the next forward as an array;
/// - the loop pipelines at depth 1: while the GPU executes step N+1, the
///   host reads back and confirms step N (EOS check, emission), so the
///   readback latency hides behind compute. EOS costs one speculative
///   forward, which is discarded.
///
/// Display emission buffers whitespace-only pieces until visible text
/// arrives, matching the chat CLIs' behavior.
public enum AutoregressiveDecodeEngine {
    public static func decode(
        _ request: AutoregressiveDecodeRequest,
        stepForward: (MLXArray) throws -> MLXArray,
        decodeToken: ((Int) -> String)? = nil,
        emitPiece: ((Int, String) -> Void)? = nil,
        shouldContinue: ((Int, String) -> Bool)? = nil,
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> AutoregressiveDecodeResult {
        guard request.tokenBudget > 0 else {
            return AutoregressiveDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        let start = Date()
        var generated: [Int] = []
        generated.reserveCapacity(request.tokenBudget)
        var firstTokenSeconds: Double?
        var repetitionHistory = repetitionHistoryArray(
            promptTokens: request.historySeedTokens,
            config: request.generationConfig
        )
        var pendingProgressWhitespace = ""

        func confirm(_ tokenArray: MLXArray) -> Bool {
            let token = tokenArray.item(Int.self)
            if request.eosTokens.contains(token) {
                return false
            }
            generated.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(start)
            }
            let piece = decodeToken?(token) ?? ""
            if let emitPiece, !piece.isEmpty {
                if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    pendingProgressWhitespace += piece
                } else {
                    emitPiece(token, pendingProgressWhitespace + piece)
                    pendingProgressWhitespace = ""
                }
            }
            return shouldContinue?(token, piece) ?? true
        }

        var buildSeconds = 0.0
        var waitSeconds = 0.0

        var logits = request.initialLogits
        var pending: MLXArray?

        while generated.count < request.tokenBudget {
            try checkCancellation?()

            // Confirm the first token before opening the deeper pipeline. This
            // keeps immediate EOS to one speculative forward, while later
            // iterations can queue the next dependent forward before the host
            // confirms the preceding token.
            if generated.isEmpty, let first = pending {
                let waitStart = CFAbsoluteTimeGetCurrent()
                pending = nil
                let confirmed = confirm(first)
                waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                guard confirmed else {
                    return AutoregressiveDecodeResult(
                        generatedTokens: generated,
                        decodeSeconds: Date().timeIntervalSince(start),
                        firstTokenSeconds: firstTokenSeconds,
                        buildSeconds: buildSeconds,
                        waitSeconds: waitSeconds
                    )
                }
                if generated.count >= request.tokenBudget {
                    break
                }
            }

            let hasPending = pending != nil
            let isFinalSample = (!hasPending && generated.count == request.tokenBudget - 1)
                || (hasPending && generated.count == request.tokenBudget - 2)
            let buildStart = CFAbsoluteTimeGetCurrent()
            let tokenArray = sampledTokenArray(
                logits: logits[0, -1, 0...],
                config: request.generationConfig,
                previousTokenIndices: repetitionHistory,
                banMask: request.banMask
            )

            if isFinalSample {
                asyncEval(tokenArray)
                buildSeconds += CFAbsoluteTimeGetCurrent() - buildStart
                if let previous = pending {
                    let waitStart = CFAbsoluteTimeGetCurrent()
                    pending = nil
                    let confirmed = confirm(previous)
                    waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                    guard confirmed else {
                        return AutoregressiveDecodeResult(
                            generatedTokens: generated,
                            decodeSeconds: Date().timeIntervalSince(start),
                            firstTokenSeconds: firstTokenSeconds,
                            buildSeconds: buildSeconds,
                            waitSeconds: waitSeconds
                        )
                    }
                }
                if generated.count < request.tokenBudget {
                    let waitStart = CFAbsoluteTimeGetCurrent()
                    _ = confirm(tokenArray)
                    waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                }
                break
            }

            repetitionHistory = appendingRepetitionHistory(
                repetitionHistory,
                token: tokenArray,
                config: request.generationConfig
            )
            logits = try stepForward(tokenArray.asType(.int32).reshaped(1, 1))
            asyncEval([logits, tokenArray])
            let buildEnd = CFAbsoluteTimeGetCurrent()
            buildSeconds += buildEnd - buildStart

            if let previous = pending {
                pending = nil
                let confirmed = confirm(previous)
                waitSeconds += CFAbsoluteTimeGetCurrent() - buildEnd
                guard confirmed else {
                    return AutoregressiveDecodeResult(
                        generatedTokens: generated,
                        decodeSeconds: Date().timeIntervalSince(start),
                        firstTokenSeconds: firstTokenSeconds,
                        buildSeconds: buildSeconds,
                        waitSeconds: waitSeconds
                    )
                }
            }
            pending = tokenArray
        }

        if let previous = pending, generated.count < request.tokenBudget {
            _ = confirm(previous)
        }

        return AutoregressiveDecodeResult(
            generatedTokens: generated,
            decodeSeconds: Date().timeIntervalSince(start),
            firstTokenSeconds: firstTokenSeconds,
            buildSeconds: buildSeconds,
            waitSeconds: waitSeconds
        )
    }

    /// Pipelined decode for generators whose next-step logits transform or
    /// sampler state depends on the previously confirmed token. The sampled
    /// token is still fed directly into the next forward on GPU; confirmation
    /// happens at the start of the following iteration while that speculative
    /// forward is already executing.
    public static func decodeStateful(
        _ request: AutoregressiveDecodeRequest,
        processLogits: ((MLXArray, [Int]) -> MLXArray)? = nil,
        stepForward: (MLXArray) -> MLXArray,
        didSampleToken: ((Int) -> Void)? = nil,
        shouldContinue: ((Int) -> Bool)? = nil
    ) -> AutoregressiveDecodeResult {
        guard request.tokenBudget > 0 else {
            return AutoregressiveDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        let start = Date()
        var generated: [Int] = []
        generated.reserveCapacity(request.tokenBudget)
        var stateTokens = request.historySeedTokens
        var firstTokenSeconds: Double?
        var repetitionHistory = repetitionHistoryArray(
            promptTokens: request.historySeedTokens,
            config: request.generationConfig
        )
        var logits = request.initialLogits
        var pending: MLXArray?
        var buildSeconds = 0.0
        var waitSeconds = 0.0

        func result() -> AutoregressiveDecodeResult {
            AutoregressiveDecodeResult(
                generatedTokens: generated,
                decodeSeconds: Date().timeIntervalSince(start),
                firstTokenSeconds: firstTokenSeconds,
                buildSeconds: buildSeconds,
                waitSeconds: waitSeconds
            )
        }

        while true {
            if let tokenArray = pending {
                let waitStart = CFAbsoluteTimeGetCurrent()
                let token = tokenArray.item(Int.self)
                waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                pending = nil

                didSampleToken?(token)
                if let shouldContinue, !shouldContinue(token) {
                    return result()
                }
                if request.eosTokens.contains(token) {
                    return result()
                }

                generated.append(token)
                stateTokens.append(token)
                if firstTokenSeconds == nil {
                    firstTokenSeconds = Date().timeIntervalSince(start)
                }
                if generated.count >= request.tokenBudget {
                    return result()
                }
            }

            let buildStart = CFAbsoluteTimeGetCurrent()
            let rawLogits = logits[0, -1, 0...]
            let nextLogits = processLogits?(rawLogits, stateTokens) ?? rawLogits
            let tokenArray = sampledTokenArray(
                logits: nextLogits,
                config: request.generationConfig,
                previousTokenIndices: repetitionHistory,
                banMask: request.banMask
            )
            repetitionHistory = appendingRepetitionHistory(
                repetitionHistory,
                token: tokenArray,
                config: request.generationConfig
            )
            logits = stepForward(tokenArray.asType(.int32).reshaped(1, 1))
            asyncEval([logits, tokenArray])
            buildSeconds += CFAbsoluteTimeGetCurrent() - buildStart
            pending = tokenArray
        }
    }
}
