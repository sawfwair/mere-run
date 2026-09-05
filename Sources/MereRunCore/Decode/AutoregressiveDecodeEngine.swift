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
    public let logprobCapture: ChatLogprobCapture
    public let logprobRegion: ChatLogprobRegion

    public init(
        initialLogits: MLXArray,
        generationConfig: GenerationConfig,
        eosTokens: Set<Int>,
        tokenBudget: Int,
        historySeedTokens: [Int] = [],
        banMask: MLXArray? = nil,
        logprobCapture: ChatLogprobCapture = .none,
        logprobRegion: ChatLogprobRegion = .visible
    ) {
        self.initialLogits = initialLogits
        self.generationConfig = generationConfig
        self.eosTokens = eosTokens
        self.tokenBudget = tokenBudget
        self.historySeedTokens = historySeedTokens
        self.banMask = banMask
        self.logprobCapture = logprobCapture
        self.logprobRegion = logprobRegion
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
    public let logprobs: ChatLogprobDiagnostics?

    public init(
        generatedTokens: [Int],
        decodeSeconds: Double,
        firstTokenSeconds: Double? = nil,
        buildSeconds: Double = 0,
        waitSeconds: Double = 0,
        logprobs: ChatLogprobDiagnostics? = nil
    ) {
        self.generatedTokens = generatedTokens
        self.decodeSeconds = decodeSeconds
        self.firstTokenSeconds = firstTokenSeconds
        self.buildSeconds = buildSeconds
        self.waitSeconds = waitSeconds
        self.logprobs = logprobs
    }
}

private struct PendingAutoregressiveSample {
    let token: MLXArray
    let logits: MLXArray
}

private struct ChatLogprobRegionTracker {
    private let defaultRegion: ChatLogprobRegion
    private var reasoning = false
    private var toolCall = false

    init(defaultRegion: ChatLogprobRegion) {
        self.defaultRegion = defaultRegion
    }

    mutating func classify(_ piece: String) -> ChatLogprobRegion {
        let normalized = piece.lowercased()
        if normalized.contains("<think>") {
            reasoning = true
            return .markup
        }
        if normalized.contains("</think>") {
            reasoning = false
            return .markup
        }
        if normalized.contains("<tool_call>") || normalized.contains("</tool_call>") {
            toolCall = !normalized.contains("</tool_call>")
            return .markup
        }
        if normalized.contains("<function=") {
            toolCall = true
            return .toolName
        }
        if normalized.contains("<parameter=") || normalized.contains("</parameter>") {
            return .toolArgument
        }
        if normalized.hasPrefix("<") && normalized.contains(">") {
            return .markup
        }
        if reasoning { return .reasoning }
        if toolCall { return .toolArgument }
        return defaultRegion
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
        decodeTokens: (([Int]) -> String)? = nil,
        emitPiece: ((Int, String) -> Void)? = nil,
        shouldContinue: ((Int, String) -> Bool)? = nil,
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> AutoregressiveDecodeResult {
        guard request.tokenBudget > 0 else {
            return AutoregressiveDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        let start = Date()
        var samplingConfig = request.generationConfig
        if request.logprobCapture.isEnabled {
            // The measured policy must be the policy that selected the token.
            // Disable the optional top-p prefilter in quality/calibration runs.
            samplingConfig.topPPrefilter = 0
        }
        var generated: [Int] = []
        generated.reserveCapacity(request.tokenBudget)
        var diagnosticHistory = request.historySeedTokens
        var measuredTokens: [ChatTokenLogprob] = []
        measuredTokens.reserveCapacity(request.tokenBudget)
        var logprobCaptureSeconds = 0.0
        var regionTracker = ChatLogprobRegionTracker(defaultRegion: request.logprobRegion)
        var firstTokenSeconds: Double?
        var repetitionHistory = repetitionHistoryArray(
            promptTokens: request.historySeedTokens,
            config: samplingConfig
        )
        var pendingProgressWhitespace = ""
        var progressDecoder = IncrementalTokenTextDecoder()

        func capturedDiagnostics() -> ChatLogprobDiagnostics? {
            guard request.logprobCapture.isEnabled else { return nil }
            return ChatLogprobDiagnostics(
                capture: request.logprobCapture,
                measuredTokens: measuredTokens,
                captureSeconds: logprobCaptureSeconds
            )
        }

        func result(
            buildSeconds: Double,
            waitSeconds: Double
        ) -> AutoregressiveDecodeResult {
            AutoregressiveDecodeResult(
                generatedTokens: generated,
                decodeSeconds: Date().timeIntervalSince(start),
                firstTokenSeconds: firstTokenSeconds,
                buildSeconds: buildSeconds,
                waitSeconds: waitSeconds,
                logprobs: capturedDiagnostics()
            )
        }

        func confirm(_ sample: PendingAutoregressiveSample) -> Bool {
            let token = sample.token.item(Int.self)
            if request.eosTokens.contains(token) {
                return false
            }
            generated.append(token)
            if firstTokenSeconds == nil {
                firstTokenSeconds = Date().timeIntervalSince(start)
            }
            let tokenPiece = decodeToken?(token) ?? ""
            let region = regionTracker.classify(tokenPiece)
            if request.logprobCapture.isEnabled {
                let captureStart = CFAbsoluteTimeGetCurrent()
                var measurement = tokenLogprobMeasurement(
                    logits: sample.logits,
                    selectedToken: token,
                    config: samplingConfig,
                    previousTokens: diagnosticHistory,
                    topLogprobs: request.logprobCapture.topLogprobs
                )
                measurement.region = region
                if request.logprobCapture.includesTokens, region != .reasoning {
                    measurement.token = tokenPiece
                    if !measurement.topLogprobs.isEmpty, let decodeToken {
                        for index in measurement.topLogprobs.indices {
                            measurement.topLogprobs[index].token = decodeToken(
                                measurement.topLogprobs[index].tokenID
                            )
                        }
                    }
                }
                measuredTokens.append(measurement)
                logprobCaptureSeconds += CFAbsoluteTimeGetCurrent() - captureStart
            }
            diagnosticHistory.append(token)
            let progressPiece = decodeTokens.map {
                progressDecoder.append(decodedText: $0(generated))
            } ?? tokenPiece
            if let emitPiece, !progressPiece.isEmpty {
                if progressPiece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    pendingProgressWhitespace += progressPiece
                } else {
                    emitPiece(token, pendingProgressWhitespace + progressPiece)
                    pendingProgressWhitespace = ""
                }
            }
            return shouldContinue?(token, tokenPiece) ?? true
        }

        var buildSeconds = 0.0
        var waitSeconds = 0.0

        var logits = request.initialLogits
        var pending: PendingAutoregressiveSample?

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
                    return result(buildSeconds: buildSeconds, waitSeconds: waitSeconds)
                }
                if generated.count >= request.tokenBudget {
                    break
                }
            }

            let hasPending = pending != nil
            let isFinalSample = (!hasPending && generated.count == request.tokenBudget - 1)
                || (hasPending && generated.count == request.tokenBudget - 2)
            let buildStart = CFAbsoluteTimeGetCurrent()
            let sampleLogits = logits[0, -1, 0...]
            let tokenArray = sampledTokenArray(
                logits: sampleLogits,
                config: samplingConfig,
                previousTokenIndices: repetitionHistory,
                banMask: request.banMask
            )
            let sample = PendingAutoregressiveSample(token: tokenArray, logits: sampleLogits)

            if isFinalSample {
                asyncEval(tokenArray)
                buildSeconds += CFAbsoluteTimeGetCurrent() - buildStart
                if let previous = pending {
                    let waitStart = CFAbsoluteTimeGetCurrent()
                    pending = nil
                    let confirmed = confirm(previous)
                    waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                    guard confirmed else {
                        return result(buildSeconds: buildSeconds, waitSeconds: waitSeconds)
                    }
                }
                if generated.count < request.tokenBudget {
                    let waitStart = CFAbsoluteTimeGetCurrent()
                    _ = confirm(sample)
                    waitSeconds += CFAbsoluteTimeGetCurrent() - waitStart
                }
                break
            }

            repetitionHistory = appendingRepetitionHistory(
                repetitionHistory,
                token: tokenArray,
                config: samplingConfig
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
                    return result(buildSeconds: buildSeconds, waitSeconds: waitSeconds)
                }
            }
            pending = sample
        }

        if let previous = pending, generated.count < request.tokenBudget {
            _ = confirm(previous)
        }

        return result(buildSeconds: buildSeconds, waitSeconds: waitSeconds)
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
