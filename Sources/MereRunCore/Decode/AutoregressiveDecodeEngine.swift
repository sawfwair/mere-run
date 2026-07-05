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
        checkCancellation: (() throws -> Void)? = nil
    ) throws -> AutoregressiveDecodeResult {
        guard request.tokenBudget > 0 else {
            return AutoregressiveDecodeResult(generatedTokens: [], decodeSeconds: 0)
        }

        let start = Date()
        var generated: [Int] = []
        generated.reserveCapacity(request.tokenBudget)
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
            if let decodeToken, let emitPiece {
                let piece = decodeToken(token)
                if !piece.isEmpty {
                    if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        pendingProgressWhitespace += piece
                    } else {
                        emitPiece(token, pendingProgressWhitespace + piece)
                        pendingProgressWhitespace = ""
                    }
                }
            }
            return true
        }

        var logits = request.initialLogits
        var pending: MLXArray?

        while generated.count < request.tokenBudget {
            try checkCancellation?()

            let tokenArray = sampledTokenArray(
                logits: logits[0, -1, 0...],
                config: request.generationConfig,
                previousTokenIndices: repetitionHistory,
                banMask: request.banMask
            )
            repetitionHistory = appendingRepetitionHistory(
                repetitionHistory,
                token: tokenArray,
                config: request.generationConfig
            )
            logits = try stepForward(tokenArray.asType(.int32).reshaped(1, 1))
            asyncEval([logits, tokenArray])

            if let previous = pending {
                pending = nil
                guard confirm(previous) else {
                    return AutoregressiveDecodeResult(
                        generatedTokens: generated,
                        decodeSeconds: Date().timeIntervalSince(start)
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
            decodeSeconds: Date().timeIntervalSince(start)
        )
    }
}
