import Foundation
import MereRunQwenModel
import MereRunTensor
import MLX
import MLXNN

extension Q35Generator {
    static func automaticDFlash2Enabled(
        modelID: String, hardwareSupported: Bool = QuantizedVerificationBackend.m5Supported,
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Bool {
        modelID == Q35Resources.q38TwentySevenB4BitModelId && hardwareSupported
            && physicalMemory >= 64 * 1_024 * 1_024 * 1_024
    }

    static func dflash2Root(
        targetRoot: URL, configuredPath: String?, enabled: Bool, automatic: Bool,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        let disabled = ["0", "false", "no", "off"].contains(environment["MERERUN_Q35_DFLASH2"]?
            .trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "")
        guard enabled, !disabled else { return nil }
        if let configuredPath { return URL(fileURLWithPath: configuredPath, isDirectory: true) }
        guard automatic else { return nil }
        let root = targetRoot.appendingPathComponent(Q35Resources.q38DFlash2ComponentPath, isDirectory: true)
        return ["config.json", "model.safetensors"].allSatisfy {
            FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path)
        } ? root : nil
    }

    func dflash2Configuration(target: Q35Config, targetRoot: URL) throws -> (config: Q35DFlash2Config, root: URL)? {
        guard let root = Self.dflash2Root(
            targetRoot: targetRoot, configuredPath: dflash2ModelPath, enabled: dflash2Enabled,
            automatic: Self.automaticDFlash2Enabled(modelID: modelId)
        ) else { return nil }
        guard modelId == Q35Resources.q38TwentySevenB4BitModelId else {
            throw Q35Error.generationFailed("DFlash 2 requires vision-chat-q38-27b-4bit.")
        }
        let config = try JSONDecoder().decode(
            Q35DFlash2Config.self, from: Data(contentsOf: root.appendingPathComponent("config.json"))
        )
        try config.validate(target: target)
        return (config, root)
    }

    func loadDFlash2(
        configuration: (config: Q35DFlash2Config, root: URL)?, progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws -> Q35DFlash2Model? {
        guard let configuration else { return nil }
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen DFlash 2 draft"))
        let draft = Q35DFlash2Model(config: configuration.config)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: configuration.root.appendingPathComponent("model.safetensors"),
            to: draft, dtype: .bfloat16, verify: .all,
            mapper: { key, value in [(Q35DFlash2Model.weightKey(key), value)] }
        )
        // Match the compact affine Q4/group-64 projection layout used by Splash.
        // Norms, convolution bases, and selector codebooks retain their BF16 values.
        draft.quantizeProjections()
        draft.prepareDrafting()
        return draft
    }

    static func dflash2Eligible(
        generationConfig: GenerationConfig, tokenBudget: Int, hasImages: Bool, hasTools: Bool,
        jsonConstrained: Bool, logprobCapture: ChatLogprobCapture, continuousBatchingEnabled: Bool
    ) -> Bool {
        tokenBudget > 1 && (generationConfig.temperature > 0
            || (generationConfig.temperature == 0 && !generationConfig.hasActivePenalties
                && generationConfig.bannedTokens.isEmpty))
            && !hasImages && !hasTools && !jsonConstrained && !logprobCapture.isEnabled
            && !continuousBatchingEnabled
    }

    func prefillDFlash2(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        promptTokens: [Int], cache: [Q35LayerCache?],
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35PrefillOutput {
        let capture = Set(draft.config.dflash.targetLayerIDs)
        let retainedStart = max(0, promptTokens.count - draft.config.slidingWindow)
        var processed = 0
        var last: Q35ForwardOutput?
        while processed < promptTokens.count {
            try Task.checkCancellation()
            let end = min(promptTokens.count, processed + Self.prefillChunkSize(
                modelId: modelId, activeRequestCount: activeChatRequestCount
            ))
            let tokens = MLXArray(promptTokens[processed..<end].map(Int32.init)).reshaped(1, end - processed)
            let output = model.forwardPrefill(
                tokens, cache: cache, captureLayerIndices: end > retainedStart ? capture : []
            )
            if end > retainedStart {
                let start = max(0, retainedStart - processed)
                draft.appendTargetContext(
                    output.capturedHiddenStates.mapValues { $0[0..., start..., 0...] }, cache: draftCache
                )
            }
            // Materialize the whole chunk in one submission. Individual draft
            // layers only build the graph; they do not synchronize separately.
            MLX.eval([output.logits] + draftCache.flatMap(\.state))
            last = output
            processed = end
            progressHandler?(ChatProgress(stage: .encoding, message: "Prefilling \(end)/\(promptTokens.count) tokens"))
            clearMLXCacheUnderPressureIfNeeded()
            await Task.yield()
        }
        guard let last else { throw Q35Error.generationFailed("Prompt is empty after tokenization.") }
        return Q35PrefillOutput(logits: last.logits, hidden: last.hidden, mtpHistoryHidden: nil, mtpSession: nil)
    }

    func decodeDFlash2(
        model: Q35Model, draft: Q35DFlash2Model, draftCache: [Q35DFlash2Cache],
        initialLogits: MLXArray, layerCaches: [Q35LayerCache?], eosSet: Set<Int>, tokenBudget: Int,
        generationConfig: GenerationConfig, promptTokens: [Int],
        decodeToken: (Int) -> String, progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws -> Q35BatchedDecodeResult {
        let started = Date()
        var firstTokenSeconds: Double?
        var generated: [Int] = []
        var pendingWhitespace = ""
        let profile = Q35MTPProfile.make()
        let observer: ((Q35DFlash2Event) -> Void)? = profile.map { profile in
            { event in
                switch event {
                case let .beginRound(depth): profile.beginRound(depth: depth)
                case let .drafted(tokens): profile.finishDraft(tokens: tokens)
                case .verificationSubmitted: profile.verificationSubmitted()
                case .verificationCompleted: profile.verificationCompleted()
                case let .accepted(count): profile.accepted(count)
                case let .finished(state):
                    MLX.eval(state)
                    profile.finishRound()
                }
            }
        }
        let banMask = tokenBanMask(
            vocabularySize: initialLogits.dim(-1), dtype: initialLogits.dtype, tokens: generationConfig.bannedTokens
        )
        let sampling: Q35DFlash2Sampling? = generationConfig.temperature > 0 ? Q35DFlash2Sampling(
            temperature: generationConfig.temperature,
            history: generationConfig.hasActivePenalties ? promptTokens : nil,
            distribution: { logits, history in
                samplingDistribution(
                    logits: logits, config: generationConfig, previousTokenIndices: history, banMask: banMask
                )
            }
        ) : nil
        func emit(_ token: Int) {
            if firstTokenSeconds == nil { firstTokenSeconds = Date().timeIntervalSince(started) }
            generated.append(token)
            let piece = decodeToken(token)
            if piece.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                pendingWhitespace += piece
            } else if !piece.isEmpty {
                progressHandler?(ChatProgress(stage: .generating, message: pendingWhitespace + piece))
                pendingWhitespace = ""
            }
        }
        let stats = try Q35DFlash2Decoder.decode(
            model: model, draft: draft, draftCache: draftCache,
            initialLogits: initialLogits, layerCaches: layerCaches, eosSet: eosSet, tokenBudget: tokenBudget,
            sampling: sampling,
            checkCancellation: { try Task.checkCancellation() },
            observer: observer,
            emit: emit
        )
        return Q35BatchedDecodeResult(
            generatedTokens: generated, decodeSeconds: Date().timeIntervalSince(started),
            firstTokenSeconds: firstTokenSeconds,
            acceleration: ChatAccelerationDiagnostics(
                route: "dflash2",
                draftModel: Q35Resources.q38DFlash2UpstreamRepoId,
                rounds: stats.rounds, draftedTokens: stats.draftedTokens, acceptedDraftTokens: stats.acceptedTokens,
                speculationProfile: profile?.result
            )
        )
    }

}
