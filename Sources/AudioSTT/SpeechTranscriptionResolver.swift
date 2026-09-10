import Foundation
import AudioCore
import MereRunCore
import MereRunExecution

/// Resolves model locations and backend policy without loading a generator.
public enum SpeechTranscriptionResolver {
    public static func resolve(
        request: ASRRequest,
        preferredBackend: ASRBackend,
        modelOverride: String? = nil,
        parakeetExecutionProvider: ParakeetExecutionProvider = .mlx,
        captureModelMetadata: Bool = false
    ) throws -> SpeechTranscriptionPlan {
        let normalizedOverride = normalized(
            modelOverride ?? parakeetExecutionProvider.bundledModelURL?.path
        )
        let inferredBackend = inferredBackendFromModelOverride(normalizedOverride)
        let effectivePreferredBackend: ASRBackend = {
            guard preferredBackend == .auto, normalizedOverride != nil else { return preferredBackend }
            return inferredBackend
        }()

        let qwenRoot = localQwenModelRoot()
        let qwenLocalAvailable = FileManager.default.fileExists(
            atPath: qwenRoot.appendingPathComponent("config.json").path
        )

        let parakeetRoot = localParakeetModelRoot()
        let parakeetLocalAvailable = FileManager.default.fileExists(
            atPath: parakeetRoot.appendingPathComponent("config.json").path
        )

        let qwenAvailable = true
        let parakeetAvailable = true
        let availability = ASRBackendAvailability(
            parakeetAvailable: parakeetAvailable,
            qwenAvailable: qwenAvailable
        )

        let parakeetCodes = loadParakeetLanguageCodesIfAvailable(
            modelOverride: normalizedOverride,
            localRoot: parakeetRoot,
            localAvailable: parakeetLocalAvailable
        )

        let decision = ASRBackendRouting.select(
            task: request.task,
            languageHint: request.language,
            preferredBackend: effectivePreferredBackend,
            availableBackends: availability,
            parakeetSupportedLanguageCodes: parakeetCodes
        )

        if case .coreML = parakeetExecutionProvider, decision.backend != .parakeet {
            throw SpeechTranscriptionIssue(
                "incompatible_execution_provider",
                "Core ML requires a Parakeet-compatible transcription request. "
                    + "This request selects Qwen (\(decision.reason))."
            )
        }

        let effectiveOverride = try compatibleModelOverride(
            normalizedOverride, inferredBackend: inferredBackend, selectedBackend: decision.backend
        )
        var plan: SpeechTranscriptionPlan
        switch decision.backend {
        case .qwen:
            plan = SpeechTranscriptionPlan(
                request: request, decision: decision,
                modelID: qwenModelId(modelOverride: effectiveOverride),
                modelPath: qwenModelPath(modelOverride: effectiveOverride, localRoot: qwenRoot, localAvailable: qwenLocalAvailable),
                provider: parakeetExecutionProvider
            )
        case .parakeet:
            plan = SpeechTranscriptionPlan(
                request: request, decision: decision,
                modelID: parakeetModelId(modelOverride: effectiveOverride),
                modelPath: parakeetModelPath(modelOverride: effectiveOverride, localRoot: parakeetRoot, localAvailable: parakeetLocalAvailable),
                provider: parakeetExecutionProvider
            )
        }
        try plan.validate()
        if captureModelMetadata { plan = try recordingPlan(plan) }
        return plan
    }

    /// Fingerprints local configuration and installation metadata, not tensor
    /// weights. Native model loading still owns checkpoint integrity checks.
    private static func recordingPlan(_ plan: SpeechTranscriptionPlan) throws -> SpeechTranscriptionPlan {
        guard let modelPath = plan.modelPath else { return plan }
        let root = URL(fileURLWithPath: modelPath)
        let modelRoot = plan.decision.backend == .parakeet ? ParakeetResources.resolveNestedIfNeeded(base: root) : root
        var files = ["config.json", "tokenizer.json", "tokenizer_config.json", MereRunModelManifest.filename]
            .map { modelRoot.appendingPathComponent($0) }
        if case .coreML(let artifactURL) = plan.provider {
            files.append(artifactURL.appendingPathComponent(ParakeetCoreMLManifest.filename))
        }
        return SpeechTranscriptionPlan(
            request: plan.request, decision: plan.decision, modelID: plan.modelID, modelPath: plan.modelPath,
            provider: plan.provider, modelMetadata: try files.map(RunFileSnapshot.capture)
        )
    }

    private static func compatibleModelOverride(
        _ modelOverride: String?, inferredBackend: ASRBackend, selectedBackend: ASRResolvedBackend
    ) throws -> String? {
        guard let modelOverride else { return nil }
        let matches = inferredBackend == .qwen && selectedBackend == .qwen
            || inferredBackend == .parakeet && selectedBackend == .parakeet
        guard !matches else { return modelOverride }
        if existingPath(from: modelOverride) != nil {
            throw SpeechTranscriptionIssue(
                "model_backend_mismatch",
                "The local model uses \(inferredBackend.rawValue), but this request requires \(selectedBackend.rawValue). "
                    + "Choose a compatible model path or omit the model override."
            )
        }
        // Task and language policy can switch a built-in default to another
        // backend. Its model identity must switch with it. Custom remote IDs
        // retain the caller's explicit backend choice.
        if ManagedModelCatalog.spec(for: modelOverride)?.category == .speechASR {
            return nil
        }
        return modelOverride
    }

    private static func localQwenModelRoot() -> URL {
        let fm = FileManager.default
        let base = MereRunModelPaths.resolveModelDir(Qwen3ASRResources.defaultModelId) { root in
            fm.fileExists(atPath: root.appendingPathComponent("config.json").path)
                || fm.fileExists(atPath: root.appendingPathComponent("\(Qwen3ASRResources.defaultModelId)/config.json").path)
        }
        let nested = base.appendingPathComponent(Qwen3ASRResources.defaultModelId, isDirectory: true)
        if fm.fileExists(atPath: nested.appendingPathComponent("config.json").path) {
            return nested
        }
        return base
    }

    private static func localParakeetModelRoot() -> URL {
        let fm = FileManager.default
        let base = MereRunModelPaths.resolveModelDir(ParakeetResources.defaultModelId) { root in
            fm.fileExists(atPath: root.appendingPathComponent("config.json").path)
                || fm.fileExists(atPath: root.appendingPathComponent("\(ParakeetResources.defaultModelId)/config.json").path)
        }
        let nested = base.appendingPathComponent(ParakeetResources.defaultModelId, isDirectory: true)
        if fm.fileExists(atPath: nested.appendingPathComponent("config.json").path) {
            return nested
        }
        return base
    }

    private static func loadParakeetLanguageCodesIfAvailable(
        modelOverride: String?,
        localRoot: URL,
        localAvailable: Bool
    ) -> Set<String>? {
        if let overridePath = existingPath(from: modelOverride) {
            let resolved = ParakeetResources.resolveNestedIfNeeded(base: overridePath)
            let configURL = resolved.appendingPathComponent("config.json")
            if let config = try? ParakeetModelConfig.load(from: configURL) {
                return Set(config.supportedLanguageCodes.map { $0.lowercased() })
            }
        }

        guard localAvailable else { return nil }
        let localConfig = localRoot.appendingPathComponent("config.json")
        guard let config = try? ParakeetModelConfig.load(from: localConfig) else { return nil }
        return Set(config.supportedLanguageCodes.map { $0.lowercased() })
    }

    private static func qwenModelPath(
        modelOverride: String?,
        localRoot: URL,
        localAvailable: Bool
    ) -> String? {
        if let overridePath = existingPath(from: modelOverride) {
            return overridePath.path
        }
        if localAvailable, modelOverride == nil || modelOverride == Qwen3ASRResources.defaultModelId {
            return localRoot.path
        }
        return nil
    }

    private static func parakeetModelPath(
        modelOverride: String?,
        localRoot: URL,
        localAvailable: Bool
    ) -> String? {
        if let overridePath = existingPath(from: modelOverride) {
            return overridePath.path
        }
        if localAvailable, modelOverride == nil || modelOverride == ParakeetResources.defaultModelId {
            return localRoot.path
        }
        return nil
    }

    private static func qwenModelId(modelOverride: String?) -> String {
        if let modelOverride, existingPath(from: modelOverride) == nil {
            return modelOverride
        }
        return Qwen3ASRResources.defaultModelId
    }

    private static func parakeetModelId(modelOverride: String?) -> String {
        if let modelOverride, existingPath(from: modelOverride) == nil {
            return modelOverride
        }
        return ParakeetResources.defaultModelId
    }

    private static func inferredBackendFromModelOverride(_ modelOverride: String?) -> ASRBackend {
        guard let modelOverride else { return .auto }
        if let existingPath = existingPath(from: modelOverride) {
            let resolved = ParakeetResources.resolveNestedIfNeeded(base: existingPath)
            if FileManager.default.fileExists(
                atPath: resolved.appendingPathComponent("config.json").path
            ),
                let config = try? ParakeetModelConfig.load(
                    from: resolved.appendingPathComponent("config.json")
                ),
                !config.target.isEmpty
            {
                return .parakeet
            }
            return .qwen
        }

        let lowered = modelOverride.lowercased()
        if lowered.contains("parakeet") {
            return .parakeet
        }
        return .qwen
    }

    private static func existingPath(from value: String?) -> URL? {
        guard let value else { return nil }
        let url = URL(fileURLWithPath: value).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        return nil
    }

    private static func normalized(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
