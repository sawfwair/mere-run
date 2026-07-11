import Foundation
import AudioCore
import AudioSTT
import MereRunCore

struct CLIASRExecutionResult {
    let result: ASRResult
    let decision: ASRBackendDecision
    let backend: ASRResolvedBackend
}

protocol CLIASRTranscriptionExecutor: Sendable {
    func transcribeQwen(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult

    func transcribeParakeet(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult
}

enum CLIASRRouting {
    static func transcribe(
        request: ASRRequest,
        preferredBackend: ASRBackend,
        modelOverride: String? = nil,
        progressHandler: (@Sendable (ASRProgress) -> Void)? = nil,
        executor: (any CLIASRTranscriptionExecutor)? = nil
    ) async throws -> CLIASRExecutionResult {
        let normalizedOverride = normalized(modelOverride)
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

        switch decision.backend {
        case .qwen:
            let modelID = qwenModelId(modelOverride: normalizedOverride)
            let modelPath = qwenModelPath(
                modelOverride: normalizedOverride,
                localRoot: qwenRoot,
                localAvailable: qwenLocalAvailable
            )
            let result: ASRResult
            if let executor {
                result = try await executor.transcribeQwen(
                    request: request,
                    modelID: modelID,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            } else {
                let generator = Qwen3ASRGenerator(modelId: modelID)
                result = try await generator.transcribe(
                    request,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            }
            return CLIASRExecutionResult(result: result, decision: decision, backend: .qwen)

        case .parakeet:
            let modelID = parakeetModelId(modelOverride: normalizedOverride)
            let modelPath = parakeetModelPath(
                modelOverride: normalizedOverride,
                localRoot: parakeetRoot,
                localAvailable: parakeetLocalAvailable
            )
            let result: ASRResult
            if let executor {
                result = try await executor.transcribeParakeet(
                    request: request,
                    modelID: modelID,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            } else {
                let generator = ParakeetGenerator(modelId: modelID)
                result = try await generator.transcribe(
                    request,
                    modelPath: modelPath,
                    progressHandler: progressHandler
                )
            }
            return CLIASRExecutionResult(result: result, decision: decision, backend: .parakeet)
        }
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
        if localAvailable {
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
        if localAvailable {
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
