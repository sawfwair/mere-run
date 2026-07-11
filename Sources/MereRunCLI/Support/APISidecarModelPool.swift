import Foundation
import AudioCore
import AudioSTT
import AudioTTS
import MereRunCore

/// A bounded, exclusive resident slot for mutable inference runtimes.
///
/// The slot keeps exactly one value resident. Requests for the same key reuse
/// it, while a different key unloads the previous value before constructing the
/// replacement. The execution admission remains held for the complete
/// operation because model generators contain mutable caches and are not
/// generally safe to re-enter while an inference is suspended.
actor APISidecarResidentSlot<Key: Equatable & Sendable, Value: Sendable> {
    private struct Resident: Sendable {
        let key: Key
        let value: Value
    }

    private let execution = RuntimeRequestAdmission(maxActiveRequests: 1)
    private var resident: Resident?

    func withValue<Result: Sendable>(
        for key: Key,
        make: @Sendable () async throws -> Value,
        unload: @Sendable (Value) async -> Void,
        operation: @Sendable (Value) async throws -> Result
    ) async throws -> Result {
        let lease = try await execution.acquire()
        do {
            let value = try await value(for: key, make: make, unload: unload)
            let result = try await operation(value)
            await lease.release()
            return result
        } catch {
            await lease.release()
            throw error
        }
    }

    func unloadResident(
        using unload: @Sendable (Value) async -> Void
    ) async throws {
        let lease = try await execution.acquire()
        if let resident {
            self.resident = nil
            await unload(resident.value)
        }
        await lease.release()
    }

    func residentKey() -> Key? {
        resident?.key
    }

    private func value(
        for key: Key,
        make: @Sendable () async throws -> Value,
        unload: @Sendable (Value) async -> Void
    ) async throws -> Value {
        if let resident, resident.key == key {
            return resident.value
        }
        if let resident {
            self.resident = nil
            await unload(resident.value)
        }
        let value = try await make()
        resident = Resident(key: key, value: value)
        return value
    }
}

enum APISidecarImageKind: String, Sendable {
    case flux2Klein
    case zImageTurbo
    case hiDreamO1
    case krea2
    case ideogram4
    case qwenImageEdit
}

private struct APISidecarImageKey: Equatable, Sendable {
    let kind: APISidecarImageKind
    let modelSpec: String
}

private struct APISidecarSpeechKey: Equatable, Sendable {
    let modelID: String
    let modelPath: String?
}

private struct APISidecarASRKey: Equatable, Sendable {
    let backend: ASRResolvedBackend
    let modelID: String
    let modelPath: String?
}

/// Non-actor image generators are safe here because the resident slot holds an
/// exclusive lease for their entire lifetime of use.
private enum APISidecarImageGenerator: @unchecked Sendable {
    case flux2Klein(Flux2KleinGenerator)
    case zImageTurbo(ZImageTurboGenerator)
    case hiDreamO1(HiDreamO1Generator)
    case krea2(Krea2Generator)
    case ideogram4(Ideogram4Generator)
    case qwenImageEdit(QwenImageEditGenerator)
}

private enum APISidecarASRGenerator: Sendable {
    case qwen(Qwen3ASRGenerator)
    case parakeet(ParakeetGenerator)
}

/// Resident runtimes for the non-chat OpenAI-compatible endpoints.
///
/// One image runtime, one TTS runtime, and one ASR runtime may be resident at a
/// time. This bounds memory even when callers provide different local model
/// paths, while repeated requests avoid model reloads.
struct APISidecarModelPool: Sendable, CLIASRTranscriptionExecutor {
    private let imageSlot = APISidecarResidentSlot<APISidecarImageKey, APISidecarImageGenerator>()
    private let speechSlot = APISidecarResidentSlot<APISidecarSpeechKey, Qwen3TTSGenerator>()
    private let asrSlot = APISidecarResidentSlot<APISidecarASRKey, APISidecarASRGenerator>()

    func generateImage(
        kind: APISidecarImageKind,
        modelSpec: String,
        request: GenerationRequest
    ) async throws -> GenerationResult {
        let key = APISidecarImageKey(kind: kind, modelSpec: normalizedPath(modelSpec))
        return try await imageSlot.withValue(
            for: key,
            make: {
                switch kind {
                case .flux2Klein:
                    return .flux2Klein(Flux2KleinGenerator())
                case .zImageTurbo:
                    return .zImageTurbo(ZImageTurboGenerator())
                case .hiDreamO1:
                    return .hiDreamO1(HiDreamO1Generator())
                case .krea2:
                    return .krea2(Krea2Generator())
                case .ideogram4:
                    return .ideogram4(Ideogram4Generator())
                case .qwenImageEdit:
                    return .qwenImageEdit(QwenImageEditGenerator())
                }
            },
            unload: { generator in
                switch generator {
                case .flux2Klein(let generator):
                    await generator.unload()
                case .zImageTurbo(let generator):
                    await generator.unload()
                case .hiDreamO1(let generator):
                    generator.unload()
                case .krea2(let generator):
                    generator.unload()
                case .ideogram4(let generator):
                    generator.unload()
                case .qwenImageEdit(let generator):
                    await generator.clearCache()
                }
            },
            operation: { generator in
                switch generator {
                case .flux2Klein(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                case .zImageTurbo(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                case .hiDreamO1(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                case .krea2(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                case .ideogram4(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                case .qwenImageEdit(let generator):
                    return try await generator.generate(request, progressHandler: nil)
                }
            }
        )
    }

    func synthesizeSpeech(
        modelID: String,
        modelPath: String?,
        request: TTSRequest
    ) async throws -> TTSResult {
        let key = APISidecarSpeechKey(
            modelID: modelID,
            modelPath: modelPath.map(normalizedPath)
        )
        return try await speechSlot.withValue(
            for: key,
            make: { Qwen3TTSGenerator(modelId: modelID) },
            unload: { generator in await generator.unload() },
            operation: { generator in
                try await generator.generate(
                    request,
                    modelPath: modelPath,
                    progressHandler: nil
                )
            }
        )
    }

    func transcribeQwen(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(
            request: request,
            key: APISidecarASRKey(
                backend: .qwen,
                modelID: modelID,
                modelPath: modelPath.map(normalizedPath)
            ),
            progressHandler: progressHandler
        )
    }

    func transcribeParakeet(
        request: ASRRequest,
        modelID: String,
        modelPath: String?,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await transcribe(
            request: request,
            key: APISidecarASRKey(
                backend: .parakeet,
                modelID: modelID,
                modelPath: modelPath.map(normalizedPath)
            ),
            progressHandler: progressHandler
        )
    }

    private func transcribe(
        request: ASRRequest,
        key: APISidecarASRKey,
        progressHandler: (@Sendable (ASRProgress) -> Void)?
    ) async throws -> ASRResult {
        try await asrSlot.withValue(
            for: key,
            make: {
                switch key.backend {
                case .qwen:
                    return .qwen(Qwen3ASRGenerator(modelId: key.modelID))
                case .parakeet:
                    return .parakeet(ParakeetGenerator(modelId: key.modelID))
                }
            },
            unload: { generator in
                switch generator {
                case .qwen(let generator):
                    await generator.unload()
                case .parakeet(let generator):
                    await generator.unload()
                }
            },
            operation: { generator in
                switch generator {
                case .qwen(let generator):
                    return try await generator.transcribe(
                        request,
                        modelPath: key.modelPath,
                        progressHandler: progressHandler
                    )
                case .parakeet(let generator):
                    return try await generator.transcribe(
                        request,
                        modelPath: key.modelPath,
                        progressHandler: progressHandler
                    )
                }
            }
        )
    }

    private func normalizedPath(_ value: String) -> String {
        URL(fileURLWithPath: value).standardizedFileURL.path
    }
}
