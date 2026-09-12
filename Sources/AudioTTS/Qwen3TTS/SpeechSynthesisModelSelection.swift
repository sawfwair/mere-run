import AudioCore
import Foundation
import MereRunCore

/// Resolves the common CLI/API selector without loading or downloading a model.
public struct SpeechSynthesisModelSelection: Sendable, Hashable {
    public let modelID: String
    public let modelPath: String?

    public static func resolve(_ selector: String, fileManager: FileManager = .default) throws -> Self {
        let normalized = selector.trimmingCharacters(in: .whitespacesAndNewlines)
        let selected = normalized.isEmpty ? Qwen3TTSResources.defaultModelId : normalized
        let path = URL(fileURLWithPath: selected).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: path.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else { throw Qwen3TTSError.unsupportedModelId(selected) }
            return Self(modelID: Qwen3TTSResources.defaultModelId, modelPath: path.path)
        }
        guard let spec = ManagedModelCatalog.spec(for: selected),
              spec.category == .speechTTS, Qwen3TTSResources.supportedModelIds.contains(spec.id) else {
            throw Qwen3TTSError.unsupportedModelId(selected)
        }
        return Self(modelID: spec.id, modelPath: nil)
    }

    public func makeGenerator() -> Qwen3TTSGenerator { Qwen3TTSGenerator(modelId: modelID) }

    public func executor(using generator: Qwen3TTSGenerator) -> Qwen3TTSSynthesisExecutor {
        Qwen3TTSSynthesisExecutor(generator: generator, modelPath: modelPath)
    }
}

/// Adapts a caller-owned Qwen generator to the shared synthesis operation.
/// The API keeps its resident generator; the CLI unloads its generator after use.
public struct Qwen3TTSSynthesisExecutor: SpeechSynthesisExecutor {
    private let generator: Qwen3TTSGenerator
    private let modelPath: String?

    public init(generator: Qwen3TTSGenerator, modelPath: String?) {
        self.generator = generator
        self.modelPath = modelPath
    }

    public func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> AudioWaveform {
        try await generator.generateAudio(request, modelPath: modelPath, progressHandler: progressHandler)
    }

    public func generateStream(
        _ request: TTSRequest,
        options: TTSStreamingOptions
    ) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
        generator.generateStream(request, options: options, modelPath: modelPath)
    }
}
