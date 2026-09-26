import AudioCore
import Foundation
import MereRunCore

/// Resolves the common CLI/API selector without loading or downloading a model.
public struct SpeechSynthesisModelSelection: Sendable, Hashable {
    public enum Backend: Sendable, Hashable { case qwen3, breeze }
    public let modelID: String
    public let modelPath: String?
    public let backend: Backend

    public static func resolve(_ selector: String, fileManager: FileManager = .default) throws -> Self {
        let normalized = selector.trimmingCharacters(in: .whitespacesAndNewlines)
        let selected = normalized.isEmpty ? Qwen3TTSResources.defaultModelId : normalized
        let path = URL(fileURLWithPath: selected).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: path.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else { throw Qwen3TTSError.unsupportedModelId(selected) }
            let configURL = path.appending(path: "config.json")
            let config = try JSONDecoder().decode(ModelTypeProbe.self, from: Data(contentsOf: configURL))
            let backend: Backend = config.modelType == "breeze" ? .breeze : .qwen3
            return Self(modelID: backend == .breeze ? ManagedModelID.breezeTTS2.rawValue
                        : Qwen3TTSResources.defaultModelId, modelPath: path.path, backend: backend)
        }
        guard let spec = ManagedModelCatalog.spec(for: selected),
              spec.category == .speechTTS,
              Qwen3TTSResources.supportedModelIds.contains(spec.id) || spec.id == ManagedModelID.breezeTTS2.rawValue else {
            throw Qwen3TTSError.unsupportedModelId(selected)
        }
        return Self(modelID: spec.id, modelPath: nil,
                    backend: spec.id == ManagedModelID.breezeTTS2.rawValue ? .breeze : .qwen3)
    }

    public func makeGenerator() -> SpeechSynthesisGenerator {
        switch backend {
        case .qwen3: .qwen3(Qwen3TTSGenerator(modelId: modelID))
        case .breeze: .breeze(BreezeTTSGenerator(modelID: modelID))
        }
    }

    public func executor(using generator: SpeechSynthesisGenerator) -> any SpeechSynthesisExecutor {
        switch generator {
        case .qwen3(let value): Qwen3TTSSynthesisExecutor(generator: value, modelPath: modelPath)
        case .breeze(let value): BreezeTTSSynthesisExecutor(generator: value, modelPath: modelPath)
        }
    }
}

private struct ModelTypeProbe: Decodable {
    let modelType: String?
    enum CodingKeys: String, CodingKey { case modelType = "model_type" }
}

public enum SpeechSynthesisGenerator: Sendable {
    case qwen3(Qwen3TTSGenerator)
    case breeze(BreezeTTSGenerator)

    public func unload() async {
        switch self {
        case .qwen3(let value): await value.unload()
        case .breeze(let value): await value.unload()
        }
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
