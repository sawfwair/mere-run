#if canImport(CoreML) && os(macOS) && arch(arm64)
import Foundation
import MereRunCoreMLModel

/// Caller-owned warm embedding runtime. Retain for repeated requests and release
/// when idle, or integrate its lifetime with MereRunResidency in a serving owner.
/// This session does not acquire admission or retain a hidden global model cache.
@available(macOS 15.0, *)
public final class EmbeddingGemma2CoreMLSession: Sendable {
    private let model: EmbeddingGemma2

    public init(root: URL) throws {
        let missing = EmbeddingGemma2CoreMLCatalog.missingFiles(root: root)
        guard missing.isEmpty else {
            throw EmbeddingGemma2Error.invalidConfiguration("Missing Core ML resources:\n" + missing.map(\.path).joined(separator: "\n"))
        }
        try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root)
        model = try EmbeddingGemma2(bundle: root)
    }

    public func embed(inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task = .raw,
                      title: String? = nil, dimensions: Int = 768, maxTokens: Int? = nil) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        try EmbeddingGemma2CoreMLRuntime.validateInputs(inputs, task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        return try await EmbeddingGemma2CoreMLRuntime.execute(model: model, inputs: inputs, task: task,
                                                            title: title, dimensions: dimensions, maxTokens: maxTokens)
    }
}
#endif
