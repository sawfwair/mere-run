import Foundation

extension MiniMaxH3AdaLNCache {
    /// Managed pulls verify bytes and tensor geometry without creating MLX arrays.
    /// The immutable pin binds the sigma values to the published sampler schedule;
    /// inference still loads and checks that schedule through `load`.
    static func validatePinnedArtifact(
        in rootURL: URL,
        pin: ModelArtifactPin,
        configuration: MiniMaxH3TransformerConfiguration,
        pointCount: Int,
        sourceIdentity: String,
        fileManager: FileManager = .default
    ) throws {
        let url = try pin.verify(in: rootURL, fileManager: fileManager)
        let metadata = try SafetensorsStreamingLoader.fileMetadata(url: url)
        guard metadata["schema_version"] == schemaVersion else {
            throw MiniMaxH3AdaLNCacheError.incompatible("unsupported schema version")
        }
        guard metadata["source_identity"] == sourceIdentity else {
            throw MiniMaxH3AdaLNCacheError.incompatible("transformer artifact changed")
        }

        let tensors = try SafetensorsStreamingLoader.metadata(url: url)
        let stepCount = pointCount - 1
        guard pointCount >= 2,
              tensors["video_sigmas"]?.shape == [pointCount],
              tensors["audio_sigmas"]?.shape == [pointCount],
              tensors["time_embeddings"]?.shape == [stepCount, 3, configuration.timeEmbeddingDimension],
              tensors["final_modulations"]?.shape == [stepCount, 3, 2 * configuration.hiddenSize],
              (0..<configuration.layerCount).allSatisfy({ index in
                  tensors["blocks.\(index).modulations"]?.shape
                      == [stepCount, 9, 6 * configuration.hiddenSize]
              }) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("source tensor geometry does not match")
        }
    }
}
