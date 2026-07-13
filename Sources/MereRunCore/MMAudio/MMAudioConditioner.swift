import Foundation
import MediaIO
import MLX

public enum MMAudioConditioningError: LocalizedError {
    case emptyVideo
    case invalidFeatureShape(name: String, shape: [Int])

    public var errorDescription: String? {
        switch self {
        case .emptyVideo:
            "The input video contains no decodable frames."
        case .invalidFeatureShape(let name, let shape):
            "Invalid MMAudio \(name) feature shape: \(shape)."
        }
    }
}

public struct MMAudioConditioningResult {
    public let positive: MMAudioConditionFeatures
    public let negativeText: MLXArray

    public init(positive: MMAudioConditionFeatures, negativeText: MLXArray) {
        self.positive = positive
        self.negativeText = negativeText
    }
}

public final class MMAudioConditioner {
    private let resources: MMAudioModelResources
    private let clip: MMAudioCLIPConditioner

    private init(resources: MMAudioModelResources, clip: MMAudioCLIPConditioner) {
        self.resources = resources
        self.clip = clip
    }

    public static func load(resources: MMAudioModelResources) async throws -> MMAudioConditioner {
        MMAudioConditioner(
            resources: resources,
            clip: try await MMAudioCLIPConditioner.load(resources: resources)
        )
    }

    public func textConditions(
        prompt: String,
        negativePrompt: String,
        config: MMAudioGenerationConfig
    ) -> MMAudioConditioningResult {
        let text = clip.encodeText([prompt])
        let negative = clip.encodeText([negativePrompt])
        return MMAudioConditioningResult(
            positive: MMAudioConditionFeatures(
                clip: MLXArray.zeros([
                    1, config.clipSequenceLength, MMAudioResources.clipDimension,
                ], dtype: text.dtype),
                sync: MLXArray.zeros([
                    1, config.syncSequenceLength, MMAudioResources.syncDimension,
                ], dtype: text.dtype),
                text: text
            ),
            negativeText: negative
        )
    }

    public func videoConditions(
        prompt: String,
        negativePrompt: String,
        videoURL: URL,
        config: MMAudioGenerationConfig,
        clipBatchSize: Int = 4,
        syncBatchSize: Int = 1
    ) throws -> MMAudioConditioningResult {
        let text = clip.encodeText([prompt])
        let negative = clip.encodeText([negativePrompt])
        let clipFeatures = try extractCLIPVideoFeatures(
            videoURL: videoURL,
            durationSeconds: config.durationSeconds,
            batchSize: clipBatchSize
        )
        let synchformer = try WooshSynchformer(
            resources: WooshSynchformerResources(rootURL: resources.rootURL)
        )
        let syncFeatures = try synchformer.extractMMAudioFeatures(
            videoURL: videoURL,
            durationSeconds: config.durationSeconds,
            segmentBatchSize: syncBatchSize
        )
        guard clipFeatures.shape == [1, config.clipSequenceLength, MMAudioResources.clipDimension] else {
            throw MMAudioConditioningError.invalidFeatureShape(name: "CLIP", shape: clipFeatures.shape)
        }
        guard syncFeatures.shape == [1, config.syncSequenceLength, MMAudioResources.syncDimension] else {
            throw MMAudioConditioningError.invalidFeatureShape(name: "Synchformer", shape: syncFeatures.shape)
        }
        return MMAudioConditioningResult(
            positive: MMAudioConditionFeatures(
                clip: clipFeatures,
                sync: syncFeatures,
                text: text
            ),
            negativeText: negative
        )
    }

    private func extractCLIPVideoFeatures(
        videoURL: URL,
        durationSeconds: Float,
        batchSize: Int
    ) throws -> MLXArray {
        precondition(batchSize > 0)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-mmaudio-clip-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let sequence = try MediaVideoIO.extractFrames(from: videoURL, into: directory)
        guard !sequence.frameURLs.isEmpty else {
            throw MMAudioConditioningError.emptyVideo
        }
        let targetCount = max(1, Int(durationSeconds * MMAudioResources.clipFramesPerSecond))
        let indices = MMAudioVideoPreprocessor.sampledFrameIndices(
            sourceFrameRate: max(1, sequence.fps),
            sourceFrameCount: sequence.frameURLs.count,
            targetFrameRate: Double(MMAudioResources.clipFramesPerSecond),
            targetFrameCount: targetCount
        )
        var featureBatches: [MLXArray] = []
        var start = 0
        while start < indices.count {
            let end = min(indices.count, start + batchSize)
            let frames = try indices[start..<end].map { index in
                MMAudioVideoPreprocessor.normalizedCLIPFrame(
                    try MediaImageIO.decode(sequence.frameURLs[index])
                )
            }
            let batch = MLX.concatenated(frames, axis: 0)
            let encoded = clip.encodeImages(batch)
            MLX.eval(encoded)
            featureBatches.append(encoded)
            start = end
        }
        let features = featureBatches.count == 1
            ? featureBatches[0]
            : MLX.concatenated(featureBatches, axis: 0)
        return features.expandedDimensions(axis: 0)
    }

}
