import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

public struct MiniMaxH3TransformerConfiguration: Hashable, Sendable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let refinerLayerCount: Int
    public let attentionHeadCount: Int
    public let attentionHeadDimension: Int
    public let feedForwardSize: Int
    public let videoLatentChannels: Int
    public let audioLatentChannels: Int
    public let patchSize: [Int]
    public let textDimension: Int
    public let timeFrequencyDimension: Int
    public let timeEmbeddingHiddenSize: Int
    public let timeEmbeddingDimension: Int
    public let ropeFrequencyCount: Int
    public let ropeTheta: Float
    public let normEpsilon: Float
    public let queryKeyNormEpsilon: Float

    public init(
        hiddenSize: Int = 5_376,
        layerCount: Int = 50,
        refinerLayerCount: Int = 2,
        attentionHeadCount: Int = 56,
        attentionHeadDimension: Int = 128,
        feedForwardSize: Int = 14_336,
        videoLatentChannels: Int = 24,
        audioLatentChannels: Int = 32,
        patchSize: [Int] = [1, 2, 2],
        textDimension: Int = 5_120,
        timeFrequencyDimension: Int = 256,
        timeEmbeddingHiddenSize: Int = 5_376,
        timeEmbeddingDimension: Int = 2_688,
        ropeFrequencyCount: Int = 16,
        ropeTheta: Float = 10_000,
        normEpsilon: Float = 1e-5,
        queryKeyNormEpsilon: Float = 1e-5
    ) {
        precondition(hiddenSize > 0 && layerCount > 0 && refinerLayerCount >= 0)
        precondition(attentionHeadCount > 0 && attentionHeadDimension > 0)
        precondition(patchSize.count == 3)
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.refinerLayerCount = refinerLayerCount
        self.attentionHeadCount = attentionHeadCount
        self.attentionHeadDimension = attentionHeadDimension
        self.feedForwardSize = feedForwardSize
        self.videoLatentChannels = videoLatentChannels
        self.audioLatentChannels = audioLatentChannels
        self.patchSize = patchSize
        self.textDimension = textDimension
        self.timeFrequencyDimension = timeFrequencyDimension
        self.timeEmbeddingHiddenSize = timeEmbeddingHiddenSize
        self.timeEmbeddingDimension = timeEmbeddingDimension
        self.ropeFrequencyCount = ropeFrequencyCount
        self.ropeTheta = ropeTheta
        self.normEpsilon = normEpsilon
        self.queryKeyNormEpsilon = queryKeyNormEpsilon
    }

    public init(_ configuration: MiniMaxH3Configuration) {
        self.init(
            hiddenSize: configuration.hiddenSize,
            layerCount: configuration.layerCount,
            refinerLayerCount: configuration.refinerLayerCount,
            attentionHeadCount: configuration.attentionHeadCount,
            attentionHeadDimension: configuration.attentionHeadDimension,
            feedForwardSize: configuration.feedForwardSize,
            videoLatentChannels: configuration.videoLatentChannels,
            audioLatentChannels: configuration.audioLatentChannels,
            patchSize: configuration.patchSize,
            textDimension: configuration.textDimension,
            timeFrequencyDimension: configuration.timeFrequencyDimension,
            timeEmbeddingHiddenSize: configuration.timeEmbeddingHiddenSize,
            timeEmbeddingDimension: configuration.timeEmbeddingDimension
        )
    }

    package var videoPatchDimension: Int {
        videoLatentChannels * patchSize.reduce(1, *)
    }
}
