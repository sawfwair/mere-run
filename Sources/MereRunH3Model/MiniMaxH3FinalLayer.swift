import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

final class MiniMaxH3FinalLayer: Module {
    @ModuleInfo(key: "norm") package var norm: RMSNorm
    @ModuleInfo(key: "adaln_proj") package var adaLN: MiniMaxH3AdaLNProjection?
    @ModuleInfo(key: "video_out") package var videoOutput: Linear
    @ModuleInfo(key: "audio_out") package var audioOutput: Linear

    package init(configuration: MiniMaxH3TransformerConfiguration, includeAdaLN: Bool) {
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.normEpsilon
        )
        self._adaLN.wrappedValue = includeAdaLN
            ? MiniMaxH3AdaLNProjection(
                inputDimension: configuration.timeEmbeddingDimension,
                hiddenSize: configuration.hiddenSize,
                partCount: 2,
                modalityCount: 1
            )
            : nil
        self._videoOutput.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.videoPatchDimension,
            bias: true
        )
        self._audioOutput.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.audioLatentChannels,
            bias: true
        )
    }

    package func discardAdaLNWeights() {
        guard adaLN != nil else { return }
        update(modules: ModuleChildren.unflattened([
            ("adaln_proj", MiniMaxH3AdaLNProjection(discarded: ())),
        ]))
    }

    package func callAsFunction(
        _ value: MLXArray,
        timeEmbedding: MLXArray,
        videoRows: Range<Int>,
        videoTimeIndex: Int32,
        audioRows: Range<Int>,
        audioTimeIndex: Int32,
        cachedModulation: MLXArray?
    ) -> MiniMaxH3TransformerOutput {
        let modulation: [MLXArray]
        if let cachedModulation {
            modulation = MLX.split(cachedModulation, parts: 2, axis: -1)
        } else {
            guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN cache is required") }
            modulation = adaLN(timeEmbedding)
        }
        let normalizedVideo = norm(value[0..., videoRows, 0...])
        let videoIndex = MLXArray([videoTimeIndex])
        let videoHidden = normalizedVideo * (1 + MLX.take(modulation[1], videoIndex, axis: 0))
            + MLX.take(modulation[0], videoIndex, axis: 0)
        let normalizedAudio = norm(value[0..., audioRows, 0...])
        let audioIndex = MLXArray([audioTimeIndex])
        let audioHidden = normalizedAudio * (1 + MLX.take(modulation[1], audioIndex, axis: 0))
            + MLX.take(modulation[0], audioIndex, axis: 0)
        return MiniMaxH3TransformerOutput(
            videoVelocityRows: miniMaxH3Linear(videoOutput, videoHidden),
            audioVelocityRows: miniMaxH3Linear(audioOutput, audioHidden)
        )
    }

    package func precomputeModulation(timeEmbedding: MLXArray) -> MLXArray {
        guard let adaLN else { preconditionFailure("MiniMax-H3 AdaLN weights are not loaded") }
        return adaLN.concatenated(timeEmbedding)
    }
}
