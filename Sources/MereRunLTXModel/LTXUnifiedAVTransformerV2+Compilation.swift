import MLX
import MLXNN

extension LTXUnifiedAVTransformerV2 {
    func compiledBlockForward(
        block: LTXUnifiedAVTransformerV2Block,
        videoHidden: MLXArray,
        audioHidden: MLXArray,
        videoAdalnParams: MLXArray,
        audioAdalnParams: MLXArray,
        videoPromptAdalnParams: MLXArray,
        audioPromptAdalnParams: MLXArray,
        avCaVideoParams: MLXArray,
        avCaAudioParams: MLXArray,
        avCaA2VGateParams: MLXArray,
        avCaV2AGateParams: MLXArray,
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        videoTextProjection: LTXAttentionProjectedContext?,
        audioTextProjection: LTXAttentionProjectedContext?,
        videoRope: LTXRope,
        videoSelfAttentionMask: MLXArray?,
        audioRope: LTXRope,
        videoCrossRope: LTXRope,
        audioCrossRope: LTXRope,
        skipsVideoSelfAttention: Bool,
        skipsAudioSelfAttention: Bool,
        skipsAudioToVideoCrossAttention: Bool,
        skipsVideoToAudioCrossAttention: Bool
    ) -> (video: MLXArray, audio: MLXArray) {
        let variant = LTXV2CompiledBlockVariant(
            hasVideoSelfAttentionMask: videoSelfAttentionMask != nil,
            hasTextProjectionCache: videoTextProjection != nil && audioTextProjection != nil,
            skipsVideoSelfAttention: skipsVideoSelfAttention,
            skipsAudioSelfAttention: skipsAudioSelfAttention,
            skipsAudioToVideoCrossAttention: skipsAudioToVideoCrossAttention,
            skipsVideoToAudioCrossAttention: skipsVideoToAudioCrossAttention
        )
        let runner: LTXUnifiedAVTransformerV2Block
        if let compiledBlockRunner {
            runner = compiledBlockRunner
        } else {
            let created = LTXUnifiedAVTransformerV2Block(
                videoFeedForwardBias: videoFeedForwardBias
            )
            compiledBlockRunner = created
            runner = created
        }
        runner.update(parameters: block.parameters())

        let forward: LTXV2CompiledBlockForward
        if let compiled = compiledBlockForwards[variant] {
            forward = compiled
        } else {
            let compiled = MLX.compile(inputs: [runner]) { inputs in
                let output = runner(
                    videoHidden: inputs[0],
                    audioHidden: inputs[1],
                    videoAdalnParams: inputs[2],
                    audioAdalnParams: inputs[3],
                    videoPromptAdalnParams: inputs[4],
                    audioPromptAdalnParams: inputs[5],
                    avCaVideoParams: inputs[6],
                    avCaAudioParams: inputs[7],
                    avCaA2VGateParams: inputs[8],
                    avCaV2AGateParams: inputs[9],
                    videoTextEmbeds: inputs[10],
                    audioTextEmbeds: inputs[11],
                    videoTextProjection: variant.hasTextProjectionCache
                        ? LTXAttentionProjectedContext(keys: inputs[20], values: inputs[21])
                        : nil,
                    audioTextProjection: variant.hasTextProjectionCache
                        ? LTXAttentionProjectedContext(keys: inputs[22], values: inputs[23])
                        : nil,
                    videoRope: (cos: inputs[12], sin: inputs[13]),
                    videoSelfAttentionMask: variant.hasVideoSelfAttentionMask
                        ? inputs[variant.hasTextProjectionCache ? 24 : 20]
                        : nil,
                    audioRope: (cos: inputs[14], sin: inputs[15]),
                    videoCrossRope: (cos: inputs[16], sin: inputs[17]),
                    audioCrossRope: (cos: inputs[18], sin: inputs[19]),
                    skipVideoSelfAttention: variant.skipsVideoSelfAttention,
                    skipAudioSelfAttention: variant.skipsAudioSelfAttention,
                    skipAudioToVideoCrossAttention: variant.skipsAudioToVideoCrossAttention,
                    skipVideoToAudioCrossAttention: variant.skipsVideoToAudioCrossAttention
                )
                return [output.video, output.audio]
            }
            compiledBlockForwards[variant] = compiled
            forward = compiled
        }

        var inputs = [
            videoHidden,
            audioHidden,
            videoAdalnParams,
            audioAdalnParams,
            videoPromptAdalnParams,
            audioPromptAdalnParams,
            avCaVideoParams,
            avCaAudioParams,
            avCaA2VGateParams,
            avCaV2AGateParams,
            videoTextEmbeds,
            audioTextEmbeds,
            videoRope.cos,
            videoRope.sin,
            audioRope.cos,
            audioRope.sin,
            videoCrossRope.cos,
            videoCrossRope.sin,
            audioCrossRope.cos,
            audioCrossRope.sin,
        ]
        if let videoTextProjection, let audioTextProjection {
            inputs.append(contentsOf: [
                videoTextProjection.keys,
                videoTextProjection.values,
                audioTextProjection.keys,
                audioTextProjection.values,
            ])
        }
        if let videoSelfAttentionMask {
            inputs.append(videoSelfAttentionMask)
        }
        let outputs = forward(inputs)
        return (outputs[0], outputs[1])
    }
}
