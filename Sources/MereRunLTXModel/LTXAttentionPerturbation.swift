import Foundation
import MLX
import MLXFast
import MLXNN

package struct LTXAudioToVideoPerturbation: Sendable, Hashable {
    package var skippedVideoSelfAttentionBlocks: Set<Int> = []
    package var skippedAudioSelfAttentionBlocks: Set<Int> = []
    package var skipsAudioToVideoCrossAttention = false
    package var skipsVideoToAudioCrossAttention = false

    package static let none = LTXAudioToVideoPerturbation()

    package static func spatioTemporal(blocks: Set<Int>) -> Self {
        Self(skippedVideoSelfAttentionBlocks: blocks)
    }

    package static func spatioTemporal(videoBlocks: Set<Int>, audioBlocks: Set<Int>) -> Self {
        Self(
            skippedVideoSelfAttentionBlocks: videoBlocks,
            skippedAudioSelfAttentionBlocks: audioBlocks
        )
    }

    package static let isolatedModalities = LTXAudioToVideoPerturbation(
        skipsAudioToVideoCrossAttention: true,
        skipsVideoToAudioCrossAttention: true
    )
}
