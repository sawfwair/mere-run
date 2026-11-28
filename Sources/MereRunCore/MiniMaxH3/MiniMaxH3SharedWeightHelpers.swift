import Foundation
import MLX
import MereRunTextEncoder

/// Shared weight-loading and conditioner-construction helpers used by both
/// `MiniMaxH3Image` (in `MereRunCore`) and `MiniMaxH3` video pipelines (in `MereRunVideo`).
package enum MiniMaxH3SharedWeightHelpers {
    package static func releasedBF16TransformerWeight(
        key: String,
        value: MLXArray,
        omitCachedAdaLNWeights: Bool,
        headCount: Int = 56,
        headDimension: Int = 128
    ) -> [(String, MLXArray)] {
        if omitCachedAdaLNWeights,
           (key.contains(".adaln_proj.") || key.hasPrefix("time_embedder.")) {
            return []
        }
        guard key.hasSuffix(".attn.qkv_proj.weight") else {
            return [(key, value)]
        }
        return [(key, deinterleavedQKVOutputRows(
            value,
            headCount: headCount,
            headDimension: headDimension
        ))]
    }

    package static func deinterleavedQKVOutputRows(
        _ value: MLXArray,
        headCount: Int,
        headDimension: Int
    ) -> MLXArray {
        let expectedRows = headCount * 3 * headDimension
        precondition(value.ndim == 2 && value.dim(0) == expectedRows)
        let trailingShape = Array(value.shape.dropFirst())
        let grouped = value.reshaped([headCount, 3, headDimension] + trailingShape)
        let pieces = MLX.split(grouped, parts: 3, axis: 1).map {
            $0.squeezed(axis: 1).reshaped([headCount * headDimension] + trailingShape)
        }
        return MLX.concatenated(pieces, axis: 0)
    }

    package static func makeConditioner() -> QwenVLEncoder {
        let textConfiguration = QwenTextEncoderConfiguration(
            vocabSize: 151_936,
            hiddenSize: 5_120,
            numHiddenLayers: 50,
            numAttentionHeads: 64,
            numKeyValueHeads: 8,
            intermediateSize: 25_600,
            ropeTheta: 5_000_000,
            maxPositionEmbeddings: 262_144,
            rmsNormEps: 1e-6,
            headDim: 128,
            mropeSection: [24, 20, 20],
            mropeInterleaved: true
        )
        let visionConfiguration = QwenVisionConfiguration(
            depth: 27,
            embedDim: 1_152,
            mlpHiddenDim: 4_304,
            hiddenAct: .geluApproximate,
            numHeads: 16,
            patchSize: 16,
            temporalPatchSize: 2,
            spatialMergeSize: 2,
            inChannels: 3,
            outHiddenDim: 5_120,
            windowSize: 112,
            fullAttentionBlockIndices: [],
            patchEmbedBias: true,
            numPositionEmbeddings: 2_304,
            useLearnedPosEmbed: true,
            deepstackVisualIndexes: [8, 16, 24]
        )
        return QwenVLEncoder(
            textEncoderConfig: textConfiguration,
            visionConfig: visionConfiguration
        )
    }

    package static func conditionerWeightKey(_ rawKey: String) -> String {
        var key = rawKey
        if key.hasPrefix("model.language_model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("model.language_model.".count))
        } else if key.hasPrefix("model.visual.") {
            key = "visionTower." + String(key.dropFirst("model.visual.".count))
        } else if key.hasPrefix("model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("model.".count))
        } else if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        }
        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc1.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc2.", with: ".patch_merger.mlp_2.")
        if key.contains(".deepstack_merger_list.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")
            key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
            key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        }
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        return key
    }

    package static func conditionerWeight(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "lm_head.weight" || key.hasPrefix("textEncoder.encoder.norm.") {
            return []
        }
        if key.hasPrefix("textEncoder.encoder.layers."),
           let suffix = key.dropFirst("textEncoder.encoder.layers.".count).split(separator: ".").first,
           let layer = Int(suffix),
           layer >= 50 {
            return []
        }
        if key == "visionTower.patch_embed.proj.weight", value.ndim == 5 {
            return [(key, value.transposed(0, 2, 3, 4, 1))]
        }
        return [(key, value)]
    }
}
