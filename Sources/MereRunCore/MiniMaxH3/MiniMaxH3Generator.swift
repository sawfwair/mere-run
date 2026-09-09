import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

public final class MiniMaxH3Generator: @unchecked Sendable {
    struct DenoisingRuntimeCacheKey: Hashable {
        let modelRoot: URL
        let modelSourceIdentity: String
        let videoSigmas: [Float]
        let audioSigmas: [Float]
        let weightMode: MiniMaxH3TransformerWeightMode
        let adapterURL: URL?
        let adapterSHA256: String?
        let adapterStrength: Float
    }

    struct ReferenceCacheKey: Hashable {
        let references: [MiniMaxH3ReferenceInput]
        let maximumFrameCount: Int
        let targetWidth: Int
        let targetHeight: Int
    }

    struct ConditionerPresentation {
        let tokenIDs: [Int]
        let tokenTags: [Int32]
        let images: [QwenVLEncoder.ConditioningImage]
    }

    struct PreparedReference {
        let kind: MiniMaxH3ReferenceKind
        let visual: MLXArray?
        let visionBlocks: [MLXArray]
        let blockTimestamps: [Double]
        let waveform: MLXArray?
        let geometry: MiniMaxH3PreparedReferenceGeometry
    }

    struct FrameCondition {
        let url: URL
        let anchor: MiniMaxH3KeyframeAnchor
    }

    struct ContinuationConditions {
        let videoRows: MLXArray
        let audioRows: MLXArray
        let videoAnchors: [MiniMaxH3KeyframeAnchor]
        let audioAnchors: [MiniMaxH3AudioConditionAnchor]
    }

    struct PreparedReferenceRows {
        let video: [MLXArray]
        let audio: [MLXArray]
    }

    let retainsRuntime: Bool
    var retainedConditioner: (root: URL, model: QwenVLEncoder)?
    var retainedVideoVAE: (root: URL, model: MiniMaxH3VideoVAE)?
    var retainedAudioVAE: (root: URL, model: MiniMaxH3AudioVAE)?
    var retainedPreparedReferences: (
        key: ReferenceCacheKey,
        values: [PreparedReference]
    )?
    var retainedPreparedReferenceRows: (
        key: ReferenceCacheKey,
        values: PreparedReferenceRows
    )?
    var retainedDenoisingRuntime: (
        key: DenoisingRuntimeCacheKey,
        transformer: MiniMaxH3Transformer,
        adaLNCache: MiniMaxH3AdaLNCache?
    )?

    public init(retainsRuntime: Bool = false) {
        self.retainsRuntime = retainsRuntime
    }

    static func mediaFrames(from decodedFrames: MLXArray) -> MLXArray {
        MLX.clip(decodedFrames * 255, min: 0, max: 255).asType(.uint8)
    }

}
