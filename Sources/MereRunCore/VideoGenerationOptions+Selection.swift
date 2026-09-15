import Foundation
import MereRunContract

extension VideoGenerationOptions {
    public func observedProfile(fileManager: FileManager = .default) -> VideoGenerationModelProfile {
        if let modelRoot {
            return .observe(root: URL(fileURLWithPath: modelRoot).standardizedFileURL, fileManager: fileManager)
        }
        let selector = resolvedRequestedModel
        let path = URL(fileURLWithPath: selector).standardizedFileURL
        if fileManager.fileExists(atPath: path.path) {
            return .observe(root: path, fileManager: fileManager)
        }
        if let id = ModelResolver.ModelID(rawValue: selector),
           let installed = ModelResolver(fileManager: fileManager).resolveIfPresent(id) {
            return .observe(root: installed.rootURL, fileManager: fileManager)
        }
        return .managed(selector)
    }

    public var usesEmbeddedFastH3Adapter: Bool {
        h3Adapter == nil
            && resolvedRequestedModel == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
    }

    public var h3AdapterInferenceRecipe: MiniMaxH3TurboAdapter.InferenceRecipe? {
        guard let reference = h3Adapter else {
            return usesEmbeddedFastH3Adapter ? MiniMaxH3TurboAdapter.fastH3VSADataFreeRecipe : nil
        }
        let filename = ManagedAdapterCatalog.spec(for: reference)?.artifact.filename
            ?? URL(fileURLWithPath: reference).lastPathComponent
        return MiniMaxH3TurboAdapter.inferenceRecipe(for: URL(fileURLWithPath: filename))
    }
    public var variant: LTXVideoVariant {
        effectiveOutputMode.compatibilityVariant
    }

    public var autoDurationRange: LTX25AutoDuration? {
        guard autoDuration.count == 2,
              autoDuration[0].isFinite, autoDuration[1].isFinite,
              autoDuration[0] > 0, autoDuration[1] >= autoDuration[0] else { return nil }
        return LTX25AutoDuration(
            minimumSeconds: autoDuration[0],
            maximumSeconds: autoDuration[1]
        )
    }

    public var requestedQuality: LTXVideoQuality {
        if audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            return .final
        }
        if let modelRoot,
           isLTX25ModelRoot(URL(fileURLWithPath: modelRoot).standardizedFileURL) {
            return .final
        }
        if let quality {
            return quality
        }
        return legacyVariant == .unifiedAV ? .final : .draft
    }

    public var effectiveOutputMode: LTXVideoOutputMode {
        if audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
            return .audioVideo
        }
        if let outputMode {
            return outputMode
        }
        if dfr {
            return .audioVideo
        }
        return legacyVariant == .unifiedAV ? .audioVideo : .videoOnly
    }

    public var productSelectionValidationMessage: String? {
        if legacyVariant != nil, quality != nil || outputMode != nil {
            return "Use --quality/--output-mode or the compatibility --variant option, not both."
        }
        let hasSourceAudio = audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        if hasSourceAudio, quality == .draft {
            return "--audio requires --quality final because source-audio conditioning uses the dev + distilled-LoRA checkpoint."
        }
        if hasSourceAudio, outputMode == .videoOnly {
            return "--audio preserves the selected soundtrack and requires --output-mode audio-video."
        }
        return nil
    }

    public var resolvedRequestedModel: String {
        let requested = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !requested.isEmpty {
            return requested
        }
        if dfr
            || ltxPreset == .hq
            || ltxPipeline != .twoStage
            || ltxSampler != nil
            || distilledLoRAStrengthStage1 != nil
            || distilledLoRAStrengthStage2 != nil {
            return ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
        }
        if hdrColorSpace != nil
            || highQualityHDR
            || textEmbeddings != nil
            || enhancePrompt
            || !autoDuration.isEmpty
            || videoDecoder != nil
            || !imageConditionings.isEmpty
            || numGeneratedKeyframes > 0
            || !generatedKeyframeIndices.isEmpty
            || !videoConditionings.isEmpty {
            return ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
        }
        let hasAudio = audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        if hasAudio || requestedQuality == .final {
            return ModelResolver.ModelID.ltxVideo23FullMLX.rawValue
        }
        return ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
    }

    public var usesLTX25RecipeGeometry: Bool {
        if let modelRoot {
            let root = URL(fileURLWithPath: modelRoot).standardizedFileURL
            if isLTX25ModelRoot(root) {
                return true
            }
        }
        let requested = resolvedRequestedModel.lowercased()
        return requested == ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
            || requested == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue
            || requested.contains("ltx25")
            || requested.contains("ltx-2.5")
    }

    public var resolvedOutputWidth: Int {
        if let width { return width }
        guard usesLTX25RecipeGeometry else { return 768 }
        if ltxPreset == .hq { return 1_920 }
        return ltxPipeline == .devOneStage ? 768 : 1_536
    }

    public var resolvedOutputHeight: Int {
        if let height { return height }
        guard usesLTX25RecipeGeometry else { return 512 }
        if ltxPreset == .hq { return 1_088 }
        return ltxPipeline == .devOneStage ? 512 : 1_024
    }

    public var numFramesSpecified: Bool { numFrames != nil }
    public var requestedFrameCount: Int { numFrames ?? 65 }
    public var hasSourceAudio: Bool {
        audio?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
    }
}
