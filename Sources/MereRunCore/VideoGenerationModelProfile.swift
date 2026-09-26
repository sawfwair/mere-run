import Foundation
import MereRunContract

/// Checkpoint layout observed without loading tensors or installing assets.
public enum VideoGenerationModelProfile: String, Sendable {
    case unknown
    case ltxMerged = "ltx_merged"
    case ltx23Distilled = "ltx23_distilled_split"
    case ltx23Full = "ltx23_full_split"
    case ltx23AudioToVideo = "ltx23_a2vid_split"
    case ltx25Full = "ltx25_full"
    case ltx25Distilled = "ltx25_distilled"
    case wan = "wan22_ti2v_mlx"
    case h3FL2VA = "minimax_h3_fl2va_mlx"
    case h3Ref2VA = "minimax_h3_ref2va_mlx"

    public var isH3: Bool { self == .h3FL2VA || self == .h3Ref2VA }
    public var isLTX25: Bool { self == .ltx25Full || self == .ltx25Distilled }
    public var quality: LTXVideoQuality? {
        switch self {
        case .ltx23Full, .ltx23AudioToVideo, .ltx25Full, .ltx25Distilled: .final
        case .ltxMerged, .ltx23Distilled: .draft
        case .unknown, .wan, .h3FL2VA, .h3Ref2VA: nil
        }
    }

    public static func observe(root: URL, fileManager: FileManager = .default) -> Self {
        let h3 = MiniMaxH3Resources(rootURL: root)
        if h3.validate(fileManager: fileManager).isEmpty, let config = try? h3.loadConfiguration() {
            return config.task == "ref2va" ? .h3Ref2VA : .h3FL2VA
        }
        let wan = Wan2Resources(rootURL: root)
        if wan.validate(fileManager: fileManager).isEmpty, (try? wan.loadConfiguration()) != nil {
            return .wan
        }
        if isLTX23FullModelRoot(root, fileManager: fileManager) { return .ltx23Full }
        if isLTX23AudioToVideoModelRoot(root, fileManager: fileManager) { return .ltx23AudioToVideo }
        if isLTX25FullModelRoot(root, fileManager: fileManager) { return .ltx25Full }
        if isLTX25ModelRoot(root, fileManager: fileManager) { return .ltx25Distilled }
        if isLTX23SplitModelRoot(root, fileManager: fileManager) { return .ltx23Distilled }
        return fileManager.fileExists(atPath: root.path) ? .ltxMerged : .unknown
    }

    /// The profile of a managed model id before resolution: the checkpoint layout of the
    /// contract's `video generate` family that lists it, so the CLI gate and this check agree by
    /// construction. An id the contract leaves to the identifier keeps its own layout, except
    /// `video-ltx-av`, which stays unchecked because resolution can pick another folder for it.
    public static func managed(_ selector: String) -> Self {
        if let family = MereRunCapabilityCatalog.videoGenerate.routing?.families.first(where: { $0.models.contains(selector) }) {
            return Self(videoGenerateFamily: family.id)
        }
        return selector == ModelResolver.ModelID.ltxVideoAV.rawValue ? .unknown : installDependentLayouts[selector] ?? .unknown
    }

    /// The managed ids whose checkpoint depends on what is installed (the contract's
    /// `identified_models`), each with the layout it names: `video-ltx-av` can run a suggested
    /// LTX 2.3 folder, the LTX 2.3 Full and A2Vid ids fall back to each other's installs, and an
    /// installed LTX 2.5 Distilled folder can hold the diffusion decoder.
    public static let installDependentLayouts: [String: Self] = [
        ModelResolver.ModelID.ltxVideoAV.rawValue: .ltxMerged,
        ModelResolver.ModelID.ltxVideo23FullMLX.rawValue: .ltx23Full,
        ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue: .ltx23AudioToVideo,
        ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue: .ltx25Distilled
    ]

    public func ltxRoute(outputMode: LTXVideoOutputMode) -> LTXVideoGenerationRoute? {
        guard !isH3, self != .wan, self != .unknown else { return nil }
        if outputMode == .audioVideo { return .unifiedAV }
        switch self {
        case .ltx23Full, .ltx23AudioToVideo, .ltx25Full: return .fullQualityVideo
        case .ltx23Distilled, .ltx25Distilled: return .splitDistilledVideo
        default: return .legacyDistilledVideo
        }
    }
}

public enum LTXVideoGenerationRoute: String, Equatable, Sendable {
    case legacyDistilledVideo = "legacy-distilled-video"
    case splitDistilledVideo = "split-distilled-video"
    case fullQualityVideo = "full-quality-video"
    case unifiedAV = "unified-av"

    public var writesAudio: Bool { self == .unifiedAV }
    public var supportsPhaseTimings: Bool { self != .legacyDistilledVideo }
}
