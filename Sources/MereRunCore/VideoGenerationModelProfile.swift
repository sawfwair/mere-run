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

    public static func managed(_ selector: String) -> Self {
        guard let id = ModelResolver.ModelID(rawValue: selector) else { return .unknown }
        switch id {
        case .ltxVideo23FullMLX: return .ltx23Full
        case .ltxVideo23A2VMLX: return .ltx23AudioToVideo
        case .ltxVideo23AVMLX: return .ltx23Distilled
        case .ltxVideo25FullBF16: return .ltx25Full
        case .ltxVideo25DistilledBF16: return .ltx25Distilled
        case .miniMaxH3Ref2VAMLX: return .h3Ref2VA
        case .miniMaxH3FL2VAMLX, .miniMaxH3FL2VABF16MLX, .miniMaxH3FL2VAQ8MLX,
             .miniMaxH3FastH3VSADataFreeMLX: return .h3FL2VA
        default: return selector == Wan2Resources.modelID ? .wan : .unknown
        }
    }

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
