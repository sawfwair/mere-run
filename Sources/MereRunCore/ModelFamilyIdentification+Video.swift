import Foundation
import MereRunContract

/// Local video checkpoints, identified by the same layout detectors the video commands run.
extension ModelFamilyIdentifier {
    /// `video generate`: `VideoGenerationModelProfile.observe`, the detector the operation uses
    /// on the resolved root. A FastH3 root is laid out like FL2VA; it runs as FastH3 only when
    /// `--model-root` names the root and `--model` names the FastH3 id, because the embedded
    /// adapter is chosen by that id (`VideoGenerationOptions.usesEmbeddedFastH3Adapter`).
    static let videoGenerate: Probe = { model, invocation in
        let profile = VideoGenerationModelProfile.observe(root: URL(fileURLWithPath: model).standardizedFileURL)
        let fastH3 = profile == .h3FL2VA
            && invocation.value("--model-root") == model
            && invocation.value("--model") == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        return profile.videoGenerateFamily(fastH3: fastH3)
    }

    /// `video retake`: the command runs any official LTX 2.5 root, the full lane when the full
    /// checkpoint validates (`VideoRetakeCommand.run`).
    static let videoRetake: Probe = { model, _ in
        let root = URL(fileURLWithPath: model).standardizedFileURL
        if isLTX25FullModelRoot(root) { return "ltx25-full" }
        return isLTX25ModelRoot(root) ? "ltx25-distilled" : nil
    }

    /// `video session`: the standalone split and full LTX 2.3 roots and both LTX 2.5 roots
    /// (`VideoSessionCommand.run`); every other layout fails there.
    static let videoSession: Probe = { model, _ in
        let root = URL(fileURLWithPath: model).standardizedFileURL
        if isLTX25FullModelRoot(root) { return "ltx25-full" }
        if isLTX25ModelRoot(root) { return "ltx25-distilled" }
        if isLTX23FullModelRoot(root) { return "ltx23-full" }
        return isLTX23SplitModelRoot(root) ? "ltx23-distilled" : nil
    }
}

extension VideoGenerationModelProfile {
    /// The layout of a `video generate` family's checkpoints; FastH3 shares FL2VA's layout.
    init(videoGenerateFamily family: String) {
        switch family {
        case "ltx-merged": self = .ltxMerged
        case "ltx23-distilled": self = .ltx23Distilled
        case "ltx23-full": self = .ltx23Full
        case "ltx23-a2vid": self = .ltx23AudioToVideo
        case "ltx25-distilled": self = .ltx25Distilled
        case "ltx25-full": self = .ltx25Full
        case "wan22-ti2v": self = .wan
        case "h3-fl2va", "h3-fast": self = .h3FL2VA
        case "h3-ref2va": self = .h3Ref2VA
        default: self = .unknown
        }
    }

    /// The `video generate` family that runs this layout; `nil` for an unrecognized one.
    func videoGenerateFamily(fastH3: Bool) -> String? {
        switch self {
        case .unknown: nil
        case .ltxMerged: "ltx-merged"
        case .ltx23Distilled: "ltx23-distilled"
        case .ltx23Full: "ltx23-full"
        case .ltx23AudioToVideo: "ltx23-a2vid"
        case .ltx25Distilled: "ltx25-distilled"
        case .ltx25Full: "ltx25-full"
        case .wan: "wan22-ti2v"
        case .h3FL2VA: fastH3 ? "h3-fast" : "h3-fl2va"
        case .h3Ref2VA: "h3-ref2va"
        }
    }
}
