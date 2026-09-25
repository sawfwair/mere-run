import Foundation
import MereRunContract

/// Video checkpoints, identified by the same resolution and layout detectors the video commands
/// run: `VideoGenerationModelResolver` finds the folder a model names without downloading, and
/// the command's detectors read its files.
extension ModelFamilyIdentifier {
    static let videoProbes: [String: Probe] = [
        "video.generate": videoGenerate,
        "video.retake": videoRetake,
        "video.session": videoSession,
    ]

    /// Managed ids whose runtime keys on their exact spelling. The identifier sends any other
    /// spelling of them (an upstream repository, another case) to the capability's probe instead
    /// of reading it as the id: FastH3's embedded adapter and fixed recipe run only for its exact
    /// id, and every other spelling runs the FL2VA layout without them.
    static let exactSpellingModels: [String: Set<String>] = [
        "video.generate": [ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue]
    ]

    /// `video generate`: `VideoGenerationModelProfile.observe` on the folder the operation will
    /// resolve. A managed id with nothing installed keeps the layout it names, however it is
    /// spelled. A FastH3 folder is laid out like FL2VA; it runs as FastH3 only when `--model`
    /// names the FastH3 id exactly and no `--h3-adapter` replaces the embedded one
    /// (`VideoGenerationOptions.usesEmbeddedFastH3Adapter`). An FL2VA folder stored as legacy
    /// Q4 is its own family: it refuses Turbo adapters. An LTX-2.5 Distilled folder that holds
    /// the diffusion decoder is its own family too: it runs `--video-decoder diffusion`.
    static let videoGenerate: Probe = { model, invocation in
        let outputMode = VideoGenerationOptions.effectiveOutputMode(
            audio: invocation.value("--audio"),
            outputMode: invocation.value("--output-mode").flatMap(LTXVideoOutputMode.init(rawValue:)),
            dfr: invocation.contains("--dfr"),
            legacyVariant: invocation.value("--variant").flatMap(LTXVideoVariant.init(rawValue:))
        )
        let root = videoRoot(model, invocation, variant: outputMode.compatibilityVariant)
        let managed = ManagedModelCatalog.spec(for: model).map { VideoGenerationModelProfile.managed($0.id) }
        guard let profile = root.map({ VideoGenerationModelProfile.observe(root: $0) })
            ?? VideoGenerationModelProfile.installDependentLayouts[model]
            ?? managed.flatMap({ $0 == .unknown ? nil : $0 }) else { return nil }
        let fastH3 = profile == .h3FL2VA
            && invocation.value("--model")?.trimmingCharacters(in: .whitespacesAndNewlines)
                == ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
            && !invocation.contains("--h3-adapter")
        if profile == .h3FL2VA, !fastH3, let root,
           (try? MiniMaxH3Resources(rootURL: root).transformerStorage())?.supportsFL2VATurboAdapters == false {
            return .family("h3-fl2va-q4")
        }
        if profile == .ltx25Distilled, let root, holdsDiffusionDecoder(root) {
            return .family("ltx25-distilled-diffusion")
        }
        return profile.videoGenerateFamily(fastH3: fastH3).map { .family($0) }
    }

    /// `video retake`: the command runs any official LTX 2.5 folder, on the full lane when the
    /// full checkpoint validates (`VideoRetakeCommand.run`).
    static let videoRetake: Probe = { model, invocation in
        guard let root = videoRoot(model, invocation, variant: .unifiedAV) else { return nil }
        if isLTX25FullModelRoot(root) { return .family("ltx25-full") }
        guard isLTX25ModelRoot(root) else { return nil }
        return .family(holdsDiffusionDecoder(root) ? "ltx25-distilled-diffusion" : "ltx25-distilled")
    }

    /// `video session`: the split and full LTX 2.3 folders and both LTX 2.5 folders
    /// (`VideoSessionCommand.run`); every other layout fails there. The LTX 2.3 Full id with
    /// nothing installed keeps its own layout.
    static let videoSession: Probe = { model, invocation in
        guard let root = videoRoot(model, invocation, variant: .unifiedAV) else {
            return model == ModelResolver.ModelID.ltxVideo23FullMLX.rawValue ? .family("ltx23-full") : nil
        }
        if isLTX25FullModelRoot(root) { return .family("ltx25-full") }
        if isLTX25ModelRoot(root) {
            return .family(holdsDiffusionDecoder(root) ? "ltx25-distilled-diffusion" : "ltx25-distilled")
        }
        if isLTX23FullModelRoot(root) { return .family("ltx23-full") }
        return isLTX23SplitModelRoot(root) ? .family("ltx23-distilled") : nil
    }

    /// The distilled LTX-2.5 runtime loads the diffusion decoder when the folder holds it and
    /// `--video-decoder diffusion` asks for it, and otherwise decodes with the convolutional one
    /// (`LTXUnifiedAVGenerator.loadStandalone`).
    private static func holdsDiffusionDecoder(_ root: URL) -> Bool {
        FileManager.default.fileExists(atPath: LTX25Resources(rootURL: root).diffusionVideoVAEURL.path)
    }

    /// The folder the video commands' resolver uses for `model` without downloading: the
    /// `--model-root` folder when `model` came from it, otherwise the folder `--model` resolves to.
    private static func videoRoot(_ model: String, _ invocation: MereRunCommandInvocation, variant: LTXVideoVariant) -> URL? {
        let fromModelRoot = invocation.value("--model-root") == model
        return VideoGenerationModelResolver.installedRoot(
            explicitModelRoot: fromModelRoot ? model : nil,
            requestedModel: fromModelRoot ? "" : model,
            variant: variant
        )
    }
}

extension VideoGenerationModelProfile {
    /// The layout of a `video generate` family's checkpoints; both FastH3 families and legacy Q4
    /// share FL2VA's, and a distilled LTX-2.5 folder with the diffusion decoder is still distilled.
    init(videoGenerateFamily family: String) {
        switch family {
        case "ltx-merged": self = .ltxMerged
        case "ltx23-distilled": self = .ltx23Distilled
        case "ltx23-full": self = .ltx23Full
        case "ltx23-a2vid": self = .ltx23AudioToVideo
        case "ltx25-distilled", "ltx25-distilled-diffusion": self = .ltx25Distilled
        case "ltx25-full": self = .ltx25Full
        case "wan22-ti2v": self = .wan
        case "h3-fl2va", "h3-fl2va-q4", "h3-fast", "h3-fast-adapter": self = .h3FL2VA
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
