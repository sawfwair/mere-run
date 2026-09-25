import Foundation
import MereRunContract

/// Model-specific presentation for capabilities shared by several runtime families. The same
/// scope drives prompt controls, task forms, the Command panel, and their generated arguments.
/// An unrecognized model keeps the full command surface so local checkpoint roots remain usable.
package enum StudioModelOptionScope {
    package static func model(for capability: MereRunCommandCapability, draft: CommandDraft) -> String {
        if capability.id == "video.generate", !draft.modelRoot.isEmpty { return draft.modelRoot }
        return draft.model
    }

    package static func model(
        for capability: MereRunCommandCapability, form: StudioConsoleDraft, seed: CommandDraft
    ) -> String {
        if capability.id == "video.generate", !form.text("--model-root").isEmpty {
            return form.text("--model-root")
        }
        let selected = form.text("--model")
        return selected.isEmpty ? seed.model : selected
    }

    package static func options(
        for capability: MereRunCommandCapability, model: String
    ) -> [MereRunCapabilityOption] {
        guard let allowed = allowedFlags(for: capability, model: model) else { return capability.options }
        return capability.options.filter { allowed.contains($0.flag) }
    }

    package static func allows(_ flag: String, in capability: MereRunCommandCapability, model: String) -> Bool {
        allowedFlags(for: capability, model: model)?.contains(flag) ?? true
    }

    /// Filter only template-generated flags. Explicit Extra arguments remain a raw CLI escape hatch.
    package static func generatedArguments(
        _ arguments: [String], capability: MereRunCommandCapability, model: String
    ) -> [String] {
        guard let allowed = allowedFlags(for: capability, model: model) else { return arguments }
        var result: [String] = []
        var index = 0
        while index < arguments.count {
            let token = arguments[index]
            guard let option = ArgumentBuilder.splitOption(token) else {
                result.append(token)
                index += 1
                continue
            }
            let carriesSeparateValue = option.value == nil && index + 1 < arguments.count
                && !arguments[index + 1].hasPrefix("--")
            if allowed.contains(option.flag) {
                result.append(token)
                if carriesSeparateValue { result.append(arguments[index + 1]) }
            }
            index += carriesSeparateValue ? 2 : 1
        }
        return result
    }

    package static func scopedDraft(
        _ draft: StudioConsoleDraft, capability: MereRunCommandCapability, model: String
    ) -> StudioConsoleDraft {
        guard let allowed = allowedFlags(for: capability, model: model) else { return draft }
        var scoped = draft
        scoped.values = draft.values.filter { allowed.contains($0.key) }
        return scoped
    }

    private static func allowedFlags(
        for capability: MereRunCommandCapability, model: String
    ) -> Set<String>? {
        let name = model.lowercased()
        let all = Set(capability.options.map(\.flag))
        switch capability.id {
        case "music.generate":
            if name.contains("yue2") { return musicShared.union(musicYuE2) }
            if name.contains("minimax-music3") { return musicShared.union(musicMiniMax) }
            if name.contains("magenta") { return musicMagenta }
            if name.contains("acestep") {
                return all.subtracting(musicYuE2.union(musicMiniMax).union(musicMagentaControls))
            }
        case "video.generate":
            if name.contains("minimax-h3") || name.contains("minimax_h3") { return videoH3 }
            if name.contains("wan") { return videoWan }
            if name.contains("ltx") { return all.subtracting(videoH3Only.union(["--shift", "--guidance-scale"])) }
        case "image.generate":
            if name.hasPrefix("image-") && !name.contains("krea") {
                return all.subtracting(imageKreaOnly)
            }
        case "sfx.generate":
            if name.contains("mmaudio") { return all.subtracting(["--renoise"]) }
            if name.contains("woosh") { return all.subtracting(["--negative-prompt"]) }
        case "audio.enhance":
            if name.contains("ap-bwe") { return all.subtracting(audioUniverSROnly) }
            if name.contains("universr") { return all.subtracting(["--overlap"]) }
        case "speech.transcribe":
            if name.contains("qwen") { return all.subtracting(["--provider", "--coreml-encoder"]) }
            if name.contains("parakeet") { return all.subtracting(["--task", "--max-tokens"]) }
        default:
            break
        }
        return nil
    }

    private static let musicShared: Set<String> = [
        "--lyrics", "--lyrics-file", "--instrumental", "--output", "--export-format", "--normalize",
        "--target-peak-db", "--fade-in-ms", "--fade-out-ms", "--no-dither", "--recipe-output",
        "--no-recipe", "--model", "--duration", "--steps", "--guidance-scale", "--seed",
        "--quiet", "--progress-json", "--receipt"
    ]

    private static let musicYuE2: Set<String> = [
        "--score-mode", "--abc-file", "--abc-output", "--abc-max-tokens",
        "--semantic-temperature", "--semantic-top-p", "--semantic-top-k",
        "--semantic-repetition-penalty", "--minimum-duration", "--min-frames", "--max-frames"
    ]

    private static let musicMiniMax: Set<String> = [
        "--compose", "--composer-model", "--composer-model-root", "--require-composer-installed",
        "--composition-output", "--lyrics-preflight", "--minimum-duration", "--min-frames",
        "--max-frames", "--sample-rate", "--memory-mode", "--performance-mode", "--sampling-tier",
        "--flow-strategy", "--flow-solver", "--ar-cfg-frames", "--flow-cfg-end",
        "--seed-strategy", "--profile-output"
    ]

    private static let musicMagentaControls: Set<String> = [
        "--temperature", "--style-conditioning", "--top-k", "--cfg-musiccoca", "--cfg-notes",
        "--cfg-drums", "--drumless", "--unmask-width", "--seed-rotation", "--prefill-silence",
        "--prefill-duration"
    ]

    private static let musicMagenta: Set<String> = Set([
        "--output", "--model", "--duration", "--quiet", "--progress-json", "--receipt"
    ]).union(musicMagentaControls)

    private static let videoH3Only: Set<String> = [
        "--h3-render-width", "--h3-render-height", "--h3-adapter", "--h3-adapter-strength",
        "--h3-frame", "--h3-window-frames", "--h3-window-overlap", "--h3-weight-mode",
        "--h3-acceleration", "--reference"
    ]

    private static let videoH3: Set<String> = Set([
        "--output", "--model", "--model-root", "--width", "--height", "--num-frames",
        "--duration", "--seed", "--steps", "--image", "--preflight", "--json", "--quiet",
        "--progress-json", "--receipt"
    ]).union(videoH3Only)

    private static let videoWan: Set<String> = [
        "--output", "--model", "--model-root", "--width", "--height", "--num-frames",
        "--duration", "--fps", "--seed", "--steps", "--guidance-scale", "--shift",
        "--negative-prompt", "--image", "--image-strength", "--end-image",
        "--end-image-strength", "--preflight", "--json", "--timings", "--timings-output",
        "--quiet", "--progress-json", "--receipt"
    ]

    private static let imageKreaOnly: Set<String> = [
        "--krea-conditioning-multiplier", "--krea-conditioning-layer-weights",
        "--krea-base-quantization-bits"
    ]

    private static let audioUniverSROnly: Set<String> = [
        "--ode-method", "--ode-steps", "--guidance-scale", "--seed", "--chunk-seconds"
    ]
}
