import MereRunContract
@testable import StudioKit

/// Video generate routed over two families, for the scoped-surface snapshots: LTX-2.5 Full takes
/// keyframes and timings, and FastH3 takes neither and runs exactly 5 steps. The shipped contract
/// routes it once the video routing lands; until then the snapshots declare it here.
enum StudioScopeSnapshotContract {
    static let fastH3Model = "video-minimax-h3-fasth3-vsa-datafree-mlx"

    static var source: StudioScopeSource {
        let video = routedVideo
        return StudioScopeSource(
            identities: StudioFixedModelIdentities(),
            capability: { $0 == video.id ? video : MereRunCapabilityCatalog.command(id: $0) }
        )
    }

    private static var routedVideo: MereRunCommandCapability {
        guard let base = MereRunCapabilityCatalog.command(id: "video.generate") else {
            preconditionFailure("video.generate is not in the contract")
        }
        let ltxOnly: Set = ["--image", "--image-strength", "--end-image", "--end-image-strength", "--timings", "--timings-output"]
        let options = base.options.map { option in
            MereRunCapabilityOption(
                flag: option.flag, aliases: option.aliases, label: option.label, kind: option.kind,
                required: option.required, repeatable: option.repeatable, choices: option.choices,
                defaultValue: option.defaultValue, group: option.group, tier: option.tier, range: option.range,
                dependsOn: option.dependsOn, families: ltxOnly.contains(option.flag) ? ["ltx25-full"] : nil,
                familyRules: option.flag == "--steps" ? [MereRunOptionFamilyRule(family: "fasth3", values: ["5"])] : []
            )
        }
        return MereRunCommandCapability(
            id: base.id, command: base.command, title: base.title, summary: base.summary,
            arguments: base.arguments, options: options, output: base.output,
            routing: MereRunCapabilityRouting(
                modelFlags: ["--model-root", "--model"],
                defaultModels: [MereRunDefaultModelRule(models: ["video-ltx25-full-bf16"])],
                families: [
                    MereRunRuntimeFamily(id: "ltx25-full", title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"]),
                    MereRunRuntimeFamily(id: "fasth3", title: "FastH3", models: [fastH3Model]),
                ]
            )
        )
    }
}
