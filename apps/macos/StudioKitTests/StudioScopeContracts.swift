import MereRunContract
@testable import StudioKit

/// Routed contracts for the mechanism tests. The shipped contract routes only single-family
/// commands until the domain changes land, so these declare two or three families over the real
/// music and video capabilities, the way their routing will: which options each family uses,
/// the rules it narrows them by, and the models that select it.
enum StudioScopeContracts {
    enum Music {
        static let ace = "ace-step"
        static let yue2 = "yue2"
        static let magenta = "magenta"
        /// ACE-Step alone takes source audio, a task type, flow edit, and a quality preset (which it
        /// runs as "song" when left off); Magenta has no steps or seed.
        static let capability = routed(
            "music.generate",
            families: [
                MereRunRuntimeFamily(id: ace, title: "ACE-Step", models: ["music-acestep"]),
                MereRunRuntimeFamily(id: yue2, title: "YuE2", models: ["music-yue2"]),
                MereRunRuntimeFamily(id: magenta, title: "Magenta RT", models: ["music-magenta-rt2-small"]),
            ],
            defaultModel: "music-acestep",
            uses: [
                "--source-audio": [ace], "--reference-audio": [ace], "--task-type": [ace], "--flow-edit": [ace],
                "--quality": [ace], "--steps": [ace, yue2], "--seed": [ace, yue2],
            ],
            ignoredBy: ["--seed": [magenta]],
            rules: ["--quality": [MereRunOptionFamilyRule(family: ace, defaultValue: "song")]]
        )
    }

    enum Video {
        static let ltx = "ltx25-full"
        static let ref2va = "h3-ref2va"
        static let fastH3 = "fasth3"
        /// LTX takes keyframes, Ref2VA requires ordered references instead, and FastH3 runs
        /// exactly 5 steps. `--model-root` wins over `--model`.
        static let capability = routed(
            "video.generate",
            families: [
                MereRunRuntimeFamily(id: ltx, title: "LTX-2.5 Full", models: ["video-ltx25-full-bf16"]),
                MereRunRuntimeFamily(id: ref2va, title: "MiniMax-H3 Ref2VA", models: ["video-minimax-h3-ref2va-mlx"]),
                MereRunRuntimeFamily(id: fastH3, title: "FastH3", models: ["video-minimax-h3-fasth3-vsa-datafree-mlx"]),
            ],
            modelFlags: ["--model-root", "--model"],
            defaultModel: "video-ltx25-full-bf16",
            uses: [
                "--image": [ltx], "--image-strength": [ltx], "--end-image": [ltx], "--end-image-strength": [ltx],
                "--reference": [ref2va], "--timings": [ltx],
            ],
            rules: [
                "--reference": [MereRunOptionFamilyRule(family: ref2va, required: true)],
                "--steps": [MereRunOptionFamilyRule(family: fastH3, values: ["5"])],
            ]
        )
    }

    /// A source that answers from `capabilities` (the shipped contract for every other id) and
    /// identifies local models from `identities`.
    static func source(
        _ capabilities: [MereRunCommandCapability],
        identities: [String: StudioModelIdentity] = [:]
    ) -> StudioScopeSource {
        let byID = Dictionary(uniqueKeysWithValues: capabilities.map { ($0.id, $0) })
        return StudioScopeSource(
            identities: StudioFixedModelIdentities(identities),
            capability: { byID[$0] ?? MereRunCapabilityCatalog.command(id: $0) }
        )
    }

    static func routed(
        _ id: String,
        families: [MereRunRuntimeFamily],
        modelFlags: [String] = ["--model"],
        defaultModel: String,
        uses: [String: [String]],
        ignoredBy: [String: [String]] = [:],
        rules: [String: [MereRunOptionFamilyRule]] = [:]
    ) -> MereRunCommandCapability {
        guard let base = MereRunCapabilityCatalog.command(id: id) else { preconditionFailure("\(id) is not in the contract") }
        let options = base.options.map { option in
            MereRunCapabilityOption(
                flag: option.flag, aliases: option.aliases, label: option.label, kind: option.kind,
                required: option.required, repeatable: option.repeatable, choices: option.choices,
                defaultValue: option.defaultValue, group: option.group, tier: option.tier, range: option.range,
                dependsOn: option.dependsOn, families: uses[option.flag], ignoredBy: ignoredBy[option.flag] ?? [],
                familyRules: rules[option.flag] ?? []
            )
        }
        return MereRunCommandCapability(
            id: base.id, command: base.command, title: base.title, summary: base.summary,
            arguments: base.arguments, options: options, output: base.output,
            routing: MereRunCapabilityRouting(
                modelFlags: modelFlags,
                defaultModels: [MereRunDefaultModelRule(models: [defaultModel])],
                families: families
            )
        )
    }
}
