import Foundation

extension ManagedModelCatalog {
    static let audioSpecs: [ManagedModelSpec] = [
        ManagedModelSpec(
            id: ModelResolver.ModelID.aukBase.rawValue, category: .audio, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: "tencent/AuK",
                revision: "790742b71a4430120daf2b2099192abae449eb9f",
                patterns: ["auk_base.safetensors", "vae.safetensors", "config.yaml", "LICENSE"]),
            upstreamRepoId: "Tencent-Hunyuan/AuK", upstreamRevision: AuKGenerator.sourceRevision,
            validationKind: .auk, estimatedDownloadBytes: 6_759_534_659,
            defaultCLICommands: ["audio edit"], companionModelIDs: [ModelResolver.ModelID.aukThinker.rawValue],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .auk,
                family: .audio,
                tier: .base,
                variant: .standard,
                precision: .fp32,
                supports: [],
                upstream: .spec
            ),
            capability: .init(
                "AuK base (experimental)",
                "Native instruction-conditioned speech generation and editing; requires audio-auk-thinker. Bounded FP32 checkpoint qualification passed on Apple Silicon.",
                minimum: 24,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aukFlash.rawValue, category: .audio, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: "tencent/AuK-Flash",
                revision: "575b92f0895f75180bf2cbd35f2e176c5732b8ed",
                patterns: ["auk_flash.safetensors", "vae.safetensors", "config.yaml", "LICENSE"]),
            upstreamRepoId: "Tencent-Hunyuan/AuK", upstreamRevision: AuKGenerator.sourceRevision,
            validationKind: .auk, estimatedDownloadBytes: 6_759_534_671,
            defaultCLICommands: ["audio edit"], companionModelIDs: [ModelResolver.ModelID.aukThinker.rawValue],
            apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .auk,
                family: .audio,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [],
                upstream: .spec
            ),
            capability: .init(
                "AuK Flash (experimental)",
                "Native four-step speech generation and editing; requires audio-auk-thinker. Bounded FP32 checkpoint qualification passed on Apple Silicon.",
                minimum: 24,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.aukThinker.rawValue, category: .audio, installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(repoId: "Qwen/Qwen2.5-Omni-3B",
                revision: "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
                patterns: ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "*.json", "LICENSE"]),
            upstreamRepoId: "Qwen/Qwen2.5-Omni-3B",
            upstreamRevision: "f75b40e3da2003cdd6e1829b1f420ca70797c34e",
            validationKind: .aukThinker, estimatedDownloadBytes: 10_009_109_352,
            defaultCLICommands: [], apiAvailability: .cliOnly,
            manifestTemplate: .init(
                engine: .auk,
                family: .audio,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [],
                upstream: .spec
            ),
            capability: .init(
                "AuK Qwen encoder",
                "Qwen2.5-Omni-3B text/audio conditioning checkpoint for AuK; not a standalone audio generator.",
                minimum: 24,
                recommended: 32
            )
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.apBWE16kTo48k.rawValue,
            category: .audio,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: APBWEResources.artifactRepository,
                revision: APBWEResources.artifactRevision,
                patterns: APBWEResources.pins.map(\.filename)
            ),
            upstreamRepoId: APBWEResources.sourceRepository,
            upstreamRevision: APBWEResources.sourceRevision,
            validationKind: .apBWE,
            estimatedDownloadBytes: 119_099_717,
            defaultCLICommands: ["audio enhance"],
            manifestTemplate: .init(
                engine: .apBWE,
                family: .audio,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                supports: [.audioEnhancement]
            ),
            capability: .init(
                "AP-BWE speech bandwidth extension",
                "Extends 16 kHz narrowband speech to 48 kHz with native MLX inference.",
                minimum: 8,
                recommended: 16
            )
        ),
        ManagedModelSpec(
            id: ModelResolver.ModelID.univerSRAudio.rawValue,
            category: .audio,
            installShape: .directoryRoot,
            hubFallback: HubFallbackConfig(
                repoId: UniverSRResources.artifactRepository,
                revision: UniverSRResources.artifactRevision,
                patterns: UniverSRResources.pins.map(\.filename)
            ),
            upstreamRepoId: UniverSRResources.sourceRepository,
            upstreamRevision: UniverSRResources.sourceRevision,
            validationKind: .univerSR,
            estimatedDownloadBytes: 229_074_334,
            defaultCLICommands: ["audio enhance"],
            manifestTemplate: .init(
                engine: .univerSR,
                family: .audio,
                tier: .small,
                variant: .standard,
                precision: .fp32,
                defaults: .init(steps: 4, cfg: 1.5),
                supports: [.audioEnhancement]
            ),
            capability: .init(
                "UniverSR general audio",
                "Extends 8, 12, 16, or 24 kHz speech, music, and sound effects to 48 kHz with native MLX inference.",
                minimum: 8,
                recommended: 16
            )
        ),
    ]
}
