import Foundation

extension ManagedModelCatalog {
    static let audioSpecs: [ManagedModelSpec] = [
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
            defaultCLICommands: ["audio enhance"]
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
            defaultCLICommands: ["audio enhance"]
        ),
    ]
}
