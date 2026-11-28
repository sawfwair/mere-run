import Foundation
import MereRunModelKit

public extension MereRunModelManifest.Components {
    /// Standard diffusers layout (`tokenizer`, `text_encoder`, `transformer`, `vae`, `scheduler`).
    static let diffusers = Self(
        tokenizer: .local(path: "tokenizer"),
        textEncoder: .local(path: "text_encoder"),
        transformer: .local(path: "transformer"),
        vae: .local(path: "vae"),
        scheduler: .local(path: "scheduler")
    )

    static let kleinShared = Self(
        tokenizer: .local(path: "tokenizer"),
        textEncoder: .local(path: "text_encoder"),
        transformer: nil,
        vae: .local(path: "vae"),
        scheduler: .local(path: "scheduler")
    )

    static let kleinHybrid = Self(
        tokenizer: .anyOf([
            .local(path: "tokenizer"),
            .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "tokenizer"),
        ]),
        textEncoder: .anyOf([
            .local(path: "text_encoder"),
            .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "text_encoder"),
        ]),
        transformer: .local(path: "transformer"),
        vae: .anyOf([
            .local(path: "vae"),
            .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "vae"),
        ]),
        scheduler: .anyOf([
            .local(path: "scheduler"),
            .model(modelID: ModelResolver.ModelID.kleinShared.rawValue, path: "scheduler"),
        ])
    )

    static let bonsaiImage = Self(
        tokenizer: .local(path: "tokenizer"),
        textEncoder: .local(path: "text_encoder-mlx-4bit"),
        transformer: .local(path: "transformer-packed-mflux"),
        vae: .local(path: "vae"),
        scheduler: .local(path: "scheduler")
    )

    static let ideogram4 = Self(
        tokenizer: .local(path: "tokenizer"),
        textEncoder: .local(path: "text_encoder"),
        transformer: .local(path: "transformer"),
        unconditionalTransformer: .local(path: "unconditional_transformer"),
        vae: .local(path: "vae"),
        scheduler: .local(path: "scheduler")
    )

    static let zetaBaseHybrid = Self(
        tokenizer: .anyOf([
            .local(path: "tokenizer"),
            .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "tokenizer"),
            .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "tokenizer"),
        ]),
        textEncoder: .anyOf([
            .local(path: "text_encoder"),
            .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "text_encoder"),
            .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "text_encoder"),
        ]),
        transformer: .local(path: "transformer"),
        vae: .anyOf([
            .local(path: "vae"),
            .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "vae"),
            .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "vae"),
        ]),
        scheduler: .anyOf([
            .local(path: "scheduler"),
            .model(modelID: ModelResolver.ModelID.zetaMax.rawValue, path: "scheduler"),
            .model(modelID: ModelResolver.ModelID.zetaNano.rawValue, path: "scheduler"),
        ])
    )

    static let mebotText = Self(
        tokenizer: .anyOf([
            .local(path: "tokenizer"),
            .model(modelID: ModelResolver.ModelID.kleinNano.rawValue, path: "tokenizer"),
            .model(modelID: ModelResolver.ModelID.kleinMax.rawValue, path: "tokenizer"),
        ]),
        textEncoder: .anyOf([
            .local(path: "text_encoder"),
            .model(modelID: ModelResolver.ModelID.kleinNano.rawValue, path: "text_encoder"),
            .model(modelID: ModelResolver.ModelID.kleinMax.rawValue, path: "text_encoder"),
        ])
    )

    /// Root-level tokenizer and text encoder (`"."`).
    static let rootText = Self(
        tokenizer: .local(path: "."),
        textEncoder: .local(path: ".")
    )

    /// Root-level tokenizer, text encoder, and transformer (`"."`).
    static let rootTextTransformer = Self(
        tokenizer: .local(path: "."),
        textEncoder: .local(path: "."),
        transformer: .local(path: ".")
    )

    /// Root-level tokenizer, text encoder, transformer, and VAE (`"."`).
    static let rootTextTransformerVAE = Self(
        tokenizer: .local(path: "."),
        textEncoder: .local(path: "."),
        transformer: .local(path: "."),
        vae: .local(path: ".")
    )

    static let denseLFM2Text = Self(
        tokenizer: .local(path: LFM2Resources.denseVariantSubdirectory),
        textEncoder: .local(path: LFM2Resources.denseVariantSubdirectory)
    )

    static let sam = Self(tokenizer: .local(path: "tokenizer"))

    static let falconPerception = Self(
        tokenizer: .anyOf([
            .local(path: "tokenizer"),
            .local(path: "."),
        ])
    )
}

extension MereRunModelManifest {
    // MARK: - Templates (known models)

    /// If `modelId` is one of `ModelResolver.ModelID`, writes a manifest into `modelRoot` and returns it.
    public static func writeTemplateIfKnown(
        modelId: String,
        to modelRoot: URL,
        usageTermsAcknowledged: Bool = false,
        createdAt: Date = Date()
    ) throws -> MereRunModelManifest? {
        guard let known = ModelResolver.ModelID(rawValue: modelId) else {
            return nil
        }
        do {
            if let existing = try MereRunModelManifest.loadIfPresent(from: modelRoot) {
                return existing
            }
        } catch {
            // Ignore decode errors and overwrite with a fresh template below.
        }

        var manifest = template(for: known, createdAt: createdAt)
        if manifest.usageTerms?.isEmpty == false {
            manifest.usageTermsAcknowledged = usageTermsAcknowledged
        }
        try manifest.write(to: modelRoot)
        return manifest
    }

    /// Returns a manifest template for a known `ModelResolver.ModelID`.
    ///
    /// Materialized directly from the canonical `ManagedModelSpec.manifestTemplate` registered
    /// in `ManagedModelCatalog`.
    public static func template(for modelID: ModelResolver.ModelID, createdAt: Date = Date()) -> MereRunModelManifest {
        guard let spec = ManagedModelCatalog.spec(forExactID: modelID.rawValue),
              let manifestTemplate = spec.manifestTemplate else {
            preconditionFailure("Missing managed model manifest template for \(modelID.rawValue)")
        }
        return manifestTemplate.makeManifest(
            for: spec,
            sources: sourceProvenance(for: spec),
            createdAt: createdAt
        )
    }

    private static func sourceProvenance(for spec: ManagedModelSpec) -> [SourceProvenance]? {
        var sources: [SourceProvenance] = []
        if let primary = spec.hubFallback {
            sources.append(
                SourceProvenance(
                    role: "primary",
                    repository: primary.repoId,
                    revision: primary.revision
                )
            )
        }
        sources.append(contentsOf: spec.mountedHubFallbacks.map { mounted in
            SourceProvenance(
                role: "component",
                repository: mounted.hubFallback.repoId,
                revision: mounted.hubFallback.revision,
                destinationPath: mounted.destinationPath
            )
        })
        if let upstreamRepoId = spec.upstreamRepoId,
           !sources.contains(where: { $0.repository == upstreamRepoId }) {
            sources.append(
                SourceProvenance(
                    role: "upstream_reference",
                    repository: upstreamRepoId,
                    revision: spec.upstreamRevision ?? "unspecified"
                )
            )
        }
        return sources.isEmpty ? nil : sources
    }
}
