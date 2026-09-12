import Foundation

// Source metadata and usage-term construction shared by catalog families.
extension ManagedModelCatalog {
    static let ltxGemma3TextEncoderRepoId = "mlx-community/gemma-3-12b-it-4bit"
    static let ltxGemma3TextEncoderRevision = "14d891e009084901c434304fe93a86fd9013e84c"
    static func usageRestriction(
        summary: String,
        component: String = "model",
        license: String,
        termSummary: String? = nil,
        sourceRepoId: String,
        sourceRevision: String,
        licenseURL: String
    ) -> ManagedModelUsageRestriction {
        ManagedModelUsageRestriction(
            summary: summary,
            terms: [
                ManagedModelUsageTerm(
                    component: component,
                    license: license,
                    summary: termSummary ?? summary,
                    sourceRepoId: sourceRepoId,
                    sourceRevision: sourceRevision,
                    licenseURL: licenseURL
                ),
            ]
        )
    }
}
