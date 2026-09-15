import Foundation

/// Ordered managed-model discovery and lookup. Family definitions stay in Core.
public enum ManagedModelCatalog {
    /// Preserve this order: repository aliases resolve to the first matching spec.
    public static let allSpecs: [ManagedModelSpec] =
        imageSpecs
            + chatSpecs
            + speechSpecs
            + textUtilitySpecs
            + visionSpecs
            + musicSpecs
            + audioSpecs
            + soundEffectSpecs
            + videoSpecs
            + geoExpansionSpecs

    public static func apiProfile(for modelID: String) -> ManagedModelAPIProfile? {
        spec(for: modelID)?.apiProfile
            ?? ManagedModelAPIProfile.companion(modelID: modelID, category: nil)
    }

    public static var allModelIDs: [String] {
        allSpecs.map(\.id)
    }

    public static func spec(for id: String) -> ManagedModelSpec? {
        let normalized = id.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return allKnownSpecs.first {
            $0.id == normalized || $0.upstreamRepoId?.lowercased() == normalized
        }
    }

    public static func missingHubSourceMessage(for modelId: String) -> String {
        "Model \(modelId) does not have a Hugging Face Hub source in this public build. Install it from a local path or choose a model listed by `mere.run model capabilities --recommended`."
    }

    private static var allKnownSpecs: [ManagedModelSpec] {
        allSpecs + companionSpecs
    }
}
