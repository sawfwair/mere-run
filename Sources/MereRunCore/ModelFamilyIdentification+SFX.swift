import Foundation
import MereRunContract

extension ModelFamilyIdentifier {
    /// `sfx generate` and `sfx video generate` run MMAudio for a folder holding its network
    /// weights, and otherwise the Woosh variant `WooshVariant.resolve` finds in the folder. A
    /// variant the command can't run names its managed model, so the folder stops at that
    /// model's exclusion instead of loading its weights.
    static let soundEffectProbes: [String: Probe] = [
        "sfx.generate": { model, _ in
            soundEffectIdentification(model) { variant in
                switch variant {
                case .dflow: "woosh-dflow"
                case .flow: "woosh-flow"
                case .vflow8s, .dvflow8s: nil
                }
            }
        },
        "sfx.video.generate": { model, _ in
            soundEffectIdentification(model) { variant in
                switch variant {
                case .dvflow8s: "woosh-dvflow"
                case .vflow8s: "woosh-vflow"
                case .dflow, .flow: nil
                }
            }
        }
    ]

    /// `family` maps a Woosh variant to the capability's family, or nil for one it excludes.
    private static func soundEffectIdentification(
        _ model: String,
        family: (WooshVariant) -> String?
    ) -> MereRunModelIdentification? {
        if MMAudioResources.isMMAudio(model: model) {
            return .family("mmaudio")
        }
        return WooshVariant.resolve(model: model).map { variant in
            family(variant).map(MereRunModelIdentification.family) ?? .managedModel(variant.managedModelId)
        }
    }
}
