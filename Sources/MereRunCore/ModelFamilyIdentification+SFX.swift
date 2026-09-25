import Foundation

extension ModelFamilyIdentifier {
    /// `sfx generate` and `sfx video generate` run MMAudio for a folder holding its network
    /// weights, and otherwise the Woosh variant `WooshVariant.resolve` finds in the folder. A
    /// variant the command can't run leaves the folder unidentified; the command refuses it.
    static let soundEffectProbes: [String: Probe] = [
        "sfx.generate": { model, _ in
            soundEffectFamily(model) { variant in
                switch variant {
                case .dflow: "woosh-dflow"
                case .flow: "woosh-flow"
                case .vflow8s, .dvflow8s: nil
                }
            }
        },
        "sfx.video.generate": { model, _ in
            soundEffectFamily(model) { variant in
                switch variant {
                case .dvflow8s: "woosh-dvflow"
                case .vflow8s: "woosh-vflow"
                case .dflow, .flow: nil
                }
            }
        }
    ]

    private static func soundEffectFamily(_ model: String, woosh: (WooshVariant) -> String?) -> String? {
        if MMAudioResources.isMMAudio(model: model) {
            return "mmaudio"
        }
        return WooshVariant.resolve(model: model).flatMap(woosh)
    }
}
