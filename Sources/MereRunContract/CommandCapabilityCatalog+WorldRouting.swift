import Foundation

extension MereRunCapabilityCatalog {
    enum WorldServeFamily: String, MereRunFamilyID {
        case dreamX = "dreamx"
        case cosmos3
    }

    /// `--backend` picks the runtime and `--model` names its checkpoint (`WorldServe.run`). The
    /// Cosmos 3 backend reads `--model`'s default, the DreamX id, as "use the Cosmos 3 default",
    /// so it lists that id too.
    static let worldServeRouting = MereRunCapabilityRouting(
        modelFlags: [],
        defaultModels: [
            MereRunDefaultModelRule(models: ["video-dreamx-world-5b-ar-mlx"], family: WorldServeFamily.dreamX.rawValue),
            MereRunDefaultModelRule(models: ["video-cosmos3-edge-mlx"], family: WorldServeFamily.cosmos3.rawValue)
        ],
        families: [
            .init(
                WorldServeFamily.dreamX, title: "DreamX World", models: ["video-dreamx-world-5b-ar-mlx"],
                selectors: [.init(flag: "--backend", values: ["dreamx"])], modelFlag: "--model"
            ),
            .init(
                WorldServeFamily.cosmos3, title: "Cosmos 3 Edge",
                models: ["video-cosmos3-edge-mlx", "video-dreamx-world-5b-ar-mlx"],
                selectors: [.init(flag: "--backend", values: ["cosmos3"])], modelFlag: "--model"
            )
        ]
    )
}
