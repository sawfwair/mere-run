import Foundation

extension MereRunCapabilityCatalog {
    enum TerraMindFloodFamily: String, MereRunFamilyID {
        case terraMindFlood = "terramind-flood"
    }

    enum TerraMindFireFamily: String, MereRunFamilyID {
        case terraMindFire = "terramind-fire"
    }

    enum OlmoEarthFamily: String, MereRunFamilyID {
        case olmoEarth = "olmoearth"
    }

    static let geoFloodRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-flood-terramind-base")],
        families: [
            .init(TerraMindFloodFamily.terraMindFlood, title: "TerraMind Flood", models: ["vision-flood-terramind-base"])
        ]
    )

    static let geoFireRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("vision-fire-terramind-base")],
        families: [
            .init(TerraMindFireFamily.terraMindFire, title: "TerraMind Fire", models: ["vision-fire-terramind-base"])
        ]
    )

    /// Every size shares one surface; the CLI picks the default by machine memory and what is
    /// installed.
    static let geoOlmoEarthRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [
            .always(
                "vision-embed-olmoearth-v12-nano", "vision-embed-olmoearth-v12-tiny",
                "vision-embed-olmoearth-v12-small", "vision-embed-olmoearth-v12-base"
            )
        ],
        families: [
            .init(
                OlmoEarthFamily.olmoEarth,
                title: "OlmoEarth v1.2",
                models: [
                    "vision-embed-olmoearth-v12-nano", "vision-embed-olmoearth-v12-tiny",
                    "vision-embed-olmoearth-v12-small", "vision-embed-olmoearth-v12-base"
                ]
            )
        ]
    )
}
