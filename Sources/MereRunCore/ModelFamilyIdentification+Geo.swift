import Foundation

extension ModelFamilyIdentifier {
    /// `geo tessera` runs a local checkpoint as the variant its `config.json` declares.
    static let geoProbes: [String: Probe] = [
        "geo.tessera": { model, _ in
            (try? TESSERAResources.declaredVariant(URL(fileURLWithPath: model))).map(tesseraFamily)
        }
    ]

    /// The `geo tessera` family that runs `variant`: the teacher alone projects to 1024 dimensions.
    static func tesseraFamily(_ variant: TESSERAVariant) -> String {
        variant == .teacher ? "tessera-teacher" : "tessera-student"
    }
}
