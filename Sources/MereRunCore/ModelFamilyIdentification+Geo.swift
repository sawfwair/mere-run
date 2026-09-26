import Foundation
import MereRunContract

extension ModelFamilyIdentifier {
    /// `geo tessera` runs a local checkpoint as the variant its `config.json` declares.
    static let geoProbes: [String: Probe] = [
        "geo.tessera": { model, _ in
            (try? TESSERAResources.declaredVariant(URL(fileURLWithPath: model))).map { .family(tesseraFamily($0)) }
        }
    ]

    /// Without `--model`, `geo tessera` runs the largest variant this machine's memory
    /// recommends that is installed, or the recommended one: the teacher on 32 GB Macs and up.
    static let geoDefaultChoosers: [String: DefaultChooser] = [
        "geo.tessera": { _ in TESSERAResources.defaultModelID() }
    ]

    /// The `geo tessera` family that runs `variant`: the teacher alone projects to 1024 dimensions.
    static func tesseraFamily(_ variant: TESSERAVariant) -> String {
        variant == .teacher ? "tessera-teacher" : "tessera-student"
    }
}
