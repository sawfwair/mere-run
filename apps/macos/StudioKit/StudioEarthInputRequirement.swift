import Foundation

// The Earth commands read one safetensors bundle and refuse it before loading weights when a
// tensor is missing (`GeoFloodCommand`, `GeoFireCommand`, `GeoTESSERACommand`,
// `GeoOlmoEarthCommand`). Studio knows the same rules, so the well says what a bundle must carry
// before anything is attached and, once a file is, reads its header and says which tensors it
// actually holds — the operator learns about a missing tensor from a checklist, not from a
// failed run.

/// What an Earth command's input bundle must hold: every `required` tensor, and at least one of
/// the `oneOf` options when there are any. An option is one or more tensors that must be present
/// together (Sentinel-1 bands with their day of year).
package struct StudioEarthInputRequirement: Equatable {
    package struct Option: Equatable, Identifiable {
        package let tensors: [String]

        package init(_ tensors: String...) {
            self.tensors = tensors
        }

        package var id: String { title }

        /// "S1_ASC + S1_ASC_DOY"
        package var title: String { tensors.joined(separator: " + ") }
    }

    package let required: [String]
    package let oneOf: [Option]

    package init(required: [String], oneOf: [Option] = []) {
        self.required = required
        self.oneOf = oneOf
    }

    /// The requirement an Earth template's command checks; nil for any other template.
    package static func requirement(for templateID: CommandTemplateID) -> StudioEarthInputRequirement? {
        switch templateID {
        case .geoFlood, .geoFire:
            return StudioEarthInputRequirement(required: ["S2L2A", "S1RTC", "DEM"])
        case .geoTessera:
            // Sentinel-1 comes in complete pairs: bands with their day of year.
            return StudioEarthInputRequirement(
                required: ["S2", "S2_DOY"],
                oneOf: [Option("S1_ASC", "S1_ASC_DOY"), Option("S1_DESC", "S1_DESC_DOY")]
            )
        case .geoOlmoEarth:
            return StudioEarthInputRequirement(
                required: ["TIMESTAMPS"],
                oneOf: [Option("S2L2A"), Option("S1RTC"), Option("LANDSAT")]
            )
        default:
            return nil
        }
    }

    /// One sentence for an empty well: "Needs S2L2A, S1RTC, and DEM." or "Needs S2 and S2_DOY,
    /// plus S1_ASC + S1_ASC_DOY or S1_DESC + S1_DESC_DOY."
    package var hint: String {
        var sentence = "Needs \(Self.list(required, conjunction: "and"))"
        if !oneOf.isEmpty {
            sentence += ", plus \(Self.list(oneOf.map(\.title), conjunction: "or"))"
        }
        return sentence + "."
    }

    /// The checklist for an attached bundle: which required tensors and which options its
    /// header carries, and which options are only half there.
    package func check(_ header: StudioSafetensorsHeader) -> StudioEarthInputCheck {
        // A safetensors header is a JSON object, so its names are unique.
        let byName = Dictionary(uniqueKeysWithValues: header.tensors.map { ($0.name, $0) })
        func row(_ tensors: [String], title: String) -> StudioEarthInputCheck.Row {
            let found = tensors.compactMap { byName[$0] }
            return StudioEarthInputCheck.Row(
                title: title,
                detail: found.count == tensors.count ? found.map(\.summary).joined(separator: " · ") : nil,
                present: found.map(\.name),
                missing: tensors.filter { byName[$0] == nil }
            )
        }
        return StudioEarthInputCheck(
            required: required.map { row([$0], title: $0) },
            oneOf: oneOf.map { row($0.tensors, title: $0.title) }
        )
    }

    /// "S2L2A, S1RTC, and DEM" / "S2 and S2_DOY" / "S2L2A"
    static func list(_ items: [String], conjunction: String) -> String {
        switch items.count {
        case 0: return ""
        case 1: return items[0]
        case 2: return "\(items[0]) \(conjunction) \(items[1])"
        default: return items.dropLast().joined(separator: ", ") + ", \(conjunction) " + items[items.count - 1]
        }
    }
}

/// An attached bundle read against its command's requirement: one row per required tensor and
/// per option, each with what the file holds for it.
package struct StudioEarthInputCheck: Equatable {
    package struct Row: Equatable, Identifiable {
        /// "S2L2A", or "S1_ASC + S1_ASC_DOY" for a pair.
        package let title: String
        /// The tensors' dtype and shape when the row is satisfied ("F32 [1, 12, 4, 256, 256]").
        package let detail: String?
        /// The row's tensors the file holds, and the ones it lacks.
        package let present: [String]
        package let missing: [String]

        package var id: String { title }

        package var isPresent: Bool { missing.isEmpty }

        /// Half a pair: bands without their day of year, or the reverse. The command refuses
        /// the bundle for it even when another option is complete.
        package var isIncomplete: Bool { !present.isEmpty && !missing.isEmpty }
    }

    package let required: [Row]
    package let oneOf: [Row]

    package var isSatisfied: Bool {
        required.allSatisfy(\.isPresent)
            && (oneOf.isEmpty || oneOf.contains(where: \.isPresent))
            && !oneOf.contains(where: \.isIncomplete)
    }

    /// The tensors the command will refuse the bundle for, in the requirement's words: "Missing
    /// S1RTC and DEM.", "S1_DESC needs S1_DESC_DOY.", or "Needs S1_ASC + S1_ASC_DOY or S1_DESC +
    /// S1_DESC_DOY."; nil when the bundle satisfies the requirement.
    package var message: String? {
        let missing = required.filter { !$0.isPresent }.map(\.title)
        var parts: [String] = []
        if !missing.isEmpty {
            parts.append("Missing \(StudioEarthInputRequirement.list(missing, conjunction: "and")).")
        }
        for option in oneOf where option.isIncomplete {
            parts.append(
                "\(StudioEarthInputRequirement.list(option.present, conjunction: "and")) needs "
                    + "\(StudioEarthInputRequirement.list(option.missing, conjunction: "and"))."
            )
        }
        if !oneOf.isEmpty, !oneOf.contains(where: { $0.isPresent || $0.isIncomplete }) {
            parts.append("Needs \(StudioEarthInputRequirement.list(oneOf.map(\.title), conjunction: "or")).")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " ")
    }
}
