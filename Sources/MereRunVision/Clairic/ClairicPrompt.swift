import MereRunCore
#if !os(iOS)
import Crypto
import Foundation

struct ClairicPrompt {
    static let header = "<|im_start|>user\n"
    static let footer = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    struct Slot {
        let field: String
        let index: Int?
        let spec: ClairicField
        let instruction: String
        var temporal: ClairicField.Kind?
    }

    static func slots(for request: ClairicRequest, compactLists: Bool) throws -> [Slot] {
        try request.fields.sorted { $0.key < $1.key }.flatMap { name, field in
            guard let item = field.listItem else {
                let spec = generationSpec(field)
                return [Slot(field: name, index: nil, spec: spec, instruction: try instruction(name, spec),
                             temporal: field.printedTemporal)]
            }
            let count = field.max_items ?? 8
            let generatedItem = generationSpec(item)
            return try (0..<count).map { offset in
                let index = offset + 1
                let childName = listName(name, index: index)
                let description = (field.instructions ?? "") + (item.printedTemporal == nil ? "" : ClairicPrintedTemporal.instruction)
                let element = "Item \(index) of \(name), following requested order and duplicate policy. "
                let missing = "Return null if this item does not exist."
                var spec = generatedItem
                spec.instructions = "Parent list \(name): \(description)\nRequested element: "
                    + (element + description + " " + missing).trimmingCharacters(in: .whitespacesAndNewlines)
                let compactSuffix = "\nList: \(name)\nRequested element: \(element)\(missing)\nField: \(childName)"
                let prompt = if compactLists, generatedItem.type != .object {
                    try scalarIntroduction(generatedItem)
                        + "\nInstructions: " + description + numberUnit(generatedItem) + compactSuffix
                } else {
                    try instruction(childName, spec)
                }
                return Slot(field: name, index: index, spec: spec, instruction: prompt, temporal: item.printedTemporal)
            }
        }
    }

    private static func generationSpec(_ field: ClairicField) -> ClairicField {
        guard field.printedTemporal != nil else { return field }
        return ClairicField(type: .text, instructions: (field.instructions ?? "") + ClairicPrintedTemporal.instruction,
                            nullable: field.nullable)
    }

    private static func listName(_ parent: String, index: Int) -> String {
        let suffix = ".item\(index)"
        guard parent.unicodeScalars.count + suffix.count > 64 else { return parent + suffix }
        let digest = SHA256.hash(data: Data(parent.utf8)).prefix(4).map { String(format: "%02x", $0) }.joined()
        return String(parent.unicodeScalars.prefix(64 - suffix.count - 9)) + "_" + digest + suffix
    }

    private static func scalarIntroduction(_ spec: ClairicField, name: String? = nil) throws -> String {
        let extract = "Extract the requested value from the supplied input, following the field instructions and scope."
        let knowledge = "Answer the requested field using reliable prior knowledge and any relevant supplied input. "
            + "This field explicitly permits knowledge: true. The answer does not need to appear in the document. "
            + "Missing source evidence does not make a known factual answer unavailable. "
            + "Return null only when neither reliable prior knowledge nor the input determines the answer. "
            + "Follow the field instructions and scope."
        let intro = (spec.knowledge == true ? knowledge : extract)
            + " Return exactly one JSON scalar matching the field schema: quote and escape strings, "
            + "write numbers without digit-group separators, and use null only when the value is unavailable and null is allowed. "
            + "Zero is a present number. Preserve repeated source words and list entries. End immediately after the value."
        return intro + (name.map { "\nField: " + $0 } ?? "") + "\nSchema: " + (try spec.schemaJSON())
    }

    private static func numberUnit(_ spec: ClairicField) -> String {
        guard spec.type == .number, let unit = spec.unit, !unit.isEmpty else { return "" }
        return " Return the value in \(unit)."
    }

    private static func instruction(_ name: String, _ spec: ClairicField) throws -> String {
        var text = spec.instructions ?? ""
        if spec.type == .object {
            return "Extract one complete requested row from the supplied document. "
                + "Return one JSON object matching the schema, or null only if that row does not exist. "
                + "Keep all properties on the same physical row. Use null for a missing cell in an existing row. "
                + "Preserve distinct rows with identical values. Follow the requested row scope and exclusions. "
                + "End immediately after the object or null."
                + "\nField: " + name + "\nSchema: " + (try spec.schemaJSON())
                + "\nInstructions: " + text + "\nProperty instructions: " + (try ClairicField.json(spec.properties ?? [:]))
        }
        if spec.type == .boolean {
            text += " This field is a Boolean statement about the input. Evaluate its condition using the supplied evidence: "
                + "return true when the condition is affirmed, false when the condition is explicitly negated, "
                + "and null only when evidence does not determine whether it holds. The literal words true and false do not need to be printed."
        }
        return try scalarIntroduction(spec, name: name) + "\nInstructions: " + text + numberUnit(spec)
    }
}
#endif
