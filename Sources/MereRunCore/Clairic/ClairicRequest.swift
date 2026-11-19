import Foundation
import MereRunQwenModel

enum ClairicError: LocalizedError {
    case invalid(String)
    case unsupported(String)
    case inference(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message), .unsupported(let message), .inference(let message): message
        }
    }
}

public struct ClairicRequest: Decodable, Sendable {
    struct Options: Decodable, Sendable {
        let return_state_text: Bool?
        let include_quantiles: Bool?
    }

    let state: ChatJSONValue?
    let fields: [String: ClairicField]
    let images: [ClairicMediaInput]?
    let documents: [ClairicMediaInput]?
    let options: Options?
    let reference_date: String?

    func validate() throws {
        guard (1...128).contains(fields.count), fields.keys.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 128 }) else {
            throw ClairicError.invalid("Clairic requires one to 128 fields with nonempty names of up to 128 bytes.")
        }
        guard (images?.count ?? 0) + (documents?.count ?? 0) <= 4 else {
            throw ClairicError.invalid("Clairic accepts at most four images or PDF pages.")
        }
        if (images?.isEmpty == false || documents?.isEmpty == false) && fields.values.contains(where: \.requiresTextOnlySource) {
            throw ClairicError.unsupported("Clairic source_only numbers currently require a text-only request.")
        }
        if reference_date != nil || options?.include_quantiles == true {
            throw ClairicError.unsupported("Reference dates and quantiles require Clairic's typed readout head.")
        }
        for (name, field) in fields { try field.validate(name: name) }
    }
}

struct ClairicField: Codable, Sendable {
    enum Kind: String, Codable, Sendable {
        case text, number, boolean, `enum`, object, list
        case date, time, datetime, duration, compute, estimate, forecast
    }

    enum Items: Codable, Sendable {
        case kind(Kind), object([String: ClairicField])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let kind = try? container.decode(Kind.self) {
                self = .kind(kind)
            } else {
                let item = try container.decode(ClairicField.self)
                guard item.type == .object, let properties = item.properties else {
                    throw ClairicError.invalid("A structured Clairic list item must be an object with properties.")
                }
                self = .object(properties)
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .kind(let kind): try container.encode(kind)
            case .object(let properties):
                try container.encode(ClairicField(type: .object, properties: properties))
            }
        }
    }

    var type: Kind
    var instructions: String?
    var nullable: Bool?
    var knowledge: Bool?
    var integer: Bool?
    var decimals: Int?
    var unit: String?
    var `enum`: [ChatJSONValue]?
    var minimum: Double?
    var maximum: Double?
    var exclusiveMinimum: Double?
    var exclusiveMaximum: Double?
    var minLength: Int?
    var maxLength: Int?
    var pattern: String?
    var properties: [String: ClairicField]?
    var items: Items?
    var max_items: Int?
    var source_only: Bool?

    var allowsNull: Bool { nullable ?? true }
    var printedTemporal: Kind? {
        source_only == true && [.date, .time, .datetime].contains(type) ? type : nil
    }

    var requiresTextOnlySource: Bool {
        source_only == true && (type == .number || listItem?.type == .number)
    }

    func validate(name: String) throws {
        switch type {
        case .text, .number, .boolean, .enum, .object, .list: break
        case _ where printedTemporal != nil: break
        default:
            throw ClairicError.unsupported("Field '\(name)' uses '\(type.rawValue)', which requires Clairic's typed readout head.")
        }
        if type == .list {
            guard let item = listItem, (0...32).contains(max_items ?? 8) else {
                throw ClairicError.invalid("List '\(name)' needs items and supports at most 32 entries.")
            }
            if source_only == true, item.type != .number, item.type != .date {
                throw ClairicError.invalid("List '\(name)' can use source_only for number or date items.")
            }
            try item.validate(name: name + "[]")
            return
        }
        if type == .object {
            guard let properties, (1...32).contains(properties.count),
                  properties.values.allSatisfy({ [.text, .number, .boolean, .enum].contains($0.type) }) else {
                throw ClairicError.invalid("Object '\(name)' requires one to 32 text, number, boolean, or enum properties.")
            }
            for (key, value) in properties { try value.validate(name: name + "." + key) }
        }
        if type == .enum, self.enum?.isEmpty != false {
            throw ClairicError.invalid("Enum '\(name)' needs a nonempty enum array.")
        }
        if let choices = self.enum, choices.isEmpty || !choices.allSatisfy({ $0 == .null ? allowsNull : acceptsType($0) }) {
            throw ClairicError.invalid("Field '\(name)' has enum choices incompatible with its type.")
        }
        if let decimals, !(0...15).contains(decimals) {
            throw ClairicError.invalid("Field '\(name)' requires decimals between zero and 15.")
        }
        if let minLength, minLength < 0 || minLength > (maxLength ?? Int.max) {
            throw ClairicError.invalid("Field '\(name)' has an invalid minimum string length.")
        }
        if let maxLength, maxLength < 0 {
            throw ClairicError.invalid("Field '\(name)' has an invalid maximum string length.")
        }
        if let pattern { _ = try NSRegularExpression(pattern: pattern) }
        if source_only == true && ((type != .number && printedTemporal == nil) || knowledge == true) {
            throw ClairicError.invalid("Field '\(name)' can use source_only for numbers or printed dates and times without knowledge answers.")
        }
    }

    var listItem: ClairicField? {
        switch items {
        case .kind(let kind) where [.text, .number, .boolean, .enum, .date].contains(kind):
            var item = self
            item.type = kind
            item.items = nil
            item.max_items = nil
            item.nullable = true
            return item
        case .object(let properties): return ClairicField(type: .object, nullable: true, properties: properties)
        default: return nil
        }
    }

    func grammar() throws -> ClairicGrammar.Value {
        if let choices = self.enum {
            let values = allowsNull ? choices + [.null] : choices
            return .choices(try values.map { try Self.json($0) })
        }
        let shape: ClairicGrammar.Value
        switch type {
        case .text: shape = .string
        case .number: shape = integer == true ? .integer : .number
        case .boolean: shape = .boolean
        case .object:
            shape = .object(try (properties ?? [:]).sorted { $0.key < $1.key }.map { (try Self.json($0.key), try $0.value.grammar()) })
        default: throw ClairicError.unsupported("Field type '\(type.rawValue)' has no slot grammar.")
        }
        return allowsNull ? .nullable(shape) : shape
    }

    /// The choice of type precedes constraints in the published slot prompt.
    func schemaJSON() throws -> String {
        if type == .enum {
            var choices = self.enum ?? []
            if allowsNull, !choices.contains(.null) { choices.append(.null) }
            return "{\"enum\":" + (try Self.json(choices)) + "}"
        }
        let kind = type == .text ? "string" : type == .number && integer == true ? "integer" : type.rawValue
        let typeValue = allowsNull ? try Self.json([kind, "null"]) : try Self.json(kind)
        var members = ["\"type\":" + typeValue]
        if let properties {
            let ordered = properties.sorted { $0.key < $1.key }
            let values = try ordered.map { (try Self.json($0.key)) + ":" + (try $0.value.schemaJSON()) }
            members += ["\"properties\":{" + values.joined(separator: ",") + "}",
                        "\"required\":" + (try Self.json(ordered.map(\.key))), "\"additionalProperties\":false"]
        }
        let constraints: [(String, ChatJSONValue?)] = [
            ("minLength", minLength.map { .number(Double($0)) }), ("maxLength", maxLength.map { .number(Double($0)) }),
            ("pattern", pattern.map(ChatJSONValue.string)), ("minimum", minimum.map(ChatJSONValue.number)),
            ("maximum", maximum.map(ChatJSONValue.number)), ("exclusiveMinimum", exclusiveMinimum.map(ChatJSONValue.number)),
            ("exclusiveMaximum", exclusiveMaximum.map(ChatJSONValue.number)), ("enum", self.enum.map(ChatJSONValue.array)),
        ]
        for (key, value) in constraints {
            if let value { members.append("\"\(key)\":" + (try Self.json(value))) }
        }
        return "{" + members.joined(separator: ",") + "}"
    }

    func normalized(_ value: ChatJSONValue) -> ChatJSONValue? {
        if value == .null { return allowsNull ? .null : nil }
        guard acceptsType(value), self.enum?.contains(value) ?? true else { return nil }
        switch value {
        case .string(let string):
            let length = string.unicodeScalars.count
            guard length >= (minLength ?? 0), length <= (maxLength ?? Int.max) else { return nil }
            if let pattern, string.range(of: pattern, options: .regularExpression) == nil { return nil }
        case .number(let number) where type == .number:
            guard containsNumber(number) else { return nil }
            if let places = integer == true ? 0 : decimals {
                let rounded = Self.round(number, places: places)
                return containsNumber(rounded) ? .number(rounded) : nil
            }
        case .object(let object) where type == .object:
            guard let properties, Set(object.keys) == Set(properties.keys) else { return nil }
            var result: [String: ChatJSONValue] = [:]
            for (name, field) in properties {
                guard let raw = object[name], let normalized = field.normalized(raw) else { return nil }
                result[name] = normalized
            }
            return .object(result)
        default: break
        }
        return value
    }

    private func acceptsType(_ value: ChatJSONValue) -> Bool {
        switch (type, value) {
        case (.text, .string), (.number, .number), (.boolean, .bool), (.object, .object), (.enum, _): true
        default: false
        }
    }

    private func containsNumber(_ value: Double) -> Bool {
        value.isFinite && value >= (minimum ?? -.infinity) && value <= (maximum ?? .infinity)
            && value > (exclusiveMinimum ?? -.infinity) && value < (exclusiveMaximum ?? .infinity)
            && (integer != true || value.rounded() == value)
    }

    private static func round(_ number: Double, places: Int) -> Double {
        if var value = Decimal(string: String(number), locale: Locale(identifier: "en_US_POSIX")) {
            var rounded = Decimal()
            NSDecimalRound(&rounded, &value, places, .plain)
            return Double(rounded.description) ?? number
        }
        // Decimal has a narrower exponent range than Double. For magnitudes
        // outside that range, the requested precision rounds to zero or
        // leaves the representable value unchanged.
        return abs(number) < 1 ? 0 : number
    }

    static func json(_ value: some Encodable) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

public struct ClairicPlan: Encodable, Sendable {
    public struct Field: Encodable, Sendable {
        public let name: String
        public let slots: Int
    }
    public let model: String
    public let input_tokens: Int
    public let image_count: Int
    public let fields: [Field]
}

public struct ClairicResponse: Codable, Sendable {
    public struct Answer: Codable, Sendable {
        public let value: ChatJSONValue
        public let valid: Bool
        public let complete: Bool
        public let confidence: Float
        public let p_null: Float
        public let span: ChatJSONValue
    }

    public struct Usage: Codable, Sendable {
        public let input_tokens: Int
        public let output_tokens: Int
    }

    public let model: String
    public let answers: [String: Answer]
    public let truncated: Bool
    public let usage: Usage
    public let state_text: String?
}
