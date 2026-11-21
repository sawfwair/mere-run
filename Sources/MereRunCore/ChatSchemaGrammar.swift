import Foundation

/// A grammar compiled from a bounded schema subset. Object keys use a stable
/// order; every completed derivation satisfies the schema. Unsupported keywords
/// fail at preparation rather than weakening constraints.
public struct ChatSchemaGrammar: Sendable {
    enum Symbol: Hashable, Sendable {
        case rule(Int), character(Unicode.Scalar), except(Set<Unicode.Scalar>), stringCharacter, hex, digit, nonzero
        func matches(_ character: Unicode.Scalar) -> Bool {
            switch self {
            case .character(let value): return value == character
            case .except(let excluded): return !excluded.contains(character)
            case .stringCharacter:
                return character != "\"" && character != "\\" && character.value >= 32
            case .hex: return character.isASCII && "0123456789abcdefABCDEF".contains(character)
            case .digit: return character.isASCII && "0123456789".contains(character)
            case .nonzero: return character.isASCII && "123456789".contains(character)
            case .rule: return false
            }
        }
    }
    struct Production: Sendable { let lhs: Int; let rhs: [Symbol] }
    struct Item: Hashable, Sendable { let production: Int; let dot: Int; let origin: Int }
    let productions: [Production]
    let rules: [Int: [Int]]
    struct Capture: Hashable, Sendable { let label: Int; let start: Int; let end: Int }
    let labels: [Int: Int]
    private var captures: [Capture] = []
    private var chart: [Set<Item>]
    private var states = 1
    var nextTerminals: [Symbol] {
        chart.last!.compactMap { item in
            let rhs = productions[item.production].rhs
            return item.dot < rhs.count ? rhs[item.dot] : nil
        }
    }
    public var isComplete: Bool { chart.last!.contains(Item(production: 0, dot: productions[0].rhs.count, origin: 0)) }

    public static let schemaKeywords: Set<String> = [
        "$schema", "$id", "$defs", "definitions", "$ref", "title", "description", "default", "examples",
        "type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf", "minItems", "maxItems"
    ]

    public init(schema: [String: ChatJSONValue]) throws {
        var compiler = Compiler(root: schema)
        let root = try compiler.schema(schema)
        compiler.productions[0] = Production(lhs: 0, rhs: [.rule(root)])
        try self.init(compiler: compiler)
    }

    init(compiler: Compiler, labels: [Int: Int] = [:]) throws {
        guard compiler.productions.count <= 4096,
              compiler.productions.reduce(0, { $0 + $1.rhs.count }) <= 131_072 else {
            throw ChatRequestIssue("output_schema", "compiled grammar exceeds complexity limits")
        }
        productions = compiler.productions
        let compiled = compiler.productions
        rules = Dictionary(grouping: compiled.indices, by: { compiled[$0].lhs })
        self.labels = labels
        chart = [[Item(production: 0, dot: 0, origin: 0)]]
        close()
        guard states <= 1_000_000 else { throw ChatRequestIssue("output_schema", "grammar closure exceeds state limits") }
    }

    mutating func takeCaptures() -> [Capture] {
        defer { captures.removeAll(keepingCapacity: true) }
        return captures
    }

    @discardableResult
    public mutating func accept(_ text: String) -> Bool {
        for character in text.unicodeScalars {
            let position = chart.count - 1
            var next: Set<Item> = []
            for item in chart[position] {
                let rhs = productions[item.production].rhs
                if item.dot < rhs.count, rhs[item.dot].matches(character) {
                    next.insert(Item(production: item.production, dot: item.dot + 1, origin: item.origin))
                }
            }
            guard !next.isEmpty, next.count <= 32_768, chart.count < 131_072 else { return false }
            chart.append(next)
            states += next.count
            close()
            guard states <= 1_000_000 else { return false }
        }
        return true
    }

    private mutating func close() {
        let position = chart.count - 1
        var pending = Array(chart[position])
        func advanced(_ item: Item) -> Item { Item(production: item.production, dot: item.dot + 1, origin: item.origin) }
        while let item = pending.popLast() {
            let production = productions[item.production]
            var additions: [Item] = []
            if item.dot == production.rhs.count {
                if let label = labels[production.lhs] {
                    let capture = Capture(label: label, start: item.origin, end: position)
                    if !captures.contains(capture) { captures.append(capture) }
                }
                for parent in chart[item.origin] {
                    let rhs = productions[parent.production].rhs
                    if parent.dot < rhs.count, rhs[parent.dot] == .rule(production.lhs) { additions.append(advanced(parent)) }
                }
            } else if case .rule(let rule) = production.rhs[item.dot] {
                additions += (rules[rule] ?? []).map { Item(production: $0, dot: 0, origin: position) }
                // Nullable completions may have been visited before this item.
                if chart[position].contains(where: {
                    $0.origin == position && productions[$0.production].lhs == rule
                        && $0.dot == productions[$0.production].rhs.count
                }) { additions.append(advanced(item)) }
            }
            for addition in additions where chart[position].insert(addition).inserted {
                states += 1
                guard states <= 1_000_000 else { return }
                pending.append(addition)
            }
        }
    }
}

extension ChatSchemaGrammar {
    struct Compiler {
        var root: [String: ChatJSONValue]
        var productions: [Production] = [Production(lhs: 0, rhs: [])]
        var nextRule = 1
        var references: Set<String> = []
        mutating func rule(_ alternatives: [[Symbol]]) -> Int {
            let id = nextRule; nextRule += 1
            for rhs in alternatives { productions.append(Production(lhs: id, rhs: rhs)) }
            return id
        }
        func literal(_ text: String) -> [Symbol] { text.unicodeScalars.map(Symbol.character) }
        func issue(_ message: String) -> ChatRequestIssue { ChatRequestIssue("output_schema", message) }
        mutating func schema(_ schema: [String: ChatJSONValue], depth: Int = 0) throws -> Int {
            guard depth < 32, productions.count < 4096 else { throw issue("schema exceeds grammar complexity limits") }
            if let keyword = Set(schema.keys).subtracting(ChatSchemaGrammar.schemaKeywords).sorted().first {
                throw issue("constrained decoding does not support '\(keyword)'; use the published constrained_schema_subset")
            }
            if let reference = schema["$ref"]?.stringValue {
                guard !references.contains(reference), reference.hasPrefix("#/"), schema.keys.allSatisfy({ ["$ref", "$defs", "definitions", "$schema", "$id", "title", "description"].contains($0) }) else {
                    throw issue("references must be nonrecursive local references without sibling constraints")
                }
                var value: ChatJSONValue = .object(root)
                for key in reference.dropFirst(2).split(separator: "/") {
                    value = value.objectValue?[String(key).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
                }
                guard let target = value.objectValue else { throw issue("reference does not name a schema") }
                references.insert(reference)
                defer { references.remove(reference) }
                return try self.schema(target, depth: depth + 1)
            }
            if let values = schema["enum"]?.arrayValue ?? schema["const"].map({ [$0] }) {
                guard !values.isEmpty, values.count <= 256 else { throw issue("enum requires 1 through 256 values") }
                for value in values { try ChatToolValidation.validate(value, schema: schema, root: root) }
                return rule(values.map { literal($0.jsonString) })
            }
            if let alternatives = schema["anyOf"]?.arrayValue {
                guard schema.keys.allSatisfy({ ["anyOf", "$defs", "definitions", "title", "description", "$schema", "$id"].contains($0) }),
                      !alternatives.isEmpty, alternatives.count <= 32 else { throw issue("anyOf requires 1 through 32 alternatives without sibling constraints") }
                var rules: [[Symbol]] = []
                for value in alternatives {
                    guard let child = value.objectValue else { throw issue("alternatives must be schemas") }
                    rules.append([.rule(try self.schema(child, depth: depth + 1))])
                }
                return rule(rules)
            }
            if let types = schema["type"]?.arrayValue {
                var alternatives: [[Symbol]] = []
                for type in types {
                    var child = schema; child["type"] = type
                    alternatives.append([.rule(try self.schema(child, depth: depth + 1))])
                }
                guard !alternatives.isEmpty else { throw issue("type union must not be empty") }
                return rule(alternatives)
            }
            switch schema["type"]?.stringValue {
            case "object":
                guard schema["additionalProperties"] == .bool(false) else { throw issue("constrained objects require additionalProperties: false") }
                let properties = schema["properties"]?.objectValue ?? [:]
                let required = Set(schema["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
                guard required.isSubset(of: Set(properties.keys)), properties.count <= 64 else { throw issue("required keys must exist; objects support at most 64 properties") }
                return try object(properties, required: required, depth: depth)
            case "array":
                guard let items = schema["items"]?.objectValue else { throw issue("arrays require an items schema") }
                let value = try self.schema(items, depth: depth + 1)
                let minimum = try quantity(schema["minItems"], fallback: 0)
                let maximum = try schema["maxItems"].map { try quantity($0, fallback: 0) }
                guard maximum == nil || maximum! >= minimum else { throw issue("maxItems must be at least minItems") }
                let list: Int
                if let maximum {
                    var tail = rule([[]])
                    if maximum > 0 {
                        for count in stride(from: maximum - 1, through: 0, by: -1) {
                            let field = (count > 0 ? literal(",") : []) + [.rule(value), .rule(tail)]
                            tail = rule(count < minimum ? [field] : [[], field])
                        }
                    }
                    list = tail
                } else {
                    let tail = nextRule; nextRule += 1
                    productions.append(Production(lhs: tail, rhs: []))
                    productions.append(Production(lhs: tail, rhs: [.rule(tail)] + literal(",") + [.rule(value)]))
                    var fields: [Symbol] = []
                    for index in 0..<max(1, minimum) { fields += (index > 0 ? literal(",") : []) + [.rule(value)] }
                    list = rule(minimum == 0 ? [[], fields + [.rule(tail)]] : [fields + [.rule(tail)]])
                }
                return rule([literal("[") + [.rule(list)] + literal("]")])
            case "string":
                let body = nextRule; nextRule += 1
                productions.append(Production(lhs: body, rhs: []))
                productions.append(Production(lhs: body, rhs: [.rule(body), .stringCharacter]))
                for value in ["\"", "\\", "/", "b", "f", "n", "r", "t"] {
                    productions.append(Production(lhs: body, rhs: [.rule(body)] + literal("\\" + value)))
                }
                let normalFirst = rule("0123456789abcefABCEF".map { [.character($0.unicodeScalars.first!)] })
                let d = rule([literal("d"), literal("D")])
                let low = rule("01234567".map { [.character($0.unicodeScalars.first!)] })
                let high = rule("89abAB".map { [.character($0.unicodeScalars.first!)] })
                let trailing = rule("cdefCDEF".map { [.character($0.unicodeScalars.first!)] })
                let unicode = rule([
                    [.rule(normalFirst), .hex, .hex, .hex], [.rule(d), .rule(low), .hex, .hex],
                    [.rule(d), .rule(high), .hex, .hex] + literal("\\u") + [.rule(d), .rule(trailing), .hex, .hex]
                ])
                productions.append(Production(lhs: body, rhs: [.rule(body)] + literal("\\u") + [.rule(unicode)]))
                return rule([literal("\"") + [.rule(body)] + literal("\"")])
            case "integer", "number":
                let digits = nextRule; nextRule += 1
                productions.append(Production(lhs: digits, rhs: []))
                productions.append(Production(lhs: digits, rhs: [.rule(digits), .digit]))
                let integer = rule([literal("0"), [.nonzero, .rule(digits)]])
                let sign = rule([[], literal("-")])
                var value: [Symbol] = [.rule(sign), .rule(integer)]
                if schema["type"] == .string("number") {
                    let fraction = rule([[], literal(".") + [.digit, .rule(digits)]])
                    let expSign = rule([[], literal("+"), literal("-")])
                    let exponent = rule([[], literal("e") + [.rule(expSign), .digit, .rule(digits)]])
                    value += [.rule(fraction), .rule(exponent)]
                }
                return rule([value])
            case "boolean": return rule([literal("true"), literal("false")])
            case "null": return rule([literal("null")])
            default: throw issue("every constrained schema must declare type, enum, const, or anyOf")
            }
        }
        mutating func object(_ properties: [String: ChatJSONValue], required: Set<String>, depth: Int) throws -> Int {
            var empty = rule([[]]), nonempty = empty
            for key in properties.keys.sorted().reversed() {
                let child = try schema(properties[key]!.objectValue!, depth: depth + 1)
                let field = literal(ChatJSONValue.string(key).jsonString + ":") + [.rule(child)]
                let previousEmpty = empty, previousNonempty = nonempty
                empty = rule(required.contains(key) ? [field + [.rule(previousNonempty)]] : [field + [.rule(previousNonempty)], [.rule(previousEmpty)]])
                nonempty = rule(required.contains(key) ? [literal(",") + field + [.rule(previousNonempty)]] : [literal(",") + field + [.rule(previousNonempty)], [.rule(previousNonempty)]])
            }
            return rule([literal("{") + [.rule(empty)] + literal("}")])
        }
        func quantity(_ value: ChatJSONValue?, fallback: Int) throws -> Int {
            guard let value else { return fallback }
            guard case .number(let number) = value, number >= 0, number <= 128, number.rounded() == number else {
                throw issue("array bounds must be integers from 0 through 128")
            }
            return Int(number)
        }
    }
}
