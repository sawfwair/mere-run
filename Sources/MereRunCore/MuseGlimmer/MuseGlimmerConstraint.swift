import Foundation
import MLX

/// Native recipient/ATEM grammar. Schema matching starts inside a completed
/// channel header, never at the reasoning-to-header boundary. Captured grammar
/// spans, rather than regexes over generated text, delimit tool arguments.
struct MuseGlimmerConstraint {
    enum CaptureKind {
        case reasoning, answer, endReasoning, endAnswer
        case parameter(String, Bool), call(ToolDefinition)
    }
    var grammar: ChatSchemaGrammar
    let kinds: [Int: CaptureKind]
    let answerSchema: [String: ChatJSONValue]?
    private var pendingUTF8: [UInt8] = []
    var isComplete: Bool { grammar.isComplete && pendingUTF8.isEmpty }
    private var raw: [Unicode.Scalar] = []
    private var body: (reasoning: Bool, offset: Int)?
    private var arguments: [String: ChatJSONValue] = [:]
    private(set) var text = ""
    private(set) var reasoning = ""
    private(set) var reasoningBlocks = 0
    private(set) var calls: [ToolCall] = []
    private(set) var events: [ChatGenerationEvent] = []

    init(request: ChatRequest) throws {
        guard let constraint = request.outputConstraint else { throw ChatRequestIssue("output_schema", "missing constraint") }
        guard request.stopSequences.isEmpty else { throw ChatRequestIssue("stop_sequences", "native constrained output does not support stop sequences") }
        answerSchema = constraint.mode == .json ? constraint.schema : nil
        var builder = Builder(compiler: .init(root: constraint.schema))
        let reasoningHeader = builder.capture(.reasoning, builder.header("self"))
        let answerHeader = builder.capture(.answer, builder.header("user"))
        let endReasoning = builder.capture(.endReasoning, builder.literal("<|eom|>"))
        let endAnswer = builder.capture(.endAnswer, builder.literal("<|eot|>"))
        let thinking = builder.until("<|eom|>")
        let answer = constraint.mode == .json ? try builder.compiler.schema(constraint.schema) : builder.until("<|eot|>")
        var packets: [Int] = []
        if constraint.mode == .tools {
            for tool in request.tools ?? [] {
                if case .function(let name) = request.toolChoice, tool.name != name { continue }
                packets.append(try builder.packet(tool))
            }
            guard !packets.isEmpty else { throw ChatRequestIssue("tools", "strict generation requires an active tool") }
        }
        let limit = constraint.mode == .tools ? (request.parallelToolCalls ? 16 : 1) : 0
        let states = (0...limit).map { _ in builder.reserve() }
        for count in 0...limit {
            let state = states[count]
            builder.add(state, [.rule(reasoningHeader), .rule(thinking), .rule(endReasoning)]
                + builder.literal("<|start|>assistant") + [.rule(state)])
            if constraint.mode == .json || request.toolChoice == .auto || count > 0 {
                builder.add(state, [.rule(answerHeader), .rule(answer), .rule(endAnswer)])
            }
            if count < limit {
                for packet in packets {
                    // One invocation per native packet; parallel requests can emit
                    // up to sixteen packets, preserving each recipient/name pair.
                    builder.add(state, [.rule(packet)] + builder.literal("<|eot|>"))
                    builder.add(state, [.rule(packet)] + builder.literal("<|eom|><|start|>assistant") + [.rule(states[count + 1])])
                }
            }
        }
        let prefix = builder.compiler.rule([[], builder.literal("<|start|>assistant")])
        builder.compiler.productions[0] = .init(lhs: 0, rhs: [.rule(prefix), .rule(states[0])])
        grammar = try ChatSchemaGrammar(compiler: builder.compiler, labels: builder.labels)
        kinds = builder.kinds
    }

    mutating func acceptBytes(_ bytes: [UInt8]) throws -> Bool {
        guard !bytes.isEmpty, let split = ChatUTF8Prefix.split(pendingUTF8 + bytes),
              try accept(split.text), split.pending.isEmpty || grammar.acceptsUTF8Prefix(split.pending) else { return false }
        pendingUTF8 = split.pending
        return true
    }

    mutating func accept(_ delta: String) throws -> Bool {
        guard grammar.accept(delta) else { return false }
        raw.append(contentsOf: delta.unicodeScalars)
        let captures = grammar.takeCaptures().sorted { $0.end < $1.end }
        do {
            for capture in captures {
                switch kinds[capture.label]! {
                case .reasoning: body = (true, capture.end); reasoningBlocks += 1
                case .answer: body = (false, capture.end)
                case .endReasoning:
                    emit(until: capture.start); body = nil
                case .endAnswer:
                    emit(until: capture.start); body = nil
                    if let answerSchema {
                        let value = try JSONDecoder().decode(ChatJSONValue.self, from: Data(text.utf8))
                        try ChatToolValidation.validate(value, schema: answerSchema)
                    }
                case .parameter(let name, let string):
                    let source = value(capture.start..<capture.end)
                    let open = "<atem:parameter name=\"" + Self.escape(name) + "\">"
                    let inner = String(String.UnicodeScalarView(source.unicodeScalars.dropFirst(open.unicodeScalars.count).dropLast("</atem:parameter>".unicodeScalars.count)))
                    guard arguments[name] == nil else { throw ChatRequestIssue("output_schema.generation", "duplicate constrained parameter") }
                    arguments[name] = string ? .string(Self.unescape(inner)) : try JSONDecoder().decode(ChatJSONValue.self, from: Data(inner.utf8))
                case .call(let tool):
                    try ChatToolValidation.validate(.object(arguments), schema: tool.parametersJSONSchema)
                    calls.append(.init(name: tool.name, jsonArguments: arguments))
                    arguments = [:]
                }
            }
        } catch {
            throw ChatRequestIssue("output_schema.generation", "native constrained output failed final schema validation: \(error.localizedDescription)")
        }
        if let body {
            let marker = Array((body.reasoning ? "<|eom|>" : "<|eot|>").unicodeScalars)
            var withheld = 0
            for length in 1..<marker.count where raw.count - body.offset >= length {
                if raw.suffix(length).elementsEqual(marker.prefix(length)) { withheld = length }
            }
            emit(until: raw.count - withheld)
        }
        return true
    }

    private mutating func emit(until end: Int) {
        guard let current = body, end > current.offset else { return }
        let delta = value(current.offset..<end)
        if current.reasoning { reasoning += delta; events.append(.reasoning(delta)) }
        else { text += delta; events.append(.text(delta)) }
        body = (current.reasoning, end)
    }
    private func value(_ range: Range<Int>) -> String { String(String.UnicodeScalarView(raw[range])) }
    static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;").replacingOccurrences(of: "'", with: "&#39;")
    }
    private static func unescape(_ text: String) -> String {
        text.replacingOccurrences(of: "&lt;", with: "<").replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"").replacingOccurrences(of: "&#39;", with: "'").replacingOccurrences(of: "&amp;", with: "&")
    }
}

private extension MuseGlimmerConstraint {
    struct Builder {
        var compiler: ChatSchemaGrammar.Compiler
        var labels: [Int: Int] = [:]
        var kinds: [Int: CaptureKind] = [:]
        func literal(_ text: String) -> [ChatSchemaGrammar.Symbol] { compiler.literal(text) }
        mutating func reserve() -> Int { let id = compiler.nextRule; compiler.nextRule += 1; return id }
        mutating func add(_ id: Int, _ symbols: [ChatSchemaGrammar.Symbol]) { compiler.productions.append(.init(lhs: id, rhs: symbols)) }
        mutating func capture(_ kind: CaptureKind, _ symbols: [ChatSchemaGrammar.Symbol]) -> Int {
            let rule = compiler.rule([symbols]); let label = kinds.count
            labels[rule] = label; kinds[label] = kind
            return rule
        }
        mutating func header(_ recipient: String) -> [ChatSchemaGrammar.Symbol] {
            let space = compiler.rule([[], literal(" ")])
            return [.rule(space)] + literal("to=" + recipient + "<|message|>")
        }
        /// A left-recursive DFA excludes complete native ending markers.
        /// Partial and overlapping marker prefixes remain ordinary body text.
        mutating func until(_ marker: String) -> Int {
            let markers = [marker, "<|eom|>", "<|eot|>"].map { Array($0.unicodeScalars) }
            var prefixes: [[Unicode.Scalar]] = [[]]
            for word in markers {
                for length in 1..<word.count {
                    let prefix = Array(word.prefix(length))
                    if !prefixes.contains(prefix) { prefixes.append(prefix) }
                }
            }
            let alphabet = Set(markers.flatMap { $0 })
            let states = prefixes.map { _ in reserve() }
            add(states[0], [])
            for index in states.indices {
                add(states[0], [.rule(states[index]), .except(alphabet)])
                for char in alphabet {
                    let candidate = prefixes[index] + [char]
                    if markers.contains(where: { candidate.suffix($0.count).elementsEqual($0) }) { continue }
                    let next = prefixes.indices.filter { candidate.suffix(prefixes[$0].count).elementsEqual(prefixes[$0]) }
                        .max { prefixes[$0].count < prefixes[$1].count }!
                    add(states[next], [.rule(states[index]), .character(char)])
                }
            }
            return compiler.rule(states.map { [.rule($0)] })
        }
        mutating func packet(_ tool: ToolDefinition) throws -> Int {
            guard tool.name != "self", tool.name != "user", !tool.name.contains("<|"), !tool.name.contains("\n"), !tool.name.contains("\r") else {
                throw ChatRequestIssue("tools.strict", "Glimmer native recipient name is reserved or contains a channel delimiter; rename tool '\(tool.name)'")
            }
            let root = tool.parametersJSONSchema
            try ChatToolValidation.validateSchema(root)
            _ = try ChatSchemaGrammar(schema: root)
            compiler.root = root
            let object = try resolve(root, root: root)
            guard object["type"] == .string("object"), object["additionalProperties"] == .bool(false), object["anyOf"] == nil else {
                throw ChatRequestIssue("tools.strict", "Glimmer strict tools require a closed object parameter schema")
            }
            let properties = object["properties"]?.objectValue ?? [:]
            let required = Set(object["required"]?.arrayValue?.compactMap(\.stringValue) ?? [])
            let space = reserve(); add(space, [])
            for char in " \n\r\t".unicodeScalars { add(space, [.rule(space), .character(char)]) }
            var tail = compiler.rule([[.rule(space)]])
            for name in properties.keys.sorted().reversed() {
                let schema = try resolve(properties[name]!.objectValue!, root: root)
                let rawString = try isString(schema)
                let argument: Int
                if rawString {
                    if let values = schema["enum"]?.arrayValue ?? schema["const"].map({ [$0] }) {
                        guard values.allSatisfy({ value in value.stringValue!.unicodeScalars.allSatisfy {
                            $0.value >= 32 || $0 == "\n" || $0 == "\r" || $0 == "\t"
                        } }) else { throw ChatRequestIssue("tools.strict", "Glimmer raw string constants cannot contain XML control characters") }
                        argument = compiler.rule(values.map { literal(MuseGlimmerConstraint.escape($0.stringValue!)) })
                    } else {
                        let string = reserve(); add(string, [])
                        add(string, [.rule(string), .xmlCharacter])
                        for entity in ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;"] { add(string, [.rule(string)] + literal(entity)) }
                        argument = string
                    }
                } else { argument = try compiler.schema(schema) }
                let parameter = capture(.parameter(name, rawString), literal("<atem:parameter name=\"" + MuseGlimmerConstraint.escape(name) + "\">")
                    + [.rule(argument)] + literal("</atem:parameter>"))
                let field: [ChatSchemaGrammar.Symbol] = [.rule(space), .rule(parameter), .rule(tail)]
                tail = compiler.rule(required.contains(name) ? [field] : [field, [.rule(tail)]])
            }
            let invocation = capture(.call(tool), literal("<atem:invoke name=\"" + MuseGlimmerConstraint.escape(tool.name) + "\">")
                + [.rule(tail)] + literal("</atem:invoke>"))
            return compiler.rule([header(tool.name) + [.rule(space)] + literal("<atem:function_calls>") + [.rule(space), .rule(invocation), .rule(space)] + literal("</atem:function_calls>") + [.rule(space)]])
        }
        func resolve(_ schema: [String: ChatJSONValue], root: [String: ChatJSONValue], depth: Int = 0) throws -> [String: ChatJSONValue] {
            guard depth < 32 else { throw ChatRequestIssue("tools.strict", "schema reference nesting exceeds the native limit") }
            guard let ref = schema["$ref"]?.stringValue else { return schema }
            var value: ChatJSONValue = .object(root)
            for part in ref.dropFirst(2).split(separator: "/") {
                value = value.objectValue?[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
            }
            guard let target = value.objectValue else { throw ChatRequestIssue("tools.strict", "invalid local schema reference") }
            return try resolve(target, root: root, depth: depth + 1)
        }
        func isString(_ schema: [String: ChatJSONValue]) throws -> Bool {
            if schema["anyOf"] != nil || schema["type"]?.arrayValue != nil {
                throw ChatRequestIssue("tools.strict", "Glimmer top-level parameters require one unambiguous type; move unions inside an object or array")
            }
            if schema["type"] == .string("string") { return true }
            if let values = schema["enum"]?.arrayValue ?? schema["const"].map({ [$0] }) {
                if values.allSatisfy({ $0.stringValue != nil }) { return true }
                if values.contains(where: { $0.stringValue != nil }) { throw ChatRequestIssue("tools.strict", "Glimmer parameter enums cannot mix raw strings and JSON types") }
            }
            return false
        }
    }
}

/// Final-target sampling keeps the committed grammar state incremental. Probes
/// are value copies, so rejected candidates cannot publish events or calls.
func museGlimmerConstrainedDecode(
    request: ChatRequest, initialLogits: MLXArray, sampling: GenerationConfig,
    budget: Int, history: [Int], decode: ([Int]) -> String,
    tokenBytes: ((Int) -> [UInt8])? = nil,
    forward: (Int) throws -> MLXArray, progress: (@Sendable (ChatProgress) -> Void)?
) throws -> (ChatSchemaDecodeResult, MuseGlimmerConstraint) {
    let started = Date()
    var state = try MuseGlimmerConstraint(request: request)
    var logits = initialLogits, tokens: [Int] = [], committed = ""
    var first: Double?
    for _ in 0..<budget {
        try Task.checkCancellation()
        var config = sampling
        var accepted: (Int, String, MuseGlimmerConstraint)?
        for _ in 0..<256 {
            try Task.checkCancellation()
            let token = sampleToken(logits: logits[0, -1, 0...], config: config, previousTokens: history + tokens)
            guard !config.bannedTokens.contains(token) else { break }
            if let tokenBytes {
                var probe = state
                if try probe.acceptBytes(tokenBytes(token)) { accepted = (token, "", probe); break }
                config.bannedTokens.append(token)
                continue
            }
            var text = decode(tokens + [token])
            let partial = text.last == "\u{FFFD}"
            while text.last == "\u{FFFD}" { text.removeLast() }
            var probe = state
            if text.utf8.starts(with: committed.utf8), text != committed || partial {
                let delta = String(decoding: text.utf8.dropFirst(committed.utf8.count), as: UTF8.self)
                if try probe.accept(delta) { accepted = (token, text, probe); break }
            }
            config.bannedTokens.append(token)
        }
        if accepted == nil, let tokenBytes {
            // Rare schema literals can lie outside the model's first candidate
            // draws. Build the exact admissible mask before applying the same
            // sampling controls; never fall back to greedy or weaken the grammar.
            var firstBytes = Array(repeating: false, count: 256)
            for byte in 0..<256 {
                var probe = state
                firstBytes[byte] = (try? probe.acceptBytes([UInt8(byte)])) == true
            }
            let vocabulary = logits.shape.last!
            var mask = Array(repeating: false, count: vocabulary)
            let banned = Set(sampling.bannedTokens)
            for id in 0..<vocabulary where !banned.contains(id) {
                if id % 256 == 0 { try Task.checkCancellation() }
                let bytes = tokenBytes(id)
                guard let first = bytes.first, firstBytes[Int(first)] else { continue }
                var probe = state
                mask[id] = (try? probe.acceptBytes(bytes)) == true
            }
            if mask.contains(true) {
                let filtered = MLX.where(MLXArray(mask), logits[0, -1, 0...], MLXArray(-Float.infinity))
                let token = sampleToken(logits: filtered, config: sampling, previousTokens: history + tokens)
                if mask.indices.contains(token), mask[token] {
                    var probe = state
                    if try probe.acceptBytes(tokenBytes(token)) { accepted = (token, "", probe) }
                }
            }
        }
        guard let (token, text, next) = accepted else { throw ChatRequestIssue("output_schema.generation", "no admissible native token for the requested schema and tokenizer") }
        for event in next.events.dropFirst(state.events.count) { progress?(.init(stage: .generating, event: event)) }
        state = next; committed = text; tokens.append(token)
        if first == nil { first = Date().timeIntervalSince(started) }
        if state.isComplete { return (.init(tokens: tokens, decodeSeconds: Date().timeIntervalSince(started), firstTokenSeconds: first), state) }
        logits = try forward(token); MLX.eval(logits)
    }
    throw ChatRequestIssue("output_schema.generation", "output budget ended before the native constrained turn completed; increase max_tokens")
}
