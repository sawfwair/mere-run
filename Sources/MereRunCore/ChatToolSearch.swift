import Foundation

public struct ChatToolSearchFailure: Error, LocalizedError, Hashable, Sendable {
    public let code: String
    public let message: String
    public var errorDescription: String? { message }
    public init(code: String, message: String) { self.code = code; self.message = message }
}

public struct ChatToolSearch: Hashable, Sendable {
    public enum Mode: String, Hashable, Sendable { case regex, bm25 }
    public let catalog: [ToolDefinition]
    public let searchTools: [String: Mode]
    public init(catalog: [ToolDefinition], searchTools: [String: Mode]) {
        self.catalog = catalog; self.searchTools = searchTools
    }

    public func search(name: String, input: [String: ChatJSONValue]) throws -> [ToolDefinition] {
        guard let mode = searchTools[name], catalog.count <= 10000 else { throw ChatRequestIssue("tool_search", "invalid catalog or search tool") }
        let key = mode == .regex ? "pattern" : "query"
        guard let query = input[key]?.stringValue, !query.isEmpty,
              query.count <= (mode == .regex ? 200 : 500) else { throw ChatRequestIssue("tool_search", "invalid or oversized \(key)") }
        let limit: Int
        if let raw = input["limit"] {
            guard case .number(let value) = raw, value.rounded() == value, (1...10000).contains(value) else {
                throw ChatRequestIssue("tool_search.limit", "must be an integer from 1 through 10000")
            }
            limit = Int(value)
        } else { limit = 5 }
        let corpus = try catalog.map { tool -> (ToolDefinition, String) in
            let text = tool.name + " " + tool.description + " " + (try tool.promptSchemaJSONString())
            guard text.utf8.count <= 256 * 1024 else { throw ChatRequestIssue("tool_search", "tool definition exceeds the local search limit") }
            return (tool, text)
        }
        guard corpus.reduce(0, { $0 + $1.1.utf8.count }) <= 32 * 1024 * 1024 else { throw ChatRequestIssue("tool_search", "catalog exceeds 32 MiB") }
        if mode == .regex {
            let expression: NSRegularExpression
            do { expression = try NSRegularExpression(pattern: query, options: [.caseInsensitive]) }
            catch { throw ChatRequestIssue("tool_search", "invalid regular expression") }
            let deadline = ProcessInfo.processInfo.systemUptime + 0.25
            var matched: [ToolDefinition] = []
            for (tool, text) in corpus.sorted(by: { $0.0.name < $1.0.name }) {
                try Task.checkCancellation()
                var found = false
                var timedOut = false
                expression.enumerateMatches(in: text, options: [.reportProgress], range: NSRange(text.startIndex..., in: text)) { result, _, stop in
                    if ProcessInfo.processInfo.systemUptime > deadline || Task.isCancelled { timedOut = true; stop.pointee = true }
                    if result != nil { found = true; stop.pointee = true }
                }
                if timedOut { throw ChatToolSearchFailure(code: "unavailable", message: "Regex search exceeded its bounded matching time.") }
                if found { matched.append(tool) }
                if matched.count == limit { break }
            }
            return matched
        }
        let deadline = ProcessInfo.processInfo.systemUptime + 1
        let terms = Self.words(query)
        let documents = corpus.map { Self.words($0.1) }
        let average = max(1, Double(documents.reduce(0) { $0 + $1.count }) / Double(max(1, documents.count)))
        let uniqueTerms = Set(terms)
        var frequencies: [String: Double] = [:]
        for document in documents {
            try Task.checkCancellation()
            guard ProcessInfo.processInfo.systemUptime <= deadline else { throw ChatToolSearchFailure(code: "unavailable", message: "BM25 search exceeded its bounded matching time.") }
            for word in Set(document).intersection(uniqueTerms) { frequencies[word, default: 0] += 1 }
        }
        var scores: [(ToolDefinition, Double)] = []
        for index in documents.indices {
            try Task.checkCancellation()
            guard ProcessInfo.processInfo.systemUptime <= deadline else { throw ChatToolSearchFailure(code: "unavailable", message: "BM25 search exceeded its bounded matching time.") }
            let words = documents[index]
            var score = 0.0
            for term in uniqueTerms.sorted() {
                let frequency = Double(words.filter { $0 == term }.count)
                guard frequency > 0 else { continue }
                let df = frequencies[term, default: 0]
                let idf = log(1 + (Double(documents.count) - df + 0.5) / (df + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * Double(words.count) / average))
            }
            if score > 0 { scores.append((corpus[index].0, score)) }
        }
        return Array(scores.sorted { $0.1 == $1.1 ? $0.0.name < $1.0.name : $0.1 > $1.1 }.prefix(limit)).map(\.0)
    }
    private static func words(_ text: String) -> [String] {
        text.lowercased().split(whereSeparator: { !$0.isLetter && !$0.isNumber }).map(String.init)
    }
}
