import Foundation

// What Text ▸ Embeddings, Text ▸ Anonymize, and Image ▸ Datasets read back. Each mirrors the
// CLI's own output type (`OpenAIEmbeddingResponse`, the anonymizer's `AnonymizeResponse`, the
// `StructuredRunEnvelope` around `LoRATrainingDatasetDiscoveryResult`) and reads only the fields
// the renderers show; a decode failure means the CLI changed. Studio does not import the CLI.

// MARK: - text embed

/// `text embed`'s OpenAI-shaped response: one vector per input text, plus the tokens it cost.
package struct StudioEmbeddingDocument: Decodable, Equatable {
    package struct Vector: Decodable, Identifiable, Equatable {
        /// The input's position in the argv, which the matrix labels "#1", "#2", …
        package let id: Int
        package let values: [Double]

        package init(id: Int, values: [Double]) {
            self.id = id
            self.values = values
        }

        private enum CodingKeys: String, CodingKey {
            case id = "index"
            case values = "embedding"
        }

        package var norm: Double {
            sqrt(values.reduce(0) { $0 + ($1 * $1) })
        }
    }

    private struct Usage: Decodable {
        let promptTokens: Int

        private enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
        }
    }

    package let model: String
    package let promptTokens: Int
    package let vectors: [Vector]

    package init(model: String, promptTokens: Int, vectors: [Vector]) {
        self.model = model
        self.promptTokens = promptTokens
        self.vectors = vectors
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case data
        case usage
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        vectors = try container.decode([Vector].self, forKey: .data)
        promptTokens = try container.decode(Usage.self, forKey: .usage).promptTokens
    }

    package static func decode(_ data: Data) -> StudioEmbeddingDocument? {
        try? JSONDecoder().decode(StudioEmbeddingDocument.self, from: data)
    }

    package var dimensions: Int {
        vectors.first?.values.count ?? 0
    }

    package func cosineSimilarity(_ lhs: Vector, _ rhs: Vector) -> Double {
        guard lhs.values.count == rhs.values.count, !lhs.values.isEmpty else { return 0 }
        let dot = zip(lhs.values, rhs.values).reduce(0) { $0 + ($1.0 * $1.1) }
        let denominator = lhs.norm * rhs.norm
        return denominator > 0 ? dot / denominator : 0
    }

    /// "2 vectors · 1024 dimensions" — the result panel's header.
    package var summary: String {
        let count = vectors.count == 1 ? "1 vector" : "\(vectors.count) vectors"
        return "\(count) · \(dimensions) dimensions"
    }
}

// MARK: - text anonymize

/// `text anonymize --json`'s response: each input beside its protected text and the spans the
/// privacy filter marked in it.
package struct StudioAnonymizationDocument: Decodable, Equatable {
    package struct Span: Identifiable, Equatable {
        /// The span's position in its result.
        package let id: Int
        package let label: String
        package let text: String
        package let startToken: Int
        package let endToken: Int

        package init(id: Int, label: String, text: String, startToken: Int, endToken: Int) {
            self.id = id
            self.label = label
            self.text = text
            self.startToken = startToken
            self.endToken = endToken
        }
    }

    package struct Result: Identifiable, Equatable {
        /// The input's position in the argv.
        package let id: Int
        package let text: String
        package let anonymizedText: String
        package let tokenCount: Int
        package let spans: [Span]

        package init(id: Int, text: String, anonymizedText: String, tokenCount: Int, spans: [Span]) {
            self.id = id
            self.text = text
            self.anonymizedText = anonymizedText
            self.tokenCount = tokenCount
            self.spans = spans
        }
    }

    /// The CLI's `OpenAIPrivacyFilterAnonymizationResult`; results and their spans are told
    /// apart by position once read.
    private struct Payload: Decodable {
        struct Span: Decodable {
            let label: String
            let text: String
            let startToken: Int
            let endToken: Int
        }

        let text: String
        let anonymizedText: String
        let tokenCount: Int
        let spans: [Span]

        private enum CodingKeys: String, CodingKey {
            case text
            case anonymizedText = "anonymized_text"
            case tokenCount = "token_count"
            case spans
        }
    }

    package let model: String
    package let results: [Result]

    package init(model: String, results: [Result]) {
        self.model = model
        self.results = results
    }

    private enum CodingKeys: String, CodingKey {
        case model
        case data
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        model = try container.decode(String.self, forKey: .model)
        results = try container.decode([Payload].self, forKey: .data).enumerated().map { index, payload in
            Result(
                id: index,
                text: payload.text,
                anonymizedText: payload.anonymizedText,
                tokenCount: payload.tokenCount,
                spans: payload.spans.enumerated().map { position, span in
                    Span(id: position, label: span.label, text: span.text, startToken: span.startToken, endToken: span.endToken)
                }
            )
        }
    }

    package static func decode(_ data: Data) -> StudioAnonymizationDocument? {
        try? JSONDecoder().decode(StudioAnonymizationDocument.self, from: data)
    }

    package var spanCount: Int {
        results.reduce(0) { $0 + $1.spans.count }
    }

    package var tokenCount: Int {
        results.reduce(0) { $0 + $1.tokenCount }
    }

    /// Every protected text, one per input, the way `text anonymize` prints them without `--json`.
    package var protectedText: String {
        results.map(\.anonymizedText).joined(separator: "\n")
    }

    /// "3 PII spans in 1 document" — the result panel's header.
    package var summary: String {
        let spans = spanCount == 1 ? "1 PII span" : "\(spanCount) PII spans"
        let documents = results.count == 1 ? "1 document" : "\(results.count) documents"
        return "\(spans) in \(documents)"
    }
}

// MARK: - image dataset discover

/// `image dataset discover --json`'s envelope: the folders under the root that hold image-caption
/// pairs, each with its counts and what keeps it from training.
package struct StudioDatasetDiscoveryDocument: Decodable, Equatable {
    package struct Diagnostic: Decodable, Equatable {
        package let title: String
        package let message: String

        package init(title: String, message: String) {
            self.title = title
            self.message = message
        }

        /// "Missing captions: Two images need captions."
        package var text: String {
            "\(title): \(message)"
        }
    }

    package struct Candidate: Decodable, Identifiable, Equatable {
        package let id: String
        package let name: String
        package let path: String
        package let status: String
        package let trainable: Bool
        package let images: Int
        package let captions: Int
        package let usablePairs: Int
        package let diagnostics: [Diagnostic]

        package init(
            id: String, name: String, path: String, status: String, trainable: Bool,
            images: Int, captions: Int, usablePairs: Int, diagnostics: [Diagnostic]
        ) {
            self.id = id
            self.name = name
            self.path = path
            self.status = status
            self.trainable = trainable
            self.images = images
            self.captions = captions
            self.usablePairs = usablePairs
            self.diagnostics = diagnostics
        }

        private enum CodingKeys: String, CodingKey {
            case id
            case name
            case path
            case status
            case trainable
            case images = "image_count"
            case captions = "caption_count"
            case usablePairs = "usable_pair_count"
            case diagnostics
        }

        /// What the CLI found wrong with the folder, one line each.
        package var problems: [String] {
            diagnostics.map(\.text)
        }
    }

    private struct Result: Decodable {
        let scannedDirectories: Int
        let candidates: [Candidate]

        private enum CodingKeys: String, CodingKey {
            case scannedDirectories = "scanned_directory_count"
            case candidates
        }
    }

    package let summary: String
    package let scannedDirectories: Int
    package let candidates: [Candidate]
    /// The run's own diagnostics (the root was not a folder, nothing was found), apart from
    /// each candidate's.
    package let diagnostics: [String]

    package init(summary: String, scannedDirectories: Int, candidates: [Candidate], diagnostics: [String]) {
        self.summary = summary
        self.scannedDirectories = scannedDirectories
        self.candidates = candidates
        self.diagnostics = diagnostics
    }

    private enum CodingKeys: String, CodingKey {
        case summary
        case result
        case diagnostics
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        summary = try container.decode(String.self, forKey: .summary)
        let result = try container.decode(Result.self, forKey: .result)
        scannedDirectories = result.scannedDirectories
        candidates = result.candidates
        diagnostics = try container.decode([Diagnostic].self, forKey: .diagnostics).map(\.text)
    }

    package static func decode(_ data: Data) -> StudioDatasetDiscoveryDocument? {
        try? JSONDecoder().decode(StudioDatasetDiscoveryDocument.self, from: data)
    }

    package var trainableCount: Int {
        candidates.filter(\.trainable).count
    }

    /// "3 candidates · 2 trainable" — the result panel's header.
    package var headline: String {
        let count = candidates.count == 1 ? "1 candidate" : "\(candidates.count) candidates"
        return "\(count) · \(trainableCount) trainable"
    }
}

// MARK: - image validate

/// What `image validate` reports on stderr: the family and suite it checked and the folder it
/// wrote its artifacts to. Read from the run's captured output, so a run that stopped before
/// "Validation complete" has no report and the panel shows the CLI's words instead.
package struct StudioImageValidationReport: Equatable {
    package let family: String
    package let suite: String
    package let artifactDirectory: URL

    package init(family: String, suite: String, artifactDirectory: URL) {
        self.family = family
        self.suite = suite
        self.artifactDirectory = artifactDirectory
    }

    private static let heading = "Image validation"
    private static let completion = "Validation complete. Artifacts written to "

    package static func decode(outputText: String) -> StudioImageValidationReport? {
        let lines = outputText.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }
        guard let start = lines.firstIndex(of: heading),
              let family = value(of: "family", in: lines[start...]),
              let suite = value(of: "suite", in: lines[start...]),
              let completed = lines.last(where: { $0.hasPrefix(completion) }) else { return nil }
        var path = String(completed.dropFirst(completion.count))
        if path.hasSuffix(".") { path.removeLast() }
        return StudioImageValidationReport(family: family, suite: suite, artifactDirectory: URL(fileURLWithPath: path, isDirectory: true))
    }

    /// "  family: zimage" → "zimage", read from the lines after the heading.
    private static func value(of key: String, in lines: ArraySlice<String>) -> String? {
        let prefix = "\(key): "
        return lines.first { $0.hasPrefix(prefix) }.map { String($0.dropFirst(prefix.count)) }
    }

    /// The artifacts on disk right now, sorted by name; empty once the folder is gone.
    package func artifacts(fileManager: FileManager = .default) -> [URL] {
        let contents = (try? fileManager.contentsOfDirectory(
            at: artifactDirectory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        )) ?? []
        return contents.sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    /// "Z-Image · all" — the result panel's header.
    package var summary: String {
        "\(familyTitle) · \(suite == "all" ? "every suite" : suite)"
    }

    package var familyTitle: String {
        switch family {
        case "zimage": return "Z-Image"
        case "klein": return "FLUX.2 Klein"
        default: return family
        }
    }
}
