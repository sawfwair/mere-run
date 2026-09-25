import Foundation
import MLX
import MereRunGLiNERModel
@preconcurrency import Tokenizers

public struct GLiNERExtractionTerm: Codable, Sendable {
    public let name: String
    public let description: String?

    public init(name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}

public struct GLiNERExtractionField: Codable, Sendable {
    public let name: String
    public let description: String?
    public let multiple: Bool

    public init(name: String, description: String? = nil, multiple: Bool = true) {
        self.name = name
        self.description = description
        self.multiple = multiple
    }

    public init(from decoder: any Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        name = try values.decode(String.self, forKey: .name)
        description = try values.decodeIfPresent(String.self, forKey: .description)
        multiple = try values.decodeIfPresent(Bool.self, forKey: .multiple) ?? true
    }

    enum CodingKeys: String, CodingKey { case name, description, multiple }
}

public struct GLiNERExtractionStructure: Codable, Sendable {
    public let name: String
    public let fields: [GLiNERExtractionField]

    public init(name: String, fields: [GLiNERExtractionField]) {
        self.name = name
        self.fields = fields
    }
}

public struct GLiNERExtractionRequest: Codable, Sendable {
    public let text: String
    public let entities: [GLiNERExtractionTerm]
    public let relations: [GLiNERExtractionTerm]
    public let structures: [GLiNERExtractionStructure]
    public let classifications: [GLiNERClassificationTask]
    public let threshold: Double

    public init(text: String, entities: [GLiNERExtractionTerm] = [], relations: [GLiNERExtractionTerm] = [],
                structures: [GLiNERExtractionStructure] = [], classifications: [GLiNERClassificationTask] = [],
                threshold: Double = 0.5) {
        self.text = text
        self.entities = entities
        self.relations = relations
        self.structures = structures
        self.classifications = classifications
        self.threshold = threshold
    }

    public init(from decoder: any Swift.Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        text = try values.decode(String.self, forKey: .text)
        entities = try values.decodeIfPresent([GLiNERExtractionTerm].self, forKey: .entities) ?? []
        relations = try values.decodeIfPresent([GLiNERExtractionTerm].self, forKey: .relations) ?? []
        structures = try values.decodeIfPresent([GLiNERExtractionStructure].self, forKey: .structures) ?? []
        classifications = try values.decodeIfPresent([GLiNERClassificationTask].self, forKey: .classifications) ?? []
        threshold = try values.decodeIfPresent(Double.self, forKey: .threshold) ?? 0.5
    }

    enum CodingKeys: String, CodingKey { case text, entities, relations, structures, classifications, threshold }

    public func validate() throws {
        guard !text.isEmpty, !entities.isEmpty || !relations.isEmpty || !structures.isEmpty || !classifications.isEmpty,
              (0...1).contains(threshold) else {
            throw GLiNERClassificationError.invalidRequest("Provide text, an extraction schema, and a threshold from 0 to 1.")
        }
        let names = entities.map(\.name) + relations.map(\.name) + structures.map(\.name)
            + classifications.map(\.name)
        guard Set(names).count == names.count else {
            throw GLiNERClassificationError.invalidRequest("Extraction schema names must be unique.")
        }
        try classifications.forEach { try $0.validate() }
        guard !(relations.map(\.name) + structures.map(\.name) + classifications.map(\.name)).contains("entities") else {
            throw GLiNERClassificationError.invalidRequest("entities is a reserved extraction schema name.")
        }
        let terms = names + entities.compactMap(\.description) + relations.compactMap(\.description)
            + structures.flatMap { $0.fields.map(\.name) + $0.fields.compactMap(\.description) }
        let markers = ["[P]", "[L]", "[E]", "[R]", "[C]", "[SEP_STRUCT]", "[SEP_TEXT]",
                       "[DESCRIPTION]", "[EXAMPLE]", "[OUTPUT]"]
        guard terms.allSatisfy({ !$0.isEmpty && $0.count <= 512 && !markers.contains(where: $0.contains) }),
              structures.allSatisfy({ !$0.fields.isEmpty && Set($0.fields.map(\.name)).count == $0.fields.count }) else {
            throw GLiNERClassificationError.invalidRequest("Extraction schema contains empty, duplicate, or reserved terms.")
        }
    }
}

public struct GLiNERExtractionSpan: Codable, Sendable {
    public let text: String
    public let confidence: Double
    public let start: Int
    public let end: Int
}

public struct GLiNERExtractionRelation: Codable, Sendable {
    public let head: GLiNERExtractionSpan
    public let tail: GLiNERExtractionSpan
}

public struct GLiNERExtractionPlan: Codable, Sendable {
    public let model: String
    public let inputTokens: Int
    public let wordCount: Int
    public let schemaNames: [String]
}

public struct GLiNERExtractionResponse: Codable, Sendable {
    public let model: String
    public let runtime: String
    public let entities: [String: [GLiNERExtractionSpan]]
    public let relations: [String: [GLiNERExtractionRelation]]
    public let structures: [String: [[String: [GLiNERExtractionSpan]]]]
    public let classifications: [String: GLiNERClassificationHead]
    public let inputTokens: Int
}

private struct GLiNERWord {
    let token: String
    let start: Int
    let end: Int
}

private struct GLiNERExtractSchema {
    enum Kind { case entities, relation, structure, classification }
    let name: String
    let kind: Kind
    let fields: [String]
    let multiple: [Bool]
    let markers: [Int]
    let classification: GLiNERClassificationTask?
}

private struct GLiNERExtractEncoding {
    let ids: [Int]
    let words: [GLiNERWord]
    let wordPositions: [Int]
    let schemas: [GLiNERExtractSchema]
}

extension GLiNERClassificationOperation {
    public func prepareLong(_ request: GLiNERExtractionRequest, chunkSize: Int = 384,
                            chunkOverlap: Int = 64) throws -> [GLiNERExtractionPlan] {
        try request.validate()
        let chunks = try GLiNERTextChunks.make(text: request.text, size: chunkSize, overlap: chunkOverlap) { text in
            (try? self.prepare(GLiNERExtractionRequest(text: text, entities: request.entities,
                                                       relations: request.relations, structures: request.structures,
                                                       classifications: request.classifications,
                                                       threshold: request.threshold))) != nil
        }
        return try chunks.map { chunk in
            try prepare(GLiNERExtractionRequest(text: chunk.text, entities: request.entities,
                                                relations: request.relations, structures: request.structures,
                                                classifications: request.classifications,
                                                threshold: request.threshold))
        }
    }

    public func predictBatch(_ requests: [GLiNERExtractionRequest]) throws -> [GLiNERExtractionResponse] {
        try requests.map { try predict($0) }
    }

    public func predictLong(_ request: GLiNERExtractionRequest, chunkSize: Int = 384,
                            chunkOverlap: Int = 64) throws -> GLiNERExtractionResponse {
        try request.validate()
        let chunks = try GLiNERTextChunks.make(text: request.text, size: chunkSize, overlap: chunkOverlap) { text in
            (try? self.prepare(GLiNERExtractionRequest(text: text, entities: request.entities,
                                                       relations: request.relations, structures: request.structures,
                                                       classifications: request.classifications,
                                                       threshold: request.threshold))) != nil
        }
        let results = try chunks.map { chunk in
            try predict(GLiNERExtractionRequest(text: chunk.text, entities: request.entities,
                                                relations: request.relations, structures: request.structures,
                                                classifications: request.classifications,
                                                threshold: request.threshold))
        }
        var entities: [String: [GLiNERExtractionSpan]] = [:]
        var relations: [String: [GLiNERExtractionRelation]] = [:]
        var structures: [String: [[String: [GLiNERExtractionSpan]]]] = [:]
        var classifications: [String: GLiNERClassificationHead] = [:]
        for (chunk, result) in zip(chunks, results) {
            for (name, spans) in result.entities {
                entities[name, default: []] += spans.map { $0.offset(by: chunk.start) }
            }
            for (name, pairs) in result.relations {
                relations[name, default: []] += pairs.map {
                    GLiNERExtractionRelation(head: $0.head.offset(by: chunk.start),
                                             tail: $0.tail.offset(by: chunk.start))
                }
            }
            for (name, records) in result.structures {
                structures[name, default: []] += records.map { record in
                    record.mapValues { $0.map { $0.offset(by: chunk.start) } }
                }
            }
        }
        for name in entities.keys { entities[name] = Self.deduplicate(entities[name] ?? []) }
        for name in relations.keys {
            var seen: Set<String> = []
            relations[name] = relations[name]?.filter { pair in
                seen.insert("\(pair.head.start):\(pair.head.end):\(pair.tail.start):\(pair.tail.end)").inserted
            }
        }
        for name in structures.keys {
            var seen: Set<String> = []
            structures[name] = structures[name]?.filter { record in
                let key = record.keys.sorted().map { field in
                    "\(field)=\(record[field, default: []].map { "\($0.start):\($0.end)" }.joined(separator: ","))"
                }.joined(separator: "|")
                return seen.insert(key).inserted
            }
        }
        for task in request.classifications {
            let heads = results.compactMap { $0.classifications[task.name] }
            classifications[task.name] = GLiNERClassificationMerge.head(task: task, chunks: heads)
        }
        return GLiNERExtractionResponse(model: modelID, runtime: "native-swift-mlx-fp32",
                                        entities: entities, relations: relations, structures: structures,
                                        classifications: classifications,
                                        inputTokens: results.reduce(0) { $0 + $1.inputTokens })
    }

    public func predictBatchLong(_ requests: [GLiNERExtractionRequest], chunkSize: Int = 384,
                                 chunkOverlap: Int = 64) throws -> [GLiNERExtractionResponse] {
        try requests.map { try predictLong($0, chunkSize: chunkSize, chunkOverlap: chunkOverlap) }
    }

    private static func deduplicate(_ spans: [GLiNERExtractionSpan]) -> [GLiNERExtractionSpan] {
        var best: [String: GLiNERExtractionSpan] = [:]
        for span in spans {
            let key = "\(span.start):\(span.end)"
            if (best[key]?.confidence ?? -1) < span.confidence { best[key] = span }
        }
        return best.values.sorted { $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start }
    }

    public func prepare(_ request: GLiNERExtractionRequest) throws -> GLiNERExtractionPlan {
        let encoding = try encodeExtraction(request)
        return GLiNERExtractionPlan(model: modelID, inputTokens: encoding.ids.count,
                                    wordCount: encoding.words.count, schemaNames: encoding.schemas.map(\.name))
    }

    public func predict(_ request: GLiNERExtractionRequest) throws -> GLiNERExtractionResponse {
        let encoding = try encodeExtraction(request)
        let model = try loadNetwork()
        let hidden = model.encode(inputIDs: MLXArray(encoding.ids).reshaped([1, encoding.ids.count]))
        var entities: [String: [GLiNERExtractionSpan]] = [:]
        var relations: [String: [GLiNERExtractionRelation]] = [:]
        var structures: [String: [[String: [GLiNERExtractionSpan]]]] = [:]
        var classifications: [String: GLiNERClassificationHead] = [:]
        for schema in encoding.schemas {
            if case .classification = schema.kind, let task = schema.classification {
                let logits = model.classify(hidden: hidden,
                    markers: MLXArray(Array(schema.markers.dropFirst())).reshaped([1, task.labels.count]))
                eval(logits)
                let values = logits[0].asArray(Float.self).map(Double.init)
                let probabilities: [Double]
                if task.multiLabel {
                    probabilities = values.map { 1 / (1 + exp(-$0)) }
                } else {
                    let peak = values.max() ?? 0
                    let exponents = values.map { exp($0 - peak) }
                    let total = exponents.reduce(0, +)
                    probabilities = exponents.map { $0 / total }
                }
                let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
                let selected = task.multiLabel
                    ? task.labels.indices.filter { probabilities[$0] >= task.threshold }
                    : [best]
                classifications[task.name] = GLiNERClassificationHead(
                    labels: (selected.isEmpty ? [best] : selected).map { task.labels[$0] },
                    probabilities: Dictionary(uniqueKeysWithValues: zip(task.labels, probabilities)))
                continue
            }
            let scores = model.scoreSpans(hidden: hidden, wordPositions: encoding.wordPositions,
                                          queryPositions: schema.markers)
            switch schema.kind {
            case .entities:
                for (index, name) in schema.fields.enumerated() {
                    entities[name] = scores.count > 0
                        ? spans(scores, instance: 0, field: index, words: encoding.words, text: request.text,
                                threshold: request.threshold) : []
                }
            case .relation:
                relations[schema.name] = (0..<scores.count).compactMap { instance in
                    let heads = spans(scores, instance: instance, field: 0, words: encoding.words,
                                      text: request.text, threshold: request.threshold)
                    let tails = spans(scores, instance: instance, field: 1, words: encoding.words,
                                      text: request.text, threshold: request.threshold)
                    guard let head = heads.first, let tail = tails.first else { return nil }
                    return GLiNERExtractionRelation(head: head, tail: tail)
                }
            case .structure:
                structures[schema.name] = (0..<scores.count).compactMap { instance in
                    var record: [String: [GLiNERExtractionSpan]] = [:]
                    for (index, name) in schema.fields.enumerated() {
                        let found = spans(scores, instance: instance, field: index, words: encoding.words,
                                          text: request.text, threshold: request.threshold)
                        record[name] = schema.multiple[index] ? found : Array(found.prefix(1))
                    }
                    return record.values.contains(where: { !$0.isEmpty }) ? record : nil
                }
            case .classification:
                continue
            }
        }
        return GLiNERExtractionResponse(model: modelID, runtime: "native-swift-mlx-fp32",
                                        entities: entities, relations: relations, structures: structures,
                                        classifications: classifications,
                                        inputTokens: encoding.ids.count)
    }

    private func spans(_ scores: GLiNERSpanScores, instance: Int, field: Int, words: [GLiNERWord],
                       text: String, threshold: Double) -> [GLiNERExtractionSpan] {
        var candidates: [GLiNERExtractionSpan] = []
        for start in words.indices {
            for width in 0..<min(8, words.count - start) {
                let score = Double(scores.probability(instance: instance, field: field, start: start, width: width))
                guard score >= threshold else { continue }
                let begin = words[start].start
                let end = words[start + width].end
                let value = String(text.unicodeScalars.dropFirst(begin).prefix(end - begin)).trimmingCharacters(in: .whitespaces)
                guard !value.isEmpty else { continue }
                candidates.append(GLiNERExtractionSpan(text: value, confidence: score, start: begin, end: end))
            }
        }
        candidates.sort { $0.confidence > $1.confidence }
        var selected: [GLiNERExtractionSpan] = []
        for candidate in candidates {
            guard !selected.contains(where: { candidate.start < $0.end && $0.start < candidate.end }) else { continue }
            selected.append(candidate)
        }
        return selected
    }

    private func encodeExtraction(_ request: GLiNERExtractionRequest) throws -> GLiNERExtractEncoding {
        try request.validate()
        let expression = try NSRegularExpression(pattern:
            #"(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|\w+(?:[-_]\w+)*|\S"#,
            options: [.caseInsensitive])
        let source = request.text as NSString
        let words = expression.matches(in: request.text, range: NSRange(location: 0, length: source.length)).map { match in
            let lower = String.Index(utf16Offset: match.range.location, in: request.text)
            let upper = String.Index(utf16Offset: NSMaxRange(match.range), in: request.text)
            return GLiNERWord(token: source.substring(with: match.range).lowercased(),
                              start: request.text.unicodeScalars.distance(from: request.text.startIndex, to: lower),
                              end: request.text.unicodeScalars.distance(from: request.text.startIndex, to: upper))
        }
        var ids: [Int] = []
        var schemas: [GLiNERExtractSchema] = []

        func append(_ part: String) {
            ids += tokenizer.tokenize(text: part).map { specialTokenIDs[$0] ?? tokenizer.convertTokenToId($0) ?? 3 }
        }

        func addSchema(name: String, kind: GLiNERExtractSchema.Kind, fields: [String],
                       descriptions: [String: String], prompt: String?, marker: String, multiple: [Bool],
                       classification: GLiNERClassificationTask? = nil) {
            if !schemas.isEmpty { append("[SEP_STRUCT]") }
            var title = name
            if let prompt { title += ": \(prompt)" }
            for field in fields {
                if let description = descriptions[field] { title += " [DESCRIPTION] \(field): \(description)" }
            }
            append("(")
            let parent = ids.count
            append("[P]")
            append(title)
            append("(")
            var markers = [parent]
            for field in fields {
                markers.append(ids.count)
                append(marker)
                append(field)
            }
            append(")")
            append(")")
            schemas.append(GLiNERExtractSchema(name: name, kind: kind, fields: fields,
                                                multiple: multiple, markers: markers,
                                                classification: classification))
        }

        for structure in request.structures {
            addSchema(name: structure.name, kind: .structure, fields: structure.fields.map(\.name),
                      descriptions: Dictionary(uniqueKeysWithValues: structure.fields.compactMap { field in
                          field.description.map { (field.name, $0) }
                      }), prompt: nil, marker: "[C]", multiple: structure.fields.map(\.multiple))
        }
        if !request.entities.isEmpty {
            addSchema(name: "entities", kind: .entities, fields: request.entities.map(\.name),
                      descriptions: Dictionary(uniqueKeysWithValues: request.entities.compactMap { term in
                          term.description.map { (term.name, $0) }
                      }), prompt: nil, marker: "[E]", multiple: Array(repeating: true, count: request.entities.count))
        }
        for relation in request.relations {
            addSchema(name: relation.name, kind: .relation, fields: ["head", "tail"], descriptions: [:],
                      prompt: relation.description, marker: "[R]", multiple: [false, false])
        }
        for task in request.classifications {
            addSchema(name: task.name, kind: .classification, fields: task.labels,
                      descriptions: task.descriptions ?? [:], prompt: task.prompt, marker: "[L]",
                      multiple: Array(repeating: task.multiLabel, count: task.labels.count), classification: task)
        }
        append("[SEP_TEXT]")
        var wordPositions: [Int] = []
        for word in words {
            wordPositions.append(ids.count)
            append(word.token)
        }
        guard !words.isEmpty, ids.count <= configuration.maxPositionEmbeddings,
              ids.allSatisfy({ (0..<configuration.vocabSize).contains($0) }) else {
            throw GLiNERClassificationError.invalidRequest("Schema and text exceed the 512 token checkpoint limit.")
        }
        return GLiNERExtractEncoding(ids: ids, words: words, wordPositions: wordPositions, schemas: schemas)
    }
}

private extension GLiNERExtractionSpan {
    func offset(by amount: Int) -> GLiNERExtractionSpan {
        GLiNERExtractionSpan(text: text, confidence: confidence, start: start + amount, end: end + amount)
    }
}
