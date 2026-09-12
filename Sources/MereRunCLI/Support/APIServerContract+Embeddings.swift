import Foundation
import MereRunCore

extension APIServerContract {
    static let maxEmbeddingInputCount = 256
    static let maxEmbeddingInputUTF8Bytes = 2 * 1_024 * 1_024

    static func embeddingTexts(from request: OpenAIEmbeddingRequest) throws -> [String] {
        guard !request.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw APIRequestValidationError.invalidField("model", "must not be empty")
        }
        if let encodingFormat = request.encoding_format?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !encodingFormat.isEmpty,
           encodingFormat != "float" {
            throw APIRequestValidationError.invalidField(
                "encoding_format",
                "only float embeddings are supported"
            )
        }
        if request.dimensions != nil {
            throw APIRequestValidationError.invalidField(
                "dimensions",
                "dimension overrides are not supported by this embedding model"
            )
        }

        let texts = request.input.texts
        guard !texts.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must contain at least one text")
        }
        guard texts.count <= maxEmbeddingInputCount else {
            throw APIRequestValidationError.invalidField(
                "input",
                "must contain at most \(maxEmbeddingInputCount) texts"
            )
        }
        var totalUTF8Bytes = 0
        for text in texts {
            let textBytes = text.utf8.count
            guard textBytes <= maxEmbeddingInputUTF8Bytes - totalUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "UTF-8 content must total at most \(maxEmbeddingInputUTF8Bytes) bytes"
                )
            }
            totalUTF8Bytes += textBytes
        }
        return texts
    }

    static func embeddingResponse(
        modelId: String,
        embeddings: [[Float]],
        tokenCounts: [Int]
    ) -> OpenAIEmbeddingResponse {
        let promptTokens = tokenCounts.reduce(0, +)
        return OpenAIEmbeddingResponse(
            model: modelId,
            data: embeddings.enumerated().map { index, vector in
                OpenAIEmbeddingDatum(index: index, embedding: vector)
            },
            usage: OpenAIEmbeddingUsage(
                prompt_tokens: promptTokens,
                total_tokens: promptTokens
            )
        )
    }
}
