import Foundation
import MediaIO
import MLX
import MereRunInkling2Model
@preconcurrency import Tokenizers

enum Inkling2Prompt {
    struct Prepared {
        let tokens: [Int]
        let images: [(offset: Int, count: Int, embeddings: MLXArray)]
    }

    static func prepare(_ request: ChatRequest, tokenizer: any Tokenizer, template: Inkling2ChatTemplate, model: Inkling2Model) throws -> Prepared {
        let effort = request.reasoningEffort ?? 0.9
        guard effort.isFinite, (0...0.99).contains(effort) else { throw ChatRequestIssue("reasoning_effort", "must be between 0 and 0.99") }
        // The released template deliberately has no assistant generation suffix.
        let rendered = try template.render(messages: request.messages, tools: request.tools, effort: effort)
        let raw = tokenizer.encode(text: rendered, addSpecialTokens: false)
        var features: [MLXArray] = []
        for message in request.messages {
            let references: [String]
            if let blocks = message.contentBlocks {
                references = blocks.compactMap { if case .image(let reference, _) = $0 { return reference }; return nil }
            } else { references = [message.imageUrl].compactMap { $0 } }
            for reference in references {
                guard message.role == .user || message.role == .tool else { throw ChatRequestIssue("image", "is only supported in user or tool messages") }
                let image = try decodeImage(reference)
                let patches = try Inkling2ImagePatches.make(rgba: image.rgba8, width: image.width, height: image.height)
                let encoded = try model.imageFeatures(pixels: patches.pixels, rows: patches.rows, columns: patches.columns)
                eval(encoded)
                features.append(encoded)
            }
        }
        let imageID = 200_054
        guard raw.filter({ $0 == imageID }).count == features.count else { throw ChatRequestIssue("image", "placeholder count differs from image count") }
        var tokens: [Int] = [], images: [(Int, Int, MLXArray)] = []
        var imageIndex = 0
        for token in raw {
            if token == imageID {
                let feature = features[imageIndex]
                images.append((tokens.count, feature.dim(0), feature))
                tokens += Array(repeating: imageID, count: feature.dim(0))
                imageIndex += 1
            } else { tokens.append(token) }
        }
        return Prepared(tokens: tokens, images: images)
    }

    private static func decodeImage(_ reference: String) throws -> MediaImage {
        if reference.hasPrefix("data:") {
            guard let comma = reference.firstIndex(of: ","), reference[..<comma].contains(";base64"),
                  let data = Data(base64Encoded: String(reference[reference.index(after: comma)...])) else {
                throw ChatRequestIssue("image", "invalid base64 image")
            }
            return try MediaImageIO.decode(data: data)
        }
        if let url = URL(string: reference), let scheme = url.scheme {
            guard scheme == "file" else { throw ChatRequestIssue("image", "use a local file or base64 data URL") }
            return try MediaImageIO.decode(url)
        }
        return try MediaImageIO.decode(URL(fileURLWithPath: (reference as NSString).expandingTildeInPath))
    }
}
