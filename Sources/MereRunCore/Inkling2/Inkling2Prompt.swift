import Foundation
import MediaIO
import MLX
import MereRunInkling2Model
@preconcurrency import Tokenizers

enum Inkling2Prompt {
    static let imageID = 200_054

    struct Prepared {
        let tokens: [Int]
        let images: [(offset: Int, count: Int, embeddings: MLXArray)]
        /// `tokens` with each image's positions holding a marker for that image's pixels, so a stored
        /// prefix matches only a prompt with the same images.
        let keyTokens: [Int]
        let imageMarkers: [Int]
    }

    /// A negative marker for an image's pixels, which no text token can equal.
    static func imageMarker(_ image: MediaImage) -> Int {
        var hasher = Hasher()
        hasher.combine(image.width)
        hasher.combine(image.height)
        image.rgba8.withUnsafeBytes { hasher.combine(bytes: $0) }
        return -Int(UInt(bitPattern: hasher.finalize()) >> 1) - 1
    }

    /// `raw` template tokens with each image placeholder expanded to its image's marker positions.
    static func keyTokens(raw: [Int], images: [(marker: Int, count: Int)]) -> [Int]? {
        var tokens: [Int] = [], index = 0
        for token in raw {
            if token == imageID {
                guard index < images.count else { return nil }
                tokens += Array(repeating: images[index].marker, count: images[index].count)
                index += 1
            } else { tokens.append(token) }
        }
        return tokens
    }

    static func prepare(_ request: ChatRequest, tokenizer: any Tokenizer, template: Inkling2ChatTemplate,
                        generator: isolated Inkling2Generator, model: Inkling2Model) throws -> Prepared {
        let effort = request.reasoningEffort ?? 0.9
        guard effort.isFinite, (0...0.99).contains(effort) else { throw ChatRequestIssue("reasoning_effort", "must be between 0 and 0.99") }
        // The released template deliberately has no assistant generation suffix.
        let rendered = try template.render(messages: request.messages, tools: request.tools, effort: effort)
        let raw = tokenizer.encode(text: rendered, addSpecialTokens: false)
        var features: [MLXArray] = [], markers: [Int] = []
        for message in request.messages {
            let references: [String]
            if let blocks = message.contentBlocks {
                references = blocks.compactMap { if case .image(let reference, _) = $0 { return reference }; return nil }
            } else { references = [message.imageUrl].compactMap { $0 } }
            for reference in references {
                guard message.role == .user || message.role == .tool else { throw ChatRequestIssue("image", "is only supported in user or tool messages") }
                let image = try decodeImage(reference)
                let marker = imageMarker(image)
                features.append(try generator.cachedImageFeatures(image, marker: marker, model: model))
                markers.append(marker)
            }
        }
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
        let keys = keyTokens(raw: raw, images: zip(markers, features).map { ($0, $1.dim(0)) }) ?? tokens
        return Prepared(tokens: tokens, images: images, keyTokens: keys, imageMarkers: markers)
    }

    /// Runs the vision tower on one image.
    static func encode(_ image: MediaImage, model: Inkling2Model) throws -> MLXArray {
        let patches = try Inkling2ImagePatches.make(rgba: image.rgba8, width: image.width, height: image.height)
        let encoded = try model.imageFeatures(pixels: patches.pixels, rows: patches.rows, columns: patches.columns)
        eval(encoded)
        return encoded
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
