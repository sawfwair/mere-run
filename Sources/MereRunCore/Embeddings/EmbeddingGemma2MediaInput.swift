import Foundation

/// Ordered content, so repeated or interleaved media stays aligned with soft-token blocks.
public enum EmbeddingGemma2Content: Sendable {
    case text(String)
    case image(URL)
    case audio(URL)
    case video(URL)
    case videoFrames([URL])
}

public struct EmbeddingGemma2Input: Sendable {
    public let content: [EmbeddingGemma2Content]
    public init(content: [EmbeddingGemma2Content]) { self.content = content }
}

public struct EmbeddingGemma2InputDocument: Decodable, Sendable {
    public let inputs: [Record]
    public struct Record: Decodable, Sendable {
        public let content: [Part]
    }
    public enum Part: Decodable, Sendable {
        case text(String), image(String), audio(String), video(String), videoFrames([String])
        enum CodingKeys: String, CodingKey { case type, text, path, frames }
        enum Kind: String, Decodable { case text, image, audio, video, videoFrames = "video-frames" }
        public init(from decoder: Decoder) throws {
            let value = try decoder.container(keyedBy: CodingKeys.self)
            switch try value.decode(Kind.self, forKey: .type) {
            case .text: self = .text(try value.decode(String.self, forKey: .text))
            case .image: self = .image(try value.decode(String.self, forKey: .path))
            case .audio: self = .audio(try value.decode(String.self, forKey: .path))
            case .video: self = .video(try value.decode(String.self, forKey: .path))
            case .videoFrames: self = .videoFrames(try value.decode([String].self, forKey: .frames))
            }
        }
    }

    public func resolved(relativeTo root: URL) throws -> [EmbeddingGemma2Input] {
        guard !inputs.isEmpty else { throw EmbeddingGemma2Error.invalidInput("Input document is empty.") }
        func url(_ path: String) throws -> URL {
            guard !path.isEmpty, !path.contains("://") else {
                throw EmbeddingGemma2Error.invalidInput("Media paths must refer to local files.")
            }
            return path.hasPrefix("/") ? URL(fileURLWithPath: path).standardizedFileURL : root.appending(path: path).standardizedFileURL
        }
        return try inputs.map { record in
            guard !record.content.isEmpty else { throw EmbeddingGemma2Error.invalidInput("Input content is empty.") }
            return EmbeddingGemma2Input(content: try record.content.map { part in
                switch part {
                case .text(let text): return .text(text)
                case .image(let path): return .image(try url(path))
                case .audio(let path): return .audio(try url(path))
                case .video(let path): return .video(try url(path))
                case .videoFrames(let paths):
                    guard (1...32).contains(paths.count) else {
                        throw EmbeddingGemma2Error.invalidInput("Provide between 1 and 32 video frames.")
                    }
                    return .videoFrames(try paths.map(url))
                }
            })
        }
    }
}
