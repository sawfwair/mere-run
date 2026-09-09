import MereRunTensor
import Foundation
import MLX

public enum MiniMaxH3LayoutError: LocalizedError, Sendable {
    case invalidGeometry(String)

    public var errorDescription: String? {
        switch self {
        case .invalidGeometry(let reason): return "Invalid MiniMax-H3 geometry: \(reason)"
        }
    }
}

public enum MiniMaxH3Modality: Int32, Sendable {
    case video = 0
    case text = 1
    case audio = 2
}

public enum MiniMaxH3KeyframeAnchor: RawRepresentable, Codable, Sendable, Equatable {
    public typealias RawValue = String

    case history(latentFrameCount: Int)
    case first
    case frame(Int)
    case last

    public init?(rawValue: String) {
        switch rawValue {
        case "first": self = .first
        case "last": self = .last
        default:
            let components = rawValue.split(separator: ":", maxSplits: 1)
            guard components.count == 2,
                  let value = Int(components[1]) else {
                return nil
            }
            switch components[0] {
            case "history": self = .history(latentFrameCount: value)
            case "frame": self = .frame(value)
            default: return nil
            }
        }
    }

    public var rawValue: String {
        switch self {
        case .history(let count): "history:\(count)"
        case .first: "first"
        case .frame(let index): "frame:\(index)"
        case .last: "last"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        guard let value = Self(rawValue: rawValue) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "invalid MiniMax-H3 keyframe anchor: \(rawValue)"
            )
        }
        self = value
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    package var latentFrameCount: Int {
        switch self {
        case .history(let count): count
        case .first, .frame, .last: 1
        }
    }
}

public enum MiniMaxH3AudioConditionAnchor: Sendable, Equatable {
    case history(latentFrameCount: Int)
    case first(latentFrameCount: Int)

    package var latentFrameCount: Int {
        switch self {
        case .history(let count), .first(let count): count
        }
    }
}

public struct MiniMaxH3ConditionSegment: Sendable {
    public let modality: MiniMaxH3Modality
    public let packedRows: Range<Int>
    public let sourceRows: Range<Int>
}

public enum MiniMaxH3ReferenceKind: String, Codable, Sendable {
    case image
    case video
    case audio
}

public struct MiniMaxH3PreparedReferenceGeometry: Sendable {
    public let kind: MiniMaxH3ReferenceKind
    public let videoLatentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let audioLatentFrames: Int

    public init(
        kind: MiniMaxH3ReferenceKind,
        videoLatentFrames: Int = 0,
        latentHeight: Int = 0,
        latentWidth: Int = 0,
        audioLatentFrames: Int = 0
    ) {
        self.kind = kind
        self.videoLatentFrames = videoLatentFrames
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.audioLatentFrames = audioLatentFrames
    }

    package var videoRowCount: Int {
        kind == .audio ? 0 : videoLatentFrames * (latentHeight / 2) * (latentWidth / 2)
    }

    package var audioRowCount: Int { audioLatentFrames * 2 }
}

public struct MiniMaxH3PackedLayout: @unchecked Sendable {
    public let positions: MLXArray
    public let tokenTags: [Int32]
    public let textRows: Range<Int>
    public let conditionRows: Range<Int>
    public let conditionSegments: [MiniMaxH3ConditionSegment]
    public let conditionVideoRowCount: Int
    public let conditionAudioRowCount: Int
    public let targetAudioRows: Range<Int>
    public let targetVideoRows: Range<Int>
    public let videoLatentFrames: Int
    public let latentHeight: Int
    public let latentWidth: Int
    public let audioLatentFrames: Int

    public var sequenceLength: Int { tokenTags.count }
}
