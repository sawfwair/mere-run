import Foundation

public enum LTXVideoVariant: String, CaseIterable, Codable, Sendable {
    case unifiedAV = "unified-av"
    case distilled
}

public enum LTXVideoQuality: String, CaseIterable, Codable, Sendable {
    case draft
    case final
}

public enum LTXVideoOutputMode: String, CaseIterable, Codable, Sendable {
    case videoOnly = "video-only"
    case audioVideo = "audio-video"

    public var writesAudio: Bool {
        self == .audioVideo
    }

    public var compatibilityVariant: LTXVideoVariant {
        writesAudio ? .unifiedAV : .distilled
    }
}

public enum TextResponseFormat: String, CaseIterable, Codable, Sendable {
    case text
    case jsonObject = "json_object"
}

public enum TextThinkingMode: String, CaseIterable, Codable, Sendable {
    case automatic
    case show
    case hide
}
