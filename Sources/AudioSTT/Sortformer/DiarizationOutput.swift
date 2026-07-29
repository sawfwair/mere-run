import Foundation

public struct DiarizationSegment: Codable, Equatable, Sendable {
    public let start: Float
    public let end: Float
    public let speaker: Int

    public init(start: Float, end: Float, speaker: Int) {
        self.start = start
        self.end = end
        self.speaker = speaker
    }
}

public struct DiarizationOutput: Equatable, Sendable {
    public let segments: [DiarizationSegment]
    public let numSpeakers: Int
    public let totalTime: Double

    public init(
        segments: [DiarizationSegment],
        numSpeakers: Int = 0,
        totalTime: Double = 0
    ) {
        self.segments = segments
        self.numSpeakers = numSpeakers
        self.totalTime = totalTime
    }

    public func rttm(fileID: String) -> String {
        segments.map { segment in
            let duration = segment.end - segment.start
            return "SPEAKER \(fileID) 1 \(Self.seconds(segment.start)) \(Self.seconds(duration)) <NA> <NA> speaker_\(segment.speaker) <NA> <NA>"
        }.joined(separator: "\n")
    }

    private static func seconds(_ value: Float) -> String {
        String(format: "%.3f", value)
    }
}
