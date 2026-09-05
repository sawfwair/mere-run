import AudioCore
import Dispatch
import Foundation

/// Stage-level timings for one prepared Parakeet transcription.
///
/// Model loading and audio-file decoding happen outside this measurement so a
/// resident benchmark can report them separately from repeatable inference.
public struct ParakeetPipelineTimings: Codable, Hashable, Sendable {
    public var featureExtractionSeconds: TimeInterval
    public var encoderSeconds: TimeInterval
    public var decoderSeconds: TimeInterval
    public var alignmentSeconds: TimeInterval
    public var windowMergeSeconds: TimeInterval
    public var cleanupSeconds: TimeInterval
    public var totalSeconds: TimeInterval
    public var windowCount: Int

    public init(
        featureExtractionSeconds: TimeInterval = 0,
        encoderSeconds: TimeInterval = 0,
        decoderSeconds: TimeInterval = 0,
        alignmentSeconds: TimeInterval = 0,
        windowMergeSeconds: TimeInterval = 0,
        cleanupSeconds: TimeInterval = 0,
        totalSeconds: TimeInterval = 0,
        windowCount: Int = 0
    ) {
        self.featureExtractionSeconds = featureExtractionSeconds
        self.encoderSeconds = encoderSeconds
        self.decoderSeconds = decoderSeconds
        self.alignmentSeconds = alignmentSeconds
        self.windowMergeSeconds = windowMergeSeconds
        self.cleanupSeconds = cleanupSeconds
        self.totalSeconds = totalSeconds
        self.windowCount = windowCount
    }

    enum CodingKeys: String, CodingKey {
        case featureExtractionSeconds = "feature_extraction_seconds"
        case encoderSeconds = "encoder_seconds"
        case decoderSeconds = "decoder_seconds"
        case alignmentSeconds = "alignment_seconds"
        case windowMergeSeconds = "window_merge_seconds"
        case cleanupSeconds = "cleanup_seconds"
        case totalSeconds = "total_seconds"
        case windowCount = "window_count"
    }
}

public struct ParakeetMeasuredTranscription: Sendable {
    public let result: ASRResult
    public let timings: ParakeetPipelineTimings

    public init(result: ASRResult, timings: ParakeetPipelineTimings) {
        self.result = result
        self.timings = timings
    }
}

struct ParakeetModelTimings {
    var encoderSeconds: TimeInterval = 0
    var decoderSeconds: TimeInterval = 0
    var alignmentSeconds: TimeInterval = 0
}

enum ParakeetMonotonicClock {
    static func now() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    static func seconds(since start: UInt64) -> TimeInterval {
        TimeInterval(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }
}
