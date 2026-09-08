import Foundation
import MLX

public enum SortformerDiarizationError: LocalizedError, Sendable {
    case emptyAudio
    case unsupportedSampleRate(actual: Int, expected: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyAudio:
            return "Speaker diarization requires non-empty audio."
        case .unsupportedSampleRate(let actual, let expected):
            return "Speaker diarization requires \(expected) Hz mono audio; received \(actual) Hz."
        }
    }
}

public final class SortformerDiarizer {
    private let model: SortformerModel

    public init(modelDirectory: URL) throws {
        model = try SortformerModel.fromModelDirectory(modelDirectory)
    }

    public func diarize(
        samples: [Float],
        sampleRate: Int = 16_000,
        threshold: Float = 0.5,
        minDuration: Float = 0,
        mergeGap: Float = 0
    ) throws -> DiarizationOutput {
        try model.generate(
            audio: MLXArray(samples),
            sampleRate: sampleRate,
            threshold: threshold,
            minDuration: minDuration,
            mergeGap: mergeGap
        )
    }
}
