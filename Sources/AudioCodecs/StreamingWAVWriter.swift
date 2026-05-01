import AVFoundation
import Foundation

public final class StreamingWAVWriter {
    public enum Error: LocalizedError {
        case invalidFormat
        case bufferAllocationFailed
        case missingChannelData

        public var errorDescription: String? {
            switch self {
            case .invalidFormat:
                return "Failed to create audio format for streaming WAV output."
            case .bufferAllocationFailed:
                return "Failed to allocate audio buffer for streaming WAV output."
            case .missingChannelData:
                return "Streaming audio buffer is missing channel data."
            }
        }
    }

    private let file: AVAudioFile
    private let format: AVAudioFormat

    public init(outputURL: URL, sampleRate: Int, channels: AVAudioChannelCount = 1) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(max(1, sampleRate)),
            channels: max(1, channels),
            interleaved: false
        ) else {
            throw Error.invalidFormat
        }

        self.format = format
        self.file = try AVAudioFile(
            forWriting: outputURL,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
    }

    public func append(samples: [Float]) throws {
        guard !samples.isEmpty else { return }
        let frameCount = AVAudioFrameCount(samples.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw Error.bufferAllocationFailed
        }

        buffer.frameLength = frameCount
        guard let channelData = buffer.floatChannelData?[0] else {
            throw Error.missingChannelData
        }

        samples.withUnsafeBufferPointer { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            channelData.update(from: baseAddress, count: samples.count)
        }

        try file.write(from: buffer)
    }
}
