import Foundation

#if canImport(AVFoundation)
@preconcurrency import AVFoundation

enum AppleMediaAudioIO {
    static func probe(_ url: URL) throws -> MediaAudioMetadata {
        let file = try open(url)
        let format = file.processingFormat
        let sampleRate = Int(format.sampleRate.rounded())
        let frameCount = Int64(file.length)
        return MediaAudioMetadata(
            sampleRate: sampleRate,
            channelCount: Int(format.channelCount),
            frameCount: frameCount,
            durationSeconds: Double(frameCount) / Double(max(1, sampleRate))
        )
    }

    static func decode(
        _ url: URL,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        let file = try open(url)

        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else {
            throw MediaIOError.audioDecodeFailed(url, "Audio file is empty.")
        }
        let buffer = try read(file, frameCount: frameCount, url: url)

        return try convert(
            buffer,
            sourceFormat: sourceFormat,
            targetSampleRate: targetSampleRate,
            channels: channels,
            url: url
        )
    }

    static func decodeSegment(
        _ url: URL,
        startTime: Double,
        duration: Double,
        targetSampleRate: Int,
        channels: Int
    ) throws -> MediaAudioBuffer {
        let file = try open(url)
        let sourceFormat = file.processingFormat
        let startFrame = AVAudioFramePosition((startTime * sourceFormat.sampleRate).rounded(.down))
        guard startFrame < file.length else {
            throw MediaIOError.audioDecodeFailed(url, "Start time is beyond the end of the audio stream.")
        }
        let requestedFrames = AVAudioFramePosition((duration * sourceFormat.sampleRate).rounded(.up))
        let availableFrames = file.length - startFrame
        let frameCount = AVAudioFrameCount(min(requestedFrames, availableFrames))
        guard frameCount > 0 else {
            throw MediaIOError.audioDecodeFailed(url, "The requested audio interval is empty.")
        }

        file.framePosition = startFrame
        let buffer = try read(file, frameCount: frameCount, url: url)
        return try convert(
            buffer,
            sourceFormat: sourceFormat,
            targetSampleRate: targetSampleRate,
            channels: channels,
            url: url
        )
    }

    // AVAudioFile may return a short read before EOF, including uncompressed WAVs.
    private static func read(_ file: AVAudioFile, frameCount: AVAudioFrameCount, url: URL) throws -> AVAudioPCMBuffer {
        guard let output = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount),
              let chunk = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: min(frameCount, 65_536)),
              let destination = output.floatChannelData, let source = chunk.floatChannelData else {
            throw MediaIOError.audioDecodeFailed(url, "Failed to allocate audio buffer.")
        }
        do {
            while output.frameLength < frameCount {
                try file.read(into: chunk, frameCount: min(chunk.frameCapacity, frameCount - output.frameLength))
                if chunk.frameLength == 0 { break }
                for channel in 0..<Int(file.processingFormat.channelCount) {
                    destination[channel].advanced(by: Int(output.frameLength)).update(from: source[channel], count: Int(chunk.frameLength))
                }
                output.frameLength += chunk.frameLength
            }
        } catch {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        }
        return output
    }

    private static func open(_ url: URL) throws -> AVAudioFile {
        do {
            return try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch {
            throw MediaIOError.audioDecodeFailed(url, error.localizedDescription)
        }
    }

    private static func convert(
        _ buffer: AVAudioPCMBuffer,
        sourceFormat: AVAudioFormat,
        targetSampleRate: Int,
        channels: Int,
        url: URL
    ) throws -> MediaAudioBuffer {
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(max(1, targetSampleRate)),
            channels: AVAudioChannelCount(max(1, channels)),
            interleaved: false
        )!

        let outputBuffer: AVAudioPCMBuffer
        if sourceFormat.sampleRate == targetFormat.sampleRate,
           sourceFormat.channelCount == targetFormat.channelCount,
           sourceFormat.commonFormat == .pcmFormatFloat32 {
            outputBuffer = buffer
        } else {
            guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
                throw MediaIOError.audioDecodeFailed(url, "Failed to create audio converter.")
            }
            let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
            let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
            guard let converted = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputCapacity) else {
                throw MediaIOError.audioDecodeFailed(url, "Failed to allocate converted audio buffer.")
            }
            var didConvert = false
            var conversionError: NSError?
            converter.convert(to: converted, error: &conversionError) { _, outStatus in
                if didConvert {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                didConvert = true
                outStatus.pointee = .haveData
                return buffer
            }
            if let conversionError {
                throw MediaIOError.audioDecodeFailed(url, conversionError.localizedDescription)
            }
            outputBuffer = converted
        }

        guard let floatData = outputBuffer.floatChannelData else {
            throw MediaIOError.audioDecodeFailed(url, "Decoded audio has no float channel data.")
        }

        let frames = Int(outputBuffer.frameLength)
        let channelCount = Int(outputBuffer.format.channelCount)
        var interleaved = [Float](repeating: 0, count: frames * channelCount)
        for frame in 0..<frames {
            for channel in 0..<channelCount {
                interleaved[(frame * channelCount) + channel] = floatData[channel][frame]
            }
        }

        return MediaAudioBuffer(
            samples: interleaved,
            sampleRate: Int(outputBuffer.format.sampleRate.rounded()),
            channelCount: channelCount,
            isInterleaved: true
        )
    }
}
#endif
