import Foundation
import MediaIO
import MLX

public enum LTXVideoMP4Writer {
    static let defaultAudioBitRate = 192_000

    public enum WriterError: LocalizedError {
        case invalidFPS(Int)
        case unsupportedShape([Int])
        case unsupportedChannels(Int)
        case unsupportedAudioShape([Int])
        case nonFiniteAudioSample
        case writerCreationFailed(URL)
        case inputRejected
        case pixelBufferPoolUnavailable
        case pixelBufferCreationFailed
        case pixelBufferBaseAddressUnavailable
        case appendFailed(Int)
        case audioTrackMissing
        case videoTrackMissing
        case exportSessionCreationFailed
        case finishFailed(String)

        public var errorDescription: String? {
            switch self {
            case .invalidFPS(let fps):
                return "FPS must be >= 1 (got \(fps))."
            case .unsupportedShape(let shape):
                return "Expected frames shaped [F, H, W, 3] (or [1, F, H, W, 3]), got \(shape)."
            case .unsupportedChannels(let channels):
                return "Expected 3-channel RGB frames, got \(channels) channels."
            case .unsupportedAudioShape(let shape):
                return "Expected audio shaped [S], [C,S], [1,C,S], or [1,S,C], got \(shape)."
            case .nonFiniteAudioSample:
                return "Audio contains a non-finite sample."
            case .writerCreationFailed(let url):
                return "Could not create MP4 writer for \(url.path)."
            case .inputRejected:
                return "AVAssetWriter rejected video input settings."
            case .pixelBufferPoolUnavailable:
                return "AVAssetWriter did not provide a pixel buffer pool."
            case .pixelBufferCreationFailed:
                return "Failed to allocate pixel buffer."
            case .pixelBufferBaseAddressUnavailable:
                return "Pixel buffer base address is unavailable."
            case .appendFailed(let frame):
                return "Failed to append frame \(frame) to MP4 stream."
            case .audioTrackMissing:
                return "Audio track is missing from source asset."
            case .videoTrackMissing:
                return "Video track is missing from source asset."
            case .exportSessionCreationFailed:
                return "Could not create AVAssetExportSession."
            case .finishFailed(let details):
                return "Failed to finish MP4 writing. \(details)"
            }
        }
    }

    public static func writeMP4(
        frames: MLXArray,
        fps: Int,
        to outputURL: URL,
        audioWaveform: MLXArray? = nil,
        audioSampleRate: Int = 24_000
    ) throws {
        let fm = FileManager.default
        if let audioWaveform {
            let tmpDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            let nonce = UUID().uuidString
            let tempVideoURL = tmpDir.appendingPathComponent("mererun-video-\(nonce)-video.mp4")
            let tempAudioURL = tmpDir.appendingPathComponent("mererun-video-\(nonce)-audio.wav")

            do {
                try writeVideoOnly(frames: frames, fps: fps, to: tempVideoURL)

                let preparedAudio = try prepareAudio(audioWaveform)
                try writeWAV(interleaved: preparedAudio.interleaved, channels: preparedAudio.channels, sampleRate: audioSampleRate, to: tempAudioURL)
                try mux(
                    videoURL: tempVideoURL,
                    audioURL: tempAudioURL,
                    outputURL: outputURL,
                    audioBitRate: Self.defaultAudioBitRate
                )
            } catch {
                try? fm.removeItem(at: tempVideoURL)
                try? fm.removeItem(at: tempAudioURL)
                throw error
            }

            try? fm.removeItem(at: tempVideoURL)
            try? fm.removeItem(at: tempAudioURL)
            return
        }

        try writeVideoOnly(frames: frames, fps: fps, to: outputURL)
    }

    private static func writeVideoOnly(
        frames: MLXArray,
        fps: Int,
        to outputURL: URL
    ) throws {
        guard fps >= 1 else {
            throw WriterError.invalidFPS(fps)
        }

        let prepared = try prepareFrames(frames)
        let frameCount = prepared.frameCount
        let height = prepared.height
        let width = prepared.width

        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        var pendingFrame = bgraFrame(prepared.tensor, at: 0)
        asyncEval(pendingFrame)
        do {
            try MediaVideoIO.writeMP4(
                bgra32FrameAt: { frameIndex in
                    let currentFrame = pendingFrame
                    if frameIndex + 1 < frameCount {
                        pendingFrame = bgraFrame(prepared.tensor, at: frameIndex + 1)
                        asyncEval(pendingFrame)
                    }
                    return currentFrame.reshaped(-1).asArray(UInt8.self)
                },
                width: width,
                height: height,
                frameCount: frameCount,
                fps: fps,
                to: outputURL
            )
        } catch {
            throw WriterError.writerCreationFailed(outputURL)
        }
    }

    private static func prepareFrames(
        _ frames: MLXArray
    ) throws -> (tensor: MLXArray, frameCount: Int, height: Int, width: Int) {
        var tensor = frames
        if tensor.ndim == 5 {
            guard tensor.dim(0) == 1 else {
                throw WriterError.unsupportedShape(tensor.shape)
            }
            tensor = tensor[0, 0..., 0..., 0..., 0...]
        }

        guard tensor.ndim == 4 else {
            throw WriterError.unsupportedShape(tensor.shape)
        }

        let frameCount = tensor.dim(0)
        let height = tensor.dim(1)
        let width = tensor.dim(2)
        let channels = tensor.dim(3)
        guard channels == 3 else {
            throw WriterError.unsupportedChannels(channels)
        }
        guard frameCount > 0, height > 0, width > 0 else {
            throw WriterError.unsupportedShape(tensor.shape)
        }

        return (tensor, frameCount, height, width)
    }

    static func bgraFrame(_ frames: MLXArray, at frameIndex: Int) -> MLXArray {
        let height = frames.dim(1)
        let width = frames.dim(2)
        let rgb = frames[frameIndex, 0..., 0..., 0...].asType(.uint8)
        let bgr = MLX.take(rgb, MLXArray([Int32(2), 1, 0]), axis: -1)
        let alpha = MLX.full([height, width, 1], values: UInt8(255))
        return MLX.concatenated([bgr, alpha], axis: -1)
    }

    static func prepareAudio(_ audio: MLXArray) throws -> (interleaved: [Float], channels: Int) {
        var sampleChannel = audio
        if sampleChannel.ndim == 1 {
            sampleChannel = sampleChannel.reshaped(sampleChannel.dim(0), 1)
        } else if sampleChannel.ndim == 2 {
            if sampleChannel.dim(1) <= 8 {
                // [samples, channels]
            } else if sampleChannel.dim(0) <= 8 {
                // [channels, samples]
                sampleChannel = sampleChannel.transposed(1, 0)
            } else {
                throw WriterError.unsupportedAudioShape(sampleChannel.shape)
            }
        } else if sampleChannel.ndim == 3 {
            guard sampleChannel.dim(0) == 1 else {
                throw WriterError.unsupportedAudioShape(sampleChannel.shape)
            }
            let squeezed = sampleChannel[0, 0..., 0...]
            if squeezed.dim(1) <= 8 {
                sampleChannel = squeezed
            } else if squeezed.dim(0) <= 8 {
                sampleChannel = squeezed.transposed(1, 0)
            } else {
                throw WriterError.unsupportedAudioShape(audio.shape)
            }
        } else {
            throw WriterError.unsupportedAudioShape(audio.shape)
        }

        let channels = sampleChannel.dim(1)
        guard channels >= 1 else {
            throw WriterError.unsupportedAudioShape(audio.shape)
        }
        let floatSamples = sampleChannel.asType(.float32).reshaped(-1).asArray(Float.self)
        let encodedSamples = try floatSamples.map { sample -> Float in
            guard sample.isFinite else {
                throw WriterError.nonFiniteAudioSample
            }
            return min(1.0, max(-1.0, sample))
        }
        return (encodedSamples, channels)
    }

    private static func writeWAV(
        interleaved: [Float],
        channels: Int,
        sampleRate: Int,
        to url: URL
    ) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: url.path) {
            try fm.removeItem(at: url)
        }
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)

        try MediaAudioIO.writeFloatWAV(
            samples: interleaved,
            sampleRate: sampleRate,
            channels: channels,
            to: url
        )
    }

    private static func mux(
        videoURL: URL,
        audioURL: URL,
        outputURL: URL,
        audioBitRate: Int
    ) throws {
        do {
            try MediaVideoIO.mux(
                videoURL: videoURL,
                audioURL: audioURL,
                outputURL: outputURL,
                audioBitRate: audioBitRate
            )
        } catch {
            throw WriterError.exportSessionCreationFailed
        }
    }
}
