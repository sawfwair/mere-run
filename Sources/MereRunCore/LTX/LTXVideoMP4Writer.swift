import AVFoundation
import CoreVideo
import Foundation
import MLX

public enum LTXVideoMP4Writer {
    public enum WriterError: LocalizedError {
        case invalidFPS(Int)
        case unsupportedShape([Int])
        case unsupportedChannels(Int)
        case unsupportedAudioShape([Int])
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
                try mux(videoURL: tempVideoURL, audioURL: tempAudioURL, outputURL: outputURL)
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
        let rgbBytes = prepared.rgbBytes

        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            throw WriterError.writerCreationFailed(outputURL)
        }

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(1_000_000, width * height * fps * 4),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw WriterError.inputRejected
        }
        writer.add(input)

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attributes
        )

        guard writer.startWriting() else {
            let details = writer.error?.localizedDescription ?? "unknown error"
            throw WriterError.finishFailed(details)
        }
        writer.startSession(atSourceTime: .zero)

        guard let pool = adaptor.pixelBufferPool else {
            throw WriterError.pixelBufferPoolUnavailable
        }

        let frameStride = height * width * 3
        for frameIndex in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }

            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
            guard status == kCVReturnSuccess, let pixelBuffer else {
                throw WriterError.pixelBufferCreationFailed
            }

            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                throw WriterError.pixelBufferBaseAddressUnavailable
            }

            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let dst = baseAddress.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
            let srcOffset = frameIndex * frameStride

            for y in 0..<height {
                let dstRow = dst.advanced(by: y * bytesPerRow)
                let srcRow = srcOffset + (y * width * 3)
                for x in 0..<width {
                    let s = srcRow + x * 3
                    let d = x * 4
                    dstRow[d] = rgbBytes[s + 2]     // B
                    dstRow[d + 1] = rgbBytes[s + 1] // G
                    dstRow[d + 2] = rgbBytes[s]     // R
                    dstRow[d + 3] = 255             // A
                }
            }

            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

            let time = CMTime(value: Int64(frameIndex), timescale: CMTimeScale(fps))
            if !adaptor.append(pixelBuffer, withPresentationTime: time) {
                throw WriterError.appendFailed(frameIndex)
            }
        }

        input.markAsFinished()

        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()

        guard writer.status == .completed else {
            let details = writer.error?.localizedDescription ?? "unknown error"
            throw WriterError.finishFailed(details)
        }
    }

    private static func prepareFrames(
        _ frames: MLXArray
    ) throws -> (rgbBytes: [UInt8], frameCount: Int, height: Int, width: Int) {
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

        let uint8Frames = tensor.asType(.uint8)
        MLX.eval(uint8Frames)
        let rgbBytes = uint8Frames.reshaped(-1).asArray(UInt8.self)
        return (rgbBytes, frameCount, height, width)
    }

    private static func prepareAudio(_ audio: MLXArray) throws -> (interleaved: [Float], channels: Int) {
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
        return (floatSamples, channels)
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

        let frameCount = interleaved.count / max(1, channels)
        var pcm = [Int16](repeating: 0, count: interleaved.count)
        for i in 0..<interleaved.count {
            let clipped = max(-1.0, min(1.0, interleaved[i]))
            pcm[i] = Int16((clipped * 32767.0).rounded())
        }

        var data = Data()
        data.reserveCapacity(44 + pcm.count * MemoryLayout<Int16>.size)

        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = pcm.count * MemoryLayout<Int16>.size
        let riffSize = 36 + dataSize

        data.append(contentsOf: Array("RIFF".utf8))
        data.appendLE(UInt32(riffSize))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        data.appendLE(UInt32(16)) // PCM fmt chunk size
        data.appendLE(UInt16(1)) // PCM
        data.appendLE(UInt16(channels))
        data.appendLE(UInt32(sampleRate))
        data.appendLE(UInt32(byteRate))
        data.appendLE(UInt16(blockAlign))
        data.appendLE(UInt16(bitsPerSample))
        data.append(contentsOf: Array("data".utf8))
        data.appendLE(UInt32(dataSize))
        pcm.withUnsafeBufferPointer { buffer in
            data.append(contentsOf: UnsafeRawBufferPointer(buffer))
        }

        try data.write(to: url)
        _ = frameCount
    }

    private static func mux(videoURL: URL, audioURL: URL, outputURL: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)

        guard let videoTrack = videoAsset.tracks(withMediaType: .video).first else {
            throw WriterError.videoTrackMissing
        }
        guard let audioTrack = audioAsset.tracks(withMediaType: .audio).first else {
            throw WriterError.audioTrackMissing
        }

        let composition = AVMutableComposition()
        guard let compositionVideoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw WriterError.videoTrackMissing
        }
        guard let compositionAudioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw WriterError.audioTrackMissing
        }

        let videoDuration = videoAsset.duration
        let audioDuration = audioAsset.duration
        let duration = CMTimeMinimum(videoDuration, audioDuration)

        try compositionVideoTrack.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: videoTrack, at: .zero)
        try compositionAudioTrack.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: audioTrack, at: .zero)
        compositionVideoTrack.preferredTransform = videoTrack.preferredTransform

        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw WriterError.exportSessionCreationFailed
        }
        export.outputURL = outputURL
        export.outputFileType = .mp4
        export.shouldOptimizeForNetworkUse = true

        let semaphore = DispatchSemaphore(value: 0)
        export.exportAsynchronously {
            semaphore.signal()
        }
        semaphore.wait()

        guard export.status == .completed else {
            let details = export.error?.localizedDescription ?? "unknown error"
            throw WriterError.finishFailed(details)
        }
    }
}

private extension Data {
    mutating func appendLE(_ value: UInt16) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }

    mutating func appendLE(_ value: UInt32) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}
