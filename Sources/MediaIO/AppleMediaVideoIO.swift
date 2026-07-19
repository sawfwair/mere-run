import Foundation

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

private typealias AppleVideoSettings = Dictionary<String, Any>

enum AppleMediaVideoIO {
    static func writeMP4(
        bgra32FrameAt frameProvider: MediaVideoIO.BGRAFrameProvider,
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        try writeMP4(
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        ) { frameIndex, destination, bytesPerRow in
            let frame = try frameProvider(frameIndex)
            let frameStride = width * height * 4
            guard frame.count == frameStride else {
                throw MediaIOError.invalidBufferSize(expected: frameStride, actual: frame.count)
            }
            frame.withUnsafeBufferPointer { source in
                guard let sourceAddress = source.baseAddress else { return }
                let rowBytes = width * 4
                for row in 0..<height {
                    destination.advanced(by: row * bytesPerRow).update(
                        from: sourceAddress.advanced(by: row * rowBytes),
                        count: rowBytes
                    )
                }
            }
        }
    }

    static func writeMP4(
        rgb24: [UInt8],
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL
    ) throws {
        let frameStride = width * height * 3
        guard rgb24.count == frameStride * frameCount else {
            throw MediaIOError.invalidBufferSize(expected: frameStride * frameCount, actual: rgb24.count)
        }
        try writeMP4(
            width: width,
            height: height,
            frameCount: frameCount,
            fps: fps,
            to: outputURL
        ) { frameIndex, destination, bytesPerRow in
            let srcOffset = frameIndex * frameStride
            for y in 0..<height {
                let dstRow = destination.advanced(by: y * bytesPerRow)
                let srcRow = srcOffset + (y * width * 3)
                for x in 0..<width {
                    let src = srcRow + (x * 3)
                    let out = x * 4
                    dstRow[out] = rgb24[src + 2]
                    dstRow[out + 1] = rgb24[src + 1]
                    dstRow[out + 2] = rgb24[src]
                    dstRow[out + 3] = 255
                }
            }
        }
    }

    private static func writeMP4(
        width: Int,
        height: Int,
        frameCount: Int,
        fps: Int,
        to outputURL: URL,
        fileType: AVFileType = .mp4,
        codec: AVVideoCodecType = .h264,
        fillFrame: (_ frameIndex: Int, _ destination: UnsafeMutablePointer<UInt8>, _ bytesPerRow: Int) throws -> Void
    ) throws {
        guard width > 0, height > 0, frameCount > 0 else {
            throw MediaIOError.videoOperationFailed("Invalid MP4 dimensions or frame rate.")
        }
        let frameRate = try MediaVideoFrameRateResolver.resolve(Double(fps))
        let integerFPS = Int(frameRate.timeScale)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? FileManager.default.removeItem(at: outputURL)

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: fileType) else {
            throw MediaIOError.videoOperationFailed("Could not create video writer for \(outputURL.path).")
        }
        var settings: AppleVideoSettings = [
            AVVideoCodecKey: codec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        var compressionProperties: AppleVideoSettings = [
            AVVideoExpectedSourceFrameRateKey: integerFPS,
        ]
        if codec == .h264 {
            compressionProperties.merge([
                AVVideoAverageBitRateKey: max(1_000_000, width * height * integerFPS * 4),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ]) { _, new in new }
        }
        settings[AVVideoCompressionPropertiesKey] = compressionProperties
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        input.mediaTimeScale = frameRate.timeScale
        guard writer.canAdd(input) else {
            throw MediaIOError.videoOperationFailed("AVAssetWriter rejected video input settings.")
        }
        writer.add(input)

        let attributes: AppleVideoSettings = [
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
            throw MediaIOError.videoOperationFailed(writer.error?.localizedDescription ?? "writer failed")
        }
        writer.startSession(atSourceTime: .zero)
        guard let pool = adaptor.pixelBufferPool else {
            throw MediaIOError.videoOperationFailed("AVAssetWriter did not provide a pixel buffer pool.")
        }

        for frameIndex in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }
            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
            guard status == kCVReturnSuccess, let pixelBuffer else {
                throw MediaIOError.videoOperationFailed("Failed to allocate pixel buffer.")
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                throw MediaIOError.videoOperationFailed("Pixel buffer base address unavailable.")
            }
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let dst = base.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
            do {
                try fillFrame(frameIndex, dst, bytesPerRow)
            } catch {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
                throw error
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            let time = CMTime(value: Int64(frameIndex), timescale: frameRate.timeScale)
            guard adaptor.append(pixelBuffer, withPresentationTime: time) else {
                throw MediaIOError.videoOperationFailed("Failed to append frame \(frameIndex).")
            }
        }

        input.markAsFinished()
        writer.endSession(atSourceTime: CMTime(value: Int64(frameCount), timescale: frameRate.timeScale))
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        guard writer.status == .completed else {
            throw MediaIOError.videoOperationFailed(writer.error?.localizedDescription ?? "writer did not complete")
        }
    }

    static func mux(videoURL: URL, audioURL: URL, outputURL: URL) throws {
        let videoAsset = AVURLAsset(url: videoURL)
        let audioAsset = AVURLAsset(url: audioURL)
        guard let videoTrack = videoAsset.tracks(withMediaType: .video).first else {
            throw MediaIOError.videoOperationFailed("Video track is missing from source asset.")
        }
        guard let audioTrack = audioAsset.tracks(withMediaType: .audio).first else {
            throw MediaIOError.videoOperationFailed("Audio track is missing from source asset.")
        }
        let composition = AVMutableComposition()
        guard let compositionVideoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
              let compositionAudioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw MediaIOError.videoOperationFailed("Could not create composition tracks.")
        }
        let duration = CMTimeMinimum(videoAsset.duration, audioAsset.duration)
        try compositionVideoTrack.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: videoTrack, at: .zero)
        try compositionAudioTrack.insertTimeRange(CMTimeRange(start: .zero, duration: duration), of: audioTrack, at: .zero)
        compositionVideoTrack.preferredTransform = videoTrack.preferredTransform
        try exportComposition(composition, outputURL: outputURL, videoComposition: nil, audioMix: nil)
    }

    static func extractFrames(
        from videoURL: URL,
        into outputDirectoryURL: URL,
        endFrame: Int?,
        validateDecodedSequence: ((Int, Int, Int) throws -> Void)?
    ) throws -> VideoFrameSequence {
        let asset = AVURLAsset(url: videoURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw MediaIOError.videoOperationFailed("Video track missing in asset: \(videoURL.path)")
        }
        let frameRate = try MediaVideoFrameRateResolver.resolve(
            Double(track.nominalFrameRate),
            fallbackFPS: 30
        )
        let fps = frameRate.framesPerSecond
        let durationSeconds = CMTimeGetSeconds(asset.duration)
        let estimatedFrameCountValue = (durationSeconds * fps).rounded(.toNearestOrEven)
        let requestedFrameCount: Int?
        if let endFrame {
            let (count, overflow) = endFrame.addingReportingOverflow(1)
            guard endFrame >= 0, !overflow else {
                throw MediaIOError.videoOperationFailed("Invalid end frame \(endFrame).")
            }
            requestedFrameCount = count
        } else {
            requestedFrameCount = nil
        }
        let estimatedFrameCount: Int
        if estimatedFrameCountValue.isFinite,
           estimatedFrameCountValue > 0,
           estimatedFrameCountValue <= Double(Int.max) {
            estimatedFrameCount = max(1, Int(estimatedFrameCountValue))
        } else if let requestedFrameCount {
            estimatedFrameCount = requestedFrameCount
        } else {
            throw MediaIOError.videoOperationFailed("Video frame count is not finite or representable.")
        }
        let frameCount = min(requestedFrameCount ?? estimatedFrameCount, estimatedFrameCount)
        if let validateDecodedSequence {
            let transformedBounds = CGRect(origin: .zero, size: track.naturalSize)
                .applying(track.preferredTransform)
            let metadataWidth = abs(transformedBounds.width).rounded(.up)
            let metadataHeight = abs(transformedBounds.height).rounded(.up)
            guard metadataWidth.isFinite,
                  metadataHeight.isFinite,
                  metadataWidth > 0,
                  metadataHeight > 0,
                  metadataWidth <= Double(Int.max),
                  metadataHeight <= Double(Int.max) else {
                throw MediaIOError.videoOperationFailed(
                    "Video frame dimensions are not finite or representable."
                )
            }
            try validateDecodedSequence(
                Int(metadataWidth),
                Int(metadataHeight),
                frameCount
            )
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var frameURLs: [URL] = []
        var frameWidth = 0
        var frameHeight = 0
        for frameIndex in 0..<frameCount {
            let time = CMTime(value: CMTimeValue(frameIndex), timescale: frameRate.timeScale)
            let image = try generator.copyCGImage(at: time, actualTime: nil)
            if frameWidth == 0 || frameHeight == 0 {
                frameWidth = image.width
                frameHeight = image.height
                try validateDecodedSequence?(frameWidth, frameHeight, frameCount)
                try FileManager.default.createDirectory(
                    at: outputDirectoryURL,
                    withIntermediateDirectories: true
                )
            } else if image.width != frameWidth || image.height != frameHeight {
                try validateDecodedSequence?(image.width, image.height, frameCount)
            }
            let frameURL = outputDirectoryURL
                .appendingPathComponent(String(format: "frame_%05d", frameIndex))
                .appendingPathExtension("png")
            try writeImage(image, to: frameURL)
            frameURLs.append(frameURL)
        }
        return VideoFrameSequence(frameURLs: frameURLs, fps: fps, frameWidth: frameWidth, frameHeight: frameHeight)
    }

    static func writeVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        let frameRate = try MediaVideoFrameRateResolver.resolve(fps, fallbackFPS: 1)
        guard let firstURL = frameURLs.first,
              let firstImage = loadCGImage(firstURL) else {
            throw MediaIOError.videoOperationFailed("No frames supplied for video writing.")
        }
        var rgb = [UInt8]()
        rgb.reserveCapacity(frameURLs.count * firstImage.width * firstImage.height * 3)
        for url in frameURLs {
            let media = try MediaImageIO.decode(url)
            guard media.width == firstImage.width, media.height == firstImage.height else {
                throw MediaIOError.videoOperationFailed("Frame dimensions do not match.")
            }
            for pixel in 0..<(media.width * media.height) {
                let src = pixel * 4
                rgb.append(media.rgba8[src])
                rgb.append(media.rgba8[src + 1])
                rgb.append(media.rgba8[src + 2])
            }
        }
        try writeMP4(
            rgb24: rgb,
            width: firstImage.width,
            height: firstImage.height,
            frameCount: frameURLs.count,
            fps: Int(frameRate.timeScale),
            to: outputURL
        )
    }

    static func writePaletteVideo(frameURLs: [URL], fps: Double, to outputURL: URL) throws {
        let frameRate = try MediaVideoFrameRateResolver.resolve(fps, fallbackFPS: 1)
        guard let firstURL = frameURLs.first else {
            throw MediaIOError.videoOperationFailed("No frames supplied for palette video writing.")
        }
        let firstImage = try MediaImageIO.decode(firstURL)
        try writeMP4(
            width: firstImage.width,
            height: firstImage.height,
            frameCount: frameURLs.count,
            fps: Int(frameRate.timeScale),
            to: outputURL,
            fileType: .mov,
            codec: .proRes4444
        ) { frameIndex, destination, bytesPerRow in
            let image = try MediaImageIO.decode(frameURLs[frameIndex])
            guard image.width == firstImage.width, image.height == firstImage.height else {
                throw MediaIOError.videoOperationFailed("Palette frame dimensions do not match.")
            }
            for y in 0..<image.height {
                let destinationRow = destination.advanced(by: y * bytesPerRow)
                for x in 0..<image.width {
                    let source = ((y * image.width) + x) * 4
                    let target = x * 4
                    destinationRow[target] = image.rgba8[source + 2]
                    destinationRow[target + 1] = image.rgba8[source + 1]
                    destinationRow[target + 2] = image.rgba8[source]
                    destinationRow[target + 3] = 255
                }
            }
        }
    }

    static func hasAudioTrack(_ url: URL) -> Bool {
        let asset = AVURLAsset(url: url)
        return !asset.tracks(withMediaType: .audio).isEmpty
    }

    private static func loadCGImage(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    private static func writeImage(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw MediaIOError.imageEncodeFailed(url)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw MediaIOError.imageEncodeFailed(url)
        }
    }

    private static func exportComposition(
        _ composition: AVComposition,
        outputURL: URL,
        videoComposition: AVVideoComposition?,
        audioMix: AVAudioMix?
    ) throws {
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: outputURL)
        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw MediaIOError.videoOperationFailed("Could not create AVAssetExportSession.")
        }
        export.outputURL = outputURL
        export.outputFileType = outputURL.pathExtension.lowercased() == "mov" ? .mov : .mp4
        export.shouldOptimizeForNetworkUse = true
        export.videoComposition = videoComposition
        export.audioMix = audioMix
        let semaphore = DispatchSemaphore(value: 0)
        export.exportAsynchronously { semaphore.signal() }
        semaphore.wait()
        guard export.status == .completed else {
            throw MediaIOError.videoOperationFailed(export.error?.localizedDescription ?? "export failed")
        }
    }
}
#endif
