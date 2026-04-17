import Foundation

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

struct SAM31VideoAsset: Sendable {
    let frameURLs: [URL]
    let fps: Double
    let frameWidth: Int
    let frameHeight: Int
}

enum SAM31VideoIO {
    enum VideoIOError: LocalizedError, Sendable {
        case missingVideoTrack(URL)
        case failedToExtractFrame(URL, Int)
        case failedToLoadFrame(URL)
        case failedToWriteFrame(URL)
        case failedToCreateWriter(URL)
        case failedToCreatePixelBuffer
        case failedToAppendFrame(Int)

        var errorDescription: String? {
            switch self {
            case .missingVideoTrack(let url):
                return "Video track missing in asset: \(url.path)"
            case .failedToExtractFrame(let url, let index):
                return "Failed to extract frame \(index) from \(url.path)."
            case .failedToLoadFrame(let url):
                return "Failed to load frame image: \(url.path)"
            case .failedToWriteFrame(let url):
                return "Failed to write frame image: \(url.path)"
            case .failedToCreateWriter(let url):
                return "Failed to create MP4 writer for \(url.path)."
            case .failedToCreatePixelBuffer:
                return "Failed to create a video pixel buffer."
            case .failedToAppendFrame(let index):
                return "Failed to append frame \(index) to the output video."
            }
        }
    }

    static func extractFrames(
        from videoURL: URL,
        into outputDirectoryURL: URL,
        endFrame: Int? = nil
    ) throws -> SAM31VideoAsset {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: outputDirectoryURL, withIntermediateDirectories: true)

        let asset = AVURLAsset(url: videoURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw VideoIOError.missingVideoTrack(videoURL)
        }

        let nominalFPS = track.nominalFrameRate > 0 ? Double(track.nominalFrameRate) : 30.0
        let fps = max(1.0, nominalFPS)
        let durationSeconds = CMTimeGetSeconds(asset.duration)
        let estimatedFrameCount = max(1, Int((durationSeconds * fps).rounded(.toNearestOrEven)))
        let frameCount = min(endFrame.map { $0 + 1 } ?? estimatedFrameCount, estimatedFrameCount)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var frameURLs: [URL] = []
        frameURLs.reserveCapacity(frameCount)
        var frameWidth = 0
        var frameHeight = 0

        for frameIndex in 0..<frameCount {
            let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(max(1, Int32(fps.rounded()))))
            let image: CGImage
            do {
                image = try generator.copyCGImage(at: time, actualTime: nil)
            } catch {
                throw VideoIOError.failedToExtractFrame(videoURL, frameIndex)
            }

            if frameWidth == 0 || frameHeight == 0 {
                frameWidth = image.width
                frameHeight = image.height
            }

            let frameURL = outputDirectoryURL
                .appendingPathComponent(String(format: "frame_%05d", frameIndex))
                .appendingPathExtension("png")
            try writeImage(image, to: frameURL)
            frameURLs.append(frameURL)
        }

        return SAM31VideoAsset(
            frameURLs: frameURLs,
            fps: fps,
            frameWidth: frameWidth,
            frameHeight: frameHeight
        )
    }

    static func writeVideo(
        frameURLs: [URL],
        fps: Double,
        to outputURL: URL
    ) throws {
        guard let firstURL = frameURLs.first,
              let firstImage = QwenVLImageLoader.loadCGImage(url: firstURL)
        else {
            throw VideoIOError.failedToLoadFrame(frameURLs.first ?? outputURL)
        }

        let width = firstImage.width
        let height = firstImage.height
        let frameDuration = CMTime(seconds: 1.0 / max(fps, 1.0), preferredTimescale: 600)

        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: outputURL)

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            throw VideoIOError.failedToCreateWriter(outputURL)
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(1_000_000, width * height * Int(fps) * 4),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false

        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: pixelBufferAttributes
        )

        guard writer.canAdd(input) else {
            throw VideoIOError.failedToCreateWriter(outputURL)
        }
        writer.add(input)

        guard writer.startWriting() else {
            throw VideoIOError.failedToCreateWriter(outputURL)
        }
        writer.startSession(atSourceTime: .zero)

        guard let pool = adaptor.pixelBufferPool else {
            throw VideoIOError.failedToCreatePixelBuffer
        }

        for (frameIndex, frameURL) in frameURLs.enumerated() {
            guard let image = QwenVLImageLoader.loadCGImage(url: frameURL) else {
                throw VideoIOError.failedToLoadFrame(frameURL)
            }

            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.002)
            }

            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
            guard status == kCVReturnSuccess, let pixelBuffer else {
                throw VideoIOError.failedToCreatePixelBuffer
            }

            try render(image, into: pixelBuffer)
            let presentationTime = CMTimeMultiply(frameDuration, multiplier: Int32(frameIndex))
            guard adaptor.append(pixelBuffer, withPresentationTime: presentationTime) else {
                throw VideoIOError.failedToAppendFrame(frameIndex)
            }
        }

        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()
    }

    private static func render(_ image: CGImage, into pixelBuffer: CVPixelBuffer) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw VideoIOError.failedToCreatePixelBuffer
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            throw VideoIOError.failedToCreatePixelBuffer
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private static func writeImage(_ image: CGImage, to url: URL) throws {
        let utType = UTType(filenameExtension: url.pathExtension.lowercased()) ?? .png
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, utType.identifier as CFString, 1, nil) else {
            throw VideoIOError.failedToWriteFrame(url)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw VideoIOError.failedToWriteFrame(url)
        }
    }
}
#endif
