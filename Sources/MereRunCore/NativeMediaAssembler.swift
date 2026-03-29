import AVFoundation
import CoreGraphics
import CoreText
import Foundation
import ImageIO
import QuartzCore

public struct NativeMediaSegment: Sendable {
    public enum Source: Sendable {
        case image(url: URL, motion: NativeKenBurnsMotion)
        case video(url: URL)
    }

    public let startSeconds: Double
    public let endSeconds: Double
    public let source: Source

    public init(startSeconds: Double, endSeconds: Double, source: Source) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.source = source
    }
}

public struct NativeKenBurnsMotion: Sendable {
    public let startZoom: Double
    public let endZoom: Double
    public let driftXFraction: Double
    public let driftYFraction: Double
    public let driftXFrequency: Double
    public let driftYFrequency: Double

    public init(
        startZoom: Double,
        endZoom: Double,
        driftXFraction: Double,
        driftYFraction: Double,
        driftXFrequency: Double,
        driftYFrequency: Double
    ) {
        self.startZoom = startZoom
        self.endZoom = endZoom
        self.driftXFraction = driftXFraction
        self.driftYFraction = driftYFraction
        self.driftXFrequency = driftXFrequency
        self.driftYFrequency = driftYFrequency
    }
}

public struct NativeMediaCaptionCue: Sendable {
    public let startSeconds: Double
    public let endSeconds: Double
    public let text: String

    public init(startSeconds: Double, endSeconds: Double, text: String) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.text = text
    }
}

public struct NativeMediaAudioLayer: Sendable {
    public let url: URL
    public let startSeconds: Double
    public let volume: Float

    public init(url: URL, startSeconds: Double, volume: Float) {
        self.url = url
        self.startSeconds = startSeconds
        self.volume = volume
    }
}

public struct NativeMediaCaptionStyle: Sendable {
    public let fontName: String
    public let fontSize: CGFloat
    public let textColor: (Double, Double, Double, Double)
    public let outlineColor: (Double, Double, Double, Double)
    public let outlineWidth: CGFloat
    public let horizontalInset: CGFloat
    public let bottomInset: CGFloat
    public let maxLines: Int

    public static let shortsDefault = NativeMediaCaptionStyle(
        fontName: "Helvetica-Bold",
        fontSize: 72,
        textColor: (1.0, 1.0, 1.0, 1.0),
        outlineColor: (0.0, 0.0, 0.0, 0.66),
        outlineWidth: -6,
        horizontalInset: 56,
        bottomInset: 120,
        maxLines: 3
    )

    public init(
        fontName: String,
        fontSize: CGFloat,
        textColor: (Double, Double, Double, Double),
        outlineColor: (Double, Double, Double, Double),
        outlineWidth: CGFloat,
        horizontalInset: CGFloat,
        bottomInset: CGFloat,
        maxLines: Int
    ) {
        self.fontName = fontName
        self.fontSize = fontSize
        self.textColor = textColor
        self.outlineColor = outlineColor
        self.outlineWidth = outlineWidth
        self.horizontalInset = horizontalInset
        self.bottomInset = bottomInset
        self.maxLines = maxLines
    }
}

public struct NativeMediaAssemblyRequest: Sendable {
    public let renderWidth: Int
    public let renderHeight: Int
    public let fps: Int
    public let segments: [NativeMediaSegment]
    public let captions: [NativeMediaCaptionCue]
    public let narrationLayers: [NativeMediaAudioLayer]
    public let backgroundLayer: NativeMediaAudioLayer?
    public let outputURL: URL
    public let workDirectory: URL
    public let captionStyle: NativeMediaCaptionStyle

    public init(
        renderWidth: Int,
        renderHeight: Int,
        fps: Int,
        segments: [NativeMediaSegment],
        captions: [NativeMediaCaptionCue],
        narrationLayers: [NativeMediaAudioLayer],
        backgroundLayer: NativeMediaAudioLayer?,
        outputURL: URL,
        workDirectory: URL,
        captionStyle: NativeMediaCaptionStyle
    ) {
        self.renderWidth = renderWidth
        self.renderHeight = renderHeight
        self.fps = fps
        self.segments = segments
        self.captions = captions
        self.narrationLayers = narrationLayers
        self.backgroundLayer = backgroundLayer
        self.outputURL = outputURL
        self.workDirectory = workDirectory
        self.captionStyle = captionStyle
    }
}

public enum NativeMediaAssemblerError: LocalizedError {
    case invalidFPS(Int)
    case emptySegments
    case imageDecodeFailed(URL)
    case writerCreationFailed(URL)
    case writerInputRejected
    case pixelBufferPoolUnavailable
    case pixelBufferCreateFailed
    case pixelBufferBaseAddressUnavailable
    case frameAppendFailed(Int)
    case missingVideoTrack(URL)
    case exportSessionCreationFailed
    case exportFailed(String)
    case compositionTrackCreationFailed

    public var errorDescription: String? {
        switch self {
        case .invalidFPS(let fps):
            return "Native assembly requires fps >= 1 (got \(fps))."
        case .emptySegments:
            return "Native assembly requires at least one segment."
        case .imageDecodeFailed(let url):
            return "Failed to decode image: \(url.path)"
        case .writerCreationFailed(let url):
            return "Failed to create media writer for: \(url.path)"
        case .writerInputRejected:
            return "AVAssetWriter rejected native segment settings."
        case .pixelBufferPoolUnavailable:
            return "AVAssetWriter did not provide a pixel buffer pool."
        case .pixelBufferCreateFailed:
            return "Failed to allocate pixel buffer for native render."
        case .pixelBufferBaseAddressUnavailable:
            return "Pixel buffer base address is unavailable."
        case .frameAppendFailed(let index):
            return "Failed to append frame \(index) during native render."
        case .missingVideoTrack(let url):
            return "Video track missing in asset: \(url.path)"
        case .exportSessionCreationFailed:
            return "Failed to create AVAssetExportSession for native assembly."
        case .exportFailed(let details):
            return "Native AV export failed: \(details)"
        case .compositionTrackCreationFailed:
            return "Failed to create AVMutableComposition track for native assembly."
        }
    }
}

public struct NativeMediaAssembler {
    public init() {}

    public func assemble(
        request: NativeMediaAssemblyRequest,
        onLog: @escaping (String) -> Void
    ) throws -> URL {
        guard request.fps >= 1 else {
            throw NativeMediaAssemblerError.invalidFPS(request.fps)
        }

        let orderedSegments = request.segments
            .filter { $0.endSeconds > $0.startSeconds }
            .sorted { $0.startSeconds < $1.startSeconds }

        guard !orderedSegments.isEmpty else {
            throw NativeMediaAssemblerError.emptySegments
        }

        let fm = FileManager.default
        if fm.fileExists(atPath: request.outputURL.path) {
            try fm.removeItem(at: request.outputURL)
        }
        try fm.createDirectory(at: request.outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let segmentsDir = request.workDirectory.appendingPathComponent("segments", isDirectory: true)
        if fm.fileExists(atPath: segmentsDir.path) {
            try fm.removeItem(at: segmentsDir)
        }
        try fm.createDirectory(at: segmentsDir, withIntermediateDirectories: true)

        onLog("  Native assembly: rendering \(orderedSegments.count) segment(s)")

        var renderedSegmentURLs: [URL] = []
        renderedSegmentURLs.reserveCapacity(orderedSegments.count)

        for (index, segment) in orderedSegments.enumerated() {
            let segmentDuration = max(0.1, segment.endSeconds - segment.startSeconds)
            let segmentURL = segmentsDir.appendingPathComponent(String(format: "segment-%03d.mp4", index))

            switch segment.source {
            case .image(let url, let motion):
                onLog("  Native assembly: beat \(index + 1)/\(orderedSegments.count) image motion")
                try renderImageSegment(
                    imageURL: url,
                    outputURL: segmentURL,
                    durationSeconds: segmentDuration,
                    motion: motion,
                    renderWidth: request.renderWidth,
                    renderHeight: request.renderHeight,
                    fps: request.fps
                )
            case .video(let url):
                onLog("  Native assembly: beat \(index + 1)/\(orderedSegments.count) video normalize")
                try renderVideoSegment(
                    videoURL: url,
                    outputURL: segmentURL,
                    durationSeconds: segmentDuration,
                    renderWidth: request.renderWidth,
                    renderHeight: request.renderHeight,
                    fps: request.fps
                )
            }

            renderedSegmentURLs.append(segmentURL)
        }

        onLog("  Native assembly: stitching timeline")

        let composition = AVMutableComposition()
        guard let compositionVideoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            throw NativeMediaAssemblerError.compositionTrackCreationFailed
        }

        var cursor = CMTime.zero
        var insertedRanges: [CMTimeRange] = []
        insertedRanges.reserveCapacity(renderedSegmentURLs.count)

        for (index, url) in renderedSegmentURLs.enumerated() {
            let asset = AVURLAsset(url: url)
            guard let track = asset.tracks(withMediaType: .video).first else {
                throw NativeMediaAssemblerError.missingVideoTrack(url)
            }

            let requestedDuration = CMTime(seconds: max(0.1, orderedSegments[index].endSeconds - orderedSegments[index].startSeconds), preferredTimescale: 600)
            let availableDuration = track.timeRange.duration
            let insertDuration = CMTimeMinimum(requestedDuration, availableDuration)

            guard insertDuration > .zero else {
                continue
            }

            try compositionVideoTrack.insertTimeRange(
                CMTimeRange(start: .zero, duration: insertDuration),
                of: track,
                at: cursor
            )

            var finalDuration = insertDuration
            if insertDuration != requestedDuration {
                compositionVideoTrack.scaleTimeRange(
                    CMTimeRange(start: cursor, duration: insertDuration),
                    toDuration: requestedDuration
                )
                finalDuration = requestedDuration
            }

            insertedRanges.append(CMTimeRange(start: cursor, duration: finalDuration))
            cursor = cursor + finalDuration
        }

        let totalDuration = cursor

        let narrationLayers = request.narrationLayers.filter { fm.fileExists(atPath: $0.url.path) }
        let hasBackground = request.backgroundLayer.map { fm.fileExists(atPath: $0.url.path) } ?? false
        let hasAnyAudio = !narrationLayers.isEmpty || hasBackground

        var audioMix: AVAudioMix?
        if hasAnyAudio {
            onLog("  Native assembly: mixing audio")
            audioMix = try addAudioTracks(
                to: composition,
                narrationLayers: narrationLayers,
                backgroundLayer: request.backgroundLayer,
                totalDuration: totalDuration
            )
        }

        let renderSize = CGSize(width: request.renderWidth, height: request.renderHeight)
        let videoComposition = makeVideoComposition(
            track: compositionVideoTrack,
            insertedRanges: insertedRanges,
            renderSize: renderSize,
            fps: request.fps,
            captions: request.captions,
            captionStyle: request.captionStyle
        )

        onLog("  Native assembly: exporting final.mp4")
        try exportComposition(
            composition,
            outputURL: request.outputURL,
            videoComposition: videoComposition,
            audioMix: audioMix
        )

        return request.outputURL
    }

    private func renderImageSegment(
        imageURL: URL,
        outputURL: URL,
        durationSeconds: Double,
        motion: NativeKenBurnsMotion,
        renderWidth: Int,
        renderHeight: Int,
        fps: Int
    ) throws {
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NativeMediaAssemblerError.imageDecodeFailed(imageURL)
        }

        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            throw NativeMediaAssemblerError.writerCreationFailed(outputURL)
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: renderWidth,
            AVVideoHeightKey: renderHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: max(1_000_000, renderWidth * renderHeight * fps * 3),
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
            ],
        ]

        let input = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw NativeMediaAssemblerError.writerInputRejected
        }
        writer.add(input)

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: renderWidth,
            kCVPixelBufferHeightKey as String: renderHeight,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attributes
        )

        guard writer.startWriting() else {
            let details = writer.error?.localizedDescription ?? "unknown error"
            throw NativeMediaAssemblerError.exportFailed(details)
        }
        writer.startSession(atSourceTime: .zero)

        guard let pool = adaptor.pixelBufferPool else {
            throw NativeMediaAssemblerError.pixelBufferPoolUnavailable
        }

        let frameCount = max(1, Int((durationSeconds * Double(fps)).rounded(.up)))

        for frameIndex in 0..<frameCount {
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.001)
            }

            var pixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pixelBuffer)
            guard status == kCVReturnSuccess, let pixelBuffer else {
                throw NativeMediaAssemblerError.pixelBufferCreateFailed
            }

            try drawKenBurnsFrame(
                image: image,
                into: pixelBuffer,
                frameIndex: frameIndex,
                frameCount: frameCount,
                motion: motion,
                renderWidth: renderWidth,
                renderHeight: renderHeight
            )

            let presentationTime = CMTime(value: Int64(frameIndex), timescale: CMTimeScale(fps))
            if !adaptor.append(pixelBuffer, withPresentationTime: presentationTime) {
                throw NativeMediaAssemblerError.frameAppendFailed(frameIndex)
            }
        }

        input.markAsFinished()
        try finishWriter(writer)
    }

    private func drawKenBurnsFrame(
        image: CGImage,
        into pixelBuffer: CVPixelBuffer,
        frameIndex: Int,
        frameCount: Int,
        motion: NativeKenBurnsMotion,
        renderWidth: Int,
        renderHeight: Int
    ) throws {
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw NativeMediaAssemblerError.pixelBufferBaseAddressUnavailable
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: baseAddress,
            width: renderWidth,
            height: renderHeight,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else {
            throw NativeMediaAssemblerError.pixelBufferBaseAddressUnavailable
        }

        let canvas = CGSize(width: Double(renderWidth), height: Double(renderHeight))
        let imageSize = CGSize(width: Double(image.width), height: Double(image.height))

        let progress = frameCount > 1 ? Double(frameIndex) / Double(frameCount - 1) : 0
        let zoom = motion.startZoom + (motion.endZoom - motion.startZoom) * progress
        let baseScale = max(canvas.width / imageSize.width, canvas.height / imageSize.height)
        let drawWidth = imageSize.width * baseScale * zoom
        let drawHeight = imageSize.height * baseScale * zoom

        let overflowX = max(0, drawWidth - canvas.width)
        let overflowY = max(0, drawHeight - canvas.height)

        let angleX = 2.0 * Double.pi * motion.driftXFrequency * progress
        let angleY = 2.0 * Double.pi * motion.driftYFrequency * progress
        let driftX = overflowX * motion.driftXFraction * sin(angleX)
        let driftY = overflowY * motion.driftYFraction * cos(angleY)

        let originX = -(overflowX / 2.0) + driftX
        let originY = -(overflowY / 2.0) + driftY

        context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: renderWidth, height: renderHeight))

        context.interpolationQuality = .high
        let rect = CGRect(x: originX, y: originY, width: drawWidth, height: drawHeight)
        context.draw(image, in: rect)
    }

    private func renderVideoSegment(
        videoURL: URL,
        outputURL: URL,
        durationSeconds: Double,
        renderWidth: Int,
        renderHeight: Int,
        fps: Int
    ) throws {
        let asset = AVURLAsset(url: videoURL)
        guard let sourceTrack = asset.tracks(withMediaType: .video).first else {
            throw NativeMediaAssemblerError.missingVideoTrack(videoURL)
        }

        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            throw NativeMediaAssemblerError.compositionTrackCreationFailed
        }

        let targetDuration = CMTime(seconds: max(0.1, durationSeconds), preferredTimescale: 600)
        let sourceDuration = sourceTrack.timeRange.duration
        let insertDuration = CMTimeMinimum(sourceDuration, targetDuration)

        try videoTrack.insertTimeRange(
            CMTimeRange(start: .zero, duration: insertDuration),
            of: sourceTrack,
            at: .zero
        )

        if insertDuration != targetDuration {
            videoTrack.scaleTimeRange(
                CMTimeRange(start: .zero, duration: insertDuration),
                toDuration: targetDuration
            )
        }

        let renderSize = CGSize(width: renderWidth, height: renderHeight)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
        layerInstruction.setTransform(
            renderTransform(for: sourceTrack, renderSize: renderSize),
            at: .zero
        )

        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: targetDuration)
        instruction.layerInstructions = [layerInstruction]

        let videoComposition = AVMutableVideoComposition()
        videoComposition.renderSize = renderSize
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(1, fps)))
        videoComposition.instructions = [instruction]

        try exportComposition(
            composition,
            outputURL: outputURL,
            videoComposition: videoComposition,
            audioMix: nil
        )
    }

    private func renderTransform(for track: AVAssetTrack, renderSize: CGSize) -> CGAffineTransform {
        let preferred = track.preferredTransform
        let naturalRect = CGRect(origin: .zero, size: track.naturalSize)
        let orientedRect = naturalRect.applying(preferred)
        let orientedSize = CGSize(width: abs(orientedRect.width), height: abs(orientedRect.height))

        let scale = max(renderSize.width / orientedSize.width, renderSize.height / orientedSize.height)

        var transform = preferred.concatenating(CGAffineTransform(scaleX: scale, y: scale))
        let transformedRect = naturalRect.applying(transform)

        let tx = (renderSize.width - transformedRect.width) / 2.0 - transformedRect.minX
        let ty = (renderSize.height - transformedRect.height) / 2.0 - transformedRect.minY
        transform = transform.concatenating(CGAffineTransform(translationX: tx, y: ty))

        return transform
    }

    private func addAudioTracks(
        to composition: AVMutableComposition,
        narrationLayers: [NativeMediaAudioLayer],
        backgroundLayer: NativeMediaAudioLayer?,
        totalDuration: CMTime
    ) throws -> AVAudioMix {
        var params: [AVAudioMixInputParameters] = []

        for layer in narrationLayers {
            let asset = AVURLAsset(url: layer.url)
            guard let sourceTrack = asset.tracks(withMediaType: .audio).first else {
                continue
            }

            guard let track = composition.addMutableTrack(
                withMediaType: .audio,
                preferredTrackID: kCMPersistentTrackID_Invalid
            ) else {
                throw NativeMediaAssemblerError.compositionTrackCreationFailed
            }

            let start = CMTime(seconds: max(0, layer.startSeconds), preferredTimescale: 600)
            guard start < totalDuration else { continue }

            let remaining = CMTimeSubtract(totalDuration, start)
            let insertDuration = CMTimeMinimum(sourceTrack.timeRange.duration, remaining)
            guard insertDuration > .zero else { continue }

            try track.insertTimeRange(
                CMTimeRange(start: .zero, duration: insertDuration),
                of: sourceTrack,
                at: start
            )

            let input = AVMutableAudioMixInputParameters(track: track)
            input.setVolume(layer.volume, at: .zero)
            params.append(input)
        }

        if let backgroundLayer {
            let asset = AVURLAsset(url: backgroundLayer.url)
            if let sourceTrack = asset.tracks(withMediaType: .audio).first,
               let track = composition.addMutableTrack(
                   withMediaType: .audio,
                   preferredTrackID: kCMPersistentTrackID_Invalid
               ) {
                let start = CMTime(seconds: max(0, backgroundLayer.startSeconds), preferredTimescale: 600)
                if start < totalDuration {
                    let remaining = CMTimeSubtract(totalDuration, start)
                    let insertDuration = CMTimeMinimum(sourceTrack.timeRange.duration, remaining)
                    if insertDuration > .zero {
                        try track.insertTimeRange(
                            CMTimeRange(start: .zero, duration: insertDuration),
                            of: sourceTrack,
                            at: start
                        )

                        let input = AVMutableAudioMixInputParameters(track: track)
                        input.setVolume(backgroundLayer.volume, at: .zero)
                        params.append(input)
                    }
                }
            }
        }

        let mix = AVMutableAudioMix()
        mix.inputParameters = params
        return mix
    }

    private func makeVideoComposition(
        track: AVCompositionTrack,
        insertedRanges: [CMTimeRange],
        renderSize: CGSize,
        fps: Int,
        captions: [NativeMediaCaptionCue],
        captionStyle: NativeMediaCaptionStyle
    ) -> AVMutableVideoComposition {
        let instruction = AVMutableVideoCompositionInstruction()
        let totalDuration = insertedRanges.reduce(CMTime.zero) { partial, range in
            max(partial, CMTimeRangeGetEnd(range))
        }
        instruction.timeRange = CMTimeRange(start: .zero, duration: totalDuration)
        instruction.enablePostProcessing = true

        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: track)
        instruction.layerInstructions = [layerInstruction]

        let composition = AVMutableVideoComposition()
        composition.renderSize = renderSize
        composition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(max(1, fps)))
        composition.instructions = [instruction]

        let cleanedCues = captions
            .filter { $0.endSeconds > $0.startSeconds && !$0.text.isEmpty }
            .sorted { $0.startSeconds < $1.startSeconds }

        guard !cleanedCues.isEmpty else {
            return composition
        }

        let parentLayer = CALayer()
        parentLayer.frame = CGRect(origin: .zero, size: renderSize)
        parentLayer.isGeometryFlipped = false

        let videoLayer = CALayer()
        videoLayer.frame = parentLayer.frame
        parentLayer.addSublayer(videoLayer)

        let overlayLayer = CALayer()
        overlayLayer.frame = parentLayer.frame
        overlayLayer.masksToBounds = true
        parentLayer.addSublayer(overlayLayer)

        for (index, cue) in cleanedCues.enumerated() {
            let cueLayer = captionLayer(
                text: cue.text,
                cue: cue,
                style: captionStyle,
                renderSize: renderSize,
                index: index
            )
            overlayLayer.addSublayer(cueLayer)
        }

        composition.animationTool = AVVideoCompositionCoreAnimationTool(
            postProcessingAsVideoLayer: videoLayer,
            in: parentLayer
        )

        return composition
    }

    private func captionLayer(
        text: String,
        cue: NativeMediaCaptionCue,
        style: NativeMediaCaptionStyle,
        renderSize: CGSize,
        index: Int
    ) -> CALayer {
        let container = CALayer()
        let maxWidth = max(100, renderSize.width - (style.horizontalInset * 2.0))
        let estimatedHeight = max(style.fontSize * 1.35 * CGFloat(max(1, style.maxLines)), style.fontSize * 1.8)
        container.frame = CGRect(
            x: style.horizontalInset,
            y: style.bottomInset,
            width: maxWidth,
            height: estimatedHeight
        )
        container.opacity = 0

        let textLayer = CATextLayer()
        textLayer.contentsScale = 2.0
        textLayer.string = text
        textLayer.font = CTFontCreateWithName(style.fontName as CFString, style.fontSize, nil)
        textLayer.fontSize = style.fontSize
        textLayer.foregroundColor = rgbaColor(style.textColor)
        textLayer.alignmentMode = .center
        textLayer.isWrapped = true
        textLayer.truncationMode = .end
        textLayer.frame = container.bounds
        textLayer.shadowColor = rgbaColor(style.outlineColor)
        textLayer.shadowOpacity = Float(max(0.0, min(1.0, style.outlineColor.3)))
        textLayer.shadowRadius = max(2.0, abs(style.outlineWidth))
        textLayer.shadowOffset = .zero
        container.addSublayer(textLayer)

        let duration = max(0.1, cue.endSeconds - cue.startSeconds)
        let animation = CAKeyframeAnimation(keyPath: "opacity")
        animation.values = [0.0, 1.0, 1.0, 0.0]
        animation.keyTimes = [0.0, 0.08, 0.92, 1.0].map { NSNumber(value: $0) }
        animation.beginTime = AVCoreAnimationBeginTimeAtZero + cue.startSeconds
        animation.duration = duration
        animation.fillMode = .forwards
        animation.isRemovedOnCompletion = false
        animation.timingFunctions = [
            CAMediaTimingFunction(name: .easeInEaseOut),
            CAMediaTimingFunction(name: .linear),
            CAMediaTimingFunction(name: .easeInEaseOut),
        ]
        container.add(animation, forKey: "caption-opacity-\(index)")

        return container
    }

    private func rgbaColor(_ rgba: (Double, Double, Double, Double)) -> CGColor {
        CGColor(red: rgba.0, green: rgba.1, blue: rgba.2, alpha: rgba.3)
    }

    private func exportComposition(
        _ composition: AVComposition,
        outputURL: URL,
        videoComposition: AVVideoComposition?,
        audioMix: AVAudioMix?
    ) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: outputURL.path) {
            try fm.removeItem(at: outputURL)
        }
        try fm.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw NativeMediaAssemblerError.exportSessionCreationFailed
        }

        export.outputURL = outputURL
        if outputURL.pathExtension.lowercased() == "mov" {
            export.outputFileType = .mov
        } else {
            export.outputFileType = .mp4
        }
        export.shouldOptimizeForNetworkUse = true
        export.videoComposition = videoComposition
        export.audioMix = audioMix

        let semaphore = DispatchSemaphore(value: 0)
        export.exportAsynchronously {
            semaphore.signal()
        }
        semaphore.wait()

        guard export.status == .completed else {
            let details = export.error?.localizedDescription ?? "status=\(export.status.rawValue)"
            throw NativeMediaAssemblerError.exportFailed(details)
        }
    }

    private func finishWriter(_ writer: AVAssetWriter) throws {
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting {
            semaphore.signal()
        }
        semaphore.wait()

        guard writer.status == .completed else {
            let details = writer.error?.localizedDescription ?? "writer status=\(writer.status.rawValue)"
            throw NativeMediaAssemblerError.exportFailed(details)
        }
    }
}
