import Foundation
import MediaIO
@preconcurrency import MLX

public enum VideoDepthAnythingProgress: Equatable, Sendable {
    case verifyingCheckpoint
    case extractingFrames
    case loadingModel
    case runningWindow(current: Int, total: Int)
    case aligningWindows
    case exportingDepth
    case writingReviewVideo

    public var message: String {
        switch self {
        case .verifyingCheckpoint: "Verifying pinned checkpoint"
        case .extractingFrames: "Extracting source video frames"
        case .loadingModel: "Loading native MLX Video Depth Anything model"
        case .runningWindow(let current, let total): "Running temporal window \(current) of \(total)"
        case .aligningWindows: "Aligning temporal depth windows"
        case .exportingDepth: "Writing depth EXRs, previews, and manifest"
        case .writingReviewVideo: "Assembling depth review MP4 at source FPS"
        }
    }
}

public struct VideoDepthReviewArtifact: Codable, Equatable, Sendable {
    public let kind: String
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String

    public init(
        kind: String = "depth-review-video",
        relativePath: String,
        mediaType: String = "video/mp4",
        byteCount: Int64,
        sha256: String
    ) {
        self.kind = kind
        self.relativePath = relativePath
        self.mediaType = mediaType
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct VideoDepthAnythingRunResult: Sendable {
    public let export: DepthSequenceExportResult
    public let reviewVideo: VideoDepthReviewArtifact
    public let checkpoint: VideoDepthAnythingCheckpoint
    public let sourceFPS: Double
    public let windowCount: Int
    public let checkpointVerificationSeconds: Double
    public let frameExtractionSeconds: Double
    public let modelLoadSeconds: Double
    public let inferenceSeconds: Double
    public let exportSeconds: Double

    public init(
        export: DepthSequenceExportResult,
        reviewVideo: VideoDepthReviewArtifact,
        checkpoint: VideoDepthAnythingCheckpoint,
        sourceFPS: Double,
        windowCount: Int,
        checkpointVerificationSeconds: Double,
        frameExtractionSeconds: Double,
        modelLoadSeconds: Double,
        inferenceSeconds: Double,
        exportSeconds: Double
    ) {
        self.export = export
        self.reviewVideo = reviewVideo
        self.checkpoint = checkpoint
        self.sourceFPS = sourceFPS
        self.windowCount = windowCount
        self.checkpointVerificationSeconds = checkpointVerificationSeconds
        self.frameExtractionSeconds = frameExtractionSeconds
        self.modelLoadSeconds = modelLoadSeconds
        self.inferenceSeconds = inferenceSeconds
        self.exportSeconds = exportSeconds
    }
}

public enum VideoDepthAnythingGeneratorError: Error, Equatable, LocalizedError, Sendable {
    case inputVideoNotFound(String)
    case noDecodedFrames
    case depthElementCountMismatch(expected: Int, actual: Int)
    case missingPreviewFrame(Int)

    public var errorDescription: String? {
        switch self {
        case .inputVideoNotFound(let path): "Input video not found: \(path)"
        case .noDecodedFrames: "Video frame extraction produced no frames."
        case .depthElementCountMismatch(let expected, let actual):
            "VDA depth output expected \(expected) values but produced \(actual)."
        case .missingPreviewFrame(let index): "Depth preview is missing for frame \(index)."
        }
    }
}

/// Result of the same bounded input/checkpoint admission used by generation,
/// returned before any MLX model is loaded or inference is run.
public struct VideoDepthAnythingPreflightResult: Equatable, Sendable {
    public let input: MeshInputRecord
    public let checkpoint: VideoDepthAnythingCheckpoint
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let sourceFPS: Double
    public let frameCount: Int
    public let effectiveInputSize: Int
    public let networkWidth: Int
    public let networkHeight: Int
    public let windowCount: Int

    public init(
        input: MeshInputRecord,
        checkpoint: VideoDepthAnythingCheckpoint,
        sourceWidth: Int,
        sourceHeight: Int,
        sourceFPS: Double,
        frameCount: Int,
        effectiveInputSize: Int,
        networkWidth: Int,
        networkHeight: Int,
        windowCount: Int
    ) {
        self.input = input
        self.checkpoint = checkpoint
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.sourceFPS = sourceFPS
        self.frameCount = frameCount
        self.effectiveInputSize = effectiveInputSize
        self.networkWidth = networkWidth
        self.networkHeight = networkHeight
        self.windowCount = windowCount
    }
}

private struct VideoDepthAnythingAdmission: Sendable {
    let inputURL: URL
    let admittedInput: VFXFileInputSnapshot
    let checkpoint: VideoDepthAnythingCheckpoint
    let sequence: VideoFrameSequence
    let preprocessingPlan: VideoDepthAnythingPreprocessingPlan
    let plans: [VideoDepthAnythingWindowPlan]
    let temporaryFrames: URL
    let checkpointVerificationSeconds: Double
    let frameExtractionSeconds: Double

    var preflightResult: VideoDepthAnythingPreflightResult {
        VideoDepthAnythingPreflightResult(
            input: admittedInput.inputRecord,
            checkpoint: checkpoint,
            sourceWidth: sequence.frameWidth,
            sourceHeight: sequence.frameHeight,
            sourceFPS: sequence.fps,
            frameCount: sequence.frameURLs.count,
            effectiveInputSize: preprocessingPlan.effectiveInputSize,
            networkWidth: preprocessingPlan.networkWidth,
            networkHeight: preprocessingPlan.networkHeight,
            windowCount: plans.count
        )
    }

    static func prepare(
        videoURL: URL,
        model requestedModel: String?,
        inputSize: Int,
        maximumFrameCount requestedMaximumFrameCount: Int?,
        progress: (@Sendable (VideoDepthAnythingProgress) -> Void)?
    ) async throws -> Self {
        let maximumFrameCount = try VideoDepthAnythingLimits.validateRequest(
            inputSize: inputSize,
            maximumFrameCount: requestedMaximumFrameCount
        )
        let inputURL = videoURL.standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw VideoDepthAnythingGeneratorError.inputVideoNotFound(inputURL.path)
        }
        let admittedInput = try VFXFileInputSnapshot.capture(
            inputURL,
            maximumEncodedBytes: VideoDepthAnythingLimits.maximumEncodedVideoBytes
        )
        let temporaryFrames = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-vda-\(UUID().uuidString)", isDirectory: true)
        do {
            progress?(.verifyingCheckpoint)
            let verificationStart = Date()
            let checkpoint = try await VideoDepthAnythingResources.resolve(
                requestedModel: requestedModel
            )
            let checkpointVerificationSeconds = Date().timeIntervalSince(verificationStart)

            progress?(.extractingFrames)
            let extractionStart = Date()
            let sequence = try MediaVideoIO.extractFrames(
                from: admittedInput.snapshotURL,
                into: temporaryFrames,
                endFrame: maximumFrameCount - 1,
                decodeLimits: MediaVideoDecodeLimits(
                    maximumPixelCountPerFrame: VideoDepthAnythingLimits.maximumDecodedPixelCountPerFrame,
                    maximumAggregatePixelCount: VideoDepthAnythingLimits.maximumAggregateDecodedPixelCount
                ),
                validateDecodedSequence: { width, height, frameCount in
                    try VideoDepthAnythingLimits.validateDecodedSequence(
                        width: width,
                        height: height,
                        frameCount: frameCount
                    )
                }
            )
            guard !sequence.frameURLs.isEmpty else {
                throw VideoDepthAnythingGeneratorError.noDecodedFrames
            }
            try VideoDepthAnythingLimits.validateDecodedSequence(
                width: sequence.frameWidth,
                height: sequence.frameHeight,
                frameCount: sequence.frameURLs.count
            )
            let preprocessingPlan = try VideoDepthAnythingPreprocessingPlan(
                sourceWidth: sequence.frameWidth,
                sourceHeight: sequence.frameHeight,
                requestedInputSize: inputSize
            )
            let plans = try VideoDepthAnythingWindowing.plans(
                originalFrameCount: sequence.frameURLs.count
            )
            return VideoDepthAnythingAdmission(
                inputURL: inputURL,
                admittedInput: admittedInput,
                checkpoint: checkpoint,
                sequence: sequence,
                preprocessingPlan: preprocessingPlan,
                plans: plans,
                temporaryFrames: temporaryFrames,
                checkpointVerificationSeconds: checkpointVerificationSeconds,
                frameExtractionSeconds: Date().timeIntervalSince(extractionStart)
            )
        } catch {
            admittedInput.cleanup()
            try? FileManager.default.removeItem(at: temporaryFrames)
            throw error
        }
    }

    func cleanup() {
        admittedInput.cleanup()
        try? FileManager.default.removeItem(at: temporaryFrames)
    }
}

public actor VideoDepthAnythingGenerator {
    private var loadedModel: VideoDepthAnythingModel?
    private var loadedCheckpoint: VideoDepthAnythingCheckpoint?

    public init() {}

    public static func preflight(
        videoURL: URL,
        model requestedModel: String? = nil,
        inputSize: Int = VideoDepthAnythingLimits.defaultInputSize,
        maximumFrameCount: Int? = nil
    ) async throws -> VideoDepthAnythingPreflightResult {
        let admission = try await VideoDepthAnythingAdmission.prepare(
            videoURL: videoURL,
            model: requestedModel,
            inputSize: inputSize,
            maximumFrameCount: maximumFrameCount,
            progress: nil
        )
        defer { admission.cleanup() }
        return admission.preflightResult
    }

    public func generate(
        videoURL: URL,
        outputDirectory: URL,
        model requestedModel: String? = nil,
        inputSize: Int = VideoDepthAnythingLimits.defaultInputSize,
        maximumFrameCount: Int? = nil,
        progress: (@Sendable (VideoDepthAnythingProgress) -> Void)? = nil
    ) async throws -> VideoDepthAnythingRunResult {
        let admission = try await VideoDepthAnythingAdmission.prepare(
            videoURL: videoURL,
            model: requestedModel,
            inputSize: inputSize,
            maximumFrameCount: maximumFrameCount,
            progress: progress
        )
        defer { admission.cleanup() }
        let inputURL = admission.inputURL
        let admittedInput = admission.admittedInput
        let checkpoint = admission.checkpoint
        let sequence = admission.sequence
        let plans = admission.plans
        let checkpointVerificationSeconds = admission.checkpointVerificationSeconds
        let frameExtractionSeconds = admission.frameExtractionSeconds

        progress?(.loadingModel)
        let loadStart = Date()
        let nativeModel = try loadModelIfNeeded(checkpoint)
        let modelLoadSeconds = Date().timeIntervalSince(loadStart)

        let pin = checkpoint.variant.pin
        let provenance = GeometryModelProvenance(
            modelID: checkpoint.variant.modelID,
            upstreamRepository: pin.repository,
            upstreamRevision: pin.revision,
            license: pin.license,
            weightsSHA256: checkpoint.weightsSHA256
        )
        let outputURL = outputDirectory.standardizedFileURL
        var aligner = try VideoDepthAnythingWindowing.StreamingAligner(
            originalFrameCount: sequence.frameURLs.count,
            semantics: checkpoint.variant.semantics
        )
        let streamingExporter = try DepthSequenceStreamingArtifactExporter(
            expectedFrameCount: sequence.frameURLs.count,
            inputURL: inputURL,
            outputDirectory: outputURL,
            width: sequence.frameWidth,
            height: sequence.frameHeight,
            fps: sequence.fps,
            semantics: checkpoint.variant.semantics,
            provenance: provenance,
            inputRecord: admittedInput.inputRecord
        )
        defer { streamingExporter.cancel() }

        let inferenceStart = Date()
        var finalizedFrameIndex = 0
        for (index, plan) in plans.enumerated() {
            progress?(.runningWindow(current: index + 1, total: plans.count))
            let frames = try plan.sourceFrameIndices.map {
                try MediaImageIO.decode(sequence.frameURLs[$0])
            }
            let preprocessed = try VideoDepthAnythingPreprocessor.normalizedVideo(
                frames: frames,
                inputSize: inputSize
            )
            let rawDepth = nativeModel(preprocessed.video).depth
            let depth = try VideoDepthAnythingPreprocessor.resizeDepth(
                rawDepth,
                sourceWidth: sequence.frameWidth,
                sourceHeight: sequence.frameHeight
            )
            MLX.eval(depth)
            let window = try Self.splitDepthWindow(
                depth.asArray(Float.self),
                width: sequence.frameWidth,
                height: sequence.frameHeight
            )
            let aligned = try aligner.append(
                window: window,
                isFinal: index == plans.count - 1
            )
            for values in aligned.finalizedFrames {
                try streamingExporter.append(try DepthSequenceFrame(
                    index: finalizedFrameIndex,
                    timeSeconds: Double(finalizedFrameIndex) / sequence.fps,
                    width: sequence.frameWidth,
                    height: sequence.frameHeight,
                    depth: values,
                    confidence: nil,
                    intrinsics: nil
                ))
                finalizedFrameIndex += 1
            }
            MLX.Memory.clearCache()
        }

        progress?(.aligningWindows)
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

        progress?(.exportingDepth)
        let exportStart = Date()
        let export = try streamingExporter.finalize()

        progress?(.writingReviewVideo)
        let previewURLs = try export.manifest.frames.map { frame -> URL in
            guard let relativePath = frame.previewPath else {
                throw VideoDepthAnythingGeneratorError.missingPreviewFrame(frame.index)
            }
            return outputURL.appendingPathComponent(relativePath)
        }
        let reviewURL = outputURL.appendingPathComponent("depth-review.mp4")
        try MediaVideoIO.writeVideo(frameURLs: previewURLs, fps: sequence.fps, to: reviewURL)
        let review = try Self.reviewArtifact(url: reviewURL, root: outputURL)
        let exportSeconds = Date().timeIntervalSince(exportStart)

        // DepthSequenceManifest v1 has no run-level artifact collection. A v2
        // schema should add one; until then the review MP4 stays in the
        // structured run result instead of being mislabeled as a frame artifact.
        return VideoDepthAnythingRunResult(
            export: export,
            reviewVideo: review,
            checkpoint: checkpoint,
            sourceFPS: sequence.fps,
            windowCount: plans.count,
            checkpointVerificationSeconds: checkpointVerificationSeconds,
            frameExtractionSeconds: frameExtractionSeconds,
            modelLoadSeconds: modelLoadSeconds,
            inferenceSeconds: inferenceSeconds,
            exportSeconds: exportSeconds
        )
    }

    public func unload() {
        loadedModel = nil
        loadedCheckpoint = nil
        MLX.Memory.clearCache()
    }

    private func loadModelIfNeeded(_ checkpoint: VideoDepthAnythingCheckpoint) throws -> VideoDepthAnythingModel {
        if let loadedModel, loadedCheckpoint == checkpoint { return loadedModel }
        let model = try VideoDepthAnythingResources.loadModel(from: checkpoint)
        loadedModel = model
        loadedCheckpoint = checkpoint
        return model
    }

    static func splitDepthWindow(_ values: [Float], width: Int, height: Int) throws -> [[Float]] {
        let frameElementCount = width * height
        let expected = VideoDepthAnythingWindowing.windowLength * frameElementCount
        guard values.count == expected else {
            throw VideoDepthAnythingGeneratorError.depthElementCountMismatch(
                expected: expected,
                actual: values.count
            )
        }
        return (0..<VideoDepthAnythingWindowing.windowLength).map { frame in
            let start = frame * frameElementCount
            return Array(values[start..<(start + frameElementCount)])
        }
    }

    static func reviewArtifact(url: URL, root: URL) throws -> VideoDepthReviewArtifact {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        let relativePath = url.path.hasPrefix(rootPath)
            ? String(url.path.dropFirst(rootPath.count))
            : url.lastPathComponent
        return VideoDepthReviewArtifact(
            relativePath: relativePath,
            byteCount: byteCount,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }
}
