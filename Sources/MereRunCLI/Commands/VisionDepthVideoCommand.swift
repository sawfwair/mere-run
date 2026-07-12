import ArgumentParser
import Foundation
import MereRunCore

struct VisionDepthVideo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "depth-video",
        abstract: "Generate temporally consistent relative or metric video depth with native VDA-S."
    )

    @Argument(help: "Input video path.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output directory.")
    var output: String?

    @Option(
        name: [.long],
        help: "Relative/metric model id, pinned .pth path, or converted model.safetensors directory."
    )
    var model: String?

    @Option(name: [.long], help: "Longest network input edge before aspect-ratio adjustment. (default: 518)")
    var inputSize: Int = 518

    @Option(name: [.long], help: "Decode at most this many source frames.")
    var maxFrames: Int?

    @Flag(name: [.long], help: "Verify inputs and checkpoint, then print the plan without inference.")
    var dryRun = false

    @Flag(name: [.long], help: "Print the structured result on stdout.")
    var json = false

    mutating func run() async throws {
        guard inputSize > 0 else { throw ValidationError("--input-size must be positive") }
        if let maxFrames, maxFrames <= 0 { throw ValidationError("--max-frames must be positive") }

        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input video not found: \(inputURL.path)")
        }
        let outputURL = Self.resolveOutputURL(output, inputURL: inputURL)

        if dryRun {
            let checkpoint = try await VideoDepthAnythingResources.resolve(requestedModel: model)
            print(try Self.jsonString(Self.makePlan(
                inputURL: inputURL,
                outputURL: outputURL,
                inputSize: inputSize,
                maximumFrameCount: maxFrames,
                checkpoint: checkpoint
            )))
            return
        }

        let generator = VideoDepthAnythingGenerator()
        do {
            let result = try await generator.generate(
                videoURL: inputURL,
                outputDirectory: outputURL,
                model: model,
                inputSize: inputSize,
                maximumFrameCount: maxFrames,
                progress: { event in CLIStderr.write("[depth-video] \(event.message)\n") }
            )
            await generator.unload()
            if json {
                print(try Self.jsonString(VisionDepthVideoRunPayload(result: result)))
            } else {
                print(result.export.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveOutputURL(_ raw: String?, inputURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            return URL(fileURLWithPath: raw).standardizedFileURL
        }
        return inputURL.deletingLastPathComponent().appendingPathComponent(
            "\(inputURL.deletingPathExtension().lastPathComponent)-depth",
            isDirectory: true
        )
    }

    static func makePlan(
        inputURL: URL,
        outputURL: URL,
        inputSize: Int,
        maximumFrameCount: Int?,
        checkpoint: VideoDepthAnythingCheckpoint
    ) -> VisionDepthVideoPlanPayload {
        VisionDepthVideoPlanPayload(
            status: "planned",
            inputPath: inputURL.path,
            outputDirectory: outputURL.path,
            modelID: checkpoint.variant.modelID,
            semantics: checkpoint.variant.semantics,
            checkpointPath: checkpoint.weightsURL.path,
            checkpointFormat: checkpoint.format,
            checkpointSHA256: checkpoint.weightsSHA256,
            checkpointVerified: true,
            inputSize: inputSize,
            maximumFrameCount: maximumFrameCount,
            temporalWindowLength: VideoDepthAnythingWindowing.windowLength,
            temporalOverlap: VideoDepthAnythingWindowing.overlap,
            frameStep: VideoDepthAnythingWindowing.frameStep,
            streamsFinalizedFrames: true,
            retainedAlignmentFrameLimit: VideoDepthAnythingWindowing.interpolationLength,
            encoderMicroBatchSize: VideoDepthAnythingMemoryConfiguration.appleSilicon.encoderMicroBatchSize,
            dptTailMicroBatchSize: VideoDepthAnythingMemoryConfiguration.appleSilicon.dptTailMicroBatchSize,
            hasConfidence: false,
            hasCameraIntrinsics: false,
            hasPointCloud: false,
            outputKinds: [
                "per-frame-depth-exr",
                "per-frame-depth-preview-png",
                "depth-review-mp4",
                "depth-sequence-manifest-json",
            ]
        )
    }

    private static func jsonString<Value: Encodable>(_ value: Value) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct VisionDepthVideoPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPath: String
    let outputDirectory: String
    let modelID: String
    let semantics: DepthSemantics
    let checkpointPath: String
    let checkpointFormat: VideoDepthAnythingCheckpointFormat
    let checkpointSHA256: String
    let checkpointVerified: Bool
    let inputSize: Int
    let maximumFrameCount: Int?
    let temporalWindowLength: Int
    let temporalOverlap: Int
    let frameStep: Int
    let streamsFinalizedFrames: Bool
    let retainedAlignmentFrameLimit: Int
    let encoderMicroBatchSize: Int?
    let dptTailMicroBatchSize: Int?
    let hasConfidence: Bool
    let hasCameraIntrinsics: Bool
    let hasPointCloud: Bool
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String,
        inputPath: String,
        outputDirectory: String,
        modelID: String,
        semantics: DepthSemantics,
        checkpointPath: String,
        checkpointFormat: VideoDepthAnythingCheckpointFormat,
        checkpointSHA256: String,
        checkpointVerified: Bool,
        inputSize: Int,
        maximumFrameCount: Int?,
        temporalWindowLength: Int,
        temporalOverlap: Int,
        frameStep: Int,
        streamsFinalizedFrames: Bool,
        retainedAlignmentFrameLimit: Int,
        encoderMicroBatchSize: Int?,
        dptTailMicroBatchSize: Int?,
        hasConfidence: Bool,
        hasCameraIntrinsics: Bool,
        hasPointCloud: Bool,
        outputKinds: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.modelID = modelID
        self.semantics = semantics
        self.checkpointPath = checkpointPath
        self.checkpointFormat = checkpointFormat
        self.checkpointSHA256 = checkpointSHA256
        self.checkpointVerified = checkpointVerified
        self.inputSize = inputSize
        self.maximumFrameCount = maximumFrameCount
        self.temporalWindowLength = temporalWindowLength
        self.temporalOverlap = temporalOverlap
        self.frameStep = frameStep
        self.streamsFinalizedFrames = streamsFinalizedFrames
        self.retainedAlignmentFrameLimit = retainedAlignmentFrameLimit
        self.encoderMicroBatchSize = encoderMicroBatchSize
        self.dptTailMicroBatchSize = dptTailMicroBatchSize
        self.hasConfidence = hasConfidence
        self.hasCameraIntrinsics = hasCameraIntrinsics
        self.hasPointCloud = hasPointCloud
        self.outputKinds = outputKinds
    }
}

struct VisionDepthVideoRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let reviewVideo: VideoDepthReviewArtifact
    let modelID: String
    let semantics: DepthSemantics
    let checkpointFormat: VideoDepthAnythingCheckpointFormat
    let checkpointSHA256: String
    let width: Int
    let height: Int
    let fps: Double
    let frameCount: Int
    let windowCount: Int
    let temporalWindowLength: Int
    let temporalOverlap: Int
    let streamsFinalizedFrames: Bool
    let retainedAlignmentFrameLimit: Int
    let encoderMicroBatchSize: Int?
    let dptTailMicroBatchSize: Int?
    let hasConfidence: Bool
    let hasCameraIntrinsics: Bool
    let hasPointCloud: Bool
    let checkpointVerificationSeconds: Double
    let frameExtractionSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let exportSeconds: Double

    init(result: VideoDepthAnythingRunResult, schemaVersion: Int = 1, status: String = "completed") {
        let manifest = result.export.manifest
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = result.export.manifestURL.path
        self.reviewVideo = result.reviewVideo
        self.modelID = manifest.model.modelID
        self.semantics = manifest.semantics
        self.checkpointFormat = result.checkpoint.format
        self.checkpointSHA256 = result.checkpoint.weightsSHA256
        self.width = manifest.width
        self.height = manifest.height
        self.fps = manifest.fps
        self.frameCount = manifest.frameCount
        self.windowCount = result.windowCount
        self.temporalWindowLength = manifest.temporalWindowLength
        self.temporalOverlap = manifest.temporalOverlap
        self.streamsFinalizedFrames = true
        self.retainedAlignmentFrameLimit = VideoDepthAnythingWindowing.interpolationLength
        self.encoderMicroBatchSize = VideoDepthAnythingMemoryConfiguration.appleSilicon.encoderMicroBatchSize
        self.dptTailMicroBatchSize = VideoDepthAnythingMemoryConfiguration.appleSilicon.dptTailMicroBatchSize
        self.hasConfidence = manifest.frames.contains { $0.confidencePath != nil }
        self.hasCameraIntrinsics = manifest.frames.contains { $0.intrinsics != nil }
        self.hasPointCloud = false
        self.checkpointVerificationSeconds = result.checkpointVerificationSeconds
        self.frameExtractionSeconds = result.frameExtractionSeconds
        self.modelLoadSeconds = result.modelLoadSeconds
        self.inferenceSeconds = result.inferenceSeconds
        self.exportSeconds = result.exportSeconds
    }
}
