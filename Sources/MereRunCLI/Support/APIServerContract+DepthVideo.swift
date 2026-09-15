import Foundation
import MereRunCore

struct APIDepthVideoArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let frameIndex: Int?
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case frameIndex = "frame_index"
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIDepthVideoTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let frameExtractionSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case frameExtractionSeconds = "frame_extraction_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIDepthVideoResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
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
    let hasConfidence: Bool
    let hasCameraIntrinsics: Bool
    let hasCameraExtrinsics: Bool
    let hasPointCloud: Bool
    let manifest: APIDepthVideoArtifactResponse
    let review: APIDepthVideoArtifactResponse
    let artifacts: [APIDepthVideoArtifactResponse]
    let timing: APIDepthVideoTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case semantics
        case checkpointFormat = "checkpoint_format"
        case checkpointSHA256 = "checkpoint_sha256"
        case width
        case height
        case fps
        case frameCount = "frame_count"
        case windowCount = "window_count"
        case temporalWindowLength = "temporal_window_length"
        case temporalOverlap = "temporal_overlap"
        case hasConfidence = "has_confidence"
        case hasCameraIntrinsics = "has_camera_intrinsics"
        case hasCameraExtrinsics = "has_camera_extrinsics"
        case hasPointCloud = "has_point_cloud"
        case manifest
        case review
        case artifacts
        case timing
    }
}

extension APIServerContract {
    static let defaultDepthVideoModelID = ModelResolver.ModelID.visionDepthVDASmall.rawValue
    static let depthVideoRoutePath = "/v1/vision/depth-video"
    static let maximumDepthVideoUploadByteCount = 512 * 1024 * 1024

    struct DepthVideoPlan: Equatable, Sendable {
        let modelID: String
        let settings: VideoDepthAnythingGenerationSettings

        var inputSize: Int { settings.inputSize }
        var maximumFrameCount: Int { settings.maximumFrameCount }

        func request(videoURL: URL, outputDirectory: URL) -> VideoDepthAnythingGenerationRequest {
            VideoDepthAnythingGenerationRequest(
                videoURL: videoURL, outputDirectory: outputDirectory, model: modelID, settings: settings
            )
        }
    }

    static func depthVideoPlan(from form: MultipartFormData) throws -> DepthVideoPlan {
        let allowedFields: Set<String> = ["model", "input_size", "max_frames"]
        try form.validateFields(
            textFields: allowedFields,
            fileFields: ["video"],
            unsupportedTextMessage: "unsupported field; client filesystem paths are not accepted",
            unsupportedFileMessage: "only a single uploaded 'video' file is accepted",
            requiresUTF8Text: false
        )

        let uploads = form.files(named: "video")
        guard uploads.count == 1, let upload = uploads.first, !upload.body.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "video",
                "exactly one non-empty uploaded video file is required"
            )
        }
        if let contentType = upload.contentType?.lowercased(),
           !contentType.hasPrefix("video/"),
           contentType != "application/octet-stream" {
            throw APIRequestValidationError.invalidField(
                "video",
                "uploaded part must have a video content type"
            )
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultDepthVideoModelID
        ).lowercased()
        let supportedModelIDs = Set(VideoDepthAnythingVariant.allCases.map(\.modelID))
        guard supportedModelIDs.contains(modelID) else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only \(supportedModelIDs.sorted().joined(separator: ", ")) are supported"
            )
        }

        let inputSize = try optionalPositiveIntField(
                form.field("input_size"),
                field: "input_size"
            ) ?? VideoDepthAnythingLimits.defaultInputSize
        let maximumFrameCount = try optionalPositiveIntField(
                form.field("max_frames"),
                field: "max_frames"
            ) ?? VideoDepthAnythingLimits.defaultMaximumFrameCount
        let settings: VideoDepthAnythingGenerationSettings
        do {
            settings = try VideoDepthAnythingGenerationSettings(
                inputSize: inputSize,
                maximumFrameCount: maximumFrameCount
            )
        } catch let error as VideoDepthAnythingLimitError {
            let field: String
            switch error {
            case .inputSizeOutOfRange:
                field = "input_size"
            default:
                field = "max_frames"
            }
            throw APIRequestValidationError.invalidField(field, error.localizedDescription)
        } catch {
            throw APIRequestValidationError.invalidField("video", error.localizedDescription)
        }
        return DepthVideoPlan(modelID: modelID, settings: settings)
    }

    static func depthVideoResponse(
        from result: VideoDepthAnythingRunResult,
        createdAt: Date = Date()
    ) throws -> APIDepthVideoResponse {
        let sequence = result.export.manifest
        let root = URL(fileURLWithPath: sequence.outputDirectory, isDirectory: true)
        let manifest = try depthVideoFileArtifact(
            kind: GeometryArtifactKind.manifest.rawValue,
            frameIndex: nil,
            url: result.export.manifestURL,
            mediaType: "application/json"
        )
        let review = APIDepthVideoArtifactResponse(
            kind: result.reviewVideo.kind,
            frameIndex: nil,
            url: root.appendingPathComponent(result.reviewVideo.relativePath).absoluteString,
            mediaType: result.reviewVideo.mediaType,
            byteCount: result.reviewVideo.byteCount,
            sha256: result.reviewVideo.sha256
        )
        let artifacts = sequence.frames.flatMap { frame in
            frame.artifacts.map { artifact in
                APIDepthVideoArtifactResponse(
                    kind: artifact.kind.rawValue,
                    frameIndex: frame.index,
                    url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                    mediaType: artifact.mediaType,
                    byteCount: artifact.byteCount,
                    sha256: artifact.sha256
                )
            }
        }.sorted { lhs, rhs in
            if lhs.frameIndex != rhs.frameIndex {
                return (lhs.frameIndex ?? -1) < (rhs.frameIndex ?? -1)
            }
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let timing = APIDepthVideoTimingResponse(
            checkpointVerificationSeconds: result.checkpointVerificationSeconds,
            frameExtractionSeconds: result.frameExtractionSeconds,
            modelLoadSeconds: result.modelLoadSeconds,
            inferenceSeconds: result.inferenceSeconds,
            exportSeconds: result.exportSeconds,
            totalSeconds: result.checkpointVerificationSeconds
                + result.frameExtractionSeconds
                + result.modelLoadSeconds
                + result.inferenceSeconds
                + result.exportSeconds
        )
        return APIDepthVideoResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.depth-video",
            status: "completed",
            model: sequence.model.modelID,
            semantics: sequence.semantics,
            checkpointFormat: result.checkpoint.format,
            checkpointSHA256: result.checkpoint.weightsSHA256,
            width: sequence.width,
            height: sequence.height,
            fps: sequence.fps,
            frameCount: sequence.frameCount,
            windowCount: result.windowCount,
            temporalWindowLength: sequence.temporalWindowLength,
            temporalOverlap: sequence.temporalOverlap,
            hasConfidence: sequence.frames.contains { $0.confidencePath != nil },
            hasCameraIntrinsics: sequence.frames.contains { $0.intrinsics != nil },
            hasCameraExtrinsics: false,
            hasPointCloud: false,
            manifest: manifest,
            review: review,
            artifacts: artifacts,
            timing: timing
        )
    }

    private static func depthVideoFileArtifact(
        kind: String,
        frameIndex: Int?,
        url: URL,
        mediaType: String
    ) throws -> APIDepthVideoArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIDepthVideoArtifactResponse(
            kind: kind,
            frameIndex: frameIndex,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }
}
