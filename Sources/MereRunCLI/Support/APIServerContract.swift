import Foundation
import AudioCore
import AudioSTT
import AudioTTS
import MediaIO
import MereRunCore
import MereRunContract

struct APIEngineCapabilities: Equatable, Sendable {
    var supportsRawProxy: Bool = false
    var supportsTools: Bool = false
    var usesNativeToolHistory: Bool = false
    var supportsToolChoice: Bool = false
    var supportsDeveloperRole: Bool = true
    var supportsStructuredOutputs: Bool = false
    var supportsReasoningEffort: Bool = false
    var supportsMaxCompletionTokens: Bool = true
    var supportsUsageInStreaming: Bool = true
    var supportsVisionContentParts: Bool = false
    var supportsAudioContentParts: Bool = false
    var supportsVideoContentParts: Bool = false
    var supportsStrictMode: Bool = false
    var supportsStopSequences: Bool = false
    var supportsSeed: Bool = false
    var supportsPenalties: Bool = false
    var supportsTopK: Bool = false
    var supportsRepetitionPenalty: Bool = false
    var supportsLogprobs: Bool = false
    var supportsProviderThinkingControls: Bool = false

    static func catalog(_ profile: ManagedModelAPIProfile) -> APIEngineCapabilities {
        APIEngineCapabilities(
            supportsRawProxy: profile.supportsRawProxy,
            supportsTools: profile.toolCall,
            usesNativeToolHistory: [.textChatQ36, .textChatLaguna, .textChatGemma4, .textChatMuseGlimmer]
                .contains(profile.servingEngine),
            supportsToolChoice: profile.supportsToolChoice,
            supportsDeveloperRole: profile.compatibility.supportsDeveloperRole,
            supportsStructuredOutputs: profile.structuredOutput,
            supportsReasoningEffort: profile.compatibility.supportsReasoningEffort,
            supportsMaxCompletionTokens: profile.compatibility.maxTokensField == .maxCompletionTokens,
            supportsUsageInStreaming: profile.compatibility.supportsUsageInStreaming,
            supportsVisionContentParts: profile.inputModalities.contains(.image),
            supportsAudioContentParts: profile.inputModalities.contains(.audio),
            supportsVideoContentParts: profile.inputModalities.contains(.video),
            supportsStrictMode: profile.compatibility.supportsStrictMode,
            supportsStopSequences: profile.supportsStopSequences,
            supportsSeed: profile.supportsSeed,
            supportsPenalties: profile.supportsPenalties,
            supportsTopK: [.textChatQ35, .textChatQ36, .textChatLaguna, .textChatLFM2,
                          .textChatMuseGlimmer, .textChatNemotronH, .textChatNemotronOmni,
                          .textChatDiffusionGemma].contains(profile.servingEngine),
            supportsRepetitionPenalty: [.textChatQ35, .textChatQ36].contains(profile.servingEngine),
            supportsLogprobs: profile.supportsLogprobs,
            supportsProviderThinkingControls: profile.supportsProviderThinkingControls
        )
    }

    static let localText = APIEngineCapabilities()

    static let localTextWithStructuredJSON = APIEngineCapabilities(
        supportsStructuredOutputs: true
    )

    static let localTextWithTools = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true
    )

    static let localTextWithToolsAndVision = APIEngineCapabilities(
        supportsTools: true,
        supportsToolChoice: true,
        supportsVisionContentParts: true
    )
}

struct APIHealthStatus: Codable, Equatable, Sendable {
    let status: String
}

struct APIGeometryArtifactResponse: Codable, Equatable, Sendable {
    let kind: GeometryArtifactKind
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIGeometryTimingResponse: Codable, Equatable, Sendable {
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessSeconds: Double

    enum CodingKeys: String, CodingKey {
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case postprocessSeconds = "postprocess_seconds"
    }
}

struct APIGeometryResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let width: Int
    let height: Int
    let units: GeometryValueUnits
    let coordinateSystem: GeometryCoordinateSystem
    let camera: GeometryCameraManifest
    let depthStatistics: GeometryDepthStatistics
    let focal: Double
    let shift: Double
    let metricScale: Double
    let tokenCount: Int
    let manifestURL: String
    let artifacts: [APIGeometryArtifactResponse]
    let timing: APIGeometryTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case width
        case height
        case units
        case coordinateSystem = "coordinate_system"
        case camera
        case depthStatistics = "depth_statistics"
        case focal
        case shift
        case metricScale = "metric_scale"
        case tokenCount = "token_count"
        case manifestURL = "manifest_url"
        case artifacts
        case timing
    }
}

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

struct APIMultiViewGeometryCheckpointResponse: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let weightsByteCount: Int64
    let weightsSHA256: String
    let configurationByteCount: Int64
    let configurationSHA256: String

    enum CodingKeys: String, CodingKey {
        case repository
        case revision
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case license
        case weightsByteCount = "weights_byte_count"
        case weightsSHA256 = "weights_sha256"
        case configurationByteCount = "configuration_byte_count"
        case configurationSHA256 = "configuration_sha256"
    }
}

struct APIMultiViewGeometryArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let viewIndex: Int?
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case viewIndex = "view_index"
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIMultiViewGeometryCameraResponse: Codable, Equatable, Sendable {
    let viewIndex: Int
    let width: Int
    let height: Int
    let intrinsics: GeometryCameraIntrinsics
    let extrinsics: GeometryCameraExtrinsics
    let selectedPointCount: Int

    enum CodingKeys: String, CodingKey {
        case viewIndex = "view_index"
        case width
        case height
        case intrinsics
        case extrinsics
        case selectedPointCount = "selected_point_count"
    }
}

struct APIMultiViewGeometryTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessingSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case decodingSeconds = "decoding_seconds"
        case preprocessingSeconds = "preprocessing_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
        case postprocessingSeconds = "postprocessing_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIMultiViewGeometryResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let checkpoint: APIMultiViewGeometryCheckpointResponse
    let units: GeometryValueUnits
    let coordinateSystem: GeometryCoordinateSystem
    let poseConditioned: Bool
    let cameraSemantics: DepthAnything3CameraSemantics
    let cameraScaleAlignment: String
    let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    let depthScaleDivisor: Float
    let processResolution: Int
    let confidencePercentile: Double
    let confidenceThreshold: Float
    let viewCount: Int
    let cameraCount: Int
    let pointCount: Int
    let pointCloudRepresentation: String
    let containsMesh: Bool
    let containsGaussianParameters: Bool
    let threeDGaussianHandoff: Geometry3DGSHandoffManifest
    let cameras: [APIMultiViewGeometryCameraResponse]
    let manifest: APIMultiViewGeometryArtifactResponse
    let artifacts: [APIMultiViewGeometryArtifactResponse]
    let timing: APIMultiViewGeometryTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case checkpoint
        case units
        case coordinateSystem = "coordinate_system"
        case poseConditioned = "pose_conditioned"
        case cameraSemantics = "camera_semantics"
        case cameraScaleAlignment = "camera_scale_alignment"
        case referenceViewStrategy = "reference_view_strategy"
        case depthScaleDivisor = "depth_scale_divisor"
        case processResolution = "process_resolution"
        case confidencePercentile = "confidence_percentile"
        case confidenceThreshold = "confidence_threshold"
        case viewCount = "view_count"
        case cameraCount = "camera_count"
        case pointCount = "point_count"
        case pointCloudRepresentation = "point_cloud_representation"
        case containsMesh = "contains_mesh"
        case containsGaussianParameters = "contains_gaussian_parameters"
        case threeDGaussianHandoff = "three_d_gaussian_handoff"
        case cameras
        case manifest
        case artifacts
        case timing
    }
}

struct APIImageTo3DCheckpointResponse: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let format: TripoSRCheckpointFormat
    let weightsByteCount: Int64
    let weightsSHA256: String
    let sourceSHA256: String
    let configurationSHA256: String

    enum CodingKeys: String, CodingKey {
        case repository
        case revision
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case license
        case format
        case weightsByteCount = "weights_byte_count"
        case weightsSHA256 = "weights_sha256"
        case sourceSHA256 = "source_sha256"
        case configurationSHA256 = "configuration_sha256"
    }
}

struct APIImageTo3DArtifactResponse: Codable, Equatable, Sendable {
    let kind: String
    let url: String
    let mediaType: String
    let byteCount: Int64
    let sha256: String

    enum CodingKeys: String, CodingKey {
        case kind
        case url
        case mediaType = "media_type"
        case byteCount = "byte_count"
        case sha256
    }
}

struct APIImageTo3DTimingResponse: Codable, Equatable, Sendable {
    let checkpointVerificationSeconds: Double
    let decodingSeconds: Double
    let preprocessingSeconds: Double
    let modelLoadSeconds: Double
    let sceneEncodingSeconds: Double
    let meshExtractionSeconds: Double
    let exportSeconds: Double
    let totalSeconds: Double

    enum CodingKeys: String, CodingKey {
        case checkpointVerificationSeconds = "checkpoint_verification_seconds"
        case decodingSeconds = "decoding_seconds"
        case preprocessingSeconds = "preprocessing_seconds"
        case modelLoadSeconds = "model_load_seconds"
        case sceneEncodingSeconds = "scene_encoding_seconds"
        case meshExtractionSeconds = "mesh_extraction_seconds"
        case exportSeconds = "export_seconds"
        case totalSeconds = "total_seconds"
    }
}

struct APIImageTo3DResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let checkpoint: APIImageTo3DCheckpointResponse
    let sourceWidth: Int
    let sourceHeight: Int
    let preparedWidth: Int
    let preparedHeight: Int
    let foregroundPolicy: String
    let foregroundRatio: Float?
    let croppedTransparentForeground: Bool
    let extractionResolution: Int
    let densityThreshold: Float
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let vertexCount: Int
    let triangleCount: Int
    let bounds: MeshBounds
    let manifest: APIImageTo3DArtifactResponse
    let meshManifest: APIImageTo3DArtifactResponse
    let artifacts: [APIImageTo3DArtifactResponse]
    let timing: APIImageTo3DTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case checkpoint
        case sourceWidth = "source_width"
        case sourceHeight = "source_height"
        case preparedWidth = "prepared_width"
        case preparedHeight = "prepared_height"
        case foregroundPolicy = "foreground_policy"
        case foregroundRatio = "foreground_ratio"
        case croppedTransparentForeground = "cropped_transparent_foreground"
        case extractionResolution = "extraction_resolution"
        case densityThreshold = "density_threshold"
        case includesVertexColors = "includes_vertex_colors"
        case meshExtractionAlgorithm = "mesh_extraction_algorithm"
        case coordinateSystem = "coordinate_system"
        case units
        case inferredUnseenGeometry = "inferred_unseen_geometry"
        case vertexCount = "vertex_count"
        case triangleCount = "triangle_count"
        case bounds
        case manifest
        case meshManifest = "mesh_manifest"
        case artifacts
        case timing
    }
}

enum APIServerContract {
    static let defaultMaxTokens = 2048
    static let maxEmbeddingInputCount = 256
    static let maxEmbeddingInputUTF8Bytes = 2 * 1_024 * 1_024
    static let maxImageInferenceSteps = 100
    static let maxSpeechPromptUTF8Bytes = 32 * 1_024
    static let maxTranscriptionTokens = 4_096
    static let defaultImageModelID = ModelResolver.ModelID.zetaNano.rawValue
    static let defaultVideoModelID = ModelResolver.ModelID.ltxVideo25DistilledBF16.rawValue
    static let defaultSpeechModelID = Qwen3TTSResources.defaultModelId
    static let defaultTranscriptionModelID = ParakeetResources.defaultModelId
    static let defaultGeometryModelID = MoGe2GenerationRequest.defaultModelID
    static let defaultMultiViewGeometryModelID = ModelResolver.ModelID.visionGeometryDA3Small.rawValue
    static let defaultImageTo3DModelID = ModelResolver.ModelID.image3DTripoSR.rawValue
    static let defaultDepthVideoModelID = ModelResolver.ModelID.visionDepthVDASmall.rawValue
    static let videoGenerationRoutePath = "/v1/videos/generations"
    static let geometryRoutePath = "/v1/vision/geometry"
    static let multiViewGeometryRoutePath = "/v1/vision/geometry/multiview"
    static let maximumMultiViewGeometryUploadByteCount = 512 * 1024 * 1024
    static let imageTo3DRoutePath = "/v1/vision/image-to-3d"
    static let maximumImageTo3DUploadByteCount = 100 * 1024 * 1024
    static let depthVideoRoutePath = "/v1/vision/depth-video"
    static let maximumDepthVideoUploadByteCount = 512 * 1024 * 1024

    static func decodeImageGenerationRequest(from data: Data) throws -> OpenAIImageGenerationRequest {
        try decodeJSONRequest(OpenAIImageGenerationRequest.self, from: data)
    }

    static func decodeVideoGenerationRequest(from data: Data) throws -> OpenAIVideoGenerationRequest {
        try decodeJSONRequest(OpenAIVideoGenerationRequest.self, from: data)
    }

    static func decodeSpeechRequest(from data: Data) throws -> OpenAIAudioSpeechRequest {
        try decodeJSONRequest(OpenAIAudioSpeechRequest.self, from: data)
    }

    struct ImageGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let responseFormat: String
        let seed: UInt64?
        let negativePrompt: String?
        let steps: Int?
        let guidanceScale: Double?
        let inputImage: URL?
        let additionalInputImages: [URL]
        let maskImage: URL?
        let strength: Double?

        init(
            modelID: String,
            prompt: String,
            width: Int,
            height: Int,
            responseFormat: String,
            seed: UInt64?,
            negativePrompt: String?,
            steps: Int?,
            guidanceScale: Double?,
            inputImage: URL?,
            additionalInputImages: [URL] = [],
            maskImage: URL? = nil,
            strength: Double?
        ) {
            self.modelID = modelID
            self.prompt = prompt
            self.width = width
            self.height = height
            self.responseFormat = responseFormat
            self.seed = seed
            self.negativePrompt = negativePrompt
            self.steps = steps
            self.guidanceScale = guidanceScale
            self.inputImage = inputImage
            self.additionalInputImages = additionalInputImages
            self.maskImage = maskImage
            self.strength = strength
        }
    }

    struct VideoGenerationPlan: Equatable, Sendable {
        let modelID: String
        let prompt: String
        let width: Int
        let height: Int
        let seconds: Double?
        let numFrames: Int?
        let fps: Int
        let seed: Int?
        let quality: LTXVideoQuality?
        let outputMode: LTXVideoOutputMode?
        let options: [String]
    }

    struct SpeechPlan: Equatable, Sendable {
        let modelID: String
        let input: String
        let voiceDescription: String
        let responseFormat: String
        let speed: Float
        let temperature: Float

        func synthesisPlan(outputURL: URL) throws -> SpeechSynthesisPlan {
            try SpeechSynthesisPlan(request: TTSRequest(
                text: input, voiceDescription: voiceDescription, speed: speed,
                temperature: temperature, outputURL: outputURL
            ))
        }

        func modelSelection() throws -> SpeechSynthesisModelSelection {
            do {
                return try SpeechSynthesisModelSelection.resolve(modelID)
            } catch Qwen3TTSError.unsupportedModelId {
                throw APIRequestValidationError.invalidField(
                    "model", "use a mere.run TTS model id or a local Qwen3-TTS model path"
                )
            }
        }
    }

    struct TranscriptionPlan: Equatable, Sendable {
        let modelID: String
        let language: String?
        let responseFormat: String
        let task: ASRTask
        let maxTokens: Int
    }

    struct GeometryPlan: Equatable, Sendable {
        let modelID: String
        let settings: MoGe2GenerationSettings

        var resolutionLevel: Int { settings.configuration.resolutionLevel }
        var tokenCount: Int? { settings.configuration.tokenCount }
        var maximumPointCount: Int? { settings.configuration.maximumPointCount }

        func request(imageURL: URL, outputDirectory: URL) -> MoGe2GenerationRequest {
            // The API accepts only the managed model; retain its default lookup.
            MoGe2GenerationRequest(imageURL: imageURL, outputDirectory: outputDirectory, settings: settings)
        }
    }

    struct MultiViewGeometryPlan: Equatable, Sendable {
        let modelID: String
        let processResolution: Int
        let referenceViewStrategy: DepthAnything3ReferenceViewStrategy
        let confidencePercentile: Double
        let maximumPointCount: Int
        let knownCameras: [DepthAnything3KnownCamera]?

        var poseConditioned: Bool { knownCameras != nil }
    }

    struct ImageTo3DPlan: Equatable, Sendable {
        let modelID: String
        let settings: TripoSRGenerationSettings

        var extractionResolution: Int { settings.extractionResolution }
        var densityThreshold: Float { settings.densityThreshold }
        var foregroundRatio: Float { settings.foregroundRatio }
        var alreadyFramed: Bool { settings.alreadyFramed }
        var includesVertexColors: Bool { settings.includesVertexColors }
        var foregroundPolicy: TripoSRForegroundPolicy { settings.foregroundPolicy }

        func request(imageURL: URL, outputDirectory: URL) -> TripoSRGenerationRequest {
            TripoSRGenerationRequest(
                imageURL: imageURL, outputDirectory: outputDirectory, model: modelID, settings: settings
            )
        }
    }

    struct DepthVideoPlan: Equatable, Sendable {
        let modelID: String
        let inputSize: Int
        let maximumFrameCount: Int
    }

    static func healthStatus() -> APIHealthStatus {
        APIHealthStatus(status: "ok")
    }

    static func modelsResponse(modelId: String, createdAt: Date = Date()) -> OpenAIModelsResponse {
        modelsResponse(modelIds: [modelId], createdAt: createdAt)
    }

    static func modelsResponse(modelIds: [String], createdAt: Date = Date()) -> OpenAIModelsResponse {
        OpenAIModelsResponse(
            object: "list",
            data: modelIds.map {
                OpenAIModel(
                    id: $0,
                    object: "model",
                    created: Int(createdAt.timeIntervalSince1970),
                    owned_by: "mere.run"
                )
            }
        )
    }

    static func chatModel(
        id: String,
        name: String,
        profile: ManagedModelAPIProfile,
        contextWindow: Int,
        maximumOutputTokens: Int,
        createdAt: Date = Date()
    ) -> OpenAIModel {
        let compatibility = profile.compatibility
        let thinkingLevels = profile.thinkingLevels.isEmpty
            ? nil
            : profile.thinkingLevels.map(\.rawValue)
        let thinkingLevelMap = profile.thinkingLevelMap.isEmpty
            ? nil
            : Dictionary(uniqueKeysWithValues: profile.thinkingLevelMap.map {
                ($0.key.rawValue, $0.value.rawValue)
            })

        return OpenAIModel(
            id: id,
            object: "model",
            created: Int(createdAt.timeIntervalSince1970),
            owned_by: "mere.run",
            name: name,
            task: profile.task.rawValue,
            reasoning: profile.reasoning,
            thinking_levels: thinkingLevels,
            tool_call: profile.toolCall,
            structured_output: profile.structuredOutput,
            modalities: OpenAIModelModalities(
                input: profile.inputModalities.map(\.rawValue),
                output: profile.outputModalities.map(\.rawValue)
            ),
            limit: OpenAIModelLimit(context: contextWindow, output: maximumOutputTokens),
            openai_compat: OpenAIModelCompatibility(
                supports_store: compatibility.supportsStore,
                supports_developer_role: compatibility.supportsDeveloperRole,
                supports_reasoning_effort: compatibility.supportsReasoningEffort,
                supports_usage_in_streaming: compatibility.supportsUsageInStreaming,
                supports_finish_reason: compatibility.supportsFinishReason,
                max_tokens_field: compatibility.maxTokensField.rawValue,
                supports_strict_mode: compatibility.supportsStrictMode,
                thinking_format: compatibility.thinkingFormat?.rawValue,
                thinking_level_map: thinkingLevelMap,
                requires_reasoning_content_on_assistant_messages: compatibility
                    .requiresReasoningContentOnAssistantMessages
            )
        )
    }

    static func companionModel(
        id: String,
        profile: ManagedModelAPIProfile,
        createdAt: Date = Date()
    ) -> OpenAIModel {
        return OpenAIModel(
            id: id,
            object: "model",
            created: Int(createdAt.timeIntervalSince1970),
            owned_by: "mere.run",
            name: id,
            task: profile.task.rawValue,
            reasoning: profile.reasoning,
            tool_call: profile.toolCall,
            structured_output: profile.structuredOutput,
            modalities: OpenAIModelModalities(
                input: profile.inputModalities.map(\.rawValue),
                output: profile.outputModalities.map(\.rawValue)
            )
        )
    }

    static func embeddingTexts(from request: OpenAIEmbeddingRequest) throws -> [String] {
        guard !request.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw APIRequestValidationError.invalidField("model", "must not be empty")
        }
        if let encodingFormat = request.encoding_format?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !encodingFormat.isEmpty,
           encodingFormat != "float" {
            throw APIRequestValidationError.invalidField(
                "encoding_format",
                "only float embeddings are supported"
            )
        }
        if request.dimensions != nil {
            throw APIRequestValidationError.invalidField(
                "dimensions",
                "dimension overrides are not supported by this embedding model"
            )
        }

        let texts = request.input.texts
        guard !texts.isEmpty else {
            throw APIRequestValidationError.invalidField("input", "must contain at least one text")
        }
        guard texts.count <= maxEmbeddingInputCount else {
            throw APIRequestValidationError.invalidField(
                "input",
                "must contain at most \(maxEmbeddingInputCount) texts"
            )
        }
        var totalUTF8Bytes = 0
        for text in texts {
            let textBytes = text.utf8.count
            guard textBytes <= maxEmbeddingInputUTF8Bytes - totalUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "UTF-8 content must total at most \(maxEmbeddingInputUTF8Bytes) bytes"
                )
            }
            totalUTF8Bytes += textBytes
        }
        return texts
    }

    static func embeddingResponse(
        modelId: String,
        embeddings: [[Float]],
        tokenCounts: [Int]
    ) -> OpenAIEmbeddingResponse {
        let promptTokens = tokenCounts.reduce(0, +)
        return OpenAIEmbeddingResponse(
            model: modelId,
            data: embeddings.enumerated().map { index, vector in
                OpenAIEmbeddingDatum(index: index, embedding: vector)
            },
            usage: OpenAIEmbeddingUsage(
                prompt_tokens: promptTokens,
                total_tokens: promptTokens
            )
        )
    }

    static func companionModelIDs(
        fileManager: FileManager = .default,
        installedModelIDs: Set<String>? = nil,
        includeLoopbackArtifactModels: Bool = true
    ) -> [String] {
        let categories: Set<ManagedModelCategory> = [
            .image, .image3D, .speechTTS, .speechASR, .textEmbed, .visionGeometry, .visionDepth,
        ]
        let ids = ManagedModelCatalog.allSpecs
            .filter { categories.contains($0.category) }
            .filter {
                includeLoopbackArtifactModels
                    || !APIVFXArtifactRoutePolicy.modelIDs.contains($0.id)
            }
            .filter { isCompanionModelInstalled($0, fileManager: fileManager, installedModelIDs: installedModelIDs) }
            .map(\.id)
        var uniqueIDs = Set(ids)
        if isQwenImageEditInstalled(fileManager: fileManager, installedModelIDs: installedModelIDs) {
            uniqueIDs.insert(QwenImageEditRepository.modelId)
        }
        return Array(uniqueIDs).sorted()
    }

    static func geometryPlan(from form: MultipartFormData) throws -> GeometryPlan {
        let modelID = normalizedModelID(form.field("model"), defaultID: defaultGeometryModelID)
        guard modelID == defaultGeometryModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only \(defaultGeometryModelID) is supported"
            )
        }
        let resolutionLevel = try optionalGeometryIntField(
            form.field("resolution_level"), field: "resolution_level", message: "must be an integer between 0 and 9"
        ) ?? MoGe2GenerationSettings.defaultResolutionLevel
        let tokenCount = try optionalGeometryIntField(form.field("token_count"), field: "token_count")
        let maximumPointCount = try optionalGeometryIntField(form.field("max_points"), field: "max_points")
        do {
            return GeometryPlan(
                modelID: modelID,
                settings: try MoGe2GenerationSettings(
                    resolutionLevel: resolutionLevel, tokenCount: tokenCount, maximumPointCount: maximumPointCount
                )
            )
        } catch let error as MoGe2GenerationError {
            switch error {
            case .invalidResolutionLevel:
                throw APIRequestValidationError.invalidField("resolution_level", "must be an integer between 0 and 9")
            case .invalidTokenCount(let value):
                throw APIRequestValidationError.invalidField(
                    "token_count", value <= 0 ? "must be greater than zero" : "must be an integer between 1 and 3600"
                )
            case .invalidMaximumPointCount:
                throw APIRequestValidationError.invalidField("max_points", "must be greater than zero")
            case .inputNotFound:
                throw error
            }
        }
    }

    private static func optionalGeometryIntField(
        _ raw: String?, field: String, message: String = "must be greater than zero"
    ) throws -> Int? {
        guard let raw = normalizedOptional(raw) else { return nil }
        guard let value = Int(raw) else { throw APIRequestValidationError.invalidField(field, message) }
        return value
    }

    static func geometryResponse(
        from result: MoGe2RunResult,
        createdAt: Date = Date()
    ) throws -> APIGeometryResponse {
        let manifest = result.export.manifest
        let root = URL(fileURLWithPath: manifest.outputDirectory, isDirectory: true)
        var artifacts = manifest.artifacts.map { artifact in
            APIGeometryArtifactResponse(
                kind: artifact.kind,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }
        let manifestURL = result.export.manifestURL
        let attributes = try FileManager.default.attributesOfItem(atPath: manifestURL.path)
        artifacts.append(
            APIGeometryArtifactResponse(
                kind: .manifest,
                url: manifestURL.absoluteString,
                mediaType: "application/json",
                byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                sha256: try ModelArtifactPin.fileSHA256(manifestURL)
            )
        )
        artifacts.sort { lhs, rhs in
            if lhs.kind.rawValue == rhs.kind.rawValue { return lhs.url < rhs.url }
            return lhs.kind.rawValue < rhs.kind.rawValue
        }
        return APIGeometryResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.geometry",
            status: "completed",
            model: manifest.model.modelID,
            width: manifest.width,
            height: manifest.height,
            units: manifest.units,
            coordinateSystem: manifest.coordinateSystem,
            camera: manifest.camera,
            depthStatistics: manifest.depthStatistics,
            focal: result.focalShift.focal,
            shift: result.focalShift.shift,
            metricScale: result.metricScale,
            tokenCount: result.tokenCount,
            manifestURL: manifestURL.absoluteString,
            artifacts: artifacts,
            timing: APIGeometryTimingResponse(
                modelLoadSeconds: result.modelLoadSeconds,
                inferenceSeconds: result.inferenceSeconds,
                postprocessSeconds: result.postprocessSeconds
            )
        )
    }

    static func multiViewGeometryPlan(from form: MultipartFormData) throws -> MultiViewGeometryPlan {
        let allowedTextFields: Set<String> = [
            "model",
            "process_resolution",
            "reference_view",
            "confidence_percentile",
            "max_points",
            "cameras",
        ]
        let allowedFileFields: Set<String> = ["image", "image[]", "cameras"]
        for part in form.parts {
            if part.filename != nil {
                guard allowedFileFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported file part; only uploaded image/image[] and cameras JSON files are accepted"
                    )
                }
            } else {
                guard allowedTextFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported field; client input, output, model, and camera filesystem paths are not accepted"
                    )
                }
                guard String(data: part.body, encoding: .utf8) != nil else {
                    throw APIRequestValidationError.invalidField(part.name, "must contain valid UTF-8 text")
                }
            }
        }
        for field in allowedTextFields
        where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

        // Multipart order is the view order. Do not regroup image and image[]
        // aliases because cameras are indexed against this exact sequence.
        let imageUploads = form.parts.filter {
            $0.filename != nil && ($0.name == "image" || $0.name == "image[]")
        }
        guard !imageUploads.isEmpty, imageUploads.allSatisfy({ !$0.body.isEmpty }) else {
            throw APIRequestValidationError.invalidField(
                "image",
                "one or more non-empty uploaded image/image[] files are required"
            )
        }
        for image in imageUploads {
            if let rawContentType = image.contentType {
                let contentType = rawContentType
                    .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                    .first?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased() ?? ""
                guard contentType.hasPrefix("image/") || contentType == "application/octet-stream" else {
                    throw APIRequestValidationError.invalidField(
                        image.name,
                        "uploaded parts must have an image content type"
                    )
                }
            }
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultMultiViewGeometryModelID
        ).lowercased()
        guard modelID == defaultMultiViewGeometryModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only the managed model id \(defaultMultiViewGeometryModelID) is supported"
            )
        }

        let processResolution = try optionalPositiveIntField(
            form.field("process_resolution"),
            field: "process_resolution"
        ) ?? 504
        do {
            try DepthAnything3Limits.validateRequest(
                viewCount: imageUploads.count,
                processResolution: processResolution
            )
        } catch let error as DepthAnything3LimitError {
            let field: String
            switch error {
            case .processResolutionOutOfRange:
                field = "process_resolution"
            case .viewCountOutOfRange, .processedPixelBudgetExceeded,
                 .invalidSourceDimensions, .sourcePixelBudgetExceeded,
                 .totalSourcePixelBudgetExceeded, .encodedByteBudgetExceeded,
                 .totalEncodedByteBudgetExceeded:
                field = "image"
            }
            throw APIRequestValidationError.invalidField(field, error.localizedDescription)
        }
        let maximumPointCount = try optionalPositiveIntField(
            form.field("max_points"),
            field: "max_points"
        ) ?? 1_000_000

        let confidencePercentile: Double
        if let raw = normalizedOptional(form.field("confidence_percentile")) {
            guard let value = Double(raw), value.isFinite, (0...100).contains(value) else {
                throw APIRequestValidationError.invalidField(
                    "confidence_percentile",
                    "must be a finite number between 0 and 100"
                )
            }
            confidencePercentile = value
        } else {
            confidencePercentile = 40
        }

        let referenceViewRaw = normalizedOptional(form.field("reference_view"))?.lowercased()
            ?? DepthAnything3ReferenceViewStrategy.saddleBalanced.rawValue
        guard let referenceViewStrategy = DepthAnything3ReferenceViewStrategy(
            rawValue: referenceViewRaw
        ) else {
            throw APIRequestValidationError.invalidField(
                "reference_view",
                "must be one of \(DepthAnything3ReferenceViewStrategy.allCases.map(\.rawValue).joined(separator: ", "))"
            )
        }

        let cameraFileParts = form.files(named: "cameras")
        let cameraTextParts = form.parts.filter { $0.name == "cameras" && $0.filename == nil }
        guard cameraFileParts.count <= 1 else {
            throw APIRequestValidationError.invalidField("cameras", "must be supplied at most once")
        }
        guard cameraFileParts.isEmpty || cameraTextParts.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "supply either an uploaded JSON document or an inline JSON document, not both"
            )
        }
        if let cameraFile = cameraFileParts.first {
            guard !cameraFile.body.isEmpty else {
                throw APIRequestValidationError.invalidField("cameras", "uploaded JSON document must not be empty")
            }
            if let rawContentType = cameraFile.contentType {
                let contentType = rawContentType
                    .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                    .first?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased() ?? ""
                guard contentType == "application/json"
                    || contentType.hasSuffix("+json")
                    || contentType == "application/octet-stream"
                    || contentType == "text/json" else {
                    throw APIRequestValidationError.invalidField(
                        "cameras",
                        "uploaded camera document must have a JSON content type"
                    )
                }
            }
        }
        let cameraData = cameraFileParts.first?.body ?? cameraTextParts.first?.body
        let knownCameras = try cameraData.map {
            try decodeMultiViewCameraDocument($0, expectedCount: imageUploads.count)
        }

        return MultiViewGeometryPlan(
            modelID: modelID,
            processResolution: processResolution,
            referenceViewStrategy: referenceViewStrategy,
            confidencePercentile: confidencePercentile,
            maximumPointCount: maximumPointCount,
            knownCameras: knownCameras
        )
    }

    static func multiViewGeometryResponse(
        from result: DepthAnything3RunResult,
        export: MultiViewGeometryExportResult,
        exportSeconds: Double,
        createdAt: Date = Date()
    ) throws -> APIMultiViewGeometryResponse {
        let scene = export.manifest
        let root = URL(fileURLWithPath: scene.outputDirectory, isDirectory: true)
        let manifest = try multiViewGeometryFileArtifact(
            kind: "manifest",
            viewIndex: nil,
            url: export.manifestURL,
            mediaType: "application/json"
        )
        let artifacts = scene.artifacts.map { artifact in
            APIMultiViewGeometryArtifactResponse(
                kind: artifact.kind.rawValue,
                viewIndex: artifact.viewIndex,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }.sorted { lhs, rhs in
            if lhs.viewIndex != rhs.viewIndex {
                return (lhs.viewIndex ?? Int.max) < (rhs.viewIndex ?? Int.max)
            }
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let cameras = try scene.views.map { view in
            guard let extrinsics = view.camera.extrinsics else {
                throw APIRequestValidationError.invalidField(
                    "result",
                    "multi-view camera \(view.index) is missing extrinsics"
                )
            }
            return APIMultiViewGeometryCameraResponse(
                viewIndex: view.index,
                width: view.width,
                height: view.height,
                intrinsics: view.camera.intrinsics,
                extrinsics: extrinsics,
                selectedPointCount: view.selectedPointCount
            )
        }.sorted { $0.viewIndex < $1.viewIndex }
        let totalSeconds = result.checkpointVerificationSeconds
            + result.decodingSeconds
            + result.preprocessingSeconds
            + result.modelLoadSeconds
            + result.inferenceSeconds
            + result.postprocessingSeconds
            + exportSeconds
        return APIMultiViewGeometryResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.geometry.multiview",
            status: "completed",
            model: result.checkpoint.modelID,
            checkpoint: APIMultiViewGeometryCheckpointResponse(
                repository: result.checkpoint.repository,
                revision: result.checkpoint.revision,
                sourceRepository: result.checkpoint.sourceRepository,
                sourceRevision: result.checkpoint.sourceRevision,
                license: result.checkpoint.license,
                weightsByteCount: result.checkpoint.weightsByteCount,
                weightsSHA256: result.checkpoint.weightsSHA256,
                configurationByteCount: result.checkpoint.configurationByteCount,
                configurationSHA256: result.checkpoint.configurationSHA256
            ),
            units: scene.units,
            coordinateSystem: scene.coordinateSystem,
            poseConditioned: scene.poseConditioned,
            cameraSemantics: result.cameraSemantics,
            cameraScaleAlignment: result.cameraScaleAlignment,
            referenceViewStrategy: result.referenceViewStrategy,
            depthScaleDivisor: result.depthScaleDivisor,
            processResolution: result.processResolution,
            confidencePercentile: scene.confidencePercentile,
            confidenceThreshold: scene.confidenceThreshold,
            viewCount: scene.views.count,
            cameraCount: cameras.count,
            pointCount: scene.pointCount,
            pointCloudRepresentation: scene.pointCloudRepresentation,
            containsMesh: false,
            containsGaussianParameters: scene.threeDGaussianHandoff.containsGaussianParameters,
            threeDGaussianHandoff: scene.threeDGaussianHandoff,
            cameras: cameras,
            manifest: manifest,
            artifacts: artifacts,
            timing: APIMultiViewGeometryTimingResponse(
                checkpointVerificationSeconds: result.checkpointVerificationSeconds,
                decodingSeconds: result.decodingSeconds,
                preprocessingSeconds: result.preprocessingSeconds,
                modelLoadSeconds: result.modelLoadSeconds,
                inferenceSeconds: result.inferenceSeconds,
                postprocessingSeconds: result.postprocessingSeconds,
                exportSeconds: exportSeconds,
                totalSeconds: totalSeconds
            )
        )
    }

    private static func decodeMultiViewCameraDocument(
        _ data: Data,
        expectedCount: Int
    ) throws -> [DepthAnything3KnownCamera] {
        let document: DepthAnything3CameraDocument
        do {
            document = try JSONDecoder().decode(DepthAnything3CameraDocument.self, from: data)
        } catch {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "must be a valid schemaVersion 1 camera JSON document"
            )
        }
        guard document.schemaVersion == 1 else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "unsupported schemaVersion \(document.schemaVersion); expected 1"
            )
        }
        guard document.cameras.count == expectedCount else {
            throw APIRequestValidationError.invalidField(
                "cameras",
                "contains \(document.cameras.count) cameras for \(expectedCount) uploaded images"
            )
        }
        for (index, camera) in document.cameras.enumerated() {
            do {
                try DepthAnything3CameraValidation.validate(camera, index: index)
            } catch {
                throw APIRequestValidationError.invalidField(
                    "cameras",
                    error.localizedDescription
                )
            }
        }
        return document.cameras
    }

    private static func multiViewGeometryFileArtifact(
        kind: String,
        viewIndex: Int?,
        url: URL,
        mediaType: String
    ) throws -> APIMultiViewGeometryArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIMultiViewGeometryArtifactResponse(
            kind: kind,
            viewIndex: viewIndex,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    static func imageTo3DPlan(from form: MultipartFormData) throws -> ImageTo3DPlan {
        let allowedTextFields: Set<String> = [
            "model",
            "resolution",
            "density_threshold",
            "foreground_ratio",
            "already_framed",
            "vertex_colors",
        ]
        for part in form.parts {
            if part.filename != nil {
                guard part.name == "image" else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "only one uploaded image file is accepted"
                    )
                }
            } else {
                guard allowedTextFields.contains(part.name) else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "unsupported field; client input, output, and checkpoint paths are not accepted"
                    )
                }
                guard String(data: part.body, encoding: .utf8) != nil else {
                    throw APIRequestValidationError.invalidField(part.name, "must contain valid UTF-8 text")
                }
            }
        }
        for field in allowedTextFields
        where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

        let uploads = form.files(named: "image")
        guard uploads.count == 1, let upload = uploads.first, !upload.body.isEmpty else {
            throw APIRequestValidationError.invalidField(
                "image",
                "exactly one non-empty uploaded image file is required"
            )
        }
        if let rawContentType = upload.contentType {
            let contentType = rawContentType
                .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                .first?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased() ?? ""
            guard contentType.hasPrefix("image/") || contentType == "application/octet-stream" else {
                throw APIRequestValidationError.invalidField(
                    "image",
                    "uploaded part must have an image content type"
                )
            }
        }

        let modelID = normalizedModelID(
            form.field("model"),
            defaultID: defaultImageTo3DModelID
        ).lowercased()
        guard modelID == defaultImageTo3DModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only the managed model id \(defaultImageTo3DModelID) is supported"
            )
        }

        let resolution = try optionalPositiveIntField(form.field("resolution"), field: "resolution")
            ?? TripoSRGenerationSettings.defaultResolution
        let density: Float
        if let raw = normalizedOptional(form.field("density_threshold")) {
            guard let value = Float(raw) else {
                throw APIRequestValidationError.invalidField("density_threshold", "must be a finite number")
            }
            density = value
        } else {
            density = TripoSRConfiguration.production.densityThreshold
        }
        let ratio: Float
        if let raw = normalizedOptional(form.field("foreground_ratio")) {
            guard let value = Float(raw) else {
                throw APIRequestValidationError.invalidField("foreground_ratio", "must be greater than 0 and at most 1")
            }
            ratio = value
        } else {
            ratio = TripoSRGenerationSettings.defaultForegroundRatio
        }
        do {
            return ImageTo3DPlan(modelID: modelID, settings: try TripoSRGenerationSettings(
                extractionResolution: resolution, densityThreshold: density, foregroundRatio: ratio,
                alreadyFramed: try multipartBoolean(
                    form.field("already_framed"), field: "already_framed", defaultValue: false
                ),
                includesVertexColors: try multipartBoolean(
                    form.field("vertex_colors"), field: "vertex_colors", defaultValue: true
                )
            ))
        } catch TripoSRGeneratorError.invalidExtractionResolution {
            throw APIRequestValidationError.invalidField("resolution", "must be an integer between 2 and 512")
        } catch TripoSRGeneratorError.invalidDensityThreshold {
            throw APIRequestValidationError.invalidField("density_threshold", "must be a finite number")
        } catch TripoSRPreprocessingError.invalidForegroundRatio {
            throw APIRequestValidationError.invalidField("foreground_ratio", "must be greater than 0 and at most 1")
        }
    }

    static func imageTo3DResponse(
        from result: TripoSRRunResult,
        createdAt: Date = Date()
    ) throws -> APIImageTo3DResponse {
        let mesh = result.export.manifest
        let root = URL(fileURLWithPath: mesh.outputDirectory, isDirectory: true)
        let artifacts = result.runManifest.manifest.artifacts.map { artifact in
            APIImageTo3DArtifactResponse(
                kind: artifact.kind,
                url: root.appendingPathComponent(artifact.relativePath).absoluteString,
                mediaType: artifact.mediaType,
                byteCount: artifact.byteCount,
                sha256: artifact.sha256
            )
        }.sorted { lhs, rhs in
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.url < rhs.url
        }
        let manifest = try imageTo3DFileArtifact(
            kind: "manifest",
            url: result.runManifest.manifestURL,
            mediaType: "application/json"
        )
        let meshManifest = try imageTo3DFileArtifact(
            kind: "mesh-manifest",
            url: result.export.manifestURL,
            mediaType: "application/json"
        )
        let totalSeconds = result.checkpointVerificationSeconds
            + result.decodingSeconds
            + result.preprocessingSeconds
            + result.modelLoadSeconds
            + result.sceneEncodingSeconds
            + result.meshExtractionSeconds
            + result.exportSeconds
        return APIImageTo3DResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.image-to-3d",
            status: "completed",
            model: result.checkpoint.modelID,
            checkpoint: APIImageTo3DCheckpointResponse(
                repository: result.checkpoint.repository,
                revision: result.checkpoint.revision,
                sourceRepository: result.checkpoint.sourceRepository,
                sourceRevision: result.checkpoint.sourceRevision,
                license: result.checkpoint.license,
                format: result.checkpoint.format,
                weightsByteCount: result.checkpoint.weightsByteCount,
                weightsSHA256: result.checkpoint.weightsSHA256,
                sourceSHA256: result.checkpoint.sourceSHA256,
                configurationSHA256: result.checkpoint.configurationSHA256
            ),
            sourceWidth: result.sourceWidth,
            sourceHeight: result.sourceHeight,
            preparedWidth: result.preparedWidth,
            preparedHeight: result.preparedHeight,
            foregroundPolicy: result.foregroundPolicy,
            foregroundRatio: result.foregroundRatio,
            croppedTransparentForeground: result.croppedTransparentForeground,
            extractionResolution: result.extractionResolution,
            densityThreshold: result.densityThreshold,
            includesVertexColors: result.includesVertexColors,
            meshExtractionAlgorithm: TripoSRRunManifestExporter.extractionAlgorithm,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.triangleCount,
            bounds: mesh.bounds,
            manifest: manifest,
            meshManifest: meshManifest,
            artifacts: artifacts,
            timing: APIImageTo3DTimingResponse(
                checkpointVerificationSeconds: result.checkpointVerificationSeconds,
                decodingSeconds: result.decodingSeconds,
                preprocessingSeconds: result.preprocessingSeconds,
                modelLoadSeconds: result.modelLoadSeconds,
                sceneEncodingSeconds: result.sceneEncodingSeconds,
                meshExtractionSeconds: result.meshExtractionSeconds,
                exportSeconds: result.exportSeconds,
                totalSeconds: totalSeconds
            )
        )
    }

    private static func imageTo3DFileArtifact(
        kind: String,
        url: URL,
        mediaType: String
    ) throws -> APIImageTo3DArtifactResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return APIImageTo3DArtifactResponse(
            kind: kind,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func multipartBoolean(
        _ raw: String?,
        field: String,
        defaultValue: Bool
    ) throws -> Bool {
        guard let value = normalizedOptional(raw)?.lowercased() else { return defaultValue }
        switch value {
        case "true", "1": return true
        case "false", "0": return false
        default:
            throw APIRequestValidationError.invalidField(
                field,
                "must be true, false, 1, or 0"
            )
        }
    }

    static func depthVideoPlan(from form: MultipartFormData) throws -> DepthVideoPlan {
        let allowedFields: Set<String> = ["model", "input_size", "max_frames"]
        for part in form.parts {
            if part.filename != nil {
                guard part.name == "video" else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "only a single uploaded 'video' file is accepted"
                    )
                }
            } else if !allowedFields.contains(part.name) {
                throw APIRequestValidationError.invalidField(
                    part.name,
                    "unsupported field; client filesystem paths are not accepted"
                )
            }
        }
        for field in allowedFields where form.parts.filter({ $0.name == field && $0.filename == nil }).count > 1 {
            throw APIRequestValidationError.invalidField(field, "must be supplied at most once")
        }

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
        do {
            _ = try VideoDepthAnythingLimits.validateRequest(
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
        return DepthVideoPlan(
            modelID: modelID,
            inputSize: inputSize,
            maximumFrameCount: maximumFrameCount
        )
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

    static func imageGenerationPlan(
        from request: OpenAIImageGenerationRequest
    ) throws -> ImageGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        let steps = try imageInferenceSteps(request.steps)
        if let guidanceScale = request.guidance_scale,
           (!guidanceScale.isFinite || guidanceScale < 0) {
            throw APIRequestValidationError.invalidField("guidance_scale", "must be a positive number")
        }
        let size = try imageSize(from: request.size)
        let responseFormat = try imageResponseFormat(request.response_format)
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(request.model),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: responseFormat,
            seed: request.seed,
            negativePrompt: normalizedOptional(request.negative_prompt),
            steps: steps,
            guidanceScale: request.guidance_scale,
            inputImage: nil,
            strength: nil
        )
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURL: URL
    ) throws -> ImageGenerationPlan {
        try imageEditPlan(from: form, inputImageURLs: [inputImageURL], maskImageURL: nil)
    }

    static func imageEditPlan(
        from form: MultipartFormData,
        inputImageURLs: [URL],
        maskImageURL: URL?
    ) throws -> ImageGenerationPlan {
        guard let inputImageURL = inputImageURLs.first else {
            throw APIRequestValidationError.invalidField("image", "image file is required")
        }
        let prompt = normalizedOptional(form.field("prompt")) ?? ""
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        if let rawN = normalizedOptional(form.field("n")) {
            guard let n = Int(rawN), n == 1 else {
                throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
            }
        }
        let steps = try imageInferenceSteps(
            optionalPositiveIntField(form.field("steps"), field: "steps")
        )
        let guidanceScale = try optionalPositiveDoubleField(form.field("guidance_scale"), field: "guidance_scale")
        let strength = try optionalUnitDoubleField(form.field("strength"), field: "strength")
        let size = try imageSize(from: form.field("size"))
        return ImageGenerationPlan(
            modelID: normalizedImageModelID(form.field("model")),
            prompt: prompt,
            width: size.width,
            height: size.height,
            responseFormat: try imageResponseFormat(form.field("response_format")),
            seed: try optionalUInt64Field(form.field("seed")),
            negativePrompt: normalizedOptional(form.field("negative_prompt")),
            steps: steps,
            guidanceScale: guidanceScale,
            inputImage: inputImageURL,
            additionalInputImages: Array(inputImageURLs.dropFirst()),
            maskImage: maskImageURL,
            strength: strength
        )
    }

    static func imageResponse(
        outputURL: URL,
        plan: ImageGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIImageGenerationResponse {
        let datum: OpenAIImageGenerationData
        switch plan.responseFormat {
        case "url":
            datum = OpenAIImageGenerationData(url: outputURL.absoluteString, revised_prompt: plan.prompt)
        case "b64_json":
            let data = try Data(contentsOf: outputURL)
            datum = OpenAIImageGenerationData(
                b64_json: data.base64EncodedString(),
                revised_prompt: plan.prompt
            )
        default:
            throw APIRequestValidationError.invalidField("response_format", "unsupported response format")
        }
        return OpenAIImageGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            data: [datum]
        )
    }

    static func videoGenerationPlan(
        from request: OpenAIVideoGenerationRequest
    ) throws -> VideoGenerationPlan {
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw APIRequestValidationError.invalidField("prompt", "must not be empty")
        }
        let modelID = normalizedOptional(request.model) ?? defaultVideoModelID
        let size = try videoSize(from: request.size)
        if let seconds = request.seconds,
           !seconds.isFinite || seconds <= 0 {
            throw APIRequestValidationError.invalidField("seconds", "must be finite and positive")
        }
        if let numFrames = request.num_frames, numFrames < 9 {
            throw APIRequestValidationError.invalidField("num_frames", "must be at least 9")
        }
        if request.seconds != nil, request.num_frames != nil {
            throw APIRequestValidationError.invalidField(
                "seconds",
                "use seconds or num_frames, not both"
            )
        }
        let fps = request.fps ?? 24
        guard fps > 0, fps <= 240 else {
            throw APIRequestValidationError.invalidField("fps", "must be between 1 and 240")
        }
        let quality: LTXVideoQuality?
        if let value = normalizedOptional(request.quality)?.lowercased() {
            guard let parsed = LTXVideoQuality(rawValue: value) else {
                throw APIRequestValidationError.invalidField("quality", "expected draft or final")
            }
            quality = parsed
        } else {
            quality = nil
        }
        let outputMode: LTXVideoOutputMode?
        if let value = normalizedOptional(request.output_mode)?.lowercased() {
            guard let parsed = LTXVideoOutputMode(rawValue: value) else {
                throw APIRequestValidationError.invalidField("output_mode", "expected video-only or audio-video")
            }
            outputMode = parsed
        } else {
            outputMode = nil
        }
        let options = try videoGenerationOptions(request.options ?? [])
        return VideoGenerationPlan(
            modelID: modelID,
            prompt: prompt,
            width: size.width,
            height: size.height,
            seconds: request.seconds,
            numFrames: request.num_frames,
            fps: fps,
            seed: request.seed,
            quality: quality,
            outputMode: outputMode,
            options: options
        )
    }

    static func videoGenerationResponse(
        outputURL: URL,
        plan: VideoGenerationPlan,
        createdAt: Date = Date()
    ) throws -> OpenAIVideoGenerationResponse {
        let attributes = try FileManager.default.attributesOfItem(atPath: outputURL.path)
        let exrDirectory = outputURL.deletingLastPathComponent().appendingPathComponent(
            outputURL.deletingPathExtension().lastPathComponent + "_exr",
            isDirectory: true
        )
        let hasEXR = FileManager.default.fileExists(atPath: exrDirectory.path)
        return OpenAIVideoGenerationResponse(
            created: Int(createdAt.timeIntervalSince1970),
            model: plan.modelID,
            artifact: OpenAIVideoGenerationArtifact(
                url: outputURL.absoluteString,
                media_type: "video/mp4",
                byte_count: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                sha256: try ModelArtifactPin.fileSHA256(outputURL)
            ),
            exr_directory_url: hasEXR ? exrDirectory.absoluteString : nil
        )
    }

    static func speechPlan(from request: OpenAIAudioSpeechRequest) throws -> SpeechPlan {
        let input = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
        let responseFormat = try speechResponseFormat(request.response_format)
        let speed = try speechSpeed(request.speed)
        let temperature = request.temperature ?? TTSRequest.defaultTemperature
        do {
            try SpeechSynthesisPlan.validateParameters(text: input, temperature: temperature, speed: speed)
        } catch SpeechSynthesisError.invalidInput(let field, let message) {
            throw APIRequestValidationError.invalidField(field == .text ? "input" : field.rawValue, message)
        }
        let voiceDescription = voiceDescription(for: request.voice, instructions: request.instructions)
        var promptUTF8Bytes = 0
        for component in [input, voiceDescription] {
            let componentBytes = component.utf8.count
            guard componentBytes <= maxSpeechPromptUTF8Bytes - promptUTF8Bytes else {
                throw APIRequestValidationError.invalidField(
                    "input",
                    "input and voice instructions must total at most \(maxSpeechPromptUTF8Bytes) UTF-8 bytes"
                )
            }
            promptUTF8Bytes += componentBytes
        }
        return SpeechPlan(
            modelID: normalizedSpeechModelID(request.model),
            input: input,
            voiceDescription: voiceDescription,
            responseFormat: responseFormat,
            speed: speed,
            temperature: temperature
        )
    }

    static func transcriptionPlan(from form: MultipartFormData) throws -> TranscriptionPlan {
        let modelID = normalizedTranscriptionModelID(form.field("model"))
        return TranscriptionPlan(
            modelID: modelID,
            language: normalizedOptional(form.field("language")),
            responseFormat: try transcriptionResponseFormat(form.field("response_format")),
            task: try transcriptionTask(form.field("task")),
            maxTokens: try transcriptionMaxTokens(form.field("max_tokens"))
        )
    }

    static func transcriptionResponse(
        from result: ASRResult,
        verbose: Bool
    ) -> OpenAIAudioTranscriptionResponse {
        OpenAIAudioTranscriptionResponse(
            text: result.text,
            language: result.language,
            duration: verbose ? result.duration : nil,
            segments: verbose ? transcriptionSegments(from: result) : nil
        )
    }

    static func transcriptionSubtitle(from result: ASRResult, format: String) -> String {
        let segments = transcriptionSegments(from: result) ?? [
            OpenAIAudioTranscriptionSegment(
                id: 0,
                start: 0,
                end: max(result.duration, 0.001),
                text: result.text
            ),
        ]
        switch format {
        case "srt":
            return segments.enumerated()
                .map { index, segment in
                    let start = subtitleTimestamp(segment.start, separator: ",")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ",")
                    return "\(index + 1)\n\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n") + "\n"
        default:
            let body = segments
                .map { segment in
                    let start = subtitleTimestamp(segment.start, separator: ".")
                    let end = subtitleTimestamp(max(segment.end, segment.start + 0.001), separator: ".")
                    return "\(start) --> \(end)\n\(segment.text)"
                }
                .joined(separator: "\n\n")
            return body.isEmpty ? "WEBVTT\n" : "WEBVTT\n\n\(body)\n"
        }
    }

    static func chatRequest(
        from openaiRequest: OpenAIChatRequest,
        fallbackLoraPath: String?,
        contextSize: Int,
        capabilities: APIEngineCapabilities = .localText,
        servedModelID: String? = nil,
        apiProfile: ManagedModelAPIProfile? = nil
    ) throws -> ChatRequest {
        guard !openaiRequest.messages.isEmpty else {
            throw APIRequestValidationError.invalidField("messages", "must contain at least one message")
        }

        try validateTopLevelOptions(openaiRequest, capabilities: capabilities)

        if let requestLora = openaiRequest.lora?.trimmingCharacters(in: .whitespacesAndNewlines),
           !requestLora.isEmpty {
            throw APIRequestValidationError.invalidField(
                "lora",
                "per-request LoRA paths are not supported; start the server with --lora instead"
            )
        }

        let maxTokens = try resolveMaxTokens(
            maxTokens: openaiRequest.max_tokens,
            maxCompletionTokens: openaiRequest.max_completion_tokens,
            capabilities: capabilities
        )
        let tools = try toolDefinitions(from: openaiRequest, capabilities: capabilities)
        let toolChoice = try chatToolChoice(from: openaiRequest, capabilities: capabilities)
        let parallelToolCalls = openaiRequest.parallel_tool_calls ?? true
        let requiresJSON = try requiresJSONResponseFormat(
            openaiRequest.response_format,
            capabilities: capabilities
        )

        var messages = try openaiRequest.messages.map { msg in
            try chatMessage(from: msg, capabilities: capabilities)
        }
        if let instruction = toolChoiceInstruction(
            choice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            tools: tools
        ) {
            if let systemIndex = messages.firstIndex(where: { $0.role == .system }) {
                messages[systemIndex].content = [messages[systemIndex].content, instruction]
                    .filter { !$0.isEmpty }
                    .joined(separator: "\n\n")
            } else {
                messages.insert(ChatMessage(role: .system, content: instruction), at: 0)
            }
        }

        let lora: LoRA?
        if let loraPath = fallbackLoraPath?.trimmingCharacters(in: .whitespacesAndNewlines),
           !loraPath.isEmpty {
            lora = LoRA.local(path: loraPath, scale: 1.0)
        } else {
            lora = nil
        }

        // Resolve each omitted sampling field independently.
        let laneModelID = servedModelID ?? ""
        let resolvedAPIProfile = apiProfile
            ?? servedModelID.flatMap { ManagedModelCatalog.apiProfile(for: $0) }
        let reasoningEffort = try reasoningEffort(
            from: openaiRequest.reasoning_effort,
            capabilities: capabilities,
            profile: resolvedAPIProfile
        )
        let logprobCapture: ChatLogprobCapture
        if openaiRequest.logprobs == true {
            if let topLogprobs = openaiRequest.top_logprobs, topLogprobs > 0 {
                logprobCapture = .top(topLogprobs)
            } else {
                logprobCapture = .tokens
            }
        } else {
            logprobCapture = .none
        }
        if openaiRequest.mere_show_unmasking == true {
            guard laneModelID == DiffusionGemmaResources.modelID else {
                throw APIRequestValidationError.invalidField(
                    "mere_show_unmasking",
                    "progressive unmasking is supported only by \(DiffusionGemmaResources.modelID)"
                )
            }
            guard openaiRequest.stream == true else {
                throw APIRequestValidationError.invalidField(
                    "mere_show_unmasking",
                    "requires stream=true"
                )
            }
        }

        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            presencePenalty: openaiRequest.presence_penalty ?? 0,
            frequencyPenalty: openaiRequest.frequency_penalty ?? 0,
            repetitionPenalty: openaiRequest.repetition_penalty ?? 1,
            seed: openaiRequest.seed.map(UInt64.init),
            reasoningEffort: reasoningEffort,
            lora: lora,
            requiresJSON: requiresJSON,
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: parallelToolCalls,
            stopSequences: openaiRequest.stop?.values ?? [],
            maxContextTokens: contextSize,
            logprobCapture: logprobCapture,
            showUnmasking: openaiRequest.mere_show_unmasking == true
        )
        do {
            let resolved = try ChatRequestResolver.resolve(
                request, modelID: laneModelID,
                sampling: ChatSamplingOptions(
                    temperature: openaiRequest.temperature, topP: openaiRequest.top_p,
                    topK: openaiRequest.top_k, minP: openaiRequest.min_p
                ), policy: .openAI, apiProfile: resolvedAPIProfile
            )
            try validateSamplingCapabilities(openaiRequest, capabilities: capabilities)
            return resolved
        } catch let issue as ChatRequestIssue {
            throw APIRequestValidationError.invalidField(issue.field, issue.message)
        }
    }


    static func includeUsageInStreaming(
        _ openaiRequest: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard openaiRequest.stream_options?.include_usage == true else {
            return false
        }
        guard capabilities.supportsUsageInStreaming else {
            throw APIRequestValidationError.invalidField(
                "stream_options.include_usage",
                "this engine cannot emit usage chunks while streaming"
            )
        }
        return true
    }

    private static func chatMessage(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> ChatMessage {
        let normalizedRole = msg.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let role: ChatMessage.Role
        switch normalizedRole {
        case "system":
            role = .system
        case "developer":
            guard capabilities.supportsDeveloperRole else {
                throw APIRequestValidationError.invalidField(
                    "messages.role",
                    "developer messages are not supported by this engine"
                )
            }
            role = .system
        case "user":
            role = .user
        case "assistant":
            role = .assistant
        case "tool":
            role = .tool
        default:
            throw APIRequestValidationError.invalidField(
                "messages.role",
                "unsupported role '\(msg.role)'"
            )
        }

        let imageURL = try firstImageURL(from: msg, capabilities: capabilities)
        let audioURL = try firstAudioURL(from: msg, capabilities: capabilities)
        let videoURL = try firstVideoURL(from: msg, capabilities: capabilities)
        let content = capabilities.usesNativeToolHistory ? msg.content : renderMessageContent(msg)
        let toolCalls = try chatMessageToolCalls(from: msg)
        return ChatMessage(
            role: role,
            content: content,
            imageUrl: imageURL,
            audioUrl: audioURL,
            videoUrl: videoURL,
            reasoningContent: msg.reasoning_content,
            name: msg.name,
            toolCallID: msg.tool_call_id,
            toolCalls: toolCalls
        )
    }

    private static func chatMessageToolCalls(
        from message: OpenAIChatMessage
    ) throws -> [ChatMessageToolCall]? {
        guard message.role.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "assistant",
              let openAIToolCalls = message.tool_calls,
              !openAIToolCalls.isEmpty else {
            return nil
        }

        return try openAIToolCalls.compactMap { toolCall in
            guard toolCall.type == "function", let function = toolCall.function else {
                return nil
            }
            guard let data = function.arguments.data(using: .utf8) else {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a UTF-8 JSON object"
                )
            }
            let arguments: [String: OpenAIJSONValue]
            do {
                arguments = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: data)
            } catch {
                throw APIRequestValidationError.invalidField(
                    "messages.tool_calls.function.arguments",
                    "must be a JSON object"
                )
            }
            return ChatMessageToolCall(
                id: toolCall.id,
                name: function.name,
                arguments: arguments
            )
        }
    }

    private static func firstImageURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.imageURLs.isEmpty else { return nil }
        guard capabilities.supportsVisionContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "image content parts are not supported by this engine"
            )
        }
        guard msg.imageURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one image content part is currently supported"
            )
        }
        return msg.imageURLs.first
    }

    private static func firstAudioURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.audioURLs.isEmpty else { return nil }
        guard capabilities.supportsAudioContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "audio content parts are not supported by this engine"
            )
        }
        guard msg.audioURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one audio content part per message is currently supported"
            )
        }
        return msg.audioURLs.first
    }

    private static func firstVideoURL(
        from msg: OpenAIChatMessage,
        capabilities: APIEngineCapabilities
    ) throws -> String? {
        guard !msg.videoURLs.isEmpty else { return nil }
        guard capabilities.supportsVideoContentParts else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "video content parts are not supported by this engine"
            )
        }
        guard msg.videoURLs.count == 1 else {
            throw APIRequestValidationError.invalidField(
                "messages.content",
                "only one video content part per message is currently supported"
            )
        }
        return msg.videoURLs.first
    }

    private static func renderMessageContent(_ msg: OpenAIChatMessage) -> String {
        guard msg.role.lowercased() == "assistant",
              let toolCalls = msg.tool_calls,
              !toolCalls.isEmpty else {
            return msg.content
        }
        let renderedCalls = toolCalls.compactMap { call -> String? in
            guard call.type == "function", let function = call.function else { return nil }
            return "<|tool_call>call:\(function.name)\(function.arguments)<tool_call|>"
        }
        guard !renderedCalls.isEmpty else {
            return msg.content
        }
        return ([msg.content] + renderedCalls)
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func validateTopLevelOptions(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let n = request.n, n != 1 {
            throw APIRequestValidationError.invalidField("n", "only n=1 is supported")
        }
        if request.store == true {
            throw APIRequestValidationError.invalidField("store", "stored chat completions are not supported")
        }
        if let modalities = request.modalities,
           modalities.contains(where: { $0 != "text" }) {
            throw APIRequestValidationError.invalidField("modalities", "only text output is supported")
        }
        if request.audio != nil {
            throw APIRequestValidationError.invalidField("audio", "audio output is not supported by /v1/chat/completions")
        }
        if request.prediction != nil {
            throw APIRequestValidationError.invalidField("prediction", "predicted outputs are not supported")
        }
        if let stop = request.stop, !stop.values.isEmpty, !capabilities.supportsStopSequences {
            throw APIRequestValidationError.invalidField("stop", "stop sequences are not supported by this engine")
        }
        if request.seed != nil, !capabilities.supportsSeed {
            throw APIRequestValidationError.invalidField("seed", "deterministic seeds are not supported by this engine")
        }
        if let seed = request.seed, seed < 0 {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        if request.logprobs == true, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("logprobs", "token log probabilities are not supported by this engine")
        }
        if request.stream == true, request.logprobs == true {
            throw APIRequestValidationError.invalidField(
                "logprobs",
                "native token log probabilities currently require stream=false"
            )
        }
        if request.logprobs == true,
           let responseType = request.response_format?.type,
           responseType != "text" {
            throw APIRequestValidationError.invalidField(
                "logprobs",
                "native token log probabilities currently require an unconstrained text response"
            )
        }
        if request.top_logprobs != nil, !capabilities.supportsLogprobs {
            throw APIRequestValidationError.invalidField("top_logprobs", "token log probabilities are not supported by this engine")
        }
        if let topLogprobs = request.top_logprobs {
            guard request.logprobs == true else {
                throw APIRequestValidationError.invalidField(
                    "top_logprobs",
                    "requires logprobs=true"
                )
            }
            guard (0...20).contains(topLogprobs) else {
                throw APIRequestValidationError.invalidField(
                    "top_logprobs",
                    "must be between 0 and 20"
                )
            }
        }
        if request.reasoning_effort != nil, !capabilities.supportsReasoningEffort {
            throw APIRequestValidationError.invalidField("reasoning_effort", "reasoning effort is not supported by this engine")
        }
        if request.think != nil || request.thinking != nil {
            guard capabilities.supportsProviderThinkingControls else {
                throw APIRequestValidationError.invalidField(
                    "thinking",
                    "provider thinking controls are not supported by this engine"
                )
            }
        }
    }

    private static func toolDefinitions(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> [ToolDefinition]? {
        guard let tools = request.tools, !tools.isEmpty else {
            switch request.tool_choice {
            case .mode("required")?, .function(_)?:
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requires at least one function in tools"
                )
            default:
                return nil
            }
        }
        guard capabilities.supportsTools else {
            throw APIRequestValidationError.invalidField("tools", "tools are not supported by this engine")
        }

        let convertedTools = try tools.map { try toolDefinition(from: $0) }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("required")?:
            return convertedTools
        case .mode("none")?:
            return nil
        case .function(let name)?:
            guard let selected = convertedTools.first(where: { $0.name == name }) else {
                throw APIRequestValidationError.invalidField(
                    "tool_choice",
                    "requested tool '\(name)' is not present in tools"
                )
            }
            return [selected]
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func chatToolChoice(
        from request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws -> ChatToolChoice {
        guard request.tool_choice != nil else { return .auto }
        guard capabilities.supportsToolChoice else {
            throw APIRequestValidationError.invalidField(
                "tool_choice",
                "tool choice is not supported by this engine"
            )
        }
        switch request.tool_choice {
        case nil, .mode("auto")?, .mode("none")?:
            return .auto
        case .mode("required")?:
            return .required
        case .function(let name)?:
            return .function(name)
        case .custom?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported object shape")
        case .mode(let value)?:
            throw APIRequestValidationError.invalidField("tool_choice", "unsupported mode '\(value)'")
        }
    }

    private static func toolChoiceInstruction(
        choice: ChatToolChoice,
        parallelToolCalls: Bool,
        tools: [ToolDefinition]?
    ) -> String? {
        guard tools?.isEmpty == false else { return nil }
        switch choice {
        case .auto where !parallelToolCalls:
            return "If you call a function, call at most one provided function."
        case .required where parallelToolCalls:
            return "You must call one or more provided functions. Do not answer without a function call."
        case .required:
            return "You must call exactly one provided function. Do not answer without a function call."
        case .function(let name) where parallelToolCalls:
            return "You must call the provided function '\(name)' at least once. Do not call any other function."
        case .function(let name):
            return "You must call the provided function '\(name)' exactly once. Do not call any other function."
        case .auto:
            return nil
        }
    }

    private static func toolDefinition(from tool: OpenAIChatTool) throws -> ToolDefinition {
        guard tool.type == "function", let function = tool.function else {
            throw APIRequestValidationError.invalidField("tools", "only function tools are supported")
        }

        let schema: [String: OpenAIJSONValue]
        if let parameters = function.parameters, parameters != .null {
            guard let object = parameters.objectValue else {
                throw APIRequestValidationError.invalidField("tools", "function parameters must be a JSON object")
            }
            schema = object
        } else {
            schema = ["type": .string("object"), "properties": .object([:]), "required": .array([])]
        }

        return ToolDefinition(
            name: function.name,
            description: function.description ?? "",
            parameterSchema: schema
        )
    }

    static func openAIToolArgumentsJSON(
        _ arguments: [String: String],
        parameterTypes: [String: String]
    ) -> String {
        let normalized = Dictionary(uniqueKeysWithValues: arguments.map { key, rawValue in
            guard let parameterType = parameterTypes[key], parameterType != "string" else {
                return (key, OpenAIJSONValue.string(rawValue))
            }
            guard let data = rawValue.data(using: .utf8),
                  let decoded = try? JSONDecoder().decode(OpenAIJSONValue.self, from: data) else {
                return (key, OpenAIJSONValue.string(rawValue))
            }
            return (key, decoded)
        })
        let data = (try? JSONEncoder().encode(normalized)) ?? Data("{}".utf8)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private static func requiresJSONResponseFormat(
        _ responseFormat: OpenAIResponseFormat?,
        capabilities: APIEngineCapabilities
    ) throws -> Bool {
        guard let responseFormat else { return false }
        switch responseFormat.type {
        case "text":
            return false
        case "json_object":
            guard capabilities.supportsStructuredOutputs else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "JSON mode is not supported by this engine"
                )
            }
            return true
        case "json_schema":
            guard capabilities.supportsStrictMode else {
                throw APIRequestValidationError.invalidField(
                    "response_format",
                    "strict JSON schema outputs are not supported by this engine"
                )
            }
            return true
        default:
            throw APIRequestValidationError.invalidField(
                "response_format",
                "unsupported response format '\(responseFormat.type)'"
            )
        }
    }

    private static func resolveMaxTokens(
        maxTokens: Int?,
        maxCompletionTokens: Int?,
        capabilities: APIEngineCapabilities
    ) throws -> Int {
        if maxCompletionTokens != nil, !capabilities.supportsMaxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "this engine does not support max_completion_tokens"
            )
        }
        if let maxTokens, let maxCompletionTokens, maxTokens != maxCompletionTokens {
            throw APIRequestValidationError.invalidField(
                "max_completion_tokens",
                "must match max_tokens when both are provided"
            )
        }
        return maxCompletionTokens ?? maxTokens ?? defaultMaxTokens
    }

    private static func reasoningEffort(
        from rawValue: String?,
        capabilities: APIEngineCapabilities,
        profile: ManagedModelAPIProfile?
    ) throws -> Double? {
        guard let rawValue else { return nil }
        guard !capabilities.supportsRawProxy else { return nil }
        let normalized = rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let level = ManagedModelThinkingLevel(rawValue: normalized),
              let strength = profile?.reasoningEffortStrengths[level] else {
            let supportedLevels = ManagedModelThinkingLevel.allCases
                .filter { profile?.reasoningEffortStrengths[$0] != nil }
                .map(\.rawValue)
                .joined(separator: ", ")
            throw APIRequestValidationError.invalidField(
                "reasoning_effort",
                "must be one of \(supportedLevels)"
            )
        }
        return strength
    }

    static func acceptsJSONContentType(_ rawValue: String?) -> Bool {
        guard let mediaType = rawValue?
            .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
            .first?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
            !mediaType.isEmpty
        else {
            return false
        }
        return mediaType == "application/json" || mediaType.hasSuffix("+json")
    }

    static func multipartBoundary(from rawValue: String?) -> String? {
        guard let pieces = rawValue?.split(separator: ";", omittingEmptySubsequences: true),
              let mediaType = pieces.first?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              mediaType == "multipart/form-data" else {
            return nil
        }
        for piece in pieces.dropFirst() {
            let pair = piece.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2,
                  pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == "boundary" else {
                continue
            }
            var boundary = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if boundary.hasPrefix("\""), boundary.hasSuffix("\""), boundary.count >= 2 {
                boundary.removeFirst()
                boundary.removeLast()
            }
            return boundary.isEmpty ? nil : String(boundary)
        }
        return nil
    }

    static func isStreamingStatusMessage(_ message: String) -> Bool {
        switch message {
        case "Generating...", "Generating response", "Retrying generation", "DS4 chat completion":
            return true
        default:
            return false
        }
    }

    private static func validateSamplingCapabilities(
        _ request: OpenAIChatRequest,
        capabilities: APIEngineCapabilities
    ) throws {
        if let topK = request.top_k, topK != 0, !capabilities.supportsTopK {
            throw APIRequestValidationError.invalidField("top_k", "top-k sampling is not supported by this engine")
        }
        if let penalty = request.presence_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("presence_penalty", "presence penalties are not supported by this engine")
        }
        if let penalty = request.frequency_penalty, penalty != 0, !capabilities.supportsPenalties {
            throw APIRequestValidationError.invalidField("frequency_penalty", "frequency penalties are not supported by this engine")
        }
        if let penalty = request.repetition_penalty, penalty != 1, !capabilities.supportsRepetitionPenalty {
            throw APIRequestValidationError.invalidField(
                "repetition_penalty", "repetition penalties are not supported by this engine"
            )
        }
    }

    private static func imageSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (1024, 1024)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width > 0,
              height > 0 else {
            throw APIRequestValidationError.invalidField("size", "expected WIDTHxHEIGHT, for example 1024x1024")
        }

        let maxDimension = 4_096
        guard width <= maxDimension, height <= maxDimension else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at most \(maxDimension) pixels"
            )
        }

        // Validate the pixel product with division so attacker-controlled
        // dimensions can never overflow an Int before the request reaches a
        // tensor allocation.
        let maxPixels = 4_194_304
        guard width <= maxPixels / height else {
            throw APIRequestValidationError.invalidField(
                "size",
                "total image area must be at most \(maxPixels) pixels"
            )
        }

        let imageAlignment = 16
        guard width >= imageAlignment,
              height >= imageAlignment,
              width % imageAlignment == 0,
              height % imageAlignment == 0 else {
            throw APIRequestValidationError.invalidField(
                "size",
                "width and height must each be at least \(imageAlignment) pixels and divisible by \(imageAlignment)"
            )
        }
        return (width, height)
    }

    private static func videoSize(from rawValue: String?) throws -> (width: Int, height: Int) {
        guard let rawValue = normalizedOptional(rawValue), rawValue.lowercased() != "auto" else {
            return (768, 512)
        }
        let parts = rawValue.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let width = Int(parts[0]),
              let height = Int(parts[1]),
              width >= 64,
              height >= 64,
              width <= 4_096,
              height <= 4_096,
              width.isMultiple(of: 64),
              height.isMultiple(of: 64),
              width <= 4_194_304 / height else {
            throw APIRequestValidationError.invalidField(
                "size",
                "expected WIDTHxHEIGHT with 64-pixel alignment and at most 4194304 pixels"
            )
        }
        return (width, height)
    }

    private static func videoGenerationOptions(_ options: [String]) throws -> [String] {
        guard options.count <= 256 else {
            throw APIRequestValidationError.invalidField("options", "must contain at most 256 arguments")
        }
        let protectedFlags: Set<String> = [
            "--output", "-o", "--model", "-m", "--width", "--height",
            "--duration", "--num-frames", "--fps", "--seed", "--quality",
            "--output-mode", "--preflight", "--json",
        ]
        var totalBytes = 0
        for option in options {
            totalBytes += option.utf8.count
            guard option.utf8.count <= 8_192, totalBytes <= 65_536 else {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "arguments must total at most 65536 UTF-8 bytes"
                )
            }
            let flag = String(option.split(separator: "=", maxSplits: 1)[0])
            if flag == "--skip-mp4" {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "--skip-mp4 is unavailable because this API route retains and hashes an MP4 artifact"
                )
            }
            if protectedFlags.contains(flag) {
                throw APIRequestValidationError.invalidField(
                    "options",
                    "\(flag) is controlled by a typed request field"
                )
            }
        }
        return options
    }

    private static func imageResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "b64_json"
        guard value == "b64_json" || value == "url" else {
            throw APIRequestValidationError.invalidField("response_format", "expected b64_json or url")
        }
        return value
    }

    private static func speechResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "wav"
        guard ["wav", "mp3", "opus", "aac", "flac"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected wav, mp3, opus, aac, or flac"
            )
        }
        return value
    }

    static func speechContentType(for responseFormat: String) -> String {
        switch responseFormat {
        case "mp3":
            return "audio/mpeg"
        case "opus":
            return "audio/ogg"
        case "aac":
            return "audio/aac"
        case "flac":
            return "audio/flac"
        default:
            return "audio/wav"
        }
    }

    private static func speechSpeed(_ rawValue: Double?) throws -> Float {
        do {
            return try SpeechSynthesisPlan.validatedSpeed(rawValue ?? Double(TTSRequest.defaultSpeed))
        } catch SpeechSynthesisError.invalidInput(_, let message) {
            throw APIRequestValidationError.invalidField("speed", message)
        }
    }

    private static func transcriptionResponseFormat(_ rawValue: String?) throws -> String {
        let value = normalizedOptional(rawValue)?.lowercased() ?? "json"
        guard ["json", "text", "verbose_json", "srt", "vtt"].contains(value) else {
            throw APIRequestValidationError.invalidField(
                "response_format",
                "expected json, text, verbose_json, srt, or vtt"
            )
        }
        return value
    }

    private static func transcriptionTask(_ rawValue: String?) throws -> ASRTask {
        let value = normalizedOptional(rawValue)?.lowercased() ?? ASRTask.transcribe.rawValue
        guard let task = ASRTask(rawValue: value) else {
            throw APIRequestValidationError.invalidField("task", "expected transcribe or translate")
        }
        return task
    }

    private static func transcriptionMaxTokens(_ rawValue: String?) throws -> Int {
        guard let rawValue = normalizedOptional(rawValue) else {
            return 448
        }
        guard let value = Int(rawValue), (1...maxTranscriptionTokens).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "max_tokens",
                "must be between 1 and \(maxTranscriptionTokens)"
            )
        }
        return value
    }

    private static func imageInferenceSteps(_ value: Int?) throws -> Int? {
        guard let value else { return nil }
        guard (1...maxImageInferenceSteps).contains(value) else {
            throw APIRequestValidationError.invalidField(
                "steps",
                "must be between 1 and \(maxImageInferenceSteps)"
            )
        }
        return value
    }

    private static func optionalUInt64Field(_ rawValue: String?) throws -> UInt64? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = UInt64(rawValue) else {
            throw APIRequestValidationError.invalidField("seed", "must be an unsigned integer")
        }
        return value
    }

    private static func optionalPositiveIntField(_ rawValue: String?, field: String) throws -> Int? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Int(rawValue), value > 0 else {
            throw APIRequestValidationError.invalidField(field, "must be greater than zero")
        }
        return value
    }

    private static func optionalPositiveDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, value >= 0 else {
            throw APIRequestValidationError.invalidField(field, "must be a positive number")
        }
        return value
    }

    private static func optionalUnitDoubleField(_ rawValue: String?, field: String) throws -> Double? {
        guard let rawValue = normalizedOptional(rawValue) else {
            return nil
        }
        guard let value = Double(rawValue), value.isFinite, (0...1).contains(value) else {
            throw APIRequestValidationError.invalidField(field, "must be between 0 and 1")
        }
        return value
    }

    private static func normalizedModelID(_ rawValue: String?, defaultID: String) -> String {
        normalizedOptional(rawValue) ?? defaultID
    }

    private static func normalizedImageModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultImageModelID)
        switch modelID.lowercased() {
        case "gpt-image-1", "dall-e-3", "dall-e-2":
            return defaultImageModelID
        default:
            return modelID
        }
    }

    private static func normalizedSpeechModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultSpeechModelID)
        switch modelID.lowercased() {
        case "tts-1", "tts-1-hd", "gpt-4o-mini-tts":
            return defaultSpeechModelID
        default:
            return modelID
        }
    }

    private static func normalizedTranscriptionModelID(_ rawValue: String?) -> String {
        let modelID = normalizedModelID(rawValue, defaultID: defaultTranscriptionModelID)
        switch modelID.lowercased() {
        case "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe":
            return defaultTranscriptionModelID
        default:
            return modelID
        }
    }

    private static func normalizedOptional(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private static func isCompanionModelInstalled(
        _ spec: ManagedModelSpec,
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(spec.id)
        }
        return spec.managedRuntimeURL(fileManager: fileManager) != nil
    }

    private static func isQwenImageEditInstalled(
        fileManager: FileManager,
        installedModelIDs: Set<String>?
    ) -> Bool {
        if let installedModelIDs {
            return installedModelIDs.contains(QwenImageEditRepository.modelId)
        }
        return QwenImageEditRepository.resolveInstalledModelRoot(fileManager: fileManager) != nil
    }

    private static func voiceDescription(for rawVoice: String?, instructions: String?) -> String {
        let instructionText = normalizedOptional(instructions)
        let voice = normalizedOptional(rawVoice)?.lowercased() ?? "nova"
        let base: String
        switch voice {
        case "alloy":
            base = "A balanced, natural voice with clear pronunciation"
        case "ash":
            base = "A calm, low voice with a steady delivery"
        case "ballad":
            base = "A warm, expressive voice with a storytelling cadence"
        case "coral":
            base = "A bright, friendly voice with gentle energy"
        case "echo":
            base = "A clear male voice with an even, conversational tone"
        case "fable":
            base = "A warm narrative voice with a measured pace"
        case "nova":
            base = TTSRequest.defaultVoiceDescription
        case "onyx":
            base = "A deep, confident voice with crisp articulation"
        case "sage":
            base = "A thoughtful, composed voice with soft emphasis"
        case "shimmer":
            base = "A bright, gentle voice with smooth pronunciation"
        default:
            base = rawVoice?.trimmingCharacters(in: .whitespacesAndNewlines)
                ?? TTSRequest.defaultVoiceDescription
        }
        if let instructionText {
            return "\(base). \(instructionText)"
        }
        return base
    }

    private static func transcriptionSegments(
        from result: ASRResult
    ) -> [OpenAIAudioTranscriptionSegment]? {
        if let sentences = result.sentenceAlignments, !sentences.isEmpty {
            return sentences.enumerated().map { index, sentence in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: sentence.startSeconds,
                    end: sentence.endSeconds,
                    text: sentence.text
                )
            }
        }
        if let tokens = result.tokenAlignments, !tokens.isEmpty {
            return tokens.enumerated().map { index, token in
                OpenAIAudioTranscriptionSegment(
                    id: index,
                    start: token.startSeconds,
                    end: token.endSeconds,
                    text: token.text
                )
            }
        }
        return nil
    }

    private static func subtitleTimestamp(_ seconds: Double, separator: String) -> String {
        let milliseconds = max(0, Int((seconds * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds % 3_600_000) / 60_000
        let secs = (milliseconds % 60_000) / 1_000
        let millis = milliseconds % 1_000
        return String(format: "%02d:%02d:%02d%@%03d", hours, minutes, secs, separator, millis)
    }

    private static func decodeJSONRequest<Request: Decodable>(
        _ type: Request.Type,
        from data: Data
    ) throws -> Request {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw APIRequestValidationError.invalidPayload
        }
    }
}

enum APIRequestValidationError: LocalizedError, Equatable {
    case invalidPayload
    case invalidField(String, String)

    var errorDescription: String? {
        switch self {
        case .invalidPayload:
            return "Invalid request payload."
        case .invalidField(let field, let reason):
            return "Invalid '\(field)': \(reason)."
        }
    }
}

struct MultipartFormData: Equatable, Sendable {
    struct Part: Equatable, Sendable {
        let name: String
        let filename: String?
        let contentType: String?
        let body: Data
    }

    enum ParseError: LocalizedError, Equatable {
        case missingBoundary
        case malformedBody
        case missingName

        var errorDescription: String? {
            switch self {
            case .missingBoundary:
                return "Missing multipart boundary."
            case .malformedBody:
                return "Malformed multipart body."
            case .missingName:
                return "Multipart part is missing a form-data name."
            }
        }
    }

    let parts: [Part]

    static func parse(body: Data, boundary: String?) throws -> MultipartFormData {
        guard let boundary, !boundary.isEmpty else {
            throw ParseError.missingBoundary
        }
        let marker = Data("--\(boundary)".utf8)
        guard !marker.isEmpty,
              let firstMarker = body.range(of: marker, options: [], in: body.startIndex..<body.endIndex) else {
            throw ParseError.malformedBody
        }

        var parts: [Part] = []
        var markerRange = firstMarker
        while true {
            let afterMarker = markerRange.upperBound
            if body.hasBytes(Data("--".utf8), at: afterMarker) {
                break
            }
            let partStart = body.indexAfterLineBreak(at: afterMarker)
            guard let nextMarker = body.range(of: marker, options: [], in: partStart..<body.endIndex) else {
                throw ParseError.malformedBody
            }
            let partEnd = body.indexTrimmingLineBreak(before: nextMarker.lowerBound)
            if partStart < partEnd {
                parts.append(try parsePart(Data(body[partStart..<partEnd])))
            }
            markerRange = nextMarker
        }

        return MultipartFormData(parts: parts)
    }

    func field(_ name: String) -> String? {
        guard let part = parts.first(where: { $0.name == name && $0.filename == nil }) else {
            return nil
        }
        return String(data: part.body, encoding: .utf8)
    }

    func file(named name: String) -> Part? {
        parts.first { $0.name == name && $0.filename != nil }
    }

    func files(named name: String) -> [Part] {
        parts.filter { $0.name == name && $0.filename != nil }
    }

    private static func parsePart(_ data: Data) throws -> Part {
        let separator = Data("\r\n\r\n".utf8)
        let fallbackSeparator = Data("\n\n".utf8)
        let separatorRange = data.range(of: separator, options: [], in: data.startIndex..<data.endIndex)
            ?? data.range(of: fallbackSeparator, options: [], in: data.startIndex..<data.endIndex)
        guard let separatorRange,
              let headerText = String(data: data[data.startIndex..<separatorRange.lowerBound], encoding: .utf8) else {
            throw ParseError.malformedBody
        }
        let body = Data(data[separatorRange.upperBound..<data.endIndex])
        let headers = parseHeaders(headerText)
        guard let disposition = headers["content-disposition"] else {
            throw ParseError.missingName
        }
        let params = parseDispositionParameters(disposition)
        guard let name = params["name"], !name.isEmpty else {
            throw ParseError.missingName
        }
        return Part(
            name: name,
            filename: params["filename"],
            contentType: headers["content-type"],
            body: body
        )
    }

    private static func parseHeaders(_ text: String) -> [String: String] {
        var headers: [String: String] = [:]
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
            headers[name] = value
        }
        return headers
    }

    private static func parseDispositionParameters(_ value: String) -> [String: String] {
        var params: [String: String] = [:]
        for part in value.split(separator: ";", omittingEmptySubsequences: false).dropFirst() {
            let pair = part.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            var rawValue = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            if rawValue.hasPrefix("\""), rawValue.hasSuffix("\""), rawValue.count >= 2 {
                rawValue.removeFirst()
                rawValue.removeLast()
            }
            params[key] = rawValue
        }
        return params
    }
}

private extension Data {
    func hasBytes(_ bytes: Data, at index: Data.Index) -> Bool {
        guard index >= startIndex, index + bytes.count <= endIndex else {
            return false
        }
        return self[index..<(index + bytes.count)].elementsEqual(bytes)
    }

    func indexAfterLineBreak(at index: Data.Index) -> Data.Index {
        if hasBytes(Data("\r\n".utf8), at: index) {
            return index + 2
        }
        if hasBytes(Data("\n".utf8), at: index) {
            return index + 1
        }
        return index
    }

    func indexTrimmingLineBreak(before index: Data.Index) -> Data.Index {
        if index >= 2, self[(index - 2)..<index].elementsEqual(Data("\r\n".utf8)) {
            return index - 2
        }
        if index >= 1, self[(index - 1)..<index].elementsEqual(Data("\n".utf8)) {
            return index - 1
        }
        return index
    }
}
