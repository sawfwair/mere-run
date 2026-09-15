import Foundation
import MereRunCore

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

extension APIServerContract {
    static let defaultMultiViewGeometryModelID = ModelResolver.ModelID.visionGeometryDA3Small.rawValue
    static let multiViewGeometryRoutePath = "/v1/vision/geometry/multiview"
    static let maximumMultiViewGeometryUploadByteCount = 512 * 1024 * 1024

    struct MultiViewGeometryPlan: Equatable, Sendable {
        let modelID: String
        let settings: DepthAnything3GenerationSettings

        var processResolution: Int { settings.processResolution }
        var referenceViewStrategy: DepthAnything3ReferenceViewStrategy { settings.referenceViewStrategy }
        var confidencePercentile: Double { settings.export.confidencePercentile }
        var maximumPointCount: Int { settings.export.maximumPointCount }
        var knownCameras: [DepthAnything3KnownCamera]? { settings.knownCameras }
        var poseConditioned: Bool { knownCameras != nil }

        func request(imageURLs: [URL], outputDirectory: URL) -> DepthAnything3GenerationRequest {
            DepthAnything3GenerationRequest(
                imageURLs: imageURLs, outputDirectory: outputDirectory, model: modelID, settings: settings
            )
        }
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
        try form.validateFields(
            textFields: allowedTextFields,
            fileFields: ["image", "image[]", "cameras"],
            unsupportedTextMessage: "unsupported field; client input, output, model, and camera filesystem paths are not accepted",
            unsupportedFileMessage: "unsupported file part; only uploaded image/image[] and cameras JSON files are accepted"
        )

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
        ) ?? DepthAnything3GenerationSettings.defaultProcessResolution
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
        ) ?? MultiViewGeometryExportConfiguration.defaultMaximumPointCount

        let confidencePercentile: Double
        if let raw = normalizedOptional(form.field("confidence_percentile")) {
            guard let value = Double(raw) else {
                throw APIRequestValidationError.invalidField(
                    "confidence_percentile",
                    "must be a finite number between 0 and 100"
                )
            }
            confidencePercentile = value
        } else {
            confidencePercentile = MultiViewGeometryExportConfiguration.defaultConfidencePercentile
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

        do {
            let settings = try DepthAnything3GenerationSettings(
                processResolution: processResolution, referenceViewStrategy: referenceViewStrategy,
                knownCameras: knownCameras, confidencePercentile: confidencePercentile,
                maximumPointCount: maximumPointCount
            )
            return MultiViewGeometryPlan(modelID: modelID, settings: settings)
        } catch MultiViewGeometryExportConfigurationError.invalidConfidencePercentile {
            throw APIRequestValidationError.invalidField(
                "confidence_percentile", "must be a finite number between 0 and 100"
            )
        }
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
}
