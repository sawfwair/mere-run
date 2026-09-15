import Foundation
import MereRunCore

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

extension APIServerContract {
    static let defaultImageTo3DModelID = ModelResolver.ModelID.image3DTripoSR.rawValue
    static let imageTo3DRoutePath = "/v1/vision/image-to-3d"
    static let maximumImageTo3DUploadByteCount = 100 * 1024 * 1024

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

    static func imageTo3DPlan(from form: MultipartFormData) throws -> ImageTo3DPlan {
        let allowedTextFields: Set<String> = [
            "model",
            "resolution",
            "density_threshold",
            "foreground_ratio",
            "already_framed",
            "vertex_colors",
        ]
        try form.validateFields(
            textFields: allowedTextFields,
            fileFields: ["image"],
            unsupportedTextMessage: "unsupported field; client input, output, and checkpoint paths are not accepted",
            unsupportedFileMessage: "only one uploaded image file is accepted"
        )

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
}
