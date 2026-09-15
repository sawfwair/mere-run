import Foundation
import MereRunCore

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

extension APIServerContract {
    static let defaultGeometryModelID = MoGe2GenerationRequest.defaultModelID
    static let geometryRoutePath = "/v1/vision/geometry"

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
}
