import Foundation
import Hummingbird
import MereRunCore

struct APIInstantMeshCheckpointResponse: Codable, Equatable, Sendable {
    let repository: String
    let revision: String
    let sourceRepository: String
    let sourceRevision: String
    let license: String
    let format: InstantMeshCheckpointFormat
    let weightsByteCount: Int64
    let weightsSHA256: String
    let sourceSHA256: String
    let configurationSHA256: String
    let sourceManifestSHA256: String
    let viewGenerationIncluded: Bool

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
        case sourceManifestSHA256 = "source_manifest_sha256"
        case viewGenerationIncluded = "view_generation_included"
    }
}

struct APIInstantMeshArtifactResponse: Codable, Equatable, Sendable {
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

struct APIInstantMeshTimingResponse: Codable, Equatable, Sendable {
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

struct APIInstantMeshResponse: Codable, Equatable, Sendable {
    let created: Int
    let object: String
    let status: String
    let model: String
    let checkpoint: APIInstantMeshCheckpointResponse
    let sourceDimensions: [InstantMeshSourceDimensions]
    let viewCount: Int
    let userSuppliedViews: Bool
    let viewGenerationIncluded: Bool
    let usedOfficialCameraRig: Bool
    let zero123PlusPlusIncluded: Bool
    let runtimePython: Bool
    let proprietaryFlexiCubesIncluded: Bool
    let extractionResolution: Int
    let includesVertexColors: Bool
    let meshExtractionAlgorithm: String
    let upstreamEmptyFieldRepairApplied: Bool
    let topologyMatchesUpstreamFlexiCubes: Bool
    let coordinateSystem: MeshCoordinateSystem
    let units: MeshUnits
    let inferredUnseenGeometry: Bool
    let vertexCount: Int
    let triangleCount: Int
    let bounds: MeshBounds
    let manifest: APIInstantMeshArtifactResponse
    let meshManifest: APIInstantMeshArtifactResponse
    let artifacts: [APIInstantMeshArtifactResponse]
    let timing: APIInstantMeshTimingResponse

    enum CodingKeys: String, CodingKey {
        case created
        case object
        case status
        case model
        case checkpoint
        case sourceDimensions = "source_dimensions"
        case viewCount = "view_count"
        case userSuppliedViews = "user_supplied_views"
        case viewGenerationIncluded = "view_generation_included"
        case usedOfficialCameraRig = "used_official_camera_rig"
        case zero123PlusPlusIncluded = "zero123_plus_plus_included"
        case runtimePython = "runtime_python"
        case proprietaryFlexiCubesIncluded = "proprietary_flexicubes_included"
        case extractionResolution = "extraction_resolution"
        case includesVertexColors = "includes_vertex_colors"
        case meshExtractionAlgorithm = "mesh_extraction_algorithm"
        case upstreamEmptyFieldRepairApplied = "upstream_empty_field_repair_applied"
        case topologyMatchesUpstreamFlexiCubes = "topology_matches_upstream_flexicubes"
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
    static var defaultInstantMeshModelID: String {
        ModelResolver.ModelID.image3DInstantMeshBase.rawValue
    }

    static var instantMeshRoutePath: String { "/v1/vision/image-to-3d-multiview" }
    static var instantMeshRouterPath: RouterPath { RouterPath(instantMeshRoutePath) }
    static var maximumInstantMeshUploadByteCount: Int { 512 * 1024 * 1024 }

    struct InstantMeshPlan: Equatable, Sendable {
        let modelID: String
        let extractionResolution: Int
        let includesVertexColors: Bool
        let cameras: [[Float]]?
    }

    static func instantMeshPlan(from form: MultipartFormData) throws -> InstantMeshPlan {
        let allowedTextFields: Set<String> = ["model", "resolution", "vertex_colors", "cameras"]
        for part in form.parts {
            if part.filename != nil {
                guard part.name == "image" || part.name == "image[]" else {
                    throw APIRequestValidationError.invalidField(
                        part.name,
                        "only uploaded image or image[] view files are accepted"
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

        let uploads = form.parts
            .filter { $0.filename != nil && ($0.name == "image" || $0.name == "image[]") }
            .filter { !$0.body.isEmpty }
        guard uploads.count == 4 || uploads.count == 6 else {
            throw APIRequestValidationError.invalidField(
                "image[]",
                "exactly 4 or 6 non-empty uploaded image views are required"
            )
        }
        for upload in uploads {
            if let rawContentType = upload.contentType {
                let contentType = rawContentType
                    .split(separator: ";", maxSplits: 1, omittingEmptySubsequences: true)
                    .first?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased() ?? ""
                guard contentType.hasPrefix("image/") || contentType == "application/octet-stream" else {
                    throw APIRequestValidationError.invalidField(
                        "image[]",
                        "every uploaded view must have an image content type"
                    )
                }
            }
        }

        let rawModel = form.field("model")?.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelID = (rawModel?.isEmpty == false ? rawModel : defaultInstantMeshModelID) ?? defaultInstantMeshModelID
        guard modelID.lowercased() == defaultInstantMeshModelID else {
            throw APIRequestValidationError.invalidField(
                "model",
                "only the managed model id \(defaultInstantMeshModelID) is supported"
            )
        }

        let extractionResolution: Int
        if let raw = form.field("resolution")?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            guard let value = Int(raw), (2...256).contains(value) else {
                throw APIRequestValidationError.invalidField(
                    "resolution",
                    "must be an integer between 2 and 256"
                )
            }
            extractionResolution = value
        } else {
            extractionResolution = InstantMeshConfiguration.production.gridResolution
        }

        let cameras: [[Float]]?
        if let raw = form.field("cameras")?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            guard let data = raw.data(using: .utf8),
                  let document = try? JSONDecoder().decode(InstantMeshCameraDocument.self, from: data),
                  document.schemaVersion == 1,
                  document.cameras.count == uploads.count else {
                throw APIRequestValidationError.invalidField(
                    "cameras",
                    "must be schemaVersion 1 JSON with one 16-value camera per uploaded view"
                )
            }
            for (index, camera) in document.cameras.enumerated()
            where camera.count != 16 || !camera.allSatisfy(\.isFinite) {
                throw APIRequestValidationError.invalidField(
                    "cameras",
                    "camera \(index) must contain 16 finite values"
                )
            }
            cameras = document.cameras
        } else {
            cameras = nil
        }

        return InstantMeshPlan(
            modelID: defaultInstantMeshModelID,
            extractionResolution: extractionResolution,
            includesVertexColors: try instantMeshBoolean(
                form.field("vertex_colors"),
                field: "vertex_colors",
                defaultValue: true
            ),
            cameras: cameras
        )
    }

    static func instantMeshResponse(
        from result: InstantMeshRunResult,
        createdAt: Date = Date()
    ) throws -> APIInstantMeshResponse {
        let mesh = result.export.manifest
        let root = URL(fileURLWithPath: mesh.outputDirectory, isDirectory: true)
        let artifacts = result.runManifest.manifest.artifacts.map { artifact in
            APIInstantMeshArtifactResponse(
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
        let manifest = try instantMeshFileArtifact(
            kind: "manifest",
            url: result.runManifest.manifestURL,
            mediaType: "application/json"
        )
        let meshManifest = try instantMeshFileArtifact(
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
        return APIInstantMeshResponse(
            created: Int(createdAt.timeIntervalSince1970),
            object: "vision.image-to-3d-multiview",
            status: "completed",
            model: result.checkpoint.modelID,
            checkpoint: APIInstantMeshCheckpointResponse(
                repository: result.checkpoint.repository,
                revision: result.checkpoint.revision,
                sourceRepository: result.checkpoint.sourceRepository,
                sourceRevision: result.checkpoint.sourceRevision,
                license: result.checkpoint.license,
                format: result.checkpoint.format,
                weightsByteCount: result.checkpoint.weightsByteCount,
                weightsSHA256: result.checkpoint.weightsSHA256,
                sourceSHA256: result.checkpoint.sourceSHA256,
                configurationSHA256: result.checkpoint.configurationSHA256,
                sourceManifestSHA256: result.checkpoint.sourceManifestSHA256,
                viewGenerationIncluded: false
            ),
            sourceDimensions: result.sourceDimensions,
            viewCount: result.viewCount,
            userSuppliedViews: true,
            viewGenerationIncluded: false,
            usedOfficialCameraRig: result.usedOfficialCameraRig,
            zero123PlusPlusIncluded: false,
            runtimePython: false,
            proprietaryFlexiCubesIncluded: false,
            extractionResolution: result.extractionResolution,
            includesVertexColors: result.includesVertexColors,
            meshExtractionAlgorithm: InstantMeshRunManifestExporter.extractionAlgorithm,
            upstreamEmptyFieldRepairApplied: result.upstreamEmptyFieldRepairApplied,
            topologyMatchesUpstreamFlexiCubes: false,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.triangleCount,
            bounds: mesh.bounds,
            manifest: manifest,
            meshManifest: meshManifest,
            artifacts: artifacts,
            timing: APIInstantMeshTimingResponse(
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

    private static func instantMeshBoolean(
        _ raw: String?,
        field: String,
        defaultValue: Bool
    ) throws -> Bool {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !value.isEmpty else { return defaultValue }
        switch value {
        case "true", "1": return true
        case "false", "0": return false
        default:
            throw APIRequestValidationError.invalidField(field, "must be true, false, 1, or 0")
        }
    }

    private static func instantMeshFileArtifact(
        kind: String,
        url: URL,
        mediaType: String
    ) throws -> APIInstantMeshArtifactResponse {
        APIInstantMeshArtifactResponse(
            kind: kind,
            url: url.absoluteString,
            mediaType: mediaType,
            byteCount: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }
}
