import Foundation

public struct InstantMeshCheckpointManifest: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let format: InstantMeshCheckpointFormat
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let sourceSHA256: String
    public let configurationSHA256: String
    public let sourceManifestSHA256: String
    public let viewGenerationIncluded: Bool

    public init(checkpoint: InstantMeshCheckpoint) {
        self.modelID = checkpoint.modelID
        self.repository = checkpoint.repository
        self.revision = checkpoint.revision
        self.sourceRepository = checkpoint.sourceRepository
        self.sourceRevision = checkpoint.sourceRevision
        self.license = checkpoint.license
        self.format = checkpoint.format
        self.weightsByteCount = checkpoint.weightsByteCount
        self.weightsSHA256 = checkpoint.weightsSHA256
        self.sourceSHA256 = checkpoint.sourceSHA256
        self.configurationSHA256 = checkpoint.configurationSHA256
        self.sourceManifestSHA256 = checkpoint.sourceManifestSHA256
        self.viewGenerationIncluded = checkpoint.viewGenerationIncluded
    }
}

public struct InstantMeshInputViewManifest: Codable, Equatable, Sendable {
    public let index: Int
    public let path: String
    public let byteCount: Int64
    public let sha256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let preparedWidth: Int
    public let preparedHeight: Int

    public init(
        index: Int,
        path: String,
        byteCount: Int64,
        sha256: String,
        sourceWidth: Int,
        sourceHeight: Int,
        preparedWidth: Int,
        preparedHeight: Int
    ) {
        self.index = index
        self.path = path
        self.byteCount = byteCount
        self.sha256 = sha256
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.preparedWidth = preparedWidth
        self.preparedHeight = preparedHeight
    }
}

public struct InstantMeshInputManifest: Codable, Equatable, Sendable {
    public let viewCount: Int
    public let userSuppliedViews: Bool
    public let cameraConditioning: String
    /// Exact row-major C2W(3x4) + normalized `(fx, fy, cx, cy)` values.
    public let cameras: [[Float]]
    public let orderedViews: [InstantMeshInputViewManifest]
}

public struct InstantMeshBoundaryManifest: Codable, Equatable, Sendable {
    public let viewGenerationIncluded: Bool
    public let zero123PlusPlusIncluded: Bool
    public let runtimePython: Bool
    public let proprietaryFlexiCubesIncluded: Bool
}

public struct InstantMeshExtractionManifest: Codable, Equatable, Sendable {
    public let resolution: Int
    public let includesVertexColors: Bool
    public let algorithm: String
    public let topologyCompatibility: String
    /// Matches the pinned upstream empty-field repair before topology extraction.
    public let upstreamEmptyFieldRepairApplied: Bool
}

public struct InstantMeshMeshSummaryManifest: Codable, Equatable, Sendable {
    public let coordinateSystem: MeshCoordinateSystem
    public let units: MeshUnits
    public let inferredUnseenGeometry: Bool
    public let vertexCount: Int
    public let triangleCount: Int
    public let bounds: MeshBounds
}

public struct InstantMeshRunArtifact: Codable, Equatable, Sendable {
    public let kind: String
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String
}

public struct InstantMeshRunManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let outputDirectory: String
    public let checkpoint: InstantMeshCheckpointManifest
    public let input: InstantMeshInputManifest
    public let boundary: InstantMeshBoundaryManifest
    public let extraction: InstantMeshExtractionManifest
    public let mesh: InstantMeshMeshSummaryManifest
    public let artifacts: [InstantMeshRunArtifact]
}

public struct InstantMeshRunManifestExport: Equatable, Sendable {
    public let manifest: InstantMeshRunManifest
    public let manifestURL: URL
}

public enum InstantMeshRunManifestExporter {
    public static let extractionAlgorithm = "native-marching-tetrahedra"
    public static let topologyCompatibility =
        "learned-field-parity-no-topology-parity-with-upstream-flexicubes"
    public static let suppliedCameraConditioning = "supplied-c2w-intrinsics"
    public static let officialCameraConditioning = "official-deterministic-conditioning-rig"

    public static func export(
        meshExport: MeshExportResult,
        checkpoint: InstantMeshCheckpoint,
        inputURLs: [URL],
        sourceDimensions: [InstantMeshSourceDimensions],
        preparedDimensions: [InstantMeshSourceDimensions],
        cameraValues: [[Float]],
        usedOfficialCameraRig: Bool,
        extractionResolution: Int,
        includesVertexColors: Bool,
        upstreamEmptyFieldRepairApplied: Bool
    ) throws -> InstantMeshRunManifestExport {
        guard inputURLs.count == sourceDimensions.count,
              inputURLs.count == preparedDimensions.count,
              inputURLs.count == cameraValues.count,
              cameraValues.allSatisfy({ $0.count == 16 && $0.allSatisfy(\.isFinite) }) else {
            throw InstantMeshPreprocessingError.cameraCountMismatch(
                expected: inputURLs.count,
                actual: cameraValues.count
            )
        }
        let mesh = meshExport.manifest
        let root = URL(fileURLWithPath: mesh.outputDirectory, isDirectory: true)
        var artifacts = mesh.artifacts.map {
            InstantMeshRunArtifact(
                kind: $0.kind.rawValue,
                relativePath: $0.relativePath,
                mediaType: $0.mediaType,
                byteCount: $0.byteCount,
                sha256: $0.sha256
            )
        }
        artifacts.append(InstantMeshRunArtifact(
            kind: "mesh-manifest",
            relativePath: meshExport.manifestURL.lastPathComponent,
            mediaType: "application/json",
            byteCount: try ModelArtifactPin.fileByteCount(meshExport.manifestURL),
            sha256: try ModelArtifactPin.fileSHA256(meshExport.manifestURL)
        ))
        artifacts.sort { lhs, rhs in
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.relativePath < rhs.relativePath
        }

        let orderedViews = try inputURLs.indices.map { index in
            let inputRecord = try mesh.inputRecord(for: inputURLs[index])
            return InstantMeshInputViewManifest(
                index: index,
                path: inputRecord.path,
                byteCount: inputRecord.byteCount,
                sha256: inputRecord.sha256,
                sourceWidth: sourceDimensions[index].width,
                sourceHeight: sourceDimensions[index].height,
                preparedWidth: preparedDimensions[index].width,
                preparedHeight: preparedDimensions[index].height
            )
        }
        let manifest = InstantMeshRunManifest(
            schemaVersion: 1,
            createdAt: mesh.createdAt,
            outputDirectory: root.path,
            checkpoint: InstantMeshCheckpointManifest(checkpoint: checkpoint),
            input: InstantMeshInputManifest(
                viewCount: inputURLs.count,
                userSuppliedViews: true,
                cameraConditioning: usedOfficialCameraRig
                    ? officialCameraConditioning
                    : suppliedCameraConditioning,
                cameras: cameraValues,
                orderedViews: orderedViews
            ),
            boundary: InstantMeshBoundaryManifest(
                viewGenerationIncluded: false,
                zero123PlusPlusIncluded: false,
                runtimePython: false,
                proprietaryFlexiCubesIncluded: false
            ),
            extraction: InstantMeshExtractionManifest(
                resolution: extractionResolution,
                includesVertexColors: includesVertexColors,
                algorithm: extractionAlgorithm,
                topologyCompatibility: topologyCompatibility,
                upstreamEmptyFieldRepairApplied: upstreamEmptyFieldRepairApplied
            ),
            mesh: InstantMeshMeshSummaryManifest(
                coordinateSystem: mesh.coordinateSystem,
                units: mesh.units,
                inferredUnseenGeometry: mesh.inferredUnseenGeometry,
                vertexCount: mesh.vertexCount,
                triangleCount: mesh.triangleCount,
                bounds: mesh.bounds
            ),
            artifacts: artifacts
        )
        let manifestURL = root.appendingPathComponent(
            runManifestFilename(meshManifestFilename: meshExport.manifestURL.lastPathComponent)
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return InstantMeshRunManifestExport(manifest: manifest, manifestURL: manifestURL)
    }

    private static func runManifestFilename(meshManifestFilename: String) -> String {
        let suffix = "-manifest.json"
        guard meshManifestFilename.hasSuffix(suffix) else {
            return "instantmesh-run-manifest.json"
        }
        return String(meshManifestFilename.dropLast(suffix.count)) + "-run-manifest.json"
    }
}
