import Foundation

public struct TripoSRCheckpointManifest: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let sourceRepository: String
    public let sourceRevision: String
    public let license: String
    public let format: TripoSRCheckpointFormat
    public let weightsByteCount: Int64
    public let weightsSHA256: String
    public let sourceSHA256: String
    public let configurationSHA256: String

    public init(checkpoint: TripoSRCheckpoint) {
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
    }
}

public struct TripoSRInputManifest: Codable, Equatable, Sendable {
    public let path: String
    public let byteCount: Int64
    public let sha256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let preparedWidth: Int
    public let preparedHeight: Int
    public let foregroundPolicy: String
    public let foregroundRatio: Float?
    public let croppedTransparentForeground: Bool
}

public struct TripoSRExtractionManifest: Codable, Equatable, Sendable {
    public let resolution: Int
    public let densityThreshold: Float
    public let includesVertexColors: Bool
    public let algorithm: String
    public let topologyCompatibility: String
}

public struct TripoSRMeshSummaryManifest: Codable, Equatable, Sendable {
    public let coordinateSystem: MeshCoordinateSystem
    public let units: MeshUnits
    public let inferredUnseenGeometry: Bool
    public let vertexCount: Int
    public let triangleCount: Int
    public let bounds: MeshBounds
}

public struct TripoSRRunArtifact: Codable, Equatable, Sendable {
    public let kind: String
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String
}

public struct TripoSRRunManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let outputDirectory: String
    public let checkpoint: TripoSRCheckpointManifest
    public let input: TripoSRInputManifest
    public let extraction: TripoSRExtractionManifest
    public let mesh: TripoSRMeshSummaryManifest
    public let artifacts: [TripoSRRunArtifact]
}

public struct TripoSRRunManifestExport: Equatable, Sendable {
    public let manifest: TripoSRRunManifest
    public let manifestURL: URL
}

public enum TripoSRRunManifestExporter {
    public static let extractionAlgorithm = "native-marching-tetrahedra"
    public static let topologyCompatibility =
        "same-sampled-isosurface-not-byte-topology-parity-with-torchmcubes"

    public static func export(
        meshExport: MeshExportResult,
        checkpoint: TripoSRCheckpoint,
        inputURL: URL,
        sourceWidth: Int,
        sourceHeight: Int,
        preparedWidth: Int,
        preparedHeight: Int,
        foregroundPolicy: String,
        foregroundRatio: Float?,
        croppedTransparentForeground: Bool,
        extractionResolution: Int,
        densityThreshold: Float,
        includesVertexColors: Bool
    ) throws -> TripoSRRunManifestExport {
        let mesh = meshExport.manifest
        let root = URL(fileURLWithPath: mesh.outputDirectory, isDirectory: true)
        let inputRecord = try mesh.inputRecord(for: inputURL)
        var artifacts = mesh.artifacts.map {
            TripoSRRunArtifact(
                kind: $0.kind.rawValue,
                relativePath: $0.relativePath,
                mediaType: $0.mediaType,
                byteCount: $0.byteCount,
                sha256: $0.sha256
            )
        }
        let meshManifestAttributes = try FileManager.default.attributesOfItem(
            atPath: meshExport.manifestURL.path
        )
        artifacts.append(TripoSRRunArtifact(
            kind: "mesh-manifest",
            relativePath: meshExport.manifestURL.lastPathComponent,
            mediaType: "application/json",
            byteCount: (meshManifestAttributes[.size] as? NSNumber)?.int64Value ?? 0,
            sha256: try ModelArtifactPin.fileSHA256(meshExport.manifestURL)
        ))
        artifacts.sort { lhs, rhs in
            if lhs.kind != rhs.kind { return lhs.kind < rhs.kind }
            return lhs.relativePath < rhs.relativePath
        }

        let manifest = TripoSRRunManifest(
            schemaVersion: 1,
            createdAt: mesh.createdAt,
            outputDirectory: root.path,
            checkpoint: TripoSRCheckpointManifest(checkpoint: checkpoint),
            input: TripoSRInputManifest(
                path: inputRecord.path,
                byteCount: inputRecord.byteCount,
                sha256: inputRecord.sha256,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                preparedWidth: preparedWidth,
                preparedHeight: preparedHeight,
                foregroundPolicy: foregroundPolicy,
                foregroundRatio: foregroundRatio,
                croppedTransparentForeground: croppedTransparentForeground
            ),
            extraction: TripoSRExtractionManifest(
                resolution: extractionResolution,
                densityThreshold: densityThreshold,
                includesVertexColors: includesVertexColors,
                algorithm: extractionAlgorithm,
                topologyCompatibility: topologyCompatibility
            ),
            mesh: TripoSRMeshSummaryManifest(
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
        return TripoSRRunManifestExport(manifest: manifest, manifestURL: manifestURL)
    }

    private static func runManifestFilename(meshManifestFilename: String) -> String {
        let suffix = "-manifest.json"
        guard meshManifestFilename.hasSuffix(suffix) else {
            return "triposr-run-manifest.json"
        }
        return String(meshManifestFilename.dropLast(suffix.count)) + "-run-manifest.json"
    }
}
