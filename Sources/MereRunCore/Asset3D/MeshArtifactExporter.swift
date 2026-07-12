import Foundation

public enum MeshArtifactKind: String, Codable, CaseIterable, Sendable {
    case obj
    case ply
    case glb
    case texture
}

public struct MeshArtifactRecord: Codable, Equatable, Sendable {
    public let kind: MeshArtifactKind
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String

    public init(kind: MeshArtifactKind, relativePath: String, mediaType: String, byteCount: Int64, sha256: String) {
        self.kind = kind
        self.relativePath = relativePath
        self.mediaType = mediaType
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public struct MeshOutputManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let inputPaths: [String]
    public let outputDirectory: String
    public let model: GeometryModelProvenance
    public let coordinateSystem: MeshCoordinateSystem
    public let units: MeshUnits
    public let inferredUnseenGeometry: Bool
    public let vertexCount: Int
    public let triangleCount: Int
    public let bounds: MeshBounds
    public let artifacts: [MeshArtifactRecord]

    public init(
        schemaVersion: Int = 1,
        createdAt: Date = Date(),
        inputPaths: [String],
        outputDirectory: String,
        model: GeometryModelProvenance,
        coordinateSystem: MeshCoordinateSystem,
        units: MeshUnits,
        inferredUnseenGeometry: Bool,
        vertexCount: Int,
        triangleCount: Int,
        bounds: MeshBounds,
        artifacts: [MeshArtifactRecord]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.inputPaths = inputPaths
        self.outputDirectory = outputDirectory
        self.model = model
        self.coordinateSystem = coordinateSystem
        self.units = units
        self.inferredUnseenGeometry = inferredUnseenGeometry
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
        self.bounds = bounds
        self.artifacts = artifacts
    }
}

public struct MeshExportResult: Equatable, Sendable {
    public let manifest: MeshOutputManifest
    public let manifestURL: URL
}

public enum MeshArtifactExporter {
    @discardableResult
    public static func export(
        mesh: MeshAsset,
        inputURLs: [URL],
        outputDirectory: URL,
        stem: String = "asset",
        provenance: GeometryModelProvenance,
        createdAt: Date = Date()
    ) throws -> MeshExportResult {
        let root = outputDirectory.standardizedFileURL
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let cleanStem = sanitizedStem(stem)
        let outputs: [(MeshArtifactKind, String, String, (MeshAsset, URL) throws -> Void)] = [
            (.obj, "obj", "model/obj", MeshOBJWriter.write),
            (.ply, "ply", "application/ply", MeshPLYWriter.write),
            (.glb, "glb", "model/gltf-binary", MeshGLBWriter.write),
        ]
        var artifacts: [MeshArtifactRecord] = []
        for (kind, extensionName, mediaType, writer) in outputs {
            let url = root.appendingPathComponent("\(cleanStem).\(extensionName)")
            try writer(mesh, url)
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            artifacts.append(MeshArtifactRecord(
                kind: kind,
                relativePath: url.lastPathComponent,
                mediaType: mediaType,
                byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0,
                sha256: try ModelArtifactPin.fileSHA256(url)
            ))
        }
        let manifest = MeshOutputManifest(
            createdAt: createdAt,
            inputPaths: inputURLs.map { $0.standardizedFileURL.path },
            outputDirectory: root.path,
            model: provenance,
            coordinateSystem: mesh.coordinateSystem,
            units: mesh.units,
            inferredUnseenGeometry: mesh.inferredUnseenGeometry,
            vertexCount: mesh.vertexCount,
            triangleCount: mesh.triangleCount,
            bounds: mesh.bounds,
            artifacts: artifacts
        )
        let manifestURL = root.appendingPathComponent("\(cleanStem)-manifest.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return MeshExportResult(manifest: manifest, manifestURL: manifestURL)
    }

    private static func sanitizedStem(_ raw: String) -> String {
        let value = raw
            .replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
        return value.isEmpty ? "asset" : value
    }
}
