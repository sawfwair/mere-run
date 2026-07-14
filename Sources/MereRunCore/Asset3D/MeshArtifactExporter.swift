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

public struct MeshInputRecord: Codable, Equatable, Sendable {
    public let path: String
    public let byteCount: Int64
    public let sha256: String

    public init(path: String, byteCount: Int64, sha256: String) {
        self.path = path
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

public enum MeshInputProvenanceError: Error, Equatable, LocalizedError, Sendable {
    case missingInputRecord(String)
    case inputRecordCountMismatch(expected: Int, actual: Int)
    case inputRecordPathMismatch(expected: String, actual: String)

    public var errorDescription: String? {
        switch self {
        case .missingInputRecord(let path):
            "Mesh manifest does not contain durable input provenance for \(path)."
        case .inputRecordCountMismatch(let expected, let actual):
            "Mesh input provenance expected \(expected) records but received \(actual)."
        case .inputRecordPathMismatch(let expected, let actual):
            "Mesh input provenance expected path \(expected) but received \(actual)."
        }
    }
}

public struct MeshOutputManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    /// Retained for readers of the original schema. New writers also populate
    /// `inputs` with durable byte counts and hashes.
    public let inputPaths: [String]
    public let inputs: [MeshInputRecord]?
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
        inputs: [MeshInputRecord]? = nil,
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
        self.inputs = inputs
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

    public func inputRecord(for url: URL) throws -> MeshInputRecord {
        let path = url.standardizedFileURL.path
        guard let record = inputs?.first(where: { $0.path == path }) else {
            throw MeshInputProvenanceError.missingInputRecord(path)
        }
        return record
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
        inputRecords admittedInputRecords: [MeshInputRecord]? = nil,
        material: MeshPBRMaterialFactors? = nil,
        createdAt: Date = Date()
    ) throws -> MeshExportResult {
        let root = outputDirectory.standardizedFileURL
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let standardizedInputs = inputURLs.map(\.standardizedFileURL)
        let inputs: [MeshInputRecord]
        if let admittedInputRecords {
            guard admittedInputRecords.count == standardizedInputs.count else {
                throw MeshInputProvenanceError.inputRecordCountMismatch(
                    expected: standardizedInputs.count,
                    actual: admittedInputRecords.count
                )
            }
            for (url, record) in zip(standardizedInputs, admittedInputRecords)
                where record.path != url.path {
                throw MeshInputProvenanceError.inputRecordPathMismatch(
                    expected: url.path,
                    actual: record.path
                )
            }
            inputs = admittedInputRecords
        } else {
            inputs = try standardizedInputs.map { url in
                MeshInputRecord(
                    path: url.path,
                    byteCount: try ModelArtifactPin.fileByteCount(url),
                    sha256: try ModelArtifactPin.fileSHA256(url)
                )
            }
        }
        let cleanStem = sanitizedStem(stem)
        let outputs: [(MeshArtifactKind, String, String, (MeshAsset, URL) throws -> Void)] = [
            (.obj, "obj", "model/obj", MeshOBJWriter.write),
            (.ply, "ply", "application/ply", MeshPLYWriter.write),
            (.glb, "glb", "model/gltf-binary", { try MeshGLBWriter.write($0, to: $1, material: material) }),
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
            inputPaths: standardizedInputs.map(\.path),
            inputs: inputs,
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
