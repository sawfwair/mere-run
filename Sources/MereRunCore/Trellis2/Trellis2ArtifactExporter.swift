import Foundation

public struct Trellis2PBRVoxelExport: Codable, Equatable, Sendable {
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String

    public init(relativePath: String, mediaType: String, byteCount: Int64, sha256: String) {
        self.relativePath = relativePath
        self.mediaType = mediaType
        self.byteCount = byteCount
        self.sha256 = sha256
    }
}

/// Deterministic little-endian storage for TRELLIS.2's six-channel sparse PBR
/// field. OBJ/PLY/GLB retain sampled RGBA vertex color; this sidecar preserves
/// base color, metallic, roughness, and alpha at every decoded O-Voxel.
public enum Trellis2PBRVoxelWriter {
    public static let mediaType = "application/vnd.mere-run.trellis2-pbrvox"

    @discardableResult
    public static func write(_ grid: Trellis2PBRVoxelGrid, to url: URL) throws -> Trellis2PBRVoxelExport {
        var data = Data("MRPBRV01".utf8)
        append(UInt32(1), to: &data)
        append(UInt32(grid.resolution), to: &data)
        append(UInt32(grid.coordinates.count), to: &data)
        append(UInt32(6), to: &data)
        for (index, coordinate) in grid.coordinates.enumerated() {
            append(coordinate.x, to: &data)
            append(coordinate.y, to: &data)
            append(coordinate.z, to: &data)
            for channel in 0..<6 {
                append(grid.attributes[index * 6 + channel], to: &data)
            }
        }
        try data.write(to: url, options: .atomic)
        return Trellis2PBRVoxelExport(
            relativePath: url.lastPathComponent,
            mediaType: mediaType,
            byteCount: Int64(data.count),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func append(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private static func append(_ value: Int32, to data: inout Data) {
        append(UInt32(bitPattern: value), to: &data)
    }

    private static func append(_ value: Float, to data: inout Data) {
        append(value.bitPattern, to: &data)
    }
}

public struct Trellis2CheckpointComponentManifest: Codable, Equatable, Sendable {
    public let name: String
    public let repository: String
    public let revision: String
    public let relativePath: String
    public let byteCount: Int64
    public let sha256: String
    public let license: String
}

public struct Trellis2InputManifest: Codable, Equatable, Sendable {
    public let path: String
    public let byteCount: Int64
    public let sha256: String
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let conditioningResolution: Int
    public let foregroundPolicy: String
    public let croppedTransparentForeground: Bool
}

public struct Trellis2GenerationManifest: Codable, Equatable, Sendable {
    public let seed: UInt64
    public let pipelineResolution: Int
    public let maximumSparseTokens: Int
    public let sparseStructureSteps: Int
    public let shapeSteps: Int
    public let textureSteps: Int
    public let extractionAlgorithm: String
    public let pbrRepresentation: String
}

public struct Trellis2MeshManifest: Codable, Equatable, Sendable {
    public let coordinateSystem: MeshCoordinateSystem
    public let units: MeshUnits
    public let inferredUnseenGeometry: Bool
    public let vertexCount: Int
    public let triangleCount: Int
    public let bounds: MeshBounds
    public let pbrVoxelCount: Int
    public let includesVertexColors: Bool
    public let includesMetallicRoughnessSidecar: Bool
}

public struct Trellis2RunArtifact: Codable, Equatable, Sendable {
    public let kind: String
    public let relativePath: String
    public let mediaType: String
    public let byteCount: Int64
    public let sha256: String
}

public struct Trellis2RunManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let outputDirectory: String
    public let modelID: String
    public let repository: String
    public let revision: String
    public let license: String
    public let inferenceBackend: String
    public let checkpointComponents: [Trellis2CheckpointComponentManifest]
    public let input: Trellis2InputManifest
    public let generation: Trellis2GenerationManifest
    public let mesh: Trellis2MeshManifest
    public let artifacts: [Trellis2RunArtifact]
}

public struct Trellis2RunManifestExport: Equatable, Sendable {
    public let manifest: Trellis2RunManifest
    public let manifestURL: URL
}

public enum Trellis2ArtifactExporter {
    public static let extractionAlgorithm = "native-ovoxel-flexible-dual-grid-small-hole-fill"
    public static let pbrRepresentation = "vertex-rgba-plus-sparse-pbrvox"

    public static func export(
        asset: Trellis2DecodedAsset,
        checkpoint: Trellis2Checkpoint,
        inputURL: URL,
        inputRecord: MeshInputRecord,
        outputDirectory: URL,
        stem: String,
        sourceWidth: Int,
        sourceHeight: Int,
        foregroundPolicy: String,
        croppedTransparentForeground: Bool,
        seed: UInt64,
        maximumSparseTokens: Int,
        createdAt: Date = Date()
    ) throws -> (mesh: MeshExportResult, pbr: Trellis2PBRVoxelExport, run: Trellis2RunManifestExport) {
        let root = outputDirectory.standardizedFileURL
        let mesh = try MeshArtifactExporter.export(
            mesh: asset.mesh,
            inputURLs: [inputURL.standardizedFileURL],
            outputDirectory: root,
            stem: stem,
            provenance: GeometryModelProvenance(
                modelID: checkpoint.modelID,
                upstreamRepository: Trellis2Resources.repository,
                upstreamRevision: Trellis2Resources.revision,
                license: "MIT",
                weightsSHA256: Trellis2Resources.primaryWeightsSHA256
            ),
            inputRecords: [inputRecord],
            material: medianMaterialFactors(
                metallic: asset.metallic,
                roughness: asset.roughness
            ),
            createdAt: createdAt
        )
        let cleanStem = mesh.manifestURL.lastPathComponent
            .replacingOccurrences(of: "-manifest.json", with: "")
        let pbr = try Trellis2PBRVoxelWriter.write(
            asset.pbrVoxels,
            to: root.appendingPathComponent("\(cleanStem).pbrvox")
        )
        let input = try mesh.manifest.inputRecord(for: inputURL)
        var artifacts = mesh.manifest.artifacts.map {
            Trellis2RunArtifact(
                kind: $0.kind.rawValue,
                relativePath: $0.relativePath,
                mediaType: $0.mediaType,
                byteCount: $0.byteCount,
                sha256: $0.sha256
            )
        }
        artifacts.append(Trellis2RunArtifact(
            kind: "pbr-voxels",
            relativePath: pbr.relativePath,
            mediaType: pbr.mediaType,
            byteCount: pbr.byteCount,
            sha256: pbr.sha256
        ))
        artifacts.append(try artifact(
            kind: "mesh-manifest",
            url: mesh.manifestURL,
            mediaType: "application/json"
        ))
        artifacts.sort {
            ($0.kind, $0.relativePath) < ($1.kind, $1.relativePath)
        }

        let manifest = Trellis2RunManifest(
            schemaVersion: 1,
            createdAt: createdAt,
            outputDirectory: root.path,
            modelID: checkpoint.modelID,
            repository: Trellis2Resources.repository,
            revision: Trellis2Resources.revision,
            license: "MIT",
            inferenceBackend: "mere.run-native-mlx",
            checkpointComponents: Trellis2Resources.checkpointComponentManifests,
            input: Trellis2InputManifest(
                path: input.path,
                byteCount: input.byteCount,
                sha256: input.sha256,
                sourceWidth: sourceWidth,
                sourceHeight: sourceHeight,
                conditioningResolution: 512,
                foregroundPolicy: foregroundPolicy,
                croppedTransparentForeground: croppedTransparentForeground
            ),
            generation: Trellis2GenerationManifest(
                seed: seed,
                pipelineResolution: 512,
                maximumSparseTokens: maximumSparseTokens,
                sparseStructureSteps: Trellis2SamplerConfiguration.sparseStructure.steps,
                shapeSteps: Trellis2SamplerConfiguration.shape.steps,
                textureSteps: Trellis2SamplerConfiguration.texture.steps,
                extractionAlgorithm: extractionAlgorithm,
                pbrRepresentation: pbrRepresentation
            ),
            mesh: Trellis2MeshManifest(
                coordinateSystem: mesh.manifest.coordinateSystem,
                units: mesh.manifest.units,
                inferredUnseenGeometry: mesh.manifest.inferredUnseenGeometry,
                vertexCount: mesh.manifest.vertexCount,
                triangleCount: mesh.manifest.triangleCount,
                bounds: mesh.manifest.bounds,
                pbrVoxelCount: asset.pbrVoxels.coordinates.count,
                includesVertexColors: asset.mesh.colorsRGBA8 != nil,
                includesMetallicRoughnessSidecar: true
            ),
            artifacts: artifacts
        )
        let manifestURL = root.appendingPathComponent("\(cleanStem)-trellis2-run-manifest.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return (mesh, pbr, Trellis2RunManifestExport(manifest: manifest, manifestURL: manifestURL))
    }

    /// Field-median metallic and roughness as uniform glTF material factors.
    /// The per-vertex values stay in the `.pbrvox` sidecar; the median keeps
    /// the GLB material honest for viewers that only read core glTF.
    static func medianMaterialFactors(
        metallic: [Float],
        roughness: [Float]
    ) -> MeshPBRMaterialFactors? {
        guard let metallicMedian = median(of: metallic),
              let roughnessMedian = median(of: roughness) else { return nil }
        return MeshPBRMaterialFactors(
            metallicFactor: metallicMedian,
            roughnessFactor: roughnessMedian
        )
    }

    private static func median(of values: [Float]) -> Float? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[middle - 1] + sorted[middle]) / 2
            : sorted[middle]
    }

    private static func artifact(kind: String, url: URL, mediaType: String) throws -> Trellis2RunArtifact {
        Trellis2RunArtifact(
            kind: kind,
            relativePath: url.lastPathComponent,
            mediaType: mediaType,
            byteCount: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }
}
