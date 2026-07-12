import Foundation

/// Writes reconstruction-only InstantMesh meshes through the canonical native
/// Asset3D contract. The exact runtime safetensors hash is retained in model
/// provenance; input views and every output artifact are independently hashed.
public enum InstantMeshAssetExporter {
    @discardableResult
    public static func export(
        mesh: MeshAsset,
        inputURLs: [URL],
        checkpoint: InstantMeshCheckpoint,
        outputDirectory: URL,
        stem: String = "instantmesh-asset",
        inputRecords: [MeshInputRecord]? = nil,
        createdAt: Date = Date()
    ) throws -> MeshExportResult {
        try MeshArtifactExporter.export(
            mesh: mesh,
            inputURLs: inputURLs.map(\.standardizedFileURL),
            outputDirectory: outputDirectory,
            stem: stem,
            provenance: GeometryModelProvenance(
                modelID: checkpoint.modelID,
                upstreamRepository: checkpoint.repository,
                upstreamRevision: checkpoint.revision,
                license: checkpoint.license,
                weightsSHA256: checkpoint.weightsSHA256
            ),
            inputRecords: inputRecords,
            createdAt: createdAt
        )
    }
}
