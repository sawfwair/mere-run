import Foundation

/// Writes TripoSR meshes through the canonical native Asset3D contract.
/// OBJ, binary PLY, and GLB share the same indexed geometry, generated normals,
/// vertex colors, normalized-object-space declaration, hashes, and provenance.
public enum TripoSRAssetExporter {
    @discardableResult
    public static func export(
        mesh: MeshAsset,
        inputURL: URL,
        checkpoint: TripoSRCheckpoint,
        outputDirectory: URL,
        stem: String = "triposr-asset",
        inputRecord: MeshInputRecord? = nil,
        createdAt: Date = Date()
    ) throws -> MeshExportResult {
        return try MeshArtifactExporter.export(
            mesh: mesh,
            inputURLs: [inputURL.standardizedFileURL],
            outputDirectory: outputDirectory,
            stem: stem,
            provenance: GeometryModelProvenance(
                modelID: checkpoint.modelID,
                upstreamRepository: checkpoint.repository,
                upstreamRevision: checkpoint.revision,
                license: checkpoint.license,
                weightsSHA256: checkpoint.weightsSHA256
            ),
            inputRecords: inputRecord.map { [$0] },
            createdAt: createdAt
        )
    }
}
