import Foundation

// What a 3D run's manifests say about the mesh it wrote, read by name from the files the CLI
// leaves in the output folder: the run manifest (`<stem>-run-manifest.json`,
// `<stem>-trellis2-run-manifest.json`) carries the counts under `mesh`, and the shared mesh
// manifest (`<stem>-manifest.json`, `MeshOutputManifest`) carries them at the top level. TRELLIS.2
// alone reports its sparse PBR voxels.

/// The vertex, triangle, and (TRELLIS.2) PBR voxel counts of a reconstructed mesh, for the one
/// line a 3D result card shows under its tile.
package struct StudioMeshSummary: Decodable, Equatable {
    package let vertexCount: Int
    package let triangleCount: Int
    package let pbrVoxelCount: Int?

    package init(vertexCount: Int, triangleCount: Int, pbrVoxelCount: Int? = nil) {
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
        self.pbrVoxelCount = pbrVoxelCount
    }

    /// "12,480 vertices · 24,956 triangles · 1,203,776 PBR voxels".
    package var text: String {
        var parts = ["\(vertexCount.formatted()) vertices", "\(triangleCount.formatted()) triangles"]
        if let pbrVoxelCount { parts.append("\(pbrVoxelCount.formatted()) PBR voxels") }
        return parts.joined(separator: " · ")
    }

    /// The counts in one manifest file: a run manifest's `mesh` object, else a mesh manifest's
    /// own top level. Nil for any other JSON.
    package static func decode(_ data: Data) -> StudioMeshSummary? {
        let decoder = JSONDecoder()
        if let run = try? decoder.decode(RunManifest.self, from: data) { return run.mesh }
        return try? decoder.decode(StudioMeshSummary.self, from: data)
    }

    /// The summary of a Library row's mesh: read from its run manifest first (the only one that
    /// counts PBR voxels), else from the shared mesh manifest; nil for a row without one.
    package static func load(item: StudioLibraryItem) -> StudioMeshSummary? {
        let manifests = item.allArtifactURLs.filter { $0.lastPathComponent.lowercased().hasSuffix("-manifest.json") }
        let runManifests = manifests.filter { $0.lastPathComponent.lowercased().hasSuffix("-run-manifest.json") }
        return (runManifests + manifests.filter { !runManifests.contains($0) }).lazy
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap(decode)
            .first
    }

    private struct RunManifest: Decodable {
        let mesh: StudioMeshSummary
    }
}
