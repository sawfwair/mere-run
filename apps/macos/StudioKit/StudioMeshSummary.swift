import Foundation

// What a 3D run's manifests say about the geometry it wrote, read by name from the files the CLI
// leaves in the output folder: the run manifest (`<stem>-run-manifest.json`,
// `<stem>-trellis2-run-manifest.json`) carries the counts under `mesh`, and the shared mesh
// manifest (`<stem>-manifest.json`, `MeshOutputManifest`) carries them at the top level. TRELLIS.2
// alone reports its sparse PBR voxels. IDOL writes a Gaussian splat rather than a mesh, and its
// `manifest.json` (`IdolRunManifest`) counts the Gaussians at the top level.

/// The vertex, triangle, and (TRELLIS.2) PBR voxel counts of a reconstructed mesh, or the
/// Gaussian count of an IDOL splat, for the one line a 3D result card shows under its tile.
package struct StudioMeshSummary: Equatable {
    package let vertexCount: Int?
    package let triangleCount: Int?
    package let pbrVoxelCount: Int?
    package let gaussianCount: Int?

    package init(vertexCount: Int, triangleCount: Int, pbrVoxelCount: Int? = nil) {
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
        self.pbrVoxelCount = pbrVoxelCount
        gaussianCount = nil
    }

    package init(gaussianCount: Int) {
        vertexCount = nil
        triangleCount = nil
        pbrVoxelCount = nil
        self.gaussianCount = gaussianCount
    }

    /// Whether the run wrote a Gaussian splat rather than a mesh.
    package var isSplat: Bool { gaussianCount != nil }

    /// "12,480 vertices · 24,956 triangles · 1,203,776 PBR voxels", or "262,144 Gaussians".
    package var text: String {
        var parts: [String] = []
        if let vertexCount { parts.append("\(vertexCount.formatted()) vertices") }
        if let triangleCount { parts.append("\(triangleCount.formatted()) triangles") }
        if let pbrVoxelCount { parts.append("\(pbrVoxelCount.formatted()) PBR voxels") }
        if let gaussianCount { parts.append("\(gaussianCount.formatted()) Gaussians") }
        return parts.joined(separator: " · ")
    }

    /// The counts in one manifest file: a run manifest's `mesh` object, else a mesh manifest's
    /// own top level, else an IDOL manifest's Gaussian count. Nil for any other JSON.
    package static func decode(_ data: Data) -> StudioMeshSummary? {
        let decoder = JSONDecoder()
        if let run = try? decoder.decode(RunManifest.self, from: data) { return run.mesh.summary }
        if let mesh = try? decoder.decode(MeshCounts.self, from: data) { return mesh.summary }
        return (try? decoder.decode(SplatManifest.self, from: data)).map { StudioMeshSummary(gaussianCount: $0.gaussianCount) }
    }

    /// The summary of a Library row's geometry: read from its run manifest first (the only one
    /// that counts PBR voxels), else from the shared mesh manifest or IDOL's `manifest.json`; nil
    /// for a row without one. The feed asks on every render, so the reading is kept per row until
    /// the row changes.
    package static func load(item: StudioLibraryItem) -> StudioMeshSummary? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let known = cache[item.id], known.updatedAt == item.updatedAt { return known.summary }
        let summary = read(item: item)
        cache[item.id] = (item.updatedAt, summary)
        return summary
    }

    private static func read(item: StudioLibraryItem) -> StudioMeshSummary? {
        let manifests = item.allArtifactURLs.filter {
            let name = $0.lastPathComponent.lowercased()
            return name == "manifest.json" || name.hasSuffix("-manifest.json")
        }
        let runManifests = manifests.filter { $0.lastPathComponent.lowercased().hasSuffix("-run-manifest.json") }
        return (runManifests + manifests.filter { !runManifests.contains($0) }).lazy
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap(decode)
            .first
    }

    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var cache: [UUID: (updatedAt: Date, summary: StudioMeshSummary?)] = [:]

    private struct RunManifest: Decodable {
        let mesh: MeshCounts
    }

    private struct MeshCounts: Decodable {
        let vertexCount: Int
        let triangleCount: Int
        let pbrVoxelCount: Int?

        var summary: StudioMeshSummary {
            StudioMeshSummary(vertexCount: vertexCount, triangleCount: triangleCount, pbrVoxelCount: pbrVoxelCount)
        }
    }

    /// `IdolRunManifest`'s top-level count of the splats in `object.ply`.
    private struct SplatManifest: Decodable {
        let gaussianCount: Int
    }
}
