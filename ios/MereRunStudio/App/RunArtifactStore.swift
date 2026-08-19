import Foundation

struct RunArtifactStore: Sendable {
    enum StoreError: LocalizedError {
        case invalidJobID
        case missingArtifact(String)
        case invalidArtifactPath(String)

        var errorDescription: String? {
            switch self {
            case .invalidJobID:
                "The run identifier cannot be used for local artifact storage."
            case .missingArtifact(let path):
                "The refreshed run is missing artifact \(path)."
            case .invalidArtifactPath(let path):
                "The refreshed run contains an invalid artifact path: \(path)."
            }
        }
    }

    let runsRoot: URL
    private var fileManager: FileManager { .default }

    init(runsRoot: URL) {
        self.runsRoot = runsRoot
    }

    func refresh(
        jobID: String,
        fetch: @Sendable (URL) async throws -> [String]
    ) async throws -> [URL] {
        guard isPathSegment(jobID) else { throw StoreError.invalidJobID }
        try fileManager.createDirectory(at: runsRoot, withIntermediateDirectories: true)
        let destination = runsRoot.appendingPathComponent(jobID, isDirectory: true)
        let staging = runsRoot.appendingPathComponent(
            ".\(jobID).refresh-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? fileManager.removeItem(at: staging) }

        let artifactPaths = try await fetch(staging)
        for path in artifactPaths {
            guard isRelativePath(path) else { throw StoreError.invalidArtifactPath(path) }
            let artifact = staging.appendingPathComponent(path).standardizedFileURL
            guard artifact.path.hasPrefix(staging.standardizedFileURL.path + "/"),
                  fileManager.fileExists(atPath: artifact.path) else {
                throw StoreError.missingArtifact(path)
            }
        }

        try replace(destination: destination, with: staging)
        return artifactPaths.map { destination.appendingPathComponent($0).standardizedFileURL }
    }

    private func replace(destination: URL, with staging: URL) throws {
        guard fileManager.fileExists(atPath: destination.path) else {
            try fileManager.moveItem(at: staging, to: destination)
            return
        }
        let backupName = ".\(destination.lastPathComponent).backup-\(UUID().uuidString)"
        _ = try fileManager.replaceItemAt(
            destination,
            withItemAt: staging,
            backupItemName: backupName,
            options: .usingNewMetadataOnly
        )
        try? fileManager.removeItem(at: runsRoot.appendingPathComponent(backupName))
    }

    private func isPathSegment(_ value: String) -> Bool {
        !value.isEmpty && value != "." && value != ".." && !value.contains("/")
    }

    private func isRelativePath(_ value: String) -> Bool {
        !value.isEmpty
            && !value.hasPrefix("/")
            && value.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
                !$0.isEmpty && $0 != "." && $0 != ".."
            }
    }
}
