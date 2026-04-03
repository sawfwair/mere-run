import Foundation

/// Shared model lookup utilities for mere.run's public model families.
///
/// Phase 1 scope:
/// - Resolve known model IDs (e.g. `image-klein-max`, `image-zimage-max`) to a local model root directory.
/// - Resolve from mere.run's configured local model store (default: `~/Library/Application Support/MereRun/models/...`).
///
/// This intentionally does *not* download models; it only resolves paths.
public struct ModelResolver {
    public enum ModelID: String, CaseIterable, Hashable, Sendable {
        case kleinNano = "image-klein-nano"
        case kleinMax = "image-klein-max"
        case kleinBase = "image-klein-base"
        case kleinShared = "image-klein-shared"
        case mebot = "text-chat-mebot"
        case q35 = "text-chat-q35"
        case q35Nano = "text-chat-q35-nano"
        case zetaNano = "image-zimage-nano"
        case zetaMax = "image-zimage-max"
        case zetaBase = "image-zimage-base"
        case visionSegmentSAM31 = "vision-segment-sam31"
    }

    public enum Source: String, Hashable, Sendable {
        /// `.../models/<id>` under the configured local mere.run model store.
        case localModelStore
    }

    public struct Resolution: Hashable, Sendable {
        public let modelID: ModelID
        public let rootURL: URL
        public let source: Source

        public init(modelID: ModelID, rootURL: URL, source: Source) {
            self.modelID = modelID
            self.rootURL = rootURL
            self.source = source
        }
    }

    public enum ResolverError: LocalizedError, Sendable {
        case applicationSupportUnavailable
        case modelNotFound(ModelID, searched: [URL], upstreamRepoId: String?)

        public var errorDescription: String? {
            switch self {
            case .applicationSupportUnavailable:
                return "Could not locate Application Support directory."
            case .modelNotFound(let id, let searched, let upstreamRepoId):
                var lines: [String] = []
                lines.append("Model not found: \(id.rawValue)")
                if let upstreamRepoId {
                    lines.append("Upstream repo: \(upstreamRepoId)")
                }
                if !searched.isEmpty {
                    lines.append("Searched:")
                    lines.append(contentsOf: searched.map { "  - \($0.path)" })
                }
                return lines.joined(separator: "\n")
            }
        }
    }

    private let fileManager: FileManager

    public init(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.fileManager = fileManager
        _ = environment
    }

    public func resolve(_ modelID: ModelID) throws -> Resolution {
        if modelID == .mebot {
            if let kleinNano = try? resolve(.kleinNano) {
                return Resolution(modelID: modelID, rootURL: kleinNano.rootURL, source: kleinNano.source)
            }

            let kleinMax = try resolve(.kleinMax)
            return Resolution(modelID: modelID, rootURL: kleinMax.rootURL, source: kleinMax.source)
        }

        var searched: [URL] = []

        for modelsRoot in candidateModelRoots() {
            let modelRoot = modelsRoot.appendingPathComponent(modelID.rawValue, isDirectory: true)
            searched.append(modelRoot)
            if isValidModelRoot(modelRoot, expectedModelID: modelID) {
                return Resolution(modelID: modelID, rootURL: modelRoot, source: .localModelStore)
            }
        }

        throw ResolverError.modelNotFound(modelID, searched: searched, upstreamRepoId: nil)
    }

    public func resolveIfPresent(_ modelID: ModelID) -> Resolution? {
        try? resolve(modelID)
    }

    private func candidateModelRoots() -> [URL] {
        [MereRunModelPaths.modelsDir.standardizedFileURL]
    }

    private func isValidModelRoot(_ url: URL, expectedModelID: ModelID? = nil) -> Bool {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }

        // Strict policy: only treat directories with a manifest as valid model roots.
        let manifest = url.appendingPathComponent(MereRunModelManifest.filename)
        guard fileManager.fileExists(atPath: manifest.path) else {
            return false
        }

        guard let expectedModelID else {
            return true
        }

        guard let loaded = try? MereRunModelManifest.loadRequired(from: url, fileManager: fileManager) else {
            return false
        }
        return loaded.id == expectedModelID.rawValue
    }
}
