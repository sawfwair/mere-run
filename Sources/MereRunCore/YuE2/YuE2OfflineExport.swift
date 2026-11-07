import Foundation

extension YuE2Resources {
    public struct OfflineExport: Codable, Sendable {
        public let schemaVersion: Int
        public let sourceRevision: String
        public let artifacts: [ModelArtifactPin]
    }

    /// Export only data and license artifacts needed by the native runtime. No executable model code is copied.
    public func export(to destination: URL) throws -> OfflineExport {
        let source = rootURL.resolvingSymlinksInPath().standardizedFileURL
        let target = destination.resolvingSymlinksInPath().standardizedFileURL
        guard source != target, !source.path.hasPrefix(target.path + "/"), !target.path.hasPrefix(source.path + "/"),
              !FileManager.default.fileExists(atPath: destination.path) else {
            throw YuE2Error.invalidRequest("Offline export requires a new directory outside its source model.")
        }
        guard validate().isEmpty else { throw YuE2Error.invalidWeights("Offline export requires the complete local YuE2 pipeline.") }
        _ = try configuration()
        _ = try vaeConfiguration()
        _ = try YuE2Tokenizer(url: rootURL.appendingPathComponent("qwen.tiktoken"))
        let manager = FileManager.default
        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(".yue2-export-\(UUID().uuidString)")
        try manager.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: temporary) }
        var paths: [String] = []
        for prefix in ["", "vae/"] {
            for name in ["config.json", "model.safetensors", "qwen.tiktoken", "yue2_generation_config.json",
                         "weights_manifest.json", "LICENSE", "THIRD_PARTY_NOTICES.md"] {
                let relative = prefix + name
                if manager.fileExists(atPath: rootURL.appendingPathComponent(relative).path) { paths.append(relative) }
            }
            paths += try licensePaths(prefix: prefix, fileManager: manager)
        }
        var artifacts: [ModelArtifactPin] = []
        for path in paths.sorted() {
            try Task.checkCancellation()
            let file = temporary.appendingPathComponent(path)
            try manager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Copy bytes rather than managed-install symlinks, so the export remains self-contained.
            try manager.copyItem(at: rootURL.appendingPathComponent(path).resolvingSymlinksInPath(), to: file)
            artifacts.append(try ModelArtifactPin(filename: path, byteCount: ModelArtifactPin.fileByteCount(file),
                sha256: ModelArtifactPin.fileSHA256(file)))
        }
        let receipt = OfflineExport(schemaVersion: 1, sourceRevision: Self.sourceRevision, artifacts: artifacts)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: temporary.appendingPathComponent("export.json"))
        try manager.moveItem(at: temporary, to: destination)
        return receipt
    }

    func licensePaths(prefix: String, fileManager: FileManager = .default) throws -> [String] {
        let licenses = rootURL.appendingPathComponent(prefix + "licenses").resolvingSymlinksInPath()
        guard fileManager.fileExists(atPath: licenses.path) else { return [] }
        return try fileManager.contentsOfDirectory(at: licenses, includingPropertiesForKeys: [.isRegularFileKey])
            .filter { try $0.pathExtension == "txt"
                && $0.resolvingSymlinksInPath().resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true }
            .map { prefix + "licenses/" + $0.lastPathComponent }
    }
}
