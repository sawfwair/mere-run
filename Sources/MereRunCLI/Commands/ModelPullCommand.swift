import ArgumentParser
import Foundation
import MereRunCore

struct ModelPull: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pull",
        abstract: "Download a managed model into the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-klein-nano or text-chat-q35). Omit when using --all.")
    var target: String?

    @Flag(name: [.long], help: "Pull every model that has an R2 archive.")
    var all: Bool = false

    @Flag(name: [.long], help: "Re-download even if the model is already installed.")
    var force: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    func validate() throws {
        if target == nil && !all {
            throw ValidationError("Provide a model id or use --all.")
        }
        if target != nil && all {
            throw ValidationError("Specify either a model id or --all, not both.")
        }
    }

    func run() async throws {
        if all {
            for entry in R2ModelRegistry.allEntries {
                try await pull(entry)
            }
        } else {
            guard let target else { return }
            if target.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == ModelResolver.ModelID.gemma4.rawValue {
                throw ValidationError("text-chat-gemma4 is not distributed as an R2 archive. It resolves through the native Hugging Face snapshot path on first chat/API use, or you can install a local Gemma 4 root and point commands at it directly.")
            }
            let entry = resolveEntry(target)
            guard let entry else {
                throw ValidationError("Unknown canonical model id: \(target)")
            }
            try await pull(entry)
        }
    }

    /// Resolve a user-supplied canonical model id to an R2ModelRegistry entry.
    private func resolveEntry(_ raw: String) -> R2ModelRegistry.ModelEntry? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return R2ModelRegistry.entry(for: normalized)
    }

    private func pull(_ entry: R2ModelRegistry.ModelEntry) async throws {
        let modelDir = MereRunModelPaths.modelDir(entry.id)

        if !force, isNonEmptyDirectory(modelDir) {
            if !quiet { stderr("[\(entry.id)] already installed, skipping (use --force to re-download)") }
            return
        }

        // Also check via ModelResolver for models already present in the active mere.run model store.
        if !force, let modelID = ModelResolver.ModelID(rawValue: entry.id) {
            if ModelResolver().resolveIfPresent(modelID) != nil {
                if !quiet { stderr("[\(entry.id)] already installed, skipping (use --force to re-download)") }
                return
            }
        }

        if !quiet { stderr("[\(entry.id)] downloading archive…") }

        let fm = FileManager.default
        let archiveFile = MereRunModelPaths.downloadsDir.appendingPathComponent(entry.archiveKey)

        try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: archiveFile.deletingLastPathComponent(), withIntermediateDirectories: true)

        try await downloadR2Archive(key: entry.archiveKey, to: archiveFile)

        if !quiet { stderr("[\(entry.id)] extracting…") }
        try extractArchive(archiveFile, to: modelDir)
        try normalizeExtractedModelLayout(for: entry.id, in: modelDir)
        try? fm.removeItem(at: archiveFile)

        _ = try MereRunModelManifest.writeTemplateIfKnown(modelId: entry.id, to: modelDir)

        let report = MereRunModelValidator.validate(modelRoot: modelDir, expectedModelID: entry.id)
        if !quiet {
            for w in report.warnings { stderr("  warning: \(w)") }
            for e in report.errors { stderr("  error: \(e)") }
        }

        print(modelDir.path)
    }

    private func isNonEmptyDirectory(_ url: URL) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return (try? fm.contentsOfDirectory(atPath: url.path))?.isEmpty == false
    }

    // MARK: - R2 Download

    private func downloadR2Archive(key: String, to archiveFile: URL) async throws {
        let request: URLRequest
        do {
            request = try await R2DownloadRequestBuilder.makeGETRequest(key: key).request
        } catch let error as R2DownloadRequestBuilder.BuildError {
            throw CleanExit.message(error.localizedDescription)
        }

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw CleanExit.message("Download failed: HTTP \(code)")
        }

        let totalSize = httpResponse.expectedContentLength

        let fm = FileManager.default
        let tempURL = archiveFile.appendingPathExtension("partial")
        try? fm.removeItem(at: tempURL)

        fm.createFile(atPath: tempURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tempURL)
        defer { try? handle.close() }

        var bytesWritten: Int64 = 0
        var buffer = Data()
        let bufferSize = 1024 * 1024
        var lastUpdate = Date.distantPast

        for try await byte in asyncBytes {
            buffer.append(byte)
            if buffer.count >= bufferSize {
                try handle.write(contentsOf: buffer)
                bytesWritten += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)

                if !quiet {
                    let now = Date()
                    if now.timeIntervalSince(lastUpdate) >= 0.25 {
                        lastUpdate = now
                        let done = ByteCountFormatter.string(fromByteCount: bytesWritten, countStyle: .file)
                        if totalSize > 0 {
                            let pct = min(100, Int(Double(bytesWritten) / Double(totalSize) * 100))
                            let total = ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file)
                            stderrRaw("\r  \(pct)%  \(done) / \(total)          ")
                        } else {
                            stderrRaw("\r  \(done)          ")
                        }
                    }
                }
            }
        }

        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
            bytesWritten += Int64(buffer.count)
        }

        try handle.synchronize()

        if !quiet { stderrRaw("\n") }

        if fm.fileExists(atPath: archiveFile.path) {
            try? fm.removeItem(at: archiveFile)
        }
        try fm.moveItem(at: tempURL, to: archiveFile)
    }

    private func extractArchive(_ archiveFile: URL, to modelDir: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", archiveFile.path, "-C", modelDir.path]
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw CleanExit.message("tar extraction failed (exit \(process.terminationStatus))")
        }
    }

    private func normalizeExtractedModelLayout(for modelID: String, in modelDir: URL) throws {
        guard modelID == "music-acestep" else {
            return
        }

        try renameDirectoryIfPresent(
            from: modelDir.appendingPathComponent(["ace", "step-v15-turbo"].joined(), isDirectory: true),
            to: modelDir.appendingPathComponent("music-acestep-v15-turbo", isDirectory: true)
        )

        let nestedRoots = [modelDir, modelDir.appendingPathComponent("checkpoints", isDirectory: true)]
        let lmPairs: [(String, String)] = [
            (["ace", "step-5Hz-lm-1.7B"].joined(), "music-acestep-5hz-lm-1.7b"),
            (["ace", "step-5hz-lm-1.7b"].joined(), "music-acestep-5hz-lm-1.7b"),
            (["ace", "step-5Hz-lm"].joined(), "music-acestep-5hz-lm"),
            (["ace", "step-5hz-lm"].joined(), "music-acestep-5hz-lm"),
            (["ace", "step-lm"].joined(), "music-acestep-lm"),
        ]

        for root in nestedRoots {
            for (oldName, newName) in lmPairs {
                try renameDirectoryIfPresent(
                    from: root.appendingPathComponent(oldName, isDirectory: true),
                    to: root.appendingPathComponent(newName, isDirectory: true)
                )
            }
        }
    }

    private func renameDirectoryIfPresent(from source: URL, to destination: URL) throws {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: source.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return
        }
        guard !fm.fileExists(atPath: destination.path) else {
            throw CleanExit.message("Model layout normalization failed: destination already exists at \(destination.path)")
        }
        try fm.moveItem(at: source, to: destination)
    }

    // MARK: - Stderr helpers

    private func stderr(_ message: String) {
        CLIStderr.write(message + "\n")
    }

    private func stderrRaw(_ message: String) {
        CLIStderr.write(message)
    }
}
