import Foundation
#if os(Linux)
import Glibc
#endif

public struct LoRAResolvedCheckpoint: Sendable {
    public let checkpointURL: URL
    public let cleanupURL: URL?
    public let optimizerStateURL: URL?
    public let iteratorStateURL: URL?
    public let lossStateURL: URL?
    public let configStateURL: URL?

    public init(
        checkpointURL: URL,
        cleanupURL: URL?,
        optimizerStateURL: URL? = nil,
        iteratorStateURL: URL? = nil,
        lossStateURL: URL? = nil,
        configStateURL: URL? = nil
    ) {
        self.checkpointURL = checkpointURL
        self.cleanupURL = cleanupURL
        self.optimizerStateURL = optimizerStateURL
        self.iteratorStateURL = iteratorStateURL
        self.lossStateURL = lossStateURL
        self.configStateURL = configStateURL
    }

    public func cleanup() {
        guard let cleanupURL else { return }
        try? FileManager.default.removeItem(at: cleanupURL)
    }
}

public enum LoRACheckpointResolver {
    public static func resolve(_ inputURL: URL) throws -> LoRAResolvedCheckpoint {
        let normalized = inputURL.standardizedFileURL
        let ext = normalized.pathExtension.lowercased()
        switch ext {
        case "safetensors":
            return LoRAResolvedCheckpoint(checkpointURL: normalized, cleanupURL: nil)
        case "zip":
            return try resolveZip(normalized)
        default:
            throw NSError(
                domain: "LoRACheckpointResolver",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Unsupported checkpoint extension .\(ext). Expected .safetensors or .zip."
                ]
            )
        }
    }

    private static func resolveZip(_ zipURL: URL) throws -> LoRAResolvedCheckpoint {
        let fm = FileManager.default
        let extractRoot = fm.temporaryDirectory.appendingPathComponent(
            "mererun-lora-resume-\(UUID().uuidString)",
            isDirectory: true
        )
        try fm.createDirectory(at: extractRoot, withIntermediateDirectories: true)
        do {
            #if os(macOS)
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
            process.arguments = ["-x", "-k", zipURL.path, extractRoot.path]
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                throw NSError(
                    domain: "LoRACheckpointResolver",
                    code: Int(process.terminationStatus),
                    userInfo: [NSLocalizedDescriptionKey: "Failed to extract checkpoint archive \(zipURL.lastPathComponent)."]
                )
            }
            #elseif os(Linux)
            let command = "unzip -q \(shellQuote(zipURL.path)) -d \(shellQuote(extractRoot.path))"
            guard system(command) == 0 else {
                throw NSError(
                    domain: "LoRACheckpointResolver",
                    code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to extract checkpoint archive \(zipURL.lastPathComponent)."]
                )
            }
            #else
            throw NSError(
                domain: "LoRACheckpointResolver",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Zip checkpoint resume is unsupported on this platform."]
            )
            #endif

            guard let checkpointURL = findCheckpoint(
                in: extractRoot,
                preferredName: zipURL.deletingPathExtension().lastPathComponent
            ) else {
                throw NSError(
                    domain: "LoRACheckpointResolver",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "No .safetensors checkpoint found in \(zipURL.lastPathComponent)."]
                )
            }

            let allFiles = allRegularFiles(in: extractRoot)
            let references = checkpointFileReferences(in: extractRoot)
            let optimizerURL = references["optimizer"].flatMap { matchFile($0, in: allFiles, root: extractRoot) }
            let iteratorURL = references["iterator"].flatMap { matchFile($0, in: allFiles, root: extractRoot) }
            let lossURL = references["loss"].flatMap { matchFile($0, in: allFiles, root: extractRoot) }
            let configURL = references["config"].flatMap { matchFile($0, in: allFiles, root: extractRoot) }

            return LoRAResolvedCheckpoint(
                checkpointURL: checkpointURL,
                cleanupURL: extractRoot,
                optimizerStateURL: optimizerURL,
                iteratorStateURL: iteratorURL,
                lossStateURL: lossURL,
                configStateURL: configURL
            )
        } catch {
            try? fm.removeItem(at: extractRoot)
            throw error
        }
    }

    private static func findCheckpoint(in root: URL, preferredName: String) -> URL? {
        let candidates = allRegularFiles(in: root).filter { $0.pathExtension.lowercased() == "safetensors" }

        if candidates.isEmpty {
            return nil
        }

        if let fromRunManifest = preferredCheckpointFileFromRunManifest(in: root),
           let matched = matchCheckpointFile(fromRunManifest, in: candidates, root: root) {
            return matched
        }

        if let fromCheckpointManifest = preferredCheckpointFileFromCheckpointManifest(in: root),
           let matched = matchCheckpointFile(fromCheckpointManifest, in: candidates, root: root) {
            return matched
        }

        if candidates.count == 1 {
            return candidates[0]
        }

        if let preferred = candidates.first(where: { $0.deletingPathExtension().lastPathComponent == preferredName }) {
            return preferred
        }

        return candidates.sorted {
            if $0.path.count == $1.path.count {
                return $0.path < $1.path
            }
            return $0.path.count < $1.path.count
        }.first
    }

    private static func preferredCheckpointFileFromRunManifest(in root: URL) -> String? {
        checkpointFilesFromRunManifest(in: root)?["lora_adapter"]
    }

    private static func checkpointFilesFromCheckpointManifest(in root: URL) -> [String: String]? {
        guard let manifestURL = findMetadataFile(named: "checkpoint.json", in: root),
              let data = try? Data(contentsOf: manifestURL),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = raw["files"] as? [String: Any] else {
            return nil
        }

        var normalized: [String: String] = [:]
        for (key, rawValue) in files {
            guard let value = rawValue as? String else { continue }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            normalized[key] = trimmed
        }
        return normalized
    }

    private static func preferredCheckpointFileFromCheckpointManifest(in root: URL) -> String? {
        checkpointFilesFromCheckpointManifest(in: root)?["lora_adapter"]
    }

    private static func checkpointFilesFromRunManifest(in root: URL) -> [String: String]? {
        guard let manifestURL = findMetadataFile(named: LoRATrainingRunManifest.filename, in: root),
              let data = try? Data(contentsOf: manifestURL),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let files = raw["checkpoint_files"] as? [String: Any] else {
            return nil
        }

        var normalized: [String: String] = [:]
        for (key, rawValue) in files {
            guard let value = rawValue as? String else { continue }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            normalized[key] = trimmed
        }
        return normalized
    }

    private static func checkpointFileReferences(in root: URL) -> [String: String] {
        let runManifestFiles = checkpointFilesFromRunManifest(in: root) ?? [:]
        let checkpointManifestFiles = checkpointFilesFromCheckpointManifest(in: root) ?? [:]
        let knownKeys = ["lora_adapter", "optimizer", "iterator", "loss", "config"]

        var merged: [String: String] = [:]
        for key in knownKeys {
            if let value = runManifestFiles[key] {
                merged[key] = value
                continue
            }
            if let value = checkpointManifestFiles[key] {
                merged[key] = value
            }
        }
        return merged
    }

    private static func findMetadataFile(named name: String, in root: URL) -> URL? {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        for case let url as URL in enumerator {
            guard url.lastPathComponent == name else { continue }
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            return url
        }

        return nil
    }

    private static func allRegularFiles(in root: URL) -> [URL] {
        let fileManager = FileManager.default
        guard let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files: [URL] = []
        for case let url as URL in enumerator {
            guard (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
            files.append(url)
        }
        return files
    }

    private static func matchCheckpointFile(_ fileName: String, in candidates: [URL], root: URL) -> URL? {
        matchFile(fileName, in: candidates, root: root)
    }

    private static func matchFile(_ fileName: String, in candidates: [URL], root: URL) -> URL? {
        let trimmed = fileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let normalized = (trimmed as NSString).standardizingPath
        let normalizedFileName = (normalized as NSString).lastPathComponent
        let normalizedNoExtension = URL(fileURLWithPath: normalizedFileName).deletingPathExtension().lastPathComponent

        if let directNameMatch = candidates.first(where: { $0.lastPathComponent == normalizedFileName }) {
            return directNameMatch
        }
        if let noExtensionMatch = candidates.first(where: { $0.deletingPathExtension().lastPathComponent == normalizedNoExtension }) {
            return noExtensionMatch
        }

        let rootPath = root.standardizedFileURL.path + "/"
        let slashNormalized = normalized.replacingOccurrences(of: "\\", with: "/")
        for candidate in candidates {
            let candidatePath = candidate.standardizedFileURL.path
            let relative = candidatePath.hasPrefix(rootPath)
                ? String(candidatePath.dropFirst(rootPath.count))
                : candidatePath
            if relative == slashNormalized || relative.hasSuffix("/" + slashNormalized) {
                return candidate
            }
        }

        return nil
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}
