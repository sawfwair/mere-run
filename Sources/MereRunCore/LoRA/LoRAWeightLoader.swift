import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if os(Linux)
import Glibc
#endif
import MLX

public enum LoRAWeightLoader {
    private static let remoteCacheDirName = "lora"
    private static let defaultTrainingAdapterRepoId = "ostris/zimage_turbo_training_adapter"
    private static let defaultTrainingAdapterFile = "zimage_turbo_training_adapter_v2.safetensors"

    private static let loraPatterns: [(down: String, up: String)] = [
        (".lora_down.", ".lora_up."),
        (".lora_A.", ".lora_B."),
    ]

    public static func load(from lora: LoRA) async throws -> LoRAWeights {
        let url = try await resolveURL(for: lora)
        return try load(from: url)
    }

    public static func resolveURL(for lora: LoRA) async throws -> URL {
        switch lora {
        case .local(let path, _):
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LoRAError.fileNotFound(url.path)
            }
            return url
        case .remote(let reference, _):
            return try await resolveRemoteReference(reference)
        }
    }

    /// Resolves a remote LoRA reference to a local file path.
    /// Supported formats:
    /// - `owner/repo:path` (Hugging Face file reference)
    /// - `https://...` / `http://...` (public URL)
    /// - legacy adapter aliases for `ostris/zimage_turbo_training_adapter`
    public static func resolveRemoteReference(_ reference: String) async throws -> URL {
        let trimmed = reference.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw LoRAError.invalidFormat("Remote LoRA reference cannot be empty.")
        }

        if let hubReference = parseHubReference(trimmed) {
            return try await downloadHubFile(hubReference)
        }

        if let remoteURL = URL(string: trimmed) {
            try validateRemoteURL(remoteURL)
            return try await downloadRemoteFile(url: remoteURL)
        }

        throw LoRAError.invalidFormat(
            "Unsupported remote LoRA reference '\(reference)'. Use a local file path, owner/repo:file, or HTTPS URL."
        )
    }

    public static func load(from url: URL) throws -> LoRAWeights {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LoRAError.fileNotFound(url.path)
        }

        let (allWeights, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let keys = allWeights.keys.sorted()

        var loraWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
        var processedKeys = Set<String>()

        for key in keys {
            if processedKeys.contains(key) { continue }
            guard let (downKey, upKey, baseKey) = resolveKeyPair(key) else { continue }
            guard let downWeight = allWeights[downKey],
                  let upWeight = allWeights[upKey] else {
                continue
            }

            let mapped = Flux2LoRAKeyMapper.map(baseKey: baseKey, down: downWeight, up: upWeight)
            if mapped.isEmpty {
                // Mapper didn't recognize the key format - use as-is (e.g., ZImage native LoRAs)
                loraWeights[baseKey] = (down: downWeight, up: upWeight)
            } else {
                for (mappedKey, pair) in mapped {
                    loraWeights[mappedKey] = pair
                }
            }

            processedKeys.insert(downKey)
            processedKeys.insert(upKey)
        }

        guard !loraWeights.isEmpty else {
            throw LoRAError.noWeightPairs
        }

        let rank = inferRank(from: loraWeights)
        let alpha = loadAlpha(from: url, metadata: metadata, fallback: Float(rank))

        return LoRAWeights(weights: loraWeights, rank: rank, alpha: alpha)
    }

    private static func resolveKeyPair(_ key: String) -> (downKey: String, upKey: String, baseKey: String)? {
        // Skip optimizer state keys (.m, .v) - only process actual weights
        if key.hasSuffix(".m") || key.hasSuffix(".v") {
            return nil
        }

        for (downPattern, upPattern) in loraPatterns {
            if key.contains(downPattern) {
                guard let base = extractBaseKey(key, pattern: downPattern) else { return nil }
                let upKey = key.replacingOccurrences(of: downPattern, with: upPattern)
                return (downKey: key, upKey: upKey, baseKey: base)
            }
            if key.contains(upPattern) {
                guard let base = extractBaseKey(key, pattern: upPattern) else { return nil }
                let downKey = key.replacingOccurrences(of: upPattern, with: downPattern)
                return (downKey: downKey, upKey: key, baseKey: base)
            }
        }
        return nil
    }

    private static func extractBaseKey(_ key: String, pattern: String) -> String? {
        guard let range = key.range(of: pattern) else { return nil }
        var base = String(key[..<range.lowerBound])

        let prefixesToRemove = [
            "base_model.model.",
            "diffusion_model.",
            "lora_unet_",
            "transformer.",
            "model.",
        ]

        for prefix in prefixesToRemove where base.hasPrefix(prefix) {
            base = String(base.dropFirst(prefix.count))
            break
        }

        return base
    }

    private static func inferRank(from weights: [String: (down: MLXArray, up: MLXArray)]) -> Int {
        for (_, pair) in weights {
            let downShape = pair.down.shape
            if downShape.count == 2 {
                return min(downShape[0], downShape[1])
            }
        }
        return 16
    }

    private static func loadAlpha(
        from url: URL,
        metadata: [String: String]?,
        fallback: Float
    ) -> Float {
        // First try safetensors metadata
        if let metadata,
           let alphaStr = metadata["lora_alpha"],
           let alpha = Float(alphaStr) {
            return alpha
        }

        // Fallback to adapter_config.json in same directory
        let configPath = url.deletingLastPathComponent().appendingPathComponent("adapter_config.json")

        guard FileManager.default.fileExists(atPath: configPath.path),
              let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return fallback
        }

        let alphaKeys = ["lora_alpha", "alpha", "network_alpha"]
        for key in alphaKeys {
            if let alpha = json[key] as? NSNumber {
                return alpha.floatValue
            }
        }

        return fallback
    }

    private struct HubLoRAReference: Hashable, Sendable {
        let repoId: String
        let revision: String
        let filePath: String
    }

    private static func parseHubReference(_ reference: String) -> HubLoRAReference? {
        let normalized = reference
            .replacingOccurrences(of: ":zimage_turbo_training_adapter_v2.safetensors.gz", with: "")
        if normalized == defaultTrainingAdapterRepoId {
            return HubLoRAReference(
                repoId: defaultTrainingAdapterRepoId,
                revision: "main",
                filePath: defaultTrainingAdapterFile
            )
        }

        let parts = reference.split(separator: ":", maxSplits: 1).map(String.init)
        guard parts.count == 2,
              parts[0].split(separator: "/").count == 2,
              !parts[1].isEmpty else {
            return nil
        }
        return HubLoRAReference(repoId: parts[0], revision: "main", filePath: parts[1])
    }

    private static func downloadHubFile(_ reference: HubLoRAReference) async throws -> URL {
        do {
            let snapshot = try HubSnapshot(
                options: HubSnapshotOptions(
                    repoId: reference.repoId,
                    revision: reference.revision,
                    patterns: [reference.filePath]
                )
            )
            return try await snapshot.fileURL(for: reference.filePath)
        } catch {
            throw LoRAError.invalidFormat("Remote LoRA download failed: \(error.localizedDescription)")
        }
    }

    private static func downloadRemoteFile(url: URL) async throws -> URL {
        let filename = url.lastPathComponent.isEmpty ? "remote-lora.safetensors" : url.lastPathComponent
        let destination = try cacheFileURL(filename: filename)

        if FileManager.default.fileExists(atPath: destination.path) {
            return try maybeDecompressGzip(destination)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        return try await downloadRequest(request, destination: destination)
    }

    private static func validateRemoteURL(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(),
              let host = url.host?.lowercased() else {
            throw LoRAError.invalidFormat("Remote LoRA reference must be a valid URL.")
        }
        if scheme == "https" {
            return
        }
        if scheme == "http", isLoopbackHost(host) {
            return
        }
        throw LoRAError.invalidFormat("Remote LoRA references must use HTTPS unless they target localhost.")
    }

    private static func isLoopbackHost(_ host: String) -> Bool {
        host == "localhost" || host == "127.0.0.1" || host == "::1"
    }

    private static func downloadRequest(
        _ request: URLRequest,
        destination: URL
    ) async throws -> URL {
        let fileManager = FileManager.default
        let partial = destination.appendingPathExtension("partial")
        try? fileManager.removeItem(at: partial)

        let (tempURL, response) = try await URLSession.shared.download(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw LoRAError.invalidFormat("Remote LoRA download failed (HTTP \(code)).")
        }

        if fileManager.fileExists(atPath: partial.path) {
            try? fileManager.removeItem(at: partial)
        }
        try fileManager.moveItem(at: tempURL, to: partial)

        if fileManager.fileExists(atPath: destination.path) {
            try? fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: partial, to: destination)

        return try maybeDecompressGzip(destination)
    }

    private static func cacheFileURL(filename: String) throws -> URL {
        let safeName = filename.replacingOccurrences(of: " ", with: "_")
        let cacheDir = MereRunModelPaths.downloadsDir.appendingPathComponent(remoteCacheDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        return cacheDir.appendingPathComponent(safeName, isDirectory: false)
    }

    private static func maybeDecompressGzip(_ url: URL) throws -> URL {
        guard url.pathExtension.lowercased() == "gz" else {
            return url
        }

        let outURL = url.deletingPathExtension()
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outURL.path) {
            return outURL
        }

        #if os(macOS)
        _ = fileManager.createFile(atPath: outURL.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: outURL)
        defer { try? outputHandle.close() }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
        process.arguments = ["-c", url.path]
        process.standardOutput = outputHandle
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw LoRAError.invalidFormat("Failed to decompress gzip LoRA archive: \(url.lastPathComponent)")
        }
        return outURL
        #elseif os(Linux)
        let command = "gzip -dc \(shellQuote(url.path)) > \(shellQuote(outURL.path))"
        guard system(command) == 0 else {
            throw LoRAError.invalidFormat("Failed to decompress gzip LoRA archive: \(url.lastPathComponent)")
        }
        return outURL
        #else
        throw LoRAError.invalidFormat("Gzip LoRA archives are unsupported on this platform.")
        #endif
    }

    private static func shellQuote(_ value: String) -> String {
        "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
    }
}
