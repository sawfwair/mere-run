import CryptoKit
import Foundation

struct Q38PLEPlacementManifest: Decodable, Sendable {
    struct File: Decodable, Sendable {
        let path: String
        let byteCount: Int
        let sha256: String

        private enum CodingKeys: String, CodingKey {
            case path
            case byteCount = "byte_count"
            case sha256
        }
    }

    let version: Int
    let format: String
    let artifactId: String
    let index: String
    let preferredPlacement: String
    let files: [File]

    private enum CodingKeys: String, CodingKey {
        case version
        case format
        case artifactId = "artifact_id"
        case index
        case preferredPlacement = "preferred_placement"
        case files
    }
}

enum Q38PLEPlacement {
    static let format = "mere-run-q38-ple-safetensors-placement-v1"
    static let manifestFilename = "MERERUN_PLE_STORE.json"

    struct Resolution: Sendable {
        let indexURL: URL
        let usedInternalCache: Bool
    }

    enum PlacementError: LocalizedError {
        case invalidManifest(String)
        case truncatedFile(URL)

        var errorDescription: String? {
            switch self {
            case .invalidManifest(let reason):
                "Invalid packaged Qwen PLE placement: \(reason)"
            case .truncatedFile(let url):
                "Packaged Qwen PLE file has an unexpected size: \(url.path)"
            }
        }
    }

    static func resolve(
        rootURL: URL,
        cacheBase: URL = MereRunModelPaths.modelCacheBase,
        fileManager: FileManager = .default,
        progressHandler: ((String) -> Void)? = nil
    ) throws -> Resolution? {
        let manifestURL = rootURL.appendingPathComponent(manifestFilename)
        guard fileManager.fileExists(atPath: manifestURL.path) else { return nil }
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Q38PLEPlacementManifest.self, from: manifestData)
        try validate(manifest: manifest, rootURL: rootURL, fileManager: fileManager)
        let sourceIndex = rootURL.appendingPathComponent(manifest.index)

        guard manifest.preferredPlacement == "internal_cache" else {
            return Resolution(indexURL: sourceIndex, usedInternalCache: false)
        }
        try fileManager.createDirectory(at: cacheBase, withIntermediateDirectories: true)
        guard try deviceIdentifier(for: rootURL, fileManager: fileManager)
                != deviceIdentifier(for: cacheBase, fileManager: fileManager) else {
            return Resolution(indexURL: sourceIndex, usedInternalCache: false)
        }

        let cacheRoot = cacheBase
            .appendingPathComponent("q38-ple", isDirectory: true)
            .appendingPathComponent(manifest.artifactId, isDirectory: true)
        let cachedIndex = cacheRoot.appendingPathComponent(manifest.index)
        let receiptName = "CACHE_RECEIPT.sha256"
        let identity = sha256(manifestData + (try Data(contentsOf: sourceIndex)))
        if cacheIsValid(
            cacheRoot,
            receiptName: receiptName,
            identity: identity,
            manifest: manifest,
            fileManager: fileManager
        ) {
            return Resolution(indexURL: cachedIndex, usedInternalCache: true)
        }

        let required = Int64(manifest.files.reduce(0) { $0 + $1.byteCount })
        let reserve: Int64 = 8 * 1_073_741_824
        let available = try availableCapacity(at: cacheBase, fileManager: fileManager)
        guard available >= required + reserve else {
            progressHandler?("Internal SSD does not have enough free space for the packaged PLE cache; using the model store directly")
            return Resolution(indexURL: sourceIndex, usedInternalCache: false)
        }

        let cacheParent = cacheRoot.deletingLastPathComponent()
        try fileManager.createDirectory(at: cacheParent, withIntermediateDirectories: true)
        let staging = cacheParent.appendingPathComponent(
            ".\(manifest.artifactId).\(UUID().uuidString).tmp",
            isDirectory: true
        )
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        do {
            progressHandler?("Caching the packaged PLE table on the internal SSD")
            for file in manifest.files {
                let source = rootURL.appendingPathComponent(file.path)
                let destination = staging.appendingPathComponent(file.path)
                try fileManager.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try fileManager.copyItem(at: source, to: destination)
                try validate(file: file, at: destination, fileManager: fileManager)
                guard try sha256(destination) == file.sha256 else {
                    throw PlacementError.invalidManifest("cached file digest differs: \(file.path)")
                }
            }
            try writePLEIndex(
                sourceIndex: sourceIndex,
                destination: staging.appendingPathComponent(manifest.index),
                cachedFiles: Set(manifest.files.map(\.path))
            )
            try Data((identity + "\n").utf8)
                .write(to: staging.appendingPathComponent(receiptName), options: .atomic)
            if fileManager.fileExists(atPath: cacheRoot.path) {
                try fileManager.removeItem(at: cacheRoot)
            }
            try fileManager.moveItem(at: staging, to: cacheRoot)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
        return Resolution(indexURL: cachedIndex, usedInternalCache: true)
    }

    static func validate(
        manifest: Q38PLEPlacementManifest,
        rootURL: URL,
        fileManager: FileManager = .default
    ) throws {
        guard manifest.version == 1,
              manifest.format == format,
              !manifest.artifactId.isEmpty,
              manifest.artifactId.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" }),
              safeRelativePath(manifest.index),
              !manifest.files.isEmpty else {
            throw PlacementError.invalidManifest("unsupported format or layout")
        }
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: rootURL.appendingPathComponent(manifest.index))
        )
        let files = Set(manifest.files.map(\.path))
        let pleFiles = Set(index.weightMap.compactMap { key, filename in
            key.contains(".ple.ple_embedding.ngram_embedding.") ? filename : nil
        })
        guard files == pleFiles else {
            throw PlacementError.invalidManifest("file list does not exactly cover the indexed PLE tensors")
        }
        for file in manifest.files {
            guard safeRelativePath(file.path), file.sha256.count == 64 else {
                throw PlacementError.invalidManifest("file path or digest is invalid")
            }
            try validate(file: file, at: rootURL.appendingPathComponent(file.path), fileManager: fileManager)
        }
    }

    private static func writePLEIndex(
        sourceIndex: URL,
        destination: URL,
        cachedFiles: Set<String>
    ) throws {
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: sourceIndex)
        )
        let filtered = HFSafetensorsIndex(
            metadata: index.metadata,
            weightMap: index.weightMap.filter { key, filename in
                key.contains(".ple.ple_embedding.ngram_embedding.")
                    && cachedFiles.contains(filename)
            }
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let output = try encoder.encode(filtered)
        try output.write(to: destination, options: .atomic)
    }

    private static func validate(
        file: Q38PLEPlacementManifest.File,
        at url: URL,
        fileManager: FileManager
    ) throws {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        guard (attributes[.size] as? NSNumber)?.intValue == file.byteCount else {
            throw PlacementError.truncatedFile(url)
        }
    }

    private static func cacheIsValid(
        _ root: URL,
        receiptName: String,
        identity: String,
        manifest: Q38PLEPlacementManifest,
        fileManager: FileManager
    ) -> Bool {
        let receiptURL = root.appendingPathComponent(receiptName)
        let indexURL = root.appendingPathComponent(manifest.index)
        guard fileManager.fileExists(atPath: indexURL.path),
              let receipt = try? String(contentsOf: receiptURL, encoding: .utf8),
              receipt.trimmingCharacters(in: .whitespacesAndNewlines) == identity else {
            return false
        }
        return manifest.files.allSatisfy { file in
            let url = root.appendingPathComponent(file.path)
            guard let attributes = try? fileManager.attributesOfItem(atPath: url.path) else {
                return false
            }
            return (attributes[.size] as? NSNumber)?.intValue == file.byteCount
        }
    }

    private static func safeRelativePath(_ path: String) -> Bool {
        guard !path.isEmpty, !path.hasPrefix("/") else { return false }
        return !path.split(separator: "/").contains("..")
    }

    private static func deviceIdentifier(for url: URL, fileManager: FileManager) throws -> UInt64 {
        var candidate = url
        while !fileManager.fileExists(atPath: candidate.path), candidate.path != "/" {
            candidate.deleteLastPathComponent()
        }
        let attributes = try fileManager.attributesOfItem(atPath: candidate.path)
        return (attributes[.systemNumber] as? NSNumber)?.uint64Value ?? 0
    }

    private static func availableCapacity(at url: URL, fileManager: FileManager) throws -> Int64 {
        let attributes = try fileManager.attributesOfFileSystem(forPath: url.path)
        return (attributes[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let data = try handle.read(upToCount: 8 * 1_024 * 1_024), !data.isEmpty {
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
