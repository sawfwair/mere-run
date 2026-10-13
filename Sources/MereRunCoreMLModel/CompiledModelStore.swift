#if canImport(CoreML) && os(macOS) && arch(arm64)
@preconcurrency import CoreML
import CryptoKit
import Foundation

/// Persistent compiled packages. Core ML can reuse device specialization at a stable URL.
/// Compilation is shared within the process; complete directories are published atomically
/// so separate CLI processes cannot observe a partially copied model.
actor CompiledModelStore {
    static let shared = CompiledModelStore(
        directory: FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appending(path: "MereRun/CoreML/EmbeddingGemma2"),
        compile: { try await MLModel.compileModel(at: $0) }
    )
    private let directory: URL
    private let compile: @Sendable (URL) async throws -> URL
    private var pending: [String: Task<URL, Error>] = [:]

    init(directory: URL, compile: @escaping @Sendable (URL) async throws -> URL) {
        self.directory = directory
        self.compile = compile
    }

    static func fingerprint(package: URL) throws -> String {
        struct WeightIdentity: Encodable {
            let path: String
            let size: UInt64
            let inode: UInt64
            let modified: Date
            let created: Date
        }
        let root = package.resolvingSymlinksInPath()
        let weight = root.appending(path: "Data/com.apple.CoreML/weights/weight.bin").resolvingSymlinksInPath()
        let attrs = try FileManager.default.attributesOfItem(atPath: weight.path)
        guard let size = attrs[.size] as? UInt64, let inode = attrs[.systemFileNumber] as? UInt64,
              let modified = attrs[.modificationDate] as? Date, let created = attrs[.creationDate] as? Date else {
            throw EmbedderError(description: "Cannot fingerprint Core ML weights: \(weight.path).")
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        var hash = SHA256()
        hash.update(data: Data(("compiled-cache-v1:arm64:" + ProcessInfo.processInfo.operatingSystemVersionString).utf8))
        for path in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel"] {
            hash.update(data: try Data(contentsOf: root.appending(path: path), options: .mappedIfSafe))
        }
        // Graph bytes are hashed; large weights use resolved file identity and metadata.
        // Replacement or modification invalidates the cache without rereading every weight.
        hash.update(data: try encoder.encode(WeightIdentity(path: weight.path, size: size, inode: inode, modified: modified, created: created)))
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func compiled(package: URL) async throws -> URL {
        let key = try Self.fingerprint(package: package)
        let destination = directory.appending(path: key)
        let output = destination.appending(path: "model.mlmodelc")
        if FileManager.default.fileExists(atPath: output.path) { return output }
        if let task = pending[key] { return try await task.value }
        let compile = self.compile
        let directory = self.directory
        let task = Task<URL, Error> {
            let manager = FileManager.default
            let temporary = try await compile(package)
            defer { try? manager.removeItem(at: temporary) }
            guard try Self.fingerprint(package: package) == key else {
                throw EmbedderError(description: "Core ML package changed during compilation; retry with a stable artifact.")
            }
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            let staging = directory.appending(path: ".staging-" + UUID().uuidString)
            try manager.createDirectory(at: staging, withIntermediateDirectories: false)
            defer { try? manager.removeItem(at: staging) }
            try manager.copyItem(at: temporary, to: staging.appending(path: "model.mlmodelc"))
            do { try manager.moveItem(at: staging, to: destination) }
            catch {
                // Another process may have published the same complete package first.
                guard manager.fileExists(atPath: output.path) else { throw error }
            }
            return output
        }
        pending[key] = task
        defer { pending[key] = nil }
        return try await task.value
    }
}
#endif
