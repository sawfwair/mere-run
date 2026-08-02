import ArgumentParser
import Foundation
import MereRunCore

struct GraphDataset: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dataset",
        abstract: "Discover graph-ready datasets without loading model runtimes.",
        subcommands: [GraphDatasetDiscover.self]
    )
}

struct WorkflowDatasetCandidate: Codable, Equatable, Sendable {
    let path: String
    let imageCount: Int
    let captionCount: Int
    let pairedCount: Int
    let totalBytes: Int64
    let ready: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case imageCount = "image_count"
        case captionCount = "caption_count"
        case pairedCount = "paired_count"
        case totalBytes = "total_bytes"
        case ready
    }
}

struct WorkflowDatasetDiscovery: Codable, Equatable, Sendable {
    let root: String
    let recommended: String?
    let candidates: [WorkflowDatasetCandidate]
}

struct GraphDatasetDiscover: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "discover",
        abstract: "Find and rank image-caption dataset directories under a root."
    )

    @Argument(help: "Root directory to inspect.")
    var root: String

    @Option(name: [.customLong("max-depth")], help: "Maximum directory depth below the root.")
    var maxDepth = 6

    @Option(name: [.long], help: "Maximum candidates to return.")
    var limit = 100

    @Flag(name: [.long], help: "Emit a structured dataset discovery result.")
    var json = false

    func run() throws {
        guard maxDepth >= 0 else { throw ValidationError("--max-depth must be at least 0.") }
        guard (1...1_000).contains(limit) else { throw ValidationError("--limit must be between 1 and 1000.") }
        let result = try WorkflowDatasetDiscoverer(
            root: URL(fileURLWithPath: root),
            maxDepth: maxDepth,
            limit: limit
        ).discover()
        if json {
            print(try StructuredRunOutput.encode(result))
        } else if result.candidates.isEmpty {
            print("No image datasets found under \(result.root).")
        } else {
            for candidate in result.candidates {
                print("[\(candidate.ready ? "ready" : "incomplete")] \(candidate.path) images=\(candidate.imageCount) pairs=\(candidate.pairedCount)")
            }
        }
    }
}

struct WorkflowDatasetDiscoverer {
    let root: URL
    let maxDepth: Int
    let limit: Int
    let fileManager: FileManager

    init(root: URL, maxDepth: Int = 6, limit: Int = 100, fileManager: FileManager = .default) {
        self.root = root.standardizedFileURL
        self.maxDepth = maxDepth
        self.limit = limit
        self.fileManager = fileManager
    }

    func discover() throws -> WorkflowDatasetDiscovery {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw ValidationError("Dataset discovery root was not found: \(root.path)")
        }
        let keys: [URLResourceKey] = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        guard let enumerator = fileManager.enumeratorResolvingSymlinks(
            at: root,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            throw ValidationError("Could not enumerate dataset discovery root: \(root.path)")
        }

        var directories: [String: DirectoryInventory] = [:]
        while let url = enumerator.nextObject() as? URL {
            let depth = enumerator.level
            let relativeComponents = url.pathComponents.suffix(depth)
            let declaredURL = relativeComponents.reduce(root) { partial, component in
                partial.appendingPathComponent(component)
            }
            let values = try url.resourceValues(forKeys: Set(keys))
            if values.isSymbolicLink == true {
                if values.isDirectory == true { enumerator.skipDescendants() }
                continue
            }
            if values.isDirectory == true {
                if depth > maxDepth { enumerator.skipDescendants() }
                continue
            }
            guard values.isRegularFile == true, depth <= maxDepth + 1 else { continue }
            let directory = declaredURL.deletingLastPathComponent().path
            var inventory = directories[directory] ?? DirectoryInventory()
            let ext = url.pathExtension.lowercased()
            let stem = url.deletingPathExtension().lastPathComponent.lowercased()
            if Self.imageExtensions.contains(ext) {
                inventory.imageStems.insert(stem)
                inventory.totalBytes += Int64(values.fileSize ?? 0)
            } else if ext == "txt" {
                inventory.captionStems.insert(stem)
                inventory.totalBytes += Int64(values.fileSize ?? 0)
            }
            directories[directory] = inventory
        }

        let candidates = directories.compactMap { path, inventory -> WorkflowDatasetCandidate? in
            guard !inventory.imageStems.isEmpty else { return nil }
            let paired = inventory.imageStems.intersection(inventory.captionStems).count
            return WorkflowDatasetCandidate(
                path: path,
                imageCount: inventory.imageStems.count,
                captionCount: inventory.captionStems.count,
                pairedCount: paired,
                totalBytes: inventory.totalBytes,
                ready: paired == inventory.imageStems.count
            )
        }.sorted { left, right in
            if left.ready != right.ready { return left.ready && !right.ready }
            if left.pairedCount != right.pairedCount { return left.pairedCount > right.pairedCount }
            if left.imageCount != right.imageCount { return left.imageCount > right.imageCount }
            return left.path < right.path
        }
        let limited = Array(candidates.prefix(limit))
        return WorkflowDatasetDiscovery(
            root: root.path,
            recommended: limited.first(where: \.ready)?.path,
            candidates: limited
        )
    }

    private static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "webp"]
}

private struct DirectoryInventory {
    var imageStems: Set<String> = []
    var captionStems: Set<String> = []
    var totalBytes: Int64 = 0
}
