import Foundation
import MLX

/// Disk-backed cache for training data (latents and embeddings).
/// Used to reduce memory usage for large datasets during LoRA training.
public struct TrainingDataCache: Sendable {
    /// Default cache directory relative to data root.
    public static let defaultCacheDir = ".zero_cache/training"

    /// Root directory for cache files.
    public let cacheDir: URL

    /// Creates a cache at the specified directory.
    public init(cacheDir: URL) {
        self.cacheDir = cacheDir
    }

    /// Creates a cache relative to the data root.
    public init(dataRoot: URL) {
        self.cacheDir = dataRoot.appendingPathComponent(Self.defaultCacheDir, isDirectory: true)
    }

    /// Initialize the cache directory, optionally wiping existing cache.
    public func initialize(wipe: Bool = false) throws {
        if wipe && FileManager.default.fileExists(atPath: cacheDir.path) {
            try FileManager.default.removeItem(at: cacheDir)
        }
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    /// Check if cache exists for a given item.
    public func exists(id: Int) -> Bool {
        let url = fileURL(for: id)
        return FileManager.default.fileExists(atPath: url.path)
    }

    /// Save encoded training data for an item.
    /// - Parameters:
    ///   - id: Unique identifier for the item
    ///   - latents: Encoded latent representation [B, C, H, W] or [B, SeqLen, D]
    ///   - cond: Conditioning embedding (prompt embeds) [B, SeqLen, D]
    ///   - referenceLatents: Optional secondary latent tensor (used by edit-style training)
    ///   - width: Original image width (for metadata)
    ///   - height: Original image height (for metadata)
    public func save(
        id: Int,
        latents: MLXArray,
        cond: MLXArray,
        referenceLatents: MLXArray? = nil,
        width: Int,
        height: Int
    ) throws {
        let url = fileURL(for: id)

        // Save as safetensors with metadata
        var arrays: [String: MLXArray] = [
            "latents": latents,
            "cond": cond,
        ]
        if let referenceLatents {
            arrays["reference_latents"] = referenceLatents
        }

        // Include shape metadata as 1D arrays
        arrays["meta_width"] = MLXArray([Int32(width)])
        arrays["meta_height"] = MLXArray([Int32(height)])

        try MLX.save(arrays: arrays, url: url)
    }

    /// Load cached training data for an item.
    /// - Parameter id: Unique identifier for the item
    /// - Returns: Tuple of (latents, conditioning, optional reference latents, width, height)
    public func load(id: Int) throws -> (
        latents: MLXArray,
        cond: MLXArray,
        referenceLatents: MLXArray?,
        width: Int,
        height: Int
    ) {
        let url = fileURL(for: id)
        let arrays = try MLX.loadArrays(url: url)

        guard let latents = arrays["latents"],
              let cond = arrays["cond"],
              let widthArr = arrays["meta_width"],
              let heightArr = arrays["meta_height"] else {
            throw TrainingDataCacheError.invalidCacheFormat(url)
        }

        let width = Int(widthArr.item(Int32.self))
        let height = Int(heightArr.item(Int32.self))

        return (
            latents: latents,
            cond: cond,
            referenceLatents: arrays["reference_latents"],
            width: width,
            height: height
        )
    }

    /// Get the cache file URL for an item.
    public func fileURL(for id: Int) -> URL {
        cacheDir.appendingPathComponent("item_\(id).safetensors")
    }

    /// Clear all cached data.
    public func clear() throws {
        if FileManager.default.fileExists(atPath: cacheDir.path) {
            try FileManager.default.removeItem(at: cacheDir)
        }
    }

    /// Get total cache size in bytes.
    public func totalSize() throws -> UInt64 {
        guard FileManager.default.fileExists(atPath: cacheDir.path) else { return 0 }

        let contents = try FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: cacheDir,
            includingPropertiesForKeys: [.fileSizeKey]
        )

        var total: UInt64 = 0
        for url in contents {
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            total += UInt64(values.fileSize ?? 0)
        }
        return total
    }

    /// Count of cached items.
    public func itemCount() throws -> Int {
        guard FileManager.default.fileExists(atPath: cacheDir.path) else { return 0 }

        let contents = try FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: cacheDir,
            includingPropertiesForKeys: nil
        )
        return contents.filter { $0.pathExtension == "safetensors" }.count
    }
}

public enum TrainingDataCacheError: Error, LocalizedError {
    case invalidCacheFormat(URL)
    case cacheNotFound(Int)

    public var errorDescription: String? {
        switch self {
        case .invalidCacheFormat(let url):
            return "Invalid cache file format: \(url.path)"
        case .cacheNotFound(let id):
            return "Cache not found for item \(id)"
        }
    }
}
