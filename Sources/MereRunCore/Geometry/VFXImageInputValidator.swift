import Crypto
import Foundation
import MediaIO

/// Bounded source-image admission for native VFX geometry and reconstruction.
///
/// The limits cover both encoded bytes and decoded dimensions. Snapshot
/// admission streams and bounds the encoded source while hashing it; header
/// and post-decode validation prevent a compact image bomb from expanding into
/// an unbounded allocation.
public struct VFXImageInputLimits: Equatable, Sendable {
    public let maximumDimension: Int
    public let maximumPixelsPerImage: Int
    public let maximumAggregatePixels: Int
    public let maximumEncodedBytesPerImage: Int64
    public let maximumAggregateEncodedBytes: Int64

    public init(
        maximumDimension: Int = 16_384,
        maximumPixelsPerImage: Int = 64_000_000,
        maximumAggregatePixels: Int = 256_000_000,
        maximumEncodedBytesPerImage: Int64 = 512 * 1_024 * 1_024,
        maximumAggregateEncodedBytes: Int64 = 1_024 * 1_024 * 1_024
    ) {
        self.maximumDimension = maximumDimension
        self.maximumPixelsPerImage = maximumPixelsPerImage
        self.maximumAggregatePixels = maximumAggregatePixels
        self.maximumEncodedBytesPerImage = maximumEncodedBytesPerImage
        self.maximumAggregateEncodedBytes = maximumAggregateEncodedBytes
    }

    public static let production = VFXImageInputLimits()
}

public struct VFXImageInputDimensions: Codable, Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let pixelCount: Int

    public init(width: Int, height: Int, pixelCount: Int) {
        self.width = width
        self.height = height
        self.pixelCount = pixelCount
    }
}

public enum VFXImageInputValidationError: Error, Equatable, LocalizedError, Sendable {
    case invalidLimits
    case invalidDimensions(path: String, width: Int, height: Int)
    case dimensionLimitExceeded(path: String, width: Int, height: Int, maximum: Int)
    case pixelLimitExceeded(path: String, pixels: Int, maximum: Int)
    case aggregatePixelLimitExceeded(pixels: Int, maximum: Int)
    case encodedByteLimitExceeded(path: String, bytes: Int64, maximum: Int64)
    case aggregateEncodedByteLimitExceeded(bytes: Int64, maximum: Int64)

    public var errorDescription: String? {
        switch self {
        case .invalidLimits:
            "VFX image limits must be positive and aggregate pixels must cover at least one image."
        case .invalidDimensions(let path, let width, let height):
            "Image header at \(path) has invalid dimensions \(width)x\(height)."
        case .dimensionLimitExceeded(let path, let width, let height, let maximum):
            "Image at \(path) is \(width)x\(height); native VFX inputs are limited to \(maximum) pixels per side."
        case .pixelLimitExceeded(let path, let pixels, let maximum):
            "Image at \(path) expands to \(pixels) pixels; native VFX inputs are limited to \(maximum) pixels per image."
        case .aggregatePixelLimitExceeded(let pixels, let maximum):
            "Native VFX inputs expand to \(pixels) pixels in total; the limit is \(maximum)."
        case .encodedByteLimitExceeded(let path, let bytes, let maximum):
            "Encoded VFX input at \(path) is \(bytes) bytes; the per-file limit is \(maximum)."
        case .aggregateEncodedByteLimitExceeded(let bytes, let maximum):
            "Encoded VFX inputs total \(bytes) bytes; the aggregate limit is \(maximum)."
        }
    }
}

public enum VFXImageInputValidator {
    /// Reads image headers and rejects oversized decoded dimensions before a
    /// decoder allocates the full RGBA image.
    @discardableResult
    public static func inspectAndValidate(
        _ urls: [URL],
        limits: VFXImageInputLimits = .production
    ) throws -> [VFXImageInputDimensions] {
        try validate(limits: limits)
        var dimensions: [VFXImageInputDimensions] = []
        dimensions.reserveCapacity(urls.count)
        var aggregatePixels = 0
        for sourceURL in urls {
            let url = sourceURL.standardizedFileURL
            let size = try MediaImageIO.size(of: url)
            let inspected = try validate(
                width: size.width,
                height: size.height,
                path: url.path,
                limits: limits
            )
            let sum = aggregatePixels.addingReportingOverflow(inspected.pixelCount)
            guard !sum.overflow, sum.partialValue <= limits.maximumAggregatePixels else {
                throw VFXImageInputValidationError.aggregatePixelLimitExceeded(
                    pixels: sum.overflow ? Int.max : sum.partialValue,
                    maximum: limits.maximumAggregatePixels
                )
            }
            aggregatePixels = sum.partialValue
            dimensions.append(inspected)
        }
        return dimensions
    }

    @discardableResult
    public static func validate(
        width: Int,
        height: Int,
        path: String,
        limits: VFXImageInputLimits = .production
    ) throws -> VFXImageInputDimensions {
        try validate(limits: limits)
        guard width > 0, height > 0 else {
            throw VFXImageInputValidationError.invalidDimensions(
                path: path,
                width: width,
                height: height
            )
        }
        guard width <= limits.maximumDimension, height <= limits.maximumDimension else {
            throw VFXImageInputValidationError.dimensionLimitExceeded(
                path: path,
                width: width,
                height: height,
                maximum: limits.maximumDimension
            )
        }
        let pixels = width.multipliedReportingOverflow(by: height)
        guard !pixels.overflow else {
            throw VFXImageInputValidationError.pixelLimitExceeded(
                path: path,
                pixels: Int.max,
                maximum: limits.maximumPixelsPerImage
            )
        }
        guard pixels.partialValue <= limits.maximumPixelsPerImage else {
            throw VFXImageInputValidationError.pixelLimitExceeded(
                path: path,
                pixels: pixels.partialValue,
                maximum: limits.maximumPixelsPerImage
            )
        }
        return VFXImageInputDimensions(
            width: width,
            height: height,
            pixelCount: pixels.partialValue
        )
    }

    private static func validate(limits: VFXImageInputLimits) throws {
        guard limits.maximumDimension > 0,
              limits.maximumPixelsPerImage > 0,
              limits.maximumAggregatePixels >= limits.maximumPixelsPerImage,
              limits.maximumEncodedBytesPerImage > 0,
              limits.maximumAggregateEncodedBytes >= limits.maximumEncodedBytesPerImage else {
            throw VFXImageInputValidationError.invalidLimits
        }
    }
}

/// Private immutable copies of caller-controlled images.
///
/// A generator validates and decodes these snapshots, then persists the
/// records below. Replacing or mutating an original path after admission can
/// therefore neither bypass decoded-size limits nor change input provenance.
struct VFXImageInputSnapshotBatch: Sendable {
    let rootURL: URL
    let sourceURLs: [URL]
    let snapshotURLs: [URL]
    let inputRecords: [MeshInputRecord]
    let dimensions: [VFXImageInputDimensions]

    static func capture(
        _ sourceURLs: [URL],
        limits: VFXImageInputLimits = .production,
        fileManager: FileManager = .default
    ) throws -> VFXImageInputSnapshotBatch {
        let sources = sourceURLs.map(\.standardizedFileURL)
        let root = fileManager.temporaryDirectory.appendingPathComponent(
            "mere-vfx-inputs-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        var committed = false
        defer {
            if !committed { try? fileManager.removeItem(at: root) }
        }

        var snapshots: [URL] = []
        var records: [MeshInputRecord] = []
        var aggregateBytes: Int64 = 0
        snapshots.reserveCapacity(sources.count)
        records.reserveCapacity(sources.count)
        for (index, source) in sources.enumerated() {
            let extensionName = source.pathExtension
            let filename = extensionName.isEmpty
                ? String(format: "%06d-input", index)
                : String(format: "%06d-input.%@", index, extensionName)
            let snapshot = root.appendingPathComponent(filename)
            let captured = try copyAndHash(
                source: source,
                destination: snapshot,
                maximumBytes: limits.maximumEncodedBytesPerImage,
                fileManager: fileManager
            )
            let sum = aggregateBytes.addingReportingOverflow(captured.byteCount)
            guard !sum.overflow, sum.partialValue <= limits.maximumAggregateEncodedBytes else {
                throw VFXImageInputValidationError.aggregateEncodedByteLimitExceeded(
                    bytes: sum.overflow ? Int64.max : sum.partialValue,
                    maximum: limits.maximumAggregateEncodedBytes
                )
            }
            aggregateBytes = sum.partialValue
            snapshots.append(snapshot)
            records.append(MeshInputRecord(
                path: source.path,
                byteCount: captured.byteCount,
                sha256: captured.sha256
            ))
        }
        var dimensions: [VFXImageInputDimensions] = []
        dimensions.reserveCapacity(snapshots.count)
        var aggregatePixels = 0
        for index in snapshots.indices {
            let size = try MediaImageIO.size(of: snapshots[index])
            let inspected = try VFXImageInputValidator.validate(
                width: size.width,
                height: size.height,
                path: sources[index].path,
                limits: limits
            )
            let sum = aggregatePixels.addingReportingOverflow(inspected.pixelCount)
            guard !sum.overflow, sum.partialValue <= limits.maximumAggregatePixels else {
                throw VFXImageInputValidationError.aggregatePixelLimitExceeded(
                    pixels: sum.overflow ? Int.max : sum.partialValue,
                    maximum: limits.maximumAggregatePixels
                )
            }
            aggregatePixels = sum.partialValue
            dimensions.append(inspected)
        }
        committed = true
        return VFXImageInputSnapshotBatch(
            rootURL: root,
            sourceURLs: sources,
            snapshotURLs: snapshots,
            inputRecords: records,
            dimensions: dimensions
        )
    }

    func cleanup(fileManager: FileManager = .default) {
        try? fileManager.removeItem(at: rootURL)
    }

    fileprivate static func copyAndHash(
        source: URL,
        destination: URL,
        maximumBytes: Int64,
        fileManager: FileManager
    ) throws -> (byteCount: Int64, sha256: String) {
        guard fileManager.createFile(
            atPath: destination.path,
            contents: nil,
            attributes: [.posixPermissions: NSNumber(value: 0o600)]
        ) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let reader = try FileHandle(forReadingFrom: source)
        let writer = try FileHandle(forWritingTo: destination)
        defer {
            try? reader.close()
            try? writer.close()
        }
        var byteCount: Int64 = 0
        var hasher = SHA256()
        while true {
            let data = try reader.read(upToCount: 1_048_576) ?? Data()
            if data.isEmpty { break }
            let next = byteCount.addingReportingOverflow(Int64(data.count))
            guard !next.overflow, next.partialValue <= maximumBytes else {
                throw VFXImageInputValidationError.encodedByteLimitExceeded(
                    path: source.path,
                    bytes: next.overflow ? Int64.max : next.partialValue,
                    maximum: maximumBytes
                )
            }
            byteCount = next.partialValue
            hasher.update(data: data)
            try writer.write(contentsOf: data)
        }
        try writer.synchronize()
        return (
            byteCount,
            hasher.finalize().map { String(format: "%02x", $0) }.joined()
        )
    }
}

/// Immutable admission for a single non-image VFX input such as a video.
/// The caller supplies the modality-specific encoded-byte ceiling.
struct VFXFileInputSnapshot: Sendable {
    let rootURL: URL
    let sourceURL: URL
    let snapshotURL: URL
    let inputRecord: MeshInputRecord

    static func capture(
        _ sourceURL: URL,
        maximumEncodedBytes: Int64,
        fileManager: FileManager = .default
    ) throws -> VFXFileInputSnapshot {
        guard maximumEncodedBytes > 0 else {
            throw VFXImageInputValidationError.invalidLimits
        }
        let source = sourceURL.standardizedFileURL
        let root = fileManager.temporaryDirectory.appendingPathComponent(
            "mere-vfx-file-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        var committed = false
        defer {
            if !committed { try? fileManager.removeItem(at: root) }
        }
        let extensionName = source.pathExtension
        let filename = extensionName.isEmpty ? "input" : "input.\(extensionName)"
        let snapshot = root.appendingPathComponent(filename)
        let captured = try VFXImageInputSnapshotBatch.copyAndHash(
            source: source,
            destination: snapshot,
            maximumBytes: maximumEncodedBytes,
            fileManager: fileManager
        )
        committed = true
        return VFXFileInputSnapshot(
            rootURL: root,
            sourceURL: source,
            snapshotURL: snapshot,
            inputRecord: MeshInputRecord(
                path: source.path,
                byteCount: captured.byteCount,
                sha256: captured.sha256
            )
        )
    }

    func cleanup(fileManager: FileManager = .default) {
        try? fileManager.removeItem(at: rootURL)
    }
}
