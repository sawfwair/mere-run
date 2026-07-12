import Crypto
import Foundation

public struct DepthSequenceExportResult: Equatable, Sendable {
    public let manifest: DepthSequenceManifest
    public let manifestURL: URL

    public init(manifest: DepthSequenceManifest, manifestURL: URL) {
        self.manifest = manifest
        self.manifestURL = manifestURL
    }
}

public enum DepthSequenceArtifactExporter {
    @discardableResult
    public static func export(
        frames: [DepthSequenceFrame],
        inputURL: URL,
        outputDirectory: URL,
        fps: Double,
        semantics: DepthSemantics,
        provenance: GeometryModelProvenance,
        temporalWindowLength: Int = VideoDepthAnythingWindowing.windowLength,
        temporalOverlap: Int = VideoDepthAnythingWindowing.overlap,
        createdAt: Date = Date()
    ) throws -> DepthSequenceExportResult {
        guard let first = frames.first else {
            throw VideoDepthAnythingWindowingError.invalidOriginalFrameCount(0)
        }
        for frame in frames {
            guard frame.width == first.width, frame.height == first.height else {
                throw GeometryError.imageDimensionMismatch(
                    expectedWidth: first.width,
                    expectedHeight: first.height,
                    actualWidth: frame.width,
                    actualHeight: frame.height
                )
            }
        }
        let outputDirectory = outputDirectory.standardizedFileURL
        let framesDirectory = outputDirectory.appendingPathComponent("frames", isDirectory: true)
        try FileManager.default.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
        let previewRange = try globalPreviewRange(frames)
        var frameManifests: [DepthSequenceFrameManifest] = []
        frameManifests.reserveCapacity(frames.count)

        for frame in frames {
            let prefix = String(format: "%06d", frame.index)
            let depthURL = framesDirectory.appendingPathComponent("\(prefix)-depth.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "Z", values: frame.depth)],
                width: frame.width,
                height: frame.height,
                to: depthURL
            )
            var artifacts = [try artifact(.depthEXR, url: depthURL, root: outputDirectory, mediaType: "image/x-exr")]

            let previewURL = framesDirectory.appendingPathComponent("\(prefix)-depth.png")
            try GeometryPreviewWriter.writeDepth(
                frame.depth,
                width: frame.width,
                height: frame.height,
                near: previewRange.near,
                far: previewRange.far,
                to: previewURL
            )
            artifacts.append(try artifact(.depthPreview, url: previewURL, root: outputDirectory, mediaType: "image/png"))

            var confidencePath: String?
            if let confidence = frame.confidence {
                let confidenceURL = framesDirectory.appendingPathComponent("\(prefix)-confidence.exr")
                try OpenEXRWriter.writeFloatChannels(
                    [(name: "Y", values: confidence)],
                    width: frame.width,
                    height: frame.height,
                    to: confidenceURL
                )
                let record = try artifact(
                    .confidenceEXR,
                    url: confidenceURL,
                    root: outputDirectory,
                    mediaType: "image/x-exr"
                )
                artifacts.append(record)
                confidencePath = record.relativePath
            }
            artifacts.sort { $0.kind.rawValue < $1.kind.rawValue }
            frameManifests.append(
                DepthSequenceFrameManifest(
                    index: frame.index,
                    timeSeconds: frame.timeSeconds,
                    depthPath: relativePath(of: depthURL, under: outputDirectory),
                    previewPath: relativePath(of: previewURL, under: outputDirectory),
                    confidencePath: confidencePath,
                    intrinsics: frame.intrinsics,
                    artifacts: artifacts
                )
            )
        }

        let manifest = DepthSequenceManifest(
            createdAt: createdAt,
            inputPath: inputURL.standardizedFileURL.path,
            outputDirectory: outputDirectory.path,
            width: first.width,
            height: first.height,
            fps: fps,
            semantics: semantics,
            model: provenance,
            temporalWindowLength: temporalWindowLength,
            temporalOverlap: temporalOverlap,
            frames: frameManifests
        )
        let manifestURL = outputDirectory.appendingPathComponent("depth-sequence-manifest.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return DepthSequenceExportResult(manifest: manifest, manifestURL: manifestURL)
    }

    private static func globalPreviewRange(_ frames: [DepthSequenceFrame]) throws -> (near: Float, far: Float) {
        let totalElementCount = frames.reduce(0) { $0 + $1.depth.count }
        let stride = max(1, totalElementCount / 2_000_000)
        var valid: [Float] = []
        valid.reserveCapacity(min(totalElementCount, 2_000_000))
        var globalIndex = 0
        for frame in frames {
            for value in frame.depth {
                if globalIndex % stride == 0, value.isFinite, value > 0 {
                    valid.append(value)
                }
                globalIndex += 1
            }
        }
        guard !valid.isEmpty else { throw GeometryError.noValidGeometry }
        valid.sort()
        return (percentile(valid, fraction: 0.02), percentile(valid, fraction: 0.98))
    }

    private static func percentile(_ sorted: [Float], fraction: Double) -> Float {
        let index = min(sorted.count - 1, max(0, Int((Double(sorted.count - 1) * fraction).rounded())))
        return sorted[index]
    }

    private static func artifact(
        _ kind: GeometryArtifactKind,
        url: URL,
        root: URL,
        mediaType: String
    ) throws -> GeometryArtifact {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? 0
        return GeometryArtifact(
            kind: kind,
            relativePath: relativePath(of: url, under: root),
            mediaType: mediaType,
            byteCount: byteCount,
            sha256: try sha256(url)
        )
    }

    private static func relativePath(of url: URL, under root: URL) -> String {
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return url.path.hasPrefix(rootPath) ? String(url.path.dropFirst(rootPath.count)) : url.lastPathComponent
    }

    private static func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1_048_576) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
