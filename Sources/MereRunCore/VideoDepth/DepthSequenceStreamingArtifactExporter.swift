import Foundation

public enum DepthSequenceStreamingExporterError: Error, Equatable, LocalizedError, Sendable {
    case invalidExpectedFrameCount(Int)
    case invalidFrameRate(Double)
    case nonSequentialFrame(expected: Int, actual: Int)
    case tooManyFrames(expected: Int)
    case incompleteSequence(expected: Int, actual: Int)
    case alreadyFinalized
    case corruptSpool(path: String, expectedBytes: Int, actualBytes: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidExpectedFrameCount(let count):
            "Depth sequence frame count must be positive; received \(count)."
        case .invalidFrameRate(let fps):
            "Depth sequence frame rate must be finite and positive; received \(fps)."
        case .nonSequentialFrame(let expected, let actual):
            "Depth sequence frames must be appended in order; expected \(expected), received \(actual)."
        case .tooManyFrames(let expected):
            "Depth sequence already contains the expected \(expected) frames."
        case .incompleteSequence(let expected, let actual):
            "Depth sequence is incomplete: expected \(expected) frames, received \(actual)."
        case .alreadyFinalized:
            "Depth sequence export has already been finalized."
        case .corruptSpool(let path, let expectedBytes, let actualBytes):
            "Depth spool \(path) expected \(expectedBytes) bytes, found \(actualBytes)."
        }
    }
}

/// Bounded-memory depth-sequence writer.
///
/// Aligned frames are immediately spooled as raw float32 and released. Final
/// EXRs and globally normalized previews are produced one frame at a time after
/// the last temporal window, while at most two million sampled depth values are
/// retained for the shared preview range.
public final class DepthSequenceStreamingArtifactExporter {
    private struct SpooledFrame {
        let index: Int
        let timeSeconds: Double
        let intrinsics: GeometryCameraIntrinsics?
        let depthURL: URL
        let confidenceURL: URL?
    }

    public let expectedFrameCount: Int
    public let width: Int
    public let height: Int

    private let inputURL: URL
    private let inputRecord: MeshInputRecord
    private let outputDirectory: URL
    private let fps: Double
    private let semantics: DepthSemantics
    private let provenance: GeometryModelProvenance
    private let temporalWindowLength: Int
    private let temporalOverlap: Int
    private let createdAt: Date
    private let spoolDirectory: URL
    private let sampleStride: Int
    private var globalElementIndex = 0
    private var previewSamples: [Float] = []
    private var frames: [SpooledFrame] = []
    private var finalized = false
    private var cleaned = false

    public init(
        expectedFrameCount: Int,
        inputURL: URL,
        outputDirectory: URL,
        width: Int,
        height: Int,
        fps: Double,
        semantics: DepthSemantics,
        provenance: GeometryModelProvenance,
        temporalWindowLength: Int = VideoDepthAnythingWindowing.windowLength,
        temporalOverlap: Int = VideoDepthAnythingWindowing.overlap,
        inputRecord admittedInputRecord: MeshInputRecord? = nil,
        createdAt: Date = Date()
    ) throws {
        guard expectedFrameCount > 0 else {
            throw DepthSequenceStreamingExporterError.invalidExpectedFrameCount(expectedFrameCount)
        }
        guard width > 0, height > 0 else {
            throw GeometryError.invalidDimensions(width: width, height: height)
        }
        guard fps.isFinite, fps > 0 else {
            throw DepthSequenceStreamingExporterError.invalidFrameRate(fps)
        }
        self.expectedFrameCount = expectedFrameCount
        let standardizedInput = inputURL.standardizedFileURL
        self.inputURL = standardizedInput
        if let admittedInputRecord {
            guard admittedInputRecord.path == standardizedInput.path else {
                throw MeshInputProvenanceError.inputRecordPathMismatch(
                    expected: standardizedInput.path,
                    actual: admittedInputRecord.path
                )
            }
            self.inputRecord = admittedInputRecord
        } else {
            self.inputRecord = MeshInputRecord(
                path: standardizedInput.path,
                byteCount: try ModelArtifactPin.fileByteCount(standardizedInput),
                sha256: try ModelArtifactPin.fileSHA256(standardizedInput)
            )
        }
        self.outputDirectory = outputDirectory.standardizedFileURL
        self.width = width
        self.height = height
        self.fps = fps
        self.semantics = semantics
        self.provenance = provenance
        self.temporalWindowLength = temporalWindowLength
        self.temporalOverlap = temporalOverlap
        self.createdAt = createdAt
        spoolDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mere-depth-spool-\(UUID().uuidString)",
            isDirectory: true
        )
        let totalElementCount = expectedFrameCount * width * height
        sampleStride = max(1, (totalElementCount + 1_999_999) / 2_000_000)
        previewSamples.reserveCapacity(min(totalElementCount, 2_000_000))
        frames.reserveCapacity(expectedFrameCount)
        try FileManager.default.createDirectory(at: spoolDirectory, withIntermediateDirectories: true)
    }

    deinit {
        cancel()
    }

    public var spooledFrameCount: Int { frames.count }
    public var sampledDepthValueCount: Int { previewSamples.count }

    public func append(_ frame: DepthSequenceFrame) throws {
        guard !finalized else { throw DepthSequenceStreamingExporterError.alreadyFinalized }
        guard frames.count < expectedFrameCount else {
            throw DepthSequenceStreamingExporterError.tooManyFrames(expected: expectedFrameCount)
        }
        guard frame.index == frames.count else {
            throw DepthSequenceStreamingExporterError.nonSequentialFrame(
                expected: frames.count,
                actual: frame.index
            )
        }
        guard frame.width == width, frame.height == height else {
            throw GeometryError.imageDimensionMismatch(
                expectedWidth: width,
                expectedHeight: height,
                actualWidth: frame.width,
                actualHeight: frame.height
            )
        }

        for value in frame.depth {
            if globalElementIndex.isMultiple(of: sampleStride), value.isFinite, value > 0 {
                previewSamples.append(value)
            }
            globalElementIndex += 1
        }

        let prefix = String(format: "%06d", frame.index)
        let depthURL = spoolDirectory.appendingPathComponent("\(prefix)-depth.f32")
        try Self.float32Data(frame.depth).write(to: depthURL, options: .atomic)
        let confidenceURL: URL?
        if let confidence = frame.confidence {
            let url = spoolDirectory.appendingPathComponent("\(prefix)-confidence.f32")
            try Self.float32Data(confidence).write(to: url, options: .atomic)
            confidenceURL = url
        } else {
            confidenceURL = nil
        }
        frames.append(
            SpooledFrame(
                index: frame.index,
                timeSeconds: frame.timeSeconds,
                intrinsics: frame.intrinsics,
                depthURL: depthURL,
                confidenceURL: confidenceURL
            )
        )
    }

    public func finalize() throws -> DepthSequenceExportResult {
        guard !finalized else { throw DepthSequenceStreamingExporterError.alreadyFinalized }
        guard frames.count == expectedFrameCount else {
            throw DepthSequenceStreamingExporterError.incompleteSequence(
                expected: expectedFrameCount,
                actual: frames.count
            )
        }
        finalized = true
        defer { cancel() }
        guard !previewSamples.isEmpty else { throw GeometryError.noValidGeometry }
        previewSamples.sort()
        let near = Self.percentile(previewSamples, fraction: 0.02)
        let far = Self.percentile(previewSamples, fraction: 0.98)

        let frameDirectory = outputDirectory.appendingPathComponent("frames", isDirectory: true)
        try FileManager.default.createDirectory(at: frameDirectory, withIntermediateDirectories: true)
        var manifests: [DepthSequenceFrameManifest] = []
        manifests.reserveCapacity(frames.count)
        let expectedElementCount = width * height
        for frame in frames {
            let prefix = String(format: "%06d", frame.index)
            let depth = try Self.readFloat32(frame.depthURL, expectedCount: expectedElementCount)
            let depthURL = frameDirectory.appendingPathComponent("\(prefix)-depth.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "Z", values: depth)],
                width: width,
                height: height,
                to: depthURL
            )
            var artifacts = [try Self.artifact(
                .depthEXR,
                url: depthURL,
                root: outputDirectory,
                mediaType: "image/x-exr"
            )]

            let previewURL = frameDirectory.appendingPathComponent("\(prefix)-depth.png")
            try GeometryPreviewWriter.writeDepth(
                depth,
                width: width,
                height: height,
                near: near,
                far: far,
                to: previewURL
            )
            artifacts.append(try Self.artifact(
                .depthPreview,
                url: previewURL,
                root: outputDirectory,
                mediaType: "image/png"
            ))

            var confidencePath: String?
            if let spooledConfidenceURL = frame.confidenceURL {
                let confidence = try Self.readFloat32(
                    spooledConfidenceURL,
                    expectedCount: expectedElementCount
                )
                let confidenceURL = frameDirectory.appendingPathComponent("\(prefix)-confidence.exr")
                try OpenEXRWriter.writeFloatChannels(
                    [(name: "Y", values: confidence)],
                    width: width,
                    height: height,
                    to: confidenceURL
                )
                let record = try Self.artifact(
                    .confidenceEXR,
                    url: confidenceURL,
                    root: outputDirectory,
                    mediaType: "image/x-exr"
                )
                artifacts.append(record)
                confidencePath = record.relativePath
            }
            artifacts.sort { $0.kind.rawValue < $1.kind.rawValue }
            manifests.append(
                DepthSequenceFrameManifest(
                    index: frame.index,
                    timeSeconds: frame.timeSeconds,
                    depthPath: Self.relativePath(of: depthURL, under: outputDirectory),
                    previewPath: Self.relativePath(of: previewURL, under: outputDirectory),
                    confidencePath: confidencePath,
                    intrinsics: frame.intrinsics,
                    artifacts: artifacts
                )
            )
        }

        let manifest = DepthSequenceManifest(
            createdAt: createdAt,
            inputPath: inputRecord.path,
            inputByteCount: inputRecord.byteCount,
            inputSHA256: inputRecord.sha256,
            outputDirectory: outputDirectory.path,
            width: width,
            height: height,
            fps: fps,
            semantics: semantics,
            model: provenance,
            temporalWindowLength: temporalWindowLength,
            temporalOverlap: temporalOverlap,
            frames: manifests
        )
        let manifestURL = outputDirectory.appendingPathComponent("depth-sequence-manifest.json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
        return DepthSequenceExportResult(manifest: manifest, manifestURL: manifestURL)
    }

    public func cancel() {
        guard !cleaned else { return }
        cleaned = true
        try? FileManager.default.removeItem(at: spoolDirectory)
    }

    private static func float32Data(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private static func readFloat32(_ url: URL, expectedCount: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let expectedBytes = expectedCount * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            throw DepthSequenceStreamingExporterError.corruptSpool(
                path: url.path,
                expectedBytes: expectedBytes,
                actualBytes: data.count
            )
        }
        var values = [Float](repeating: 0, count: expectedCount)
        values.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination)
        }
        return values
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
        GeometryArtifact(
            kind: kind,
            relativePath: relativePath(of: url, under: root),
            mediaType: mediaType,
            byteCount: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func relativePath(of url: URL, under root: URL) -> String {
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return url.path.hasPrefix(rootPath)
            ? String(url.path.dropFirst(rootPath.count))
            : url.lastPathComponent
    }
}
