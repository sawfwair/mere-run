import Foundation

/// Provenance and artifact record for one Marigold V2 depth run.
public struct MarigoldV2DepthManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let createdAt: Date
    public let inputPath: String
    public let inputByteCount: Int64
    public let inputSHA256: String
    public let outputDirectory: String
    /// Size of the written artifacts, matching the source image.
    public let width: Int
    public let height: Int
    /// Size the transformer actually ran at, after alignment to the patch grid.
    public let inferenceWidth: Int
    public let inferenceHeight: Int
    public let semantics: DepthSemantics
    public let parameterization: MarigoldV2DepthParameterization
    public let checkpoint: String
    public let seeThrough: Bool
    public let depthStatistics: MarigoldV2DepthStatistics
    public let model: GeometryModelProvenance
    public let artifacts: [GeometryArtifact]

    public init(
        schemaVersion: Int = 1,
        createdAt: Date = Date(),
        inputPath: String,
        inputByteCount: Int64,
        inputSHA256: String,
        outputDirectory: String,
        width: Int,
        height: Int,
        inferenceWidth: Int,
        inferenceHeight: Int,
        semantics: DepthSemantics = .affineRelative,
        parameterization: MarigoldV2DepthParameterization,
        checkpoint: String,
        seeThrough: Bool,
        depthStatistics: MarigoldV2DepthStatistics,
        model: GeometryModelProvenance,
        artifacts: [GeometryArtifact]
    ) {
        self.schemaVersion = schemaVersion
        self.createdAt = createdAt
        self.inputPath = inputPath
        self.inputByteCount = inputByteCount
        self.inputSHA256 = inputSHA256
        self.outputDirectory = outputDirectory
        self.width = width
        self.height = height
        self.inferenceWidth = inferenceWidth
        self.inferenceHeight = inferenceHeight
        self.semantics = semantics
        self.parameterization = parameterization
        self.checkpoint = checkpoint
        self.seeThrough = seeThrough
        self.depthStatistics = depthStatistics
        self.model = model
        self.artifacts = artifacts
    }
}

public struct MarigoldV2DepthExportResult: Equatable, Sendable {
    public let manifest: MarigoldV2DepthManifest
    public let manifestURL: URL

    public init(manifest: MarigoldV2DepthManifest, manifestURL: URL) {
        self.manifest = manifest
        self.manifestURL = manifestURL
    }
}

/// Writes the EXR, preview, and manifest for a single affine-relative depth map.
///
/// No point cloud or camera is written: Marigold recovers depth up to an unknown
/// scale and shift and never estimates intrinsics, so projecting its output would
/// imply a camera the model does not predict.
public enum MarigoldV2DepthArtifactExporter {
    @discardableResult
    public static func export(
        depth: [Float],
        width: Int,
        height: Int,
        inferenceWidth: Int,
        inferenceHeight: Int,
        statistics: MarigoldV2DepthStatistics,
        checkpoint: MarigoldV2DepthCheckpoint,
        inputURL: URL,
        outputDirectory: URL,
        provenance: GeometryModelProvenance,
        inputRecord admittedInputRecord: MeshInputRecord? = nil,
        createdAt: Date = Date(),
        stem: String? = nil
    ) throws -> MarigoldV2DepthExportResult {
        let pixelCount = width * height
        guard depth.count == pixelCount else {
            throw GeometryError.invalidElementCount(
                field: "marigold depth",
                expected: pixelCount,
                actual: depth.count
            )
        }

        let fileManager = FileManager.default
        let outputDirectory = outputDirectory.standardizedFileURL
        let standardizedInput = inputURL.standardizedFileURL
        let inputRecord: MeshInputRecord
        if let admittedInputRecord {
            guard admittedInputRecord.path == standardizedInput.path else {
                throw MeshInputProvenanceError.inputRecordPathMismatch(
                    expected: standardizedInput.path,
                    actual: admittedInputRecord.path
                )
            }
            inputRecord = admittedInputRecord
        } else {
            inputRecord = MeshInputRecord(
                path: standardizedInput.path,
                byteCount: try ModelArtifactPin.fileByteCount(standardizedInput),
                sha256: try ModelArtifactPin.fileSHA256(standardizedInput)
            )
        }
        try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let resolvedStem = sanitize(stem ?? standardizedInput.deletingPathExtension().lastPathComponent)
        var artifacts: [GeometryArtifact] = []

        let depthURL = outputDirectory.appendingPathComponent("\(resolvedStem)-depth.exr")
        try OpenEXRWriter.writeFloatChannels(
            [(name: "Z", values: depth)],
            width: width,
            height: height,
            to: depthURL
        )
        artifacts.append(
            try artifact(.depthEXR, url: depthURL, root: outputDirectory, mediaType: "image/x-exr")
        )

        let previewURL = outputDirectory.appendingPathComponent("\(resolvedStem)-depth.png")
        try GeometryPreviewWriter.writeDepth(
            depth,
            width: width,
            height: height,
            near: statistics.normalizedFloor,
            far: 1,
            to: previewURL
        )
        artifacts.append(
            try artifact(.depthPreview, url: previewURL, root: outputDirectory, mediaType: "image/png")
        )

        let manifest = MarigoldV2DepthManifest(
            createdAt: createdAt,
            inputPath: inputRecord.path,
            inputByteCount: inputRecord.byteCount,
            inputSHA256: inputRecord.sha256,
            outputDirectory: outputDirectory.path,
            width: width,
            height: height,
            inferenceWidth: inferenceWidth,
            inferenceHeight: inferenceHeight,
            parameterization: checkpoint.parameterization,
            checkpoint: checkpoint.rawValue,
            seeThrough: checkpoint.isSeeThrough,
            depthStatistics: statistics,
            model: provenance,
            artifacts: artifacts
        )

        let manifestURL = outputDirectory.appendingPathComponent("\(resolvedStem)-depth.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)

        return MarigoldV2DepthExportResult(manifest: manifest, manifestURL: manifestURL)
    }

    static func sanitize(_ stem: String) -> String {
        let cleaned = stem
            .replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
        return cleaned.isEmpty ? "depth" : cleaned
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
