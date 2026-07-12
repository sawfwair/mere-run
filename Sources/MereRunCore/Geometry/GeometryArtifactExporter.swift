import Crypto
import Foundation
import MediaIO

public struct GeometryExportOptions: Equatable, Sendable {
    public let stem: String
    public let maximumPointCount: Int?
    public let pointSamplingStride: Int?
    public let writeNormalFiles: Bool
    public let writeConfidenceFile: Bool

    public init(
        stem: String = "geometry",
        maximumPointCount: Int? = nil,
        pointSamplingStride: Int? = nil,
        writeNormalFiles: Bool = true,
        writeConfidenceFile: Bool = true
    ) {
        let clean = stem
            .replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
        self.stem = clean.isEmpty ? "geometry" : clean
        self.maximumPointCount = maximumPointCount
        self.pointSamplingStride = pointSamplingStride
        self.writeNormalFiles = writeNormalFiles
        self.writeConfidenceFile = writeConfidenceFile
    }
}

public struct GeometryExportResult: Equatable, Sendable {
    public let manifest: GeometryOutputManifest
    public let manifestURL: URL

    public init(manifest: GeometryOutputManifest, manifestURL: URL) {
        self.manifest = manifest
        self.manifestURL = manifestURL
    }
}

public enum GeometryArtifactExporter {
    @discardableResult
    public static func export(
        frame: DenseGeometryFrame,
        inputURL: URL,
        sourceImageURL: URL? = nil,
        outputDirectory: URL,
        provenance: GeometryModelProvenance,
        frameIndex: Int? = nil,
        frameTimeSeconds: Double? = nil,
        createdAt: Date = Date(),
        options: GeometryExportOptions = GeometryExportOptions()
    ) throws -> GeometryExportResult {
        let fileManager = FileManager.default
        let outputDirectory = outputDirectory.standardizedFileURL
        try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        var sourcePixels: [UInt8]?
        if let sourceImageURL {
            let image = try MediaImageIO.decode(sourceImageURL)
            guard image.width == frame.width, image.height == frame.height else {
                throw GeometryError.imageDimensionMismatch(
                    expectedWidth: frame.width,
                    expectedHeight: frame.height,
                    actualWidth: image.width,
                    actualHeight: image.height
                )
            }
            sourcePixels = image.rgba8
        }

        let stem = options.stem
        var artifacts: [GeometryArtifact] = []

        let depthEXR = outputDirectory.appendingPathComponent("\(stem)-depth.exr")
        try OpenEXRWriter.writeFloatChannels(
            [(name: "Z", values: frame.depth)],
            width: frame.width,
            height: frame.height,
            to: depthEXR
        )
        artifacts.append(try artifact(.depthEXR, url: depthEXR, root: outputDirectory, mediaType: "image/x-exr"))

        let depthPreview = outputDirectory.appendingPathComponent("\(stem)-depth.png")
        try GeometryPreviewWriter.writeDepth(frame, to: depthPreview)
        artifacts.append(try artifact(.depthPreview, url: depthPreview, root: outputDirectory, mediaType: "image/png"))

        if options.writeNormalFiles, let normals = frame.normals {
            let nx = strideChannel(normals, channel: 0)
            let ny = strideChannel(normals, channel: 1)
            let nz = strideChannel(normals, channel: 2)
            let normalEXR = outputDirectory.appendingPathComponent("\(stem)-normal.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "NX", values: nx), (name: "NY", values: ny), (name: "NZ", values: nz)],
                width: frame.width,
                height: frame.height,
                to: normalEXR
            )
            artifacts.append(try artifact(.normalEXR, url: normalEXR, root: outputDirectory, mediaType: "image/x-exr"))

            let normalPreview = outputDirectory.appendingPathComponent("\(stem)-normal.png")
            try GeometryPreviewWriter.writeNormals(frame, to: normalPreview)
            artifacts.append(try artifact(.normalPreview, url: normalPreview, root: outputDirectory, mediaType: "image/png"))
        }

        let validity = outputDirectory.appendingPathComponent("\(stem)-validity.png")
        try GeometryPreviewWriter.writeValidity(frame, to: validity)
        artifacts.append(try artifact(.validityMask, url: validity, root: outputDirectory, mediaType: "image/png"))

        if options.writeConfidenceFile, let confidence = frame.confidence {
            let confidenceEXR = outputDirectory.appendingPathComponent("\(stem)-confidence.exr")
            try OpenEXRWriter.writeFloatChannels(
                [(name: "Y", values: confidence)],
                width: frame.width,
                height: frame.height,
                to: confidenceEXR
            )
            artifacts.append(try artifact(.confidenceEXR, url: confidenceEXR, root: outputDirectory, mediaType: "image/x-exr"))
        }

        let pointCloud = outputDirectory.appendingPathComponent("\(stem)-points.ply")
        try GeometryPLYWriter.write(
            frame: frame,
            rgba8: sourcePixels,
            options: GeometryPLYOptions(
                maximumPointCount: options.maximumPointCount,
                samplingStride: options.pointSamplingStride
            ),
            to: pointCloud
        )
        artifacts.append(try artifact(.pointCloud, url: pointCloud, root: outputDirectory, mediaType: "application/ply"))

        let camera = GeometryCameraManifest(intrinsics: frame.intrinsics, extrinsics: frame.extrinsics)
        let cameraURL = outputDirectory.appendingPathComponent("\(stem)-camera.json")
        try encode(camera, to: cameraURL)
        artifacts.append(try artifact(.camera, url: cameraURL, root: outputDirectory, mediaType: "application/json"))

        let statistics = try GeometryProjection.depthStatistics(for: frame)
        let manifest = GeometryOutputManifest(
            createdAt: createdAt,
            inputPath: inputURL.standardizedFileURL.path,
            outputDirectory: outputDirectory.path,
            frameIndex: frameIndex,
            frameTimeSeconds: frameTimeSeconds,
            width: frame.width,
            height: frame.height,
            units: frame.units,
            coordinateSystem: frame.coordinateSystem,
            model: provenance,
            camera: camera,
            depthStatistics: statistics,
            artifacts: artifacts.sorted { lhs, rhs in
                if lhs.kind.rawValue == rhs.kind.rawValue { return lhs.relativePath < rhs.relativePath }
                return lhs.kind.rawValue < rhs.kind.rawValue
            }
        )
        let manifestURL = outputDirectory.appendingPathComponent("\(stem)-manifest.json")
        try encode(manifest, to: manifestURL)
        return GeometryExportResult(manifest: manifest, manifestURL: manifestURL)
    }

    private static func strideChannel(_ values: [Float], channel: Int) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(values.count / 3)
        for index in Swift.stride(from: channel, to: values.count, by: 3) {
            output.append(values[index])
        }
        return output
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
        let rootPath = root.standardizedFileURL.path.hasSuffix("/")
            ? root.standardizedFileURL.path
            : root.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        return path.hasPrefix(rootPath) ? String(path.dropFirst(rootPath.count)) : url.lastPathComponent
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

    private static func encode<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: .atomic)
    }
}
