import Foundation
import MediaIO
@preconcurrency import MLX

public struct MoGe2InferenceConfiguration: Equatable, Sendable {
    public let resolutionLevel: Int
    public let tokenCount: Int?
    public let maximumPointCount: Int?

    public init(
        resolutionLevel: Int = 9,
        tokenCount: Int? = nil,
        maximumPointCount: Int? = nil
    ) {
        self.resolutionLevel = min(9, max(0, resolutionLevel))
        self.tokenCount = tokenCount.map { max(1, $0) }
        self.maximumPointCount = maximumPointCount.map { max(1, $0) }
    }

    public var effectiveTokenCount: Int {
        tokenCount ?? Int(1_200 + (Double(resolutionLevel) / 9) * 2_400)
    }
}

public struct MoGe2RunResult: Sendable {
    public let export: GeometryExportResult
    public let focalShift: MoGe2FocalShiftSolution
    public let metricScale: Double
    public let tokenCount: Int
    public let modelLoadSeconds: Double
    public let inferenceSeconds: Double
    public let postprocessSeconds: Double
}

public enum MoGe2GeneratorError: Error, Equatable, LocalizedError, Sendable {
    case modelFileNotFound(String)
    case unsupportedBatch

    public var errorDescription: String? {
        switch self {
        case .modelFileNotFound(let path): "MoGe-2 model.onnx was not found at \(path)."
        case .unsupportedBatch: "MoGe-2 production export currently accepts one image per run."
        }
    }
}

public actor MoGe2Generator {
    private var loadedModel: MoGe2Model?
    private var loadedModelURL: URL?

    public init() {}

    public func generate(
        imageURL: URL,
        outputDirectory: URL,
        model: String? = nil,
        configuration: MoGe2InferenceConfiguration = MoGe2InferenceConfiguration(),
        progress: (@Sendable (String) -> Void)? = nil
    ) async throws -> MoGe2RunResult {
        let standardizedImage = imageURL.standardizedFileURL
        progress?("Resolving pinned MoGe-2 weights")
        let loadStart = Date()
        let nativeModel = try await loadModelIfNeeded(requestedModel: model)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        progress?("Decoding \(standardizedImage.lastPathComponent)")
        let image = try MediaImageIO.decode(standardizedImage)
        let input = Self.rgbNHWC(image)
        let tokenCount = configuration.effectiveTokenCount

        progress?("Running native MLX geometry inference at \(tokenCount) tokens")
        let inferenceStart = Date()
        let raw = nativeModel(input, tokenCount: tokenCount)
        MLX.eval(raw.points, raw.normals, raw.maskProbability, raw.metricScale)
        let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

        progress?("Recovering camera and metric geometry")
        let postprocessStart = Date()
        let processed = try MoGe2Postprocessor.process(raw)
        let postprocessSeconds = Date().timeIntervalSince(postprocessStart)

        progress?("Writing EXR, PNG, camera, and point-cloud artifacts")
        let provenance = GeometryModelProvenance(
            modelID: GeometryModelPins.moge2Small.modelID,
            upstreamRepository: GeometryModelPins.moge2Small.repository,
            upstreamRevision: GeometryModelPins.moge2Small.revision,
            license: GeometryModelPins.moge2Small.license,
            weightsSHA256: GeometryModelPins.moge2Small.artifacts[0].sha256
        )
        let export = try GeometryArtifactExporter.export(
            frame: processed.frame,
            inputURL: standardizedImage,
            sourceImageURL: standardizedImage,
            outputDirectory: outputDirectory,
            provenance: provenance,
            options: GeometryExportOptions(
                stem: standardizedImage.deletingPathExtension().lastPathComponent,
                maximumPointCount: configuration.maximumPointCount
            )
        )
        return MoGe2RunResult(
            export: export,
            focalShift: processed.focalShift,
            metricScale: processed.metricScale,
            tokenCount: tokenCount,
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds,
            postprocessSeconds: postprocessSeconds
        )
    }

    public func unload() {
        loadedModel = nil
        loadedModelURL = nil
        MLX.Memory.clearCache()
    }

    private func loadModelIfNeeded(requestedModel: String?) async throws -> MoGe2Model {
        let modelURL: URL
        if let requestedModel, !requestedModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let explicit = URL(fileURLWithPath: requestedModel).standardizedFileURL
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: explicit.path, isDirectory: &isDirectory) else {
                throw MoGe2GeneratorError.modelFileNotFound(explicit.path)
            }
            modelURL = isDirectory.boolValue ? explicit.appendingPathComponent("model.onnx") : explicit
        } else {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: nil,
                defaultModelID: ModelResolver.ModelID.visionGeometryMoGe2Small.rawValue,
                allowAutoDownload: true
            )
            var isDirectory: ObjCBool = false
            let exists = FileManager.default.fileExists(
                atPath: resolved.url.path,
                isDirectory: &isDirectory
            )
            modelURL = exists && isDirectory.boolValue
                ? resolved.url.appendingPathComponent("model.onnx")
                : resolved.url
        }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw MoGe2GeneratorError.modelFileNotFound(modelURL.path)
        }
        if let loadedModel, loadedModelURL == modelURL { return loadedModel }

        let canonicalPin = GeometryModelPins.moge2Small.artifacts[0]
        let explicitPin = ModelArtifactPin(
            filename: modelURL.lastPathComponent,
            byteCount: canonicalPin.byteCount,
            sha256: canonicalPin.sha256
        )
        _ = try explicitPin.verify(in: modelURL.deletingLastPathComponent())
        let model = MoGe2Model()
        try MoGe2ONNXWeights.load(model: model, archive: ONNXInitializerArchive(url: modelURL))
        self.loadedModel = model
        self.loadedModelURL = modelURL
        return model
    }

    private static func rgbNHWC(_ image: MediaImage) -> MLXArray {
        var values = [Float](repeating: 0, count: image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            let source = pixel * 4
            let destination = pixel * 3
            values[destination] = Float(image.rgba8[source]) / 255
            values[destination + 1] = Float(image.rgba8[source + 1]) / 255
            values[destination + 2] = Float(image.rgba8[source + 2]) / 255
        }
        return MLXArray(values).reshaped(1, image.height, image.width, 3)
    }
}
