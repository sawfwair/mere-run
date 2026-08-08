import ArgumentParser
import Foundation
@preconcurrency import MLX
import MereRunCore

struct GeoOlmoEarth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "olmoearth",
        abstract: "Encode multisensor Earth observations with native OlmoEarth v1.2."
    )

    @Argument(help: "Safetensors input containing TIMESTAMPS and one or more raw imagery tensors.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path for spatial embeddings.")
    var output: String

    @Option(name: [.long], help: "Managed model id or converted OlmoEarth v1.2 model root.")
    var model: String?

    @Option(name: [.long], help: "Spatial patch size: 1, 2, 4, or 8 pixels.")
    var patchSize = 4

    @Option(name: [.long], help: "Input ground sample distance in meters.")
    var inputResolution: Float = 10

    @Flag(name: [.long], help: "Also write full space-time tokens in addition to time-pooled grids.")
    var includeTokens = false

    @Flag(name: [.long], help: "Validate inputs and model availability without loading weights.")
    var preflight = false

    @Flag(name: [.long], help: "Print structured execution metadata on stdout.")
    var json = false

    mutating func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        guard [1, 2, 4, 8].contains(patchSize) else {
            throw ValidationError("--patch-size must be 1, 2, 4, or 8")
        }
        guard inputResolution > 0 else {
            throw ValidationError("--input-resolution must be greater than zero")
        }
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input safetensors not found: \(inputURL.path)")
        }
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        let checkpoint = try await OlmoEarthResources.resolve(requestedModel: model)
        guard patchSize <= checkpoint.source.architecture.maximumPatchSize else {
            throw ValidationError(
                "Model \(checkpoint.source.modelID) supports patch sizes up to "
                    + "\(checkpoint.source.architecture.maximumPatchSize)"
            )
        }
        let inputs = try MLX.loadArrays(url: inputURL)
        guard let timestamps = inputs["TIMESTAMPS"] else {
            throw ValidationError("Input safetensors must contain a TIMESTAMPS tensor")
        }
        let rawModalities = Dictionary(uniqueKeysWithValues: OlmoEarthModality.allCases.compactMap { modality in
            inputs[modality.inputTensorName].map { (modality, $0) }
        })
        try Self.validate(
            modalities: rawModalities,
            timestamps: timestamps,
            patchSize: patchSize,
            maximumSequenceLength: checkpoint.source.architecture.maximumSequenceLength
        )
        let plan = GeoModelPayload(
            status: preflight ? "ready" : "completed",
            operation: "multisensor-spatial-embedding",
            modelID: checkpoint.source.modelID,
            variant: checkpoint.source.variant.rawValue,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: timestamps.dim(0),
            device: "metal",
            modelLoadSeconds: nil,
            inferenceSeconds: nil
        )
        if preflight {
            print(try GeoCommandSupport.jsonString(plan))
            return
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let loadStart = ContinuousClock.now
        let nativeModel = try OlmoEarthResources.loadModel(from: checkpoint)
        let loadSeconds = GeoCommandSupport.seconds(since: loadStart)
        var normalized: [OlmoEarthModality: MLXArray] = [:]
        for (modality, value) in rawModalities {
            normalized[modality] = OlmoEarthPreprocessor.normalize(value, modality: modality)
        }
        let inferenceStart = ContinuousClock.now
        let embeddings = try nativeModel(
            modalities: normalized,
            timestamps: timestamps,
            patchSize: patchSize,
            inputResolutionMeters: inputResolution
        )
        var outputs: [String: MLXArray] = [:]
        for modality in OlmoEarthModality.allCases where rawModalities[modality] != nil {
            outputs[modality.outputTensorName] = embeddings.pooled[modality]!.asType(.float32)
            if includeTokens {
                outputs["\(modality.outputTensorName)_TOKENS"] = embeddings.tokens[modality]!.asType(.float32)
            }
        }
        MLX.eval(Array(outputs.values))
        let inferenceSeconds = GeoCommandSupport.seconds(since: inferenceStart)
        try MLX.save(
            arrays: outputs,
            metadata: [
                "format": "mere.run/olmoearth-v1.2-embeddings-v1",
                "model_id": checkpoint.source.modelID,
                "source_revision": checkpoint.source.sourceRevision,
                "patch_size": String(patchSize),
                "input_resolution_meters": String(inputResolution),
            ],
            url: outputURL
        )
        let result = GeoModelPayload(
            status: "completed",
            operation: "multisensor-spatial-embedding",
            modelID: checkpoint.source.modelID,
            variant: checkpoint.source.variant.rawValue,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: timestamps.dim(0),
            device: "metal",
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds
        )
        print(json ? try GeoCommandSupport.jsonString(result) : outputURL.path)
    }

    private static func validate(
        modalities: [OlmoEarthModality: MLXArray],
        timestamps: MLXArray,
        patchSize: Int,
        maximumSequenceLength: Int
    ) throws {
        guard !modalities.isEmpty else {
            throw ValidationError("Input must contain at least one of S2L2A, S1RTC, or LANDSAT")
        }
        guard timestamps.ndim == 3,
              timestamps.dim(0) > 0,
              timestamps.dim(1) > 0,
              timestamps.dim(1) <= maximumSequenceLength,
              timestamps.dim(2) == 3 else {
            throw ValidationError(
                "TIMESTAMPS must have shape [batch, 1...\(maximumSequenceLength), 3]"
            )
        }
        let timestampValues = timestamps.asType(.int32).asArray(Int32.self)
        for index in stride(from: 0, to: timestampValues.count, by: 3) {
            guard (1...31).contains(Int(timestampValues[index])),
                  (0...11).contains(Int(timestampValues[index + 1])) else {
                throw ValidationError("TIMESTAMPS uses (day 1-31, zero-indexed month 0-11, year)")
            }
        }
        var spatialShape: [Int]?
        for (modality, value) in modalities {
            guard value.ndim == 5,
                  value.dim(0) == timestamps.dim(0),
                  value.dim(1) > 0,
                  value.dim(2) > 0,
                  value.dim(3) == timestamps.dim(1),
                  value.dim(4) == modality.channelCount else {
                throw ValidationError(
                    "\(modality.inputTensorName) must have shape "
                        + "[batch, height, width, timestamps, \(modality.channelCount)]"
                )
            }
            guard value.dim(1).isMultiple(of: patchSize), value.dim(2).isMultiple(of: patchSize) else {
                throw ValidationError("Imagery height and width must be divisible by --patch-size")
            }
            let current = [value.dim(1), value.dim(2)]
            if let spatialShape, spatialShape != current {
                throw ValidationError("All imagery modalities must use the same height and width")
            }
            spatialShape = current
        }
    }
}
