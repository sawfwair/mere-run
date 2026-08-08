import ArgumentParser
import Foundation
@preconcurrency import MLX
import MereRunCore

struct GeoTESSERA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tessera",
        abstract: "Encode local Sentinel-1/2 time series with a native TESSERA v2 student."
    )

    @Argument(help: "Safetensors input containing raw S2/S1 observations and day-of-year tensors.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path for embeddings.")
    var output: String

    @Option(name: [.long], help: "Managed model id or converted TESSERA v2 model root.")
    var model: String?

    @Option(name: [.long], help: "Output dimensions. Students: 16/32/64/128; teacher: 1024.")
    var dimensions: Int?

    @Flag(name: [.long], help: "Validate inputs and model availability without loading weights.")
    var preflight = false

    @Flag(name: [.long], help: "Print structured execution metadata on stdout.")
    var json = false

    mutating func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input safetensors not found: \(inputURL.path)")
        }
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        let checkpoint = try await TESSERAResources.resolve(requestedModel: model)
        let outputDimensions = dimensions ?? checkpoint.source.architecture.representationDimension
        let allowedDimensions = checkpoint.source.variant == .teacher
            ? [1_024]
            : Array(TESSERAModel.supportedOutputDimensions)
        guard allowedDimensions.contains(outputDimensions) else {
            throw ValidationError(
                checkpoint.source.variant == .teacher
                    ? "TESSERA v2 Teacher requires --dimensions 1024"
                    : "TESSERA v2 students require --dimensions 16, 32, 64, or 128"
            )
        }
        let inputs = try MLX.loadArrays(url: inputURL)
        guard let sentinel2 = inputs["S2"], let sentinel2DayOfYear = inputs["S2_DOY"] else {
            throw ValidationError("Input safetensors must contain S2 and S2_DOY tensors")
        }
        let preparedSentinel2 = try TESSERAPreprocessor.prepareSentinel2(
            bands: sentinel2,
            dayOfYear: sentinel2DayOfYear
        )
        let preparedSentinel1 = try TESSERAPreprocessor.prepareSentinel1(
            ascendingBands: inputs["S1_ASC"],
            ascendingDayOfYear: inputs["S1_ASC_DOY"],
            descendingBands: inputs["S1_DESC"],
            descendingDayOfYear: inputs["S1_DESC_DOY"],
            variant: checkpoint.source.variant
        )
        guard preparedSentinel2.dim(0) == preparedSentinel1.dim(0) else {
            throw ValidationError("S2 and S1 inputs must have the same batch size")
        }
        guard preparedSentinel2.dim(1) <= checkpoint.source.architecture.maximumSequenceLength,
              preparedSentinel1.dim(1) <= checkpoint.source.architecture.maximumSequenceLength else {
            throw ValidationError(
                "TESSERA sequences may contain at most \(checkpoint.source.architecture.maximumSequenceLength) observations"
            )
        }
        let plan = GeoModelPayload(
            status: preflight ? "ready" : "completed",
            operation: "time-series-embedding",
            modelID: checkpoint.source.modelID,
            variant: checkpoint.source.variant.rawValue,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: preparedSentinel2.dim(0),
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
        let nativeModel = try TESSERAResources.loadModel(from: checkpoint)
        let loadSeconds = GeoCommandSupport.seconds(since: loadStart)
        let inferenceStart = ContinuousClock.now
        let embeddings = try nativeModel(
            s2: preparedSentinel2,
            s1: preparedSentinel1,
            outputDimensions: outputDimensions
        )
        MLX.eval(embeddings)
        let inferenceSeconds = GeoCommandSupport.seconds(since: inferenceStart)
        try MLX.save(
            arrays: ["embeddings": embeddings.asType(.float32)],
            metadata: [
                "format": "mere.run/tessera-v2-embeddings-v1",
                "model_id": checkpoint.source.modelID,
                "source_revision": checkpoint.source.sourceRevision,
                "dimensions": String(outputDimensions),
            ],
            url: outputURL
        )
        let result = GeoModelPayload(
            status: "completed",
            operation: "time-series-embedding",
            modelID: checkpoint.source.modelID,
            variant: checkpoint.source.variant.rawValue,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: preparedSentinel2.dim(0),
            device: "metal",
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds
        )
        print(json ? try GeoCommandSupport.jsonString(result) : outputURL.path)
    }
}
