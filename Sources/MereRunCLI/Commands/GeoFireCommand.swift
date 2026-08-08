import ArgumentParser
import Foundation
@preconcurrency import MLX
import MereRunCore

struct GeoFire: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fire",
        abstract: "Run native TerraMind Fire tile inference with MLX on Apple silicon."
    )

    @Argument(help: "Safetensors input containing normalized S2L2A, S1RTC, and DEM tile batches.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path for fire logits.")
    var output: String

    @Option(name: [.long], help: "Managed model id or converted TerraMind Fire model root.")
    var model: String?

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
        let checkpoint = try await TerraMindFireResources.resolve(requestedModel: model)
        let inputs = try MLX.loadArrays(url: inputURL)
        guard let s2l2a = inputs["S2L2A"], let s1rtc = inputs["S1RTC"], let dem = inputs["DEM"] else {
            throw ValidationError("Input safetensors must contain S2L2A, S1RTC, and DEM tensors")
        }
        let plan = GeoModelPayload(
            status: preflight ? "ready" : "completed",
            operation: "fire-segmentation",
            modelID: checkpoint.configuration.modelID,
            variant: "base",
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: s2l2a.dim(0),
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
        let nativeModel = try TerraMindFireResources.loadModel(from: checkpoint)
        let loadSeconds = GeoCommandSupport.seconds(since: loadStart)
        let inferenceStart = ContinuousClock.now
        let logits = try nativeModel(s2l2a: s2l2a, s1rtc: s1rtc, dem: dem)
        MLX.eval(logits)
        let inferenceSeconds = GeoCommandSupport.seconds(since: inferenceStart)
        try MLX.save(
            arrays: ["logits": logits.asType(.float32)],
            metadata: [
                "format": "mere.run/terramind-fire-logits-v1",
                "model_id": checkpoint.configuration.modelID,
                "source_revision": TerraMindFireResources.sourceRevision,
            ],
            url: outputURL
        )
        let result = GeoModelPayload(
            status: "completed",
            operation: "fire-segmentation",
            modelID: checkpoint.configuration.modelID,
            variant: "base",
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: s2l2a.dim(0),
            device: "metal",
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds
        )
        print(json ? try GeoCommandSupport.jsonString(result) : outputURL.path)
    }
}
