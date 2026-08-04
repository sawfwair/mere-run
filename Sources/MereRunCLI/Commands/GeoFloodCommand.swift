import ArgumentParser
import Foundation
@preconcurrency import MLX
import MereRunCore

struct GeoFlood: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flood",
        abstract: "Run native TerraMind Flood tile inference with MLX on Apple silicon."
    )

    @Argument(help: "Safetensors input containing normalized S2L2A, S1RTC, and DEM tile batches.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output safetensors path for logits.")
    var output: String

    @Option(name: [.long], help: "Managed model id or converted TerraMind Flood model root.")
    var model: String?

    @Flag(name: [.long], help: "Validate inputs and model availability without loading weights.")
    var preflight = false

    @Flag(name: [.long], help: "Print structured execution metadata on stdout.")
    var json = false

    mutating func run() async throws {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input safetensors not found: \(inputURL.path)")
        }
        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        let checkpoint = try await TerraMindFloodResources.resolve(requestedModel: model)
        let inputs = try MLX.loadArrays(url: inputURL)
        guard let s2l2a = inputs["S2L2A"], let s1rtc = inputs["S1RTC"], let dem = inputs["DEM"] else {
            throw ValidationError("Input safetensors must contain S2L2A, S1RTC, and DEM tensors")
        }
        let batchSize = s2l2a.dim(0)
        let plan = GeoFloodPayload(
            status: preflight ? "ready" : "completed",
            modelID: TerraMindFloodResources.defaultModelID,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: batchSize,
            device: "metal",
            modelLoadSeconds: nil,
            inferenceSeconds: nil
        )
        if preflight {
            print(try Self.jsonString(plan))
            return
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let loadStart = ContinuousClock.now
        let nativeModel = try TerraMindFloodResources.loadModel(from: checkpoint)
        let loadSeconds = Self.seconds(since: loadStart)
        let inferenceStart = ContinuousClock.now
        let logits = try nativeModel(s2l2a: s2l2a, s1rtc: s1rtc, dem: dem)
        MLX.eval(logits)
        let inferenceSeconds = Self.seconds(since: inferenceStart)
        try MLX.save(
            arrays: ["logits": logits.asType(.float32)],
            metadata: [
                "format": "mere.run/terramind-flood-logits-v1",
                "model_id": TerraMindFloodResources.defaultModelID,
                "source_revision": TerraMindFloodResources.sourceRevision,
            ],
            url: outputURL
        )
        let result = GeoFloodPayload(
            status: "completed",
            modelID: TerraMindFloodResources.defaultModelID,
            modelPath: checkpoint.rootURL.path,
            inputPath: inputURL.path,
            outputPath: outputURL.path,
            batchSize: batchSize,
            device: "metal",
            modelLoadSeconds: loadSeconds,
            inferenceSeconds: inferenceSeconds
        )
        if json {
            print(try Self.jsonString(result))
        } else {
            print(outputURL.path)
        }
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    private static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct GeoFloodPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let modelID: String
    let modelPath: String
    let inputPath: String
    let outputPath: String
    let batchSize: Int
    let device: String
    let modelLoadSeconds: Double?
    let inferenceSeconds: Double?

    init(
        schemaVersion: Int = 1,
        status: String,
        modelID: String,
        modelPath: String,
        inputPath: String,
        outputPath: String,
        batchSize: Int,
        device: String,
        modelLoadSeconds: Double?,
        inferenceSeconds: Double?
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.modelID = modelID
        self.modelPath = modelPath
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.batchSize = batchSize
        self.device = device
        self.modelLoadSeconds = modelLoadSeconds
        self.inferenceSeconds = inferenceSeconds
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case modelID = "model_id"
        case modelPath = "model_path"
        case inputPath = "input_path"
        case outputPath = "output_path"
        case batchSize = "batch_size"
        case device
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
    }
}
