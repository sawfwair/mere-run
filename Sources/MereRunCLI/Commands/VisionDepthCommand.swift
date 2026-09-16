import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

struct VisionDepth: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "depth",
        abstract: "Estimate affine-invariant depth for a still image with native Marigold V2."
    )

    @Argument(help: "Input image path.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output directory.")
    var output: String?

    @Option(name: [.long], help: "Managed model id or model root.")
    var model: String?

    @Option(
        name: [.long],
        help: "Longest inference edge, rounded to a multiple of 16. (default: 1024)"
    )
    var maxEdge: Int?

    @Flag(name: [.long], help: "Run at the source resolution instead of capping the longest edge.")
    var native = false

    @Option(
        name: [.long],
        help: "Depth checkpoint: \(MarigoldV2DepthCheckpoint.allCases.map(\.rawValue).joined(separator: ", "))."
    )
    var checkpoint: String?

    @Flag(name: [.long], help: "Validate and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print the structured result on stdout.")
    var json = false

    mutating func run() async throws {
        if native && maxEdge != nil {
            throw ValidationError("--native and --max-edge cannot be combined")
        }
        if let maxEdge, maxEdge < MarigoldV2InferenceConfiguration.alignment {
            throw ValidationError(
                "--max-edge must be at least \(MarigoldV2InferenceConfiguration.alignment)"
            )
        }
        let resolvedCheckpoint = try Self.resolveCheckpoint(checkpoint)

        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input image not found: \(inputURL.path)")
        }
        let outputURL = Self.resolveOutputURL(output, inputURL: inputURL)
        let size = try MediaImageIO.size(of: inputURL)
        let configuration = MarigoldV2InferenceConfiguration(
            checkpoint: resolvedCheckpoint,
            maximumEdge: native ? nil : (maxEdge ?? MarigoldV2InferenceConfiguration.defaultMaximumEdge)
        )
        let plan = Self.makePlan(
            inputURL: inputURL,
            outputURL: outputURL,
            imageWidth: size.width,
            imageHeight: size.height,
            model: model,
            configuration: configuration
        )
        if dryRun {
            print(try Self.jsonString(plan))
            return
        }

        let generator = MarigoldV2Generator()
        do {
            let result = try await generator.generate(
                imageURL: inputURL,
                outputDirectory: outputURL,
                model: model,
                configuration: configuration,
                progress: { message in CLIStderr.write("[depth] \(message)\n") }
            )
            await generator.unload()
            let payload = VisionDepthRunPayload(result: result)
            if json {
                print(try Self.jsonString(payload))
            } else {
                print(result.export.manifestURL.path)
            }
        } catch {
            await generator.unload()
            throw error
        }
    }

    static func resolveCheckpoint(_ raw: String?) throws -> MarigoldV2DepthCheckpoint {
        guard let raw, !raw.isEmpty else {
            return MarigoldV2Repository.installedCheckpoint
        }
        guard let checkpoint = MarigoldV2DepthCheckpoint(rawValue: raw.lowercased()) else {
            throw ValidationError(
                "Unknown --checkpoint '\(raw)'. Expected one of: "
                    + MarigoldV2DepthCheckpoint.allCases.map(\.rawValue).joined(separator: ", ")
            )
        }
        return checkpoint
    }

    static func resolveOutputURL(_ raw: String?, inputURL: URL) -> URL {
        if let raw, !raw.isEmpty {
            return URL(fileURLWithPath: raw).standardizedFileURL
        }
        return inputURL.deletingLastPathComponent().appendingPathComponent(
            "\(inputURL.deletingPathExtension().lastPathComponent)-depth",
            isDirectory: true
        )
    }

    static func makePlan(
        inputURL: URL,
        outputURL: URL,
        imageWidth: Int,
        imageHeight: Int,
        model: String?,
        configuration: MarigoldV2InferenceConfiguration
    ) -> VisionDepthPlanPayload {
        let inference = configuration.inferenceSize(width: imageWidth, height: imageHeight)
        let installed = ManagedModelResolver.resolveInstalledModel(
            id: ModelResolver.ModelID.visionDepthMarigoldV2.rawValue
        )
        return VisionDepthPlanPayload(
            status: "planned",
            inputPath: inputURL.path,
            outputDirectory: outputURL.path,
            model: model ?? ModelResolver.ModelID.visionDepthMarigoldV2.rawValue,
            managedModelInstalled: installed != nil,
            checkpoint: configuration.checkpoint.rawValue,
            parameterization: configuration.checkpoint.parameterization.rawValue,
            seeThrough: configuration.checkpoint.isSeeThrough,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            inferenceWidth: inference.width,
            inferenceHeight: inference.height,
            semantics: DepthSemantics.affineRelative.rawValue,
            outputKinds: ["depth-exr", "depth-preview-png", "manifest-json"]
        )
    }

    private static func jsonString<T: Encodable>(_ payload: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        return String(decoding: try encoder.encode(payload), as: UTF8.self)
    }
}

struct VisionDepthPlanPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let inputPath: String
    let outputDirectory: String
    let model: String
    let managedModelInstalled: Bool
    let checkpoint: String
    let parameterization: String
    let seeThrough: Bool
    let imageWidth: Int
    let imageHeight: Int
    let inferenceWidth: Int
    let inferenceHeight: Int
    let semantics: String
    let outputKinds: [String]

    init(
        schemaVersion: Int = 1,
        status: String,
        inputPath: String,
        outputDirectory: String,
        model: String,
        managedModelInstalled: Bool,
        checkpoint: String,
        parameterization: String,
        seeThrough: Bool,
        imageWidth: Int,
        imageHeight: Int,
        inferenceWidth: Int,
        inferenceHeight: Int,
        semantics: String,
        outputKinds: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.inputPath = inputPath
        self.outputDirectory = outputDirectory
        self.model = model
        self.managedModelInstalled = managedModelInstalled
        self.checkpoint = checkpoint
        self.parameterization = parameterization
        self.seeThrough = seeThrough
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.inferenceWidth = inferenceWidth
        self.inferenceHeight = inferenceHeight
        self.semantics = semantics
        self.outputKinds = outputKinds
    }
}

struct VisionDepthRunPayload: Codable {
    let schemaVersion: Int
    let status: String
    let manifestPath: String
    let modelID: String
    let checkpoint: String
    let parameterization: String
    let seeThrough: Bool
    let semantics: DepthSemantics
    let width: Int
    let height: Int
    let inferenceWidth: Int
    let inferenceHeight: Int
    let promptTokenCount: Int
    let adapterPairCount: Int
    let vaeDecoderTensorCount: Int
    let depthStatistics: MarigoldV2DepthStatistics
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let postprocessSeconds: Double
    let artifacts: [GeometryArtifact]

    init(result: MarigoldV2RunResult, schemaVersion: Int = 1, status: String = "completed") {
        let manifest = result.export.manifest
        self.schemaVersion = schemaVersion
        self.status = status
        self.manifestPath = result.export.manifestURL.path
        self.modelID = manifest.model.modelID
        self.checkpoint = manifest.checkpoint
        self.parameterization = manifest.parameterization.rawValue
        self.seeThrough = manifest.seeThrough
        self.semantics = manifest.semantics
        self.width = manifest.width
        self.height = manifest.height
        self.inferenceWidth = result.inferenceWidth
        self.inferenceHeight = result.inferenceHeight
        self.promptTokenCount = result.promptTokenCount
        self.adapterPairCount = result.adapterPairCount
        self.vaeDecoderTensorCount = result.vaeDecoderTensorCount
        self.depthStatistics = manifest.depthStatistics
        self.modelLoadSeconds = result.modelLoadSeconds
        self.inferenceSeconds = result.inferenceSeconds
        self.postprocessSeconds = result.postprocessSeconds
        self.artifacts = manifest.artifacts
    }
}
