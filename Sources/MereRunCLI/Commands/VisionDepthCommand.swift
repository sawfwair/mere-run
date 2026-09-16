import ArgumentParser
import Foundation
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
        help: "Depth checkpoint: \(MarigoldV2GenerationSettings.checkpointNames.joined(separator: ", "))."
    )
    var checkpoint: String?

    @Flag(name: [.long], help: "Validate and print the execution plan without loading weights.")
    var dryRun = false

    @Flag(name: [.long], help: "Print the structured result on stdout.")
    var json = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt = false

    mutating func run() async throws {
        try RunReceipt.validate(receipt: receipt, dryRun: dryRun)
        do {
            let request = try makeGenerationRequest()
            if dryRun {
                let plan = try MarigoldV2GenerationOperation.prepare(request)
                print(try Self.jsonString(Self.makePlan(plan)))
                return
            }
            let result = try await MarigoldV2GenerationOperation.execute(
                request,
                prepareRuntime: { try MLXBundleSupport.ensureAvailable(quiet: true) },
                progress: { message in CLIStderr.write("[depth] \(message)\n") }
            )
            if json {
                print(try Self.jsonString(VisionDepthRunPayload(result: result)))
            } else {
                print(result.export.manifestURL.path)
            }
            try RunReceipt.emit(
                RunReceipt.depthOutputs(
                    depth: result.export.depthURL,
                    preview: result.export.previewURL,
                    manifest: result.export.manifestURL
                ),
                enabled: receipt
            )
        } catch let error as MarigoldV2GenerationError {
            switch error {
            case .nativeResolutionConflictsWithMaximumEdge:
                throw ValidationError("--native and --max-edge cannot be combined")
            case .maximumEdgeBelowAlignment:
                throw ValidationError(
                    "--max-edge must be at least \(MarigoldV2GenerationSettings.minimumMaximumEdge)"
                )
            case .unknownCheckpoint(let name):
                throw ValidationError(
                    "Unknown --checkpoint '\(name)'. Expected one of: "
                        + MarigoldV2GenerationSettings.checkpointNames.joined(separator: ", ")
                )
            case .inputNotFound:
                throw ValidationError(error.localizedDescription)
            }
        }
    }

    func makeGenerationRequest() throws -> MarigoldV2GenerationRequest {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        return MarigoldV2GenerationRequest(
            imageURL: inputURL,
            outputDirectory: Self.resolveOutputURL(output, inputURL: inputURL),
            model: model,
            settings: try MarigoldV2GenerationSettings(
                checkpoint: checkpoint, maximumEdge: maxEdge, nativeResolution: native
            )
        )
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

    static func makePlan(_ plan: MarigoldV2GenerationPlan) -> VisionDepthPlanPayload {
        let request = plan.request
        let checkpoint = request.settings.configuration.checkpoint
        return VisionDepthPlanPayload(
            status: "planned",
            inputPath: request.imageURL.path,
            outputDirectory: request.outputDirectory.path,
            model: request.model ?? MarigoldV2GenerationRequest.defaultModelID,
            managedModelInstalled: plan.managedModelInstalled,
            checkpoint: checkpoint.rawValue,
            parameterization: checkpoint.parameterization.rawValue,
            seeThrough: checkpoint.isSeeThrough,
            imageWidth: plan.dimensions.width,
            imageHeight: plan.dimensions.height,
            inferenceWidth: plan.inferenceWidth,
            inferenceHeight: plan.inferenceHeight,
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
