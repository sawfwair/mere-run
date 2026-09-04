import ArgumentParser
import Foundation
import MereRunCore

struct VisionGround: AsyncParsableCommand {
    struct PreflightReport: Codable, Equatable {
        let status: String
        let capability: String
        let modelID: String
        let image: String
        let queries: [String]
        let annotatedImage: String
        let detectionsJSON: String
        let maskOutputDirectory: String?

        enum CodingKeys: String, CodingKey {
            case status
            case capability
            case modelID = "model_id"
            case image
            case queries
            case annotatedImage = "annotated_image"
            case detectionsJSON = "detections_json"
            case maskOutputDirectory = "mask_output_directory"
        }
    }

    struct ResolvedModel {
        let modelID: String
        let rootURL: URL
        let isManaged: Bool
    }

    static let defaultManagedModelID: ModelResolver.ModelID = .visionGroundFalconPerception

    static let configuration = CommandConfiguration(
        commandName: "ground",
        abstract: "Ground text queries in an image with the native Falcon Perception runtime.",
        discussion: """
        Uses the native Swift/MLX Falcon Perception detector path in MereRunCore.
        If --model is omitted, this command resolves the managed model id
        `vision-ground-falcon-perception` from the local mere.run model store.
        """
    )

    @Argument(help: "Image file path.")
    var image: String

    @Option(
        name: [.customLong("query"), .customLong("prompt")],
        parsing: .upToNextOption,
        help: "One or more grounding expressions. Example: --query \"cat\" \"person in red\"."
    )
    var query: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Falcon Perception model root.")
    var model: String?

    @Option(name: [.customShort("o"), .long], help: "Annotated output image path (default: <image>_grounded.<ext>).")
    var output: String?

    @Option(name: [.customLong("json-output")], help: "JSON metadata path (default: <image>_grounded.json).")
    var jsonOutput: String?

    @Option(name: [.customLong("mask-output-dir")], help: "Optional directory for per-detection PNG mask exports.")
    var maskOutputDir: String?

    @Flag(name: [.customLong("preflight")], help: "Validate the image, model, queries, and outputs without loading Falcon Perception.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Print only the annotated output image path.")
    var quiet: Bool = false

    @Flag(name: [.customLong(RunReceipt.flagName)], help: RunReceipt.flagHelp)
    var receipt: Bool = false

    func validate() throws {
        guard !query.isEmpty else {
            throw ValidationError("Provide at least one --query or --prompt value.")
        }
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for vision ground.")
        }
        try RunReceipt.validate(receipt: receipt, preflight: preflight)
    }

    func run() async throws {
        let fileManager = FileManager.default
        let imageURL = URL(fileURLWithPath: image).standardizedFileURL
        guard fileManager.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image not found: \(imageURL.path)")
        }

        let resolvedModel = try Self.resolveModelRoot(model, fileManager: fileManager)
        let outputImageURL = Self.resolveAnnotatedOutputURL(output, inputImageURL: imageURL)
        let outputJSONURL = Self.resolveJSONOutputURL(jsonOutput, inputImageURL: imageURL)
        let maskOutputDirectoryURL = Self.resolveDirectoryURL(maskOutputDir)

        if preflight {
            let report = PreflightReport(
                status: "ready",
                capability: "vision.ground",
                modelID: resolvedModel.modelID,
                image: imageURL.path,
                queries: query,
                annotatedImage: outputImageURL.path,
                detectionsJSON: outputJSONURL.path,
                maskOutputDirectory: maskOutputDirectoryURL?.path
            )
            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                print(String(decoding: try encoder.encode(report), as: UTF8.self))
            } else {
                print("Ready: \(report.capability) with \(report.modelID)")
                print("Image: \(report.image)")
                print("Queries: \(report.queries.joined(separator: ", "))")
                print("Annotated image: \(report.annotatedImage)")
                print("Detections JSON: \(report.detectionsJSON)")
                if let maskOutputDirectory = report.maskOutputDirectory {
                    print("Masks: \(maskOutputDirectory)")
                }
            }
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        let grounder = try FalconPerceptionGrounder(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        defer { grounder.unload() }
        let result = try grounder.ground(
            imageURL: imageURL,
            queries: query,
            annotatedImageURL: outputImageURL,
            jsonOutputURL: outputJSONURL,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )

        if quiet {
            print(result.annotatedImageURL.path)
        } else {
            print("Model: \(result.modelID)")
            print("Detections: \(result.detections.count)")
            print("Image: \(result.annotatedImageURL.path)")
            print("JSON: \(result.jsonOutputURL.path)")
            if let maskOutputDirectoryURL {
                print("Masks: \(maskOutputDirectoryURL.path)")
            }
        }
        try RunReceipt.emit(
            RunReceipt.annotatedImageOutputs(
                image: result.annotatedImageURL,
                detections: result.jsonOutputURL,
                masks: maskOutputDirectoryURL
            ),
            enabled: receipt
        )
    }

    static func resolveModelRoot(
        _ rawModel: String?,
        fileManager: FileManager = .default,
        resolver: ModelResolver = ModelResolver()
    ) throws -> ResolvedModel {
        if let rawModel, !rawModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let url = URL(fileURLWithPath: rawModel).standardizedFileURL
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                guard isDirectory.boolValue else {
                    throw ValidationError("Model path must be a directory: \(url.path)")
                }
                return ResolvedModel(modelID: rawModel, rootURL: url, isManaged: false)
            }

            if let modelID = ModelResolver.ModelID(rawValue: rawModel) {
                do {
                    let resolved = try resolver.resolve(modelID)
                    return ResolvedModel(modelID: modelID.rawValue, rootURL: resolved.rootURL, isManaged: true)
                } catch {
                    throw ValidationError(
                        "Model \(modelID.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: modelID.rawValue))` or point --model at a local path."
                    )
                }
            }

            throw ValidationError("Model path not found: \(rawModel). Pass a local model path or the managed id vision-ground-falcon-perception.")
        }

        do {
            let resolved = try resolver.resolve(defaultManagedModelID)
            return ResolvedModel(modelID: defaultManagedModelID.rawValue, rootURL: resolved.rootURL, isManaged: true)
        } catch {
            throw ValidationError(
                "Model \(defaultManagedModelID.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: defaultManagedModelID.rawValue))` or point --model at a local path."
            )
        }
    }

    static func resolveAnnotatedOutputURL(_ rawOutput: String?, inputImageURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            return FalconPerceptionGrounder.defaultAnnotatedOutputURL(for: inputImageURL)
        }

        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        if outputURL.pathExtension.isEmpty {
            let ext = inputImageURL.pathExtension.isEmpty ? "png" : inputImageURL.pathExtension
            return outputURL.appendingPathExtension(ext)
        }
        return outputURL
    }

    static func resolveJSONOutputURL(_ rawOutput: String?, inputImageURL: URL) -> URL {
        guard let rawOutput, !rawOutput.isEmpty else {
            return FalconPerceptionGrounder.defaultJSONOutputURL(for: inputImageURL)
        }

        let outputURL = URL(fileURLWithPath: rawOutput).standardizedFileURL
        if outputURL.pathExtension.isEmpty {
            return outputURL.appendingPathExtension("json")
        }
        return outputURL
    }

    static func resolveDirectoryURL(_ rawDirectory: String?) -> URL? {
        guard let rawDirectory, !rawDirectory.isEmpty else { return nil }
        return URL(fileURLWithPath: rawDirectory).standardizedFileURL
    }
}
