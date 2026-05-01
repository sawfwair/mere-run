import ArgumentParser
import Foundation
import MereRunCore

struct VisionGround: AsyncParsableCommand {
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

    func validate() throws {
        guard !query.isEmpty else {
            throw ValidationError("Provide at least one --query or --prompt value.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let fileManager = FileManager.default
        let imageURL = URL(fileURLWithPath: image).standardizedFileURL
        guard fileManager.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image not found: \(imageURL.path)")
        }

        let resolvedModel = try Self.resolveModelRoot(model, fileManager: fileManager)
        let outputImageURL = Self.resolveAnnotatedOutputURL(output, inputImageURL: imageURL)
        let outputJSONURL = Self.resolveJSONOutputURL(jsonOutput, inputImageURL: imageURL)
        let maskOutputDirectoryURL = Self.resolveDirectoryURL(maskOutputDir)

        let grounder = try FalconPerceptionGrounder(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        let result = try grounder.ground(
            imageURL: imageURL,
            queries: query,
            annotatedImageURL: outputImageURL,
            jsonOutputURL: outputJSONURL,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )

        print("Model: \(result.modelID)")
        print("Detections: \(result.detections.count)")
        print("Image: \(result.annotatedImageURL.path)")
        print("JSON: \(result.jsonOutputURL.path)")
        if let maskOutputDirectoryURL {
            print("Masks: \(maskOutputDirectoryURL.path)")
        }
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
                        "Model \(modelID.rawValue) not found. Pull it with `mere.run model pull \(modelID.rawValue)` or point --model at a local path."
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
                "Model \(defaultManagedModelID.rawValue) not found. Pull it with `mere.run model pull \(defaultManagedModelID.rawValue)` or point --model at a local path."
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
