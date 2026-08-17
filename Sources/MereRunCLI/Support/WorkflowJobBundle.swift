import ArgumentParser
import MereRunRelayKit
import Crypto
import Foundation
import MereRunCore

// The materializer and bundle helpers live in MereRunRelayKit and take a
// WorkflowMaterializationEnvironment. These wrappers keep the CLI's original
// signatures, supplying the managed model catalog, per-command default model
// ids, and discovered plugin providers only the CLI can know.

extension WorkflowMaterializationEnvironment {
    static var cli: WorkflowMaterializationEnvironment {
        WorkflowMaterializationEnvironment(
            mereRunVersion: MereRunCLIVersion.current,
            pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes,
            defaultModelID: { nodeKind in
                switch nodeKind {
                case "image.train-lora": ImageTrainLoRA.defaultManagedModelID.rawValue
                case "image.generate": ImageGenerate.defaultManagedModelID.rawValue
                case "vision.ground": VisionGround.defaultManagedModelID.rawValue
                case "vision.segment", "vision.track": VisionSegment.defaultManagedModelID.rawValue
                case "video.generate": ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
                default: nil
                }
            },
            modelProvenance: { modelID in
                let spec = ManagedModelCatalog.spec(for: modelID)
                return .fromCatalogIdentity(
                    id: modelID,
                    repository: spec?.upstreamRepoId,
                    revision: spec?.upstreamRevision
                )
            },
            providerRequirement: { providerID in
                WorkflowGraphProviderRegistry.discoveredCatalog().provider(id: providerID)?.requirement
            }
        )
    }
}

extension WorkflowBundleMaterializer {
    init(
        graph: WorkflowGraphDocument,
        suppliedInputs: WorkflowInputsDocument,
        destination: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init,
        jobID: @escaping () -> UUID = UUID.init,
        seed: @escaping () -> Int64 = { Int64.random(in: 0...Int64.max) }
    ) {
        self.init(
            graph: graph,
            suppliedInputs: suppliedInputs,
            destination: destination,
            environment: .cli,
            fileManager: fileManager,
            now: now,
            jobID: jobID,
            seed: seed
        )
    }
}

func verifiedPortableWorkflowBundle(
    at directory: URL,
    fileManager: FileManager = .default
) throws -> WorkflowBundleMaterialization {
    try verifiedPortableWorkflowBundle(
        at: directory,
        fileManager: fileManager,
        pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes
    )
}


struct WorkflowInvocationOutput: Codable, Equatable, Sendable {
    let type: WorkflowPortType
    let path: String?
    let optional: Bool
    let contentTypes: [String]

    enum CodingKeys: String, CodingKey {
        case type
        case path
        case optional
        case contentTypes = "content_types"
    }
}

struct WorkflowPluginNodeInvocationDocument: Codable, Equatable, Sendable {
    static let contractVersion = "mere.run/plugin-graph-invocation.v1"

    let contractVersion: String
    let jobID: String
    let nodeID: String
    let kind: String
    let arguments: [String: WorkflowValue]
    let outputs: [String: WorkflowInvocationOutput]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case jobID = "job_id"
        case nodeID = "node_id"
        case kind
        case arguments
        case outputs
    }
}

struct WorkflowNodeInvocation: Equatable {
    let command: [String]
    let executable: URL
    let preflightArguments: [String]
    let runArguments: [String]
    let outputs: [String: WorkflowInvocationOutput]
    let streamsEvents: Bool
    let intrinsic: WorkflowIntrinsicInvocation?
    let stdoutOutputName: String?

    init(
        command: [String],
        executable: URL,
        preflightArguments: [String],
        runArguments: [String],
        outputs: [String: WorkflowInvocationOutput],
        streamsEvents: Bool,
        intrinsic: WorkflowIntrinsicInvocation? = nil,
        stdoutOutputName: String? = nil
    ) {
        self.command = command
        self.executable = executable
        self.preflightArguments = preflightArguments
        self.runArguments = runArguments
        self.outputs = outputs
        self.streamsEvents = streamsEvents
        self.intrinsic = intrinsic
        self.stdoutOutputName = stdoutOutputName
    }
}

struct WorkflowIntrinsicInvocation: Equatable {
    let kind: String
    let arguments: [String: WorkflowValue]

    func evaluate() throws -> [String: WorkflowValue] {
        switch kind {
        case "text.value":
            return ["text": try required("value")]
        case "integer.value", "number.value", "boolean.value", "json.value":
            return ["value": try required("value")]
        case "seed.value":
            return ["seed": try required("seed")]
        case "choice.value":
            guard case .array(let options) = try required("options"),
                  options.allSatisfy({ $0.stringValue != nil }),
                  case .string(let selected) = try required("selected") else {
                throw ValidationError("choice.value requires string options and a selected string.")
            }
            guard options.contains(.string(selected)) else {
                throw ValidationError("choice.value selected value must appear in options.")
            }
            return ["value": .string(selected)]
        case "text.join":
            guard case .array(let parts) = try required("parts"),
                  parts.allSatisfy({ $0.stringValue != nil }) else {
                throw ValidationError("text.join parts must resolve to an array of strings.")
            }
            let separator = arguments["separator"]?.stringValue ?? "\n"
            return ["text": .string(parts.compactMap(\.stringValue).joined(separator: separator))]
        case "text.template":
            guard case .string(let template) = try required("template"),
                  case .object(let variables) = try required("variables"),
                  variables.values.allSatisfy({ $0.stringValue != nil }) else {
                throw ValidationError("text.template requires a string template and string variables.")
            }
            return ["text": .string(try renderTemplate(template, variables: variables))]
        default:
            throw ValidationError("Unsupported intrinsic workflow node kind '\(kind)'.")
        }
    }

    private func required(_ name: String) throws -> WorkflowValue {
        guard let value = arguments[name], value != .null else {
            throw ValidationError("\(kind) requires argument '\(name)'.")
        }
        return value
    }

    private func renderTemplate(
        _ template: String,
        variables: [String: WorkflowValue]
    ) throws -> String {
        let expression = try NSRegularExpression(pattern: #"(?<!\\)\{\{([^{}]+)\}\}"#)
        let source = template as NSString
        let matches = expression.matches(
            in: template,
            range: NSRange(location: 0, length: source.length)
        )
        var rendered = template
        for match in matches.reversed() {
            let name = source.substring(with: match.range(at: 1))
            guard name.range(
                of: "^[a-z][a-z0-9_]{0,63}$",
                options: .regularExpression
            ) != nil else {
                throw ValidationError("text.template placeholder '{{\(name)}}' is invalid.")
            }
            guard let replacement = variables[name]?.stringValue else {
                throw ValidationError("text.template is missing variable '\(name)'.")
            }
            guard let range = Range(match.range, in: rendered) else {
                throw ValidationError("text.template could not resolve placeholder '\(name)'.")
            }
            rendered.replaceSubrange(range, with: replacement)
        }
        return rendered.replacingOccurrences(of: #"\{{"#, with: "{{")
    }
}

enum WorkflowNodeCommandBuilder {
    static func invocation(
        node: WorkflowNode,
        arguments: [String: WorkflowValue],
        nodeDirectory: URL,
        jobID: String = UUID().uuidString.lowercased()
    ) throws -> WorkflowNodeInvocation {
        if node.resolvedProviderID != WorkflowNodeProviderIdentity.builtInID {
            return try pluginInvocation(
                node: node,
                arguments: arguments,
                nodeDirectory: nodeDirectory,
                jobID: jobID
            )
        }
        let artifacts = nodeDirectory.appendingPathComponent("artifacts", isDirectory: true)
        switch node.kind {
        case "text.value", "integer.value", "number.value", "boolean.value",
             "json.value", "seed.value", "choice.value", "text.join", "text.template":
            guard let entry = WorkflowNodeRegistry.entry(for: node) else {
                throw ValidationError("Unsupported workflow node kind '\(node.kind)'.")
            }
            return .init(
                command: ["intrinsic", node.kind],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: [],
                outputs: Dictionary(uniqueKeysWithValues: entry.outputs.map { output in
                    (
                        output.name,
                        WorkflowInvocationOutput(
                            type: output.type,
                            path: nil,
                            optional: output.optional,
                            contentTypes: output.contentTypes
                        )
                    )
                }),
                streamsEvents: false,
                intrinsic: WorkflowIntrinsicInvocation(kind: node.kind, arguments: arguments)
            )
        case "text.enhance":
            let instruction = try requiredString("instruction", in: arguments)
            let text = try requiredString("text", in: arguments)
            var args = [
                "text", "chat",
                "--prompt", "\(instruction)\n\nText:\n\(text)",
                "--model", try requiredString("model", in: arguments),
            ]
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            appendNumber("temperature", flag: "--temperature", from: arguments, to: &args)
            args += ["--no-thinking", "--require-installed", "--quiet"]
            return .init(
                command: ["text", "chat"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args,
                outputs: ["text": valueOutput(.string)],
                streamsEvents: false,
                stdoutOutputName: "text"
            )
        case "image.describe":
            var args = [
                "text", "chat",
                "--prompt", try requiredString("instruction", in: arguments),
                "--image", try requiredString("image", in: arguments),
                "--model", try requiredString("model", in: arguments),
            ]
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            appendNumber("temperature", flag: "--temperature", from: arguments, to: &args)
            args += ["--no-thinking", "--require-installed", "--quiet"]
            return .init(
                command: ["text", "chat"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args,
                outputs: ["text": valueOutput(.string)],
                streamsEvents: false,
                stdoutOutputName: "text"
            )
        case "vision.ground":
            let outputImage = artifacts.appendingPathComponent("image.png")
            let detections = artifacts.appendingPathComponent("detections.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "ground", try requiredString("image", in: arguments),
                "--query",
            ]
            args.append(contentsOf: try requiredStringArray("queries", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            args += [
                "--output", outputImage.path,
                "--json-output", detections.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "ground"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "image": fileOutput(outputImage, contentTypes: ["image/png"]),
                    "detections": fileOutput(detections, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
        case "vision.segment":
            let outputImage = artifacts.appendingPathComponent("image.png")
            let segments = artifacts.appendingPathComponent("segments.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "segment", try requiredString("image", in: arguments),
                "--prompt",
            ]
            args.append(contentsOf: try requiredStringArray("prompts", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendNumber("threshold", flag: "--threshold", from: arguments, to: &args)
            appendInteger("resolution", flag: "--resolution", from: arguments, to: &args)
            appendFlag("show_boxes", flag: "--show-boxes", from: arguments, to: &args)
            appendFlag("multimask", flag: "--multimask", from: arguments, to: &args)
            args += [
                "--output", outputImage.path,
                "--json-output", segments.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "segment"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "image": fileOutput(outputImage, contentTypes: ["image/png"]),
                    "segments": fileOutput(segments, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
        case "vision.track":
            let outputVideo = artifacts.appendingPathComponent("video.mp4")
            let tracks = artifacts.appendingPathComponent("tracks.json")
            let masks = artifacts.appendingPathComponent("masks", isDirectory: true)
            var args = [
                "vision", "track", try requiredString("video", in: arguments),
                "--prompt",
            ]
            args.append(contentsOf: try requiredStringArray("prompts", in: arguments))
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendNumber("threshold", flag: "--threshold", from: arguments, to: &args)
            appendInteger("resolution", flag: "--resolution", from: arguments, to: &args)
            appendInteger("init_frame", flag: "--init-frame", from: arguments, to: &args)
            appendInteger("end_frame", flag: "--end-frame", from: arguments, to: &args)
            appendFlag("show_boxes", flag: "--show-boxes", from: arguments, to: &args)
            appendFlag("show_labels", flag: "--show-labels", from: arguments, to: &args)
            args += [
                "--output", outputVideo.path,
                "--json-output", tracks.path,
                "--mask-output-dir", masks.path,
            ]
            return .init(
                command: ["vision", "track"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: [
                    "video": fileOutput(outputVideo, contentTypes: ["video/mp4"]),
                    "tracks": fileOutput(tracks, contentTypes: ["application/json"]),
                    "masks": directoryOutput(masks, contentTypes: ["image/png"]),
                ],
                streamsEvents: false
            )
        case "image.train-lora":
            let output = artifacts.appendingPathComponent("adapter.safetensors")
            var args = ["image", "train-lora", "--data", try requiredString("data", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("recipe", flag: "--recipe", from: arguments, to: &args)
            appendInteger("training_steps", flag: "--training-steps", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("max_text_length", flag: "--max-text-length", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendInteger("rank", flag: "--rank", from: arguments, to: &args)
            appendFlag("lite", flag: "--lite", from: arguments, to: &args)
            appendInteger(
                "base_quantization_bits",
                flag: "--base-quantization-bits",
                from: arguments,
                to: &args
            )
            appendString("sample_prompt", flag: "--sample-prompt", from: arguments, to: &args)
            return .init(
                command: ["image", "train-lora"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["adapter": fileOutput(output, contentTypes: ["application/x-safetensors"])],
                streamsEvents: false
            )
        case "image.generate":
            let output = artifacts.appendingPathComponent("image.png")
            var args = ["image", "generate", "--prompt", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("negative_prompt", flag: "--negative-prompt", from: arguments, to: &args)
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendString("input", flag: "--input", from: arguments, to: &args)
            appendString("lora", flag: "--lora", from: arguments, to: &args)
            appendNumber("lora_scale", flag: "--lora-scale", from: arguments, to: &args)
            appendNumber("cfg_scale", flag: "--cfg", from: arguments, to: &args)
            appendNumber("strength", flag: "--strength", from: arguments, to: &args)
            appendInteger(
                "krea_base_quantization_bits",
                flag: "--krea-base-quantization-bits",
                from: arguments,
                to: &args
            )
            if case .array(let references)? = arguments["reference_images"] {
                for reference in references {
                    guard let path = reference.stringValue else {
                        throw ValidationError("image.generate reference_images values must resolve to paths.")
                    }
                    args += ["--ref-image", path]
                }
            }
            return .init(
                command: ["image", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["image": fileOutput(output, contentTypes: ["image/png"])],
                streamsEvents: false
            )
        case "video.generate":
            let output = artifacts.appendingPathComponent("video.mp4")
            var args = ["video", "generate", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("width", flag: "--width", from: arguments, to: &args)
            appendInteger("height", flag: "--height", from: arguments, to: &args)
            appendInteger("num_frames", flag: "--num-frames", from: arguments, to: &args)
            appendNumber("duration", flag: "--duration", from: arguments, to: &args)
            appendInteger("fps", flag: "--fps", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendString("image", flag: "--image", from: arguments, to: &args)
            appendString("end_image", flag: "--end-image", from: arguments, to: &args)
            return .init(
                command: ["video", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: args + ["--preflight", "--json"],
                runArguments: args + ["--quiet"],
                outputs: ["video": fileOutput(output, contentTypes: ["video/mp4"])],
                streamsEvents: false
            )
        case "music.generate":
            let output = artifacts.appendingPathComponent("music.wav")
            var args = ["music", "generate", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("lyrics", flag: "--lyrics", from: arguments, to: &args)
            appendNumber("duration", flag: "--duration", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            return .init(
                command: ["music", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["audio": fileOutput(output, contentTypes: ["audio/wav"])],
                streamsEvents: false
            )
        case "sfx.generate":
            let output = artifacts.appendingPathComponent("sfx.wav")
            var args = ["sfx", "generate", try requiredString("prompt", in: arguments), "--output", output.path]
            appendString("negative_prompt", flag: "--negative-prompt", from: arguments, to: &args)
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendNumber("duration", flag: "--duration", from: arguments, to: &args)
            appendInteger("steps", flag: "--steps", from: arguments, to: &args)
            appendNumber("cfg_scale", flag: "--cfg", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            args += ["--quiet"]
            return .init(
                command: ["sfx", "generate"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["audio": fileOutput(output, contentTypes: ["audio/wav"])],
                streamsEvents: false
            )
        case "speech.synthesize":
            let output = artifacts.appendingPathComponent("speech.wav")
            var args = ["speech", "synthesize", try requiredString("text", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("voice", flag: "--voice", from: arguments, to: &args)
            appendString("language", flag: "--language", from: arguments, to: &args)
            return .init(
                command: ["speech", "synthesize"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["audio": fileOutput(output, contentTypes: ["audio/wav"])],
                streamsEvents: false
            )
        case "speech.transcribe":
            let output = artifacts.appendingPathComponent("transcript.txt")
            var args = ["speech", "transcribe", try requiredString("audio", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("backend", flag: "--backend", from: arguments, to: &args)
            return .init(
                command: ["speech", "transcribe"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["text": fileOutput(output, contentTypes: ["text/plain"])],
                streamsEvents: false
            )
        case "vision.caption":
            let output = artifacts.appendingPathComponent("captions", isDirectory: true)
            var args = ["vision", "caption", try requiredString("image", in: arguments), "--output-dir", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("prompt", flag: "--prompt", from: arguments, to: &args)
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            appendNumber("temperature", flag: "--temperature", from: arguments, to: &args)
            return .init(
                command: ["vision", "caption"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["captions": directoryOutput(output, contentTypes: ["text/plain"])],
                streamsEvents: false
            )
        case "vision.ocr":
            let output = artifacts.appendingPathComponent("ocr", isDirectory: true)
            var args = ["vision", "ocr", try requiredString("image", in: arguments), "--output-dir", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("backend", flag: "--backend", from: arguments, to: &args)
            return .init(
                command: ["vision", "ocr"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["text": directoryOutput(output, contentTypes: ["text/plain", "text/markdown", "application/json"])],
                streamsEvents: false
            )
        case "vision.geometry":
            let output = artifacts.appendingPathComponent("geometry", isDirectory: true)
            var args = ["vision", "geometry", try requiredString("image", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("resolution_level", flag: "--resolution-level", from: arguments, to: &args)
            appendInteger("max_points", flag: "--max-points", from: arguments, to: &args)
            return .init(
                command: ["vision", "geometry"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: [
                    "geometry": directoryOutput(
                        output,
                        contentTypes: ["image/x-exr", "application/json", "application/octet-stream"]
                    ),
                ],
                streamsEvents: false
            )
        case "vision.image-to-3d":
            let output = artifacts.appendingPathComponent("mesh", isDirectory: true)
            var args = ["vision", "image-to-3d", try requiredString("image", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("resolution", flag: "--resolution", from: arguments, to: &args)
            appendNumber("foreground_ratio", flag: "--foreground-ratio", from: arguments, to: &args)
            return .init(
                command: ["vision", "image-to-3d"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: [
                    "mesh": directoryOutput(
                        output,
                        contentTypes: ["model/gltf-binary", "model/obj", "application/octet-stream"]
                    ),
                ],
                streamsEvents: false
            )
        case "audio.enhance":
            let output = artifacts.appendingPathComponent("enhanced.wav")
            var args = ["audio", "enhance", try requiredString("audio", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("seed", flag: "--seed", from: arguments, to: &args)
            return .init(
                command: ["audio", "enhance"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["audio": fileOutput(output, contentTypes: ["audio/wav"])],
                streamsEvents: false
            )
        case "music.separate":
            let output = artifacts.appendingPathComponent("stems", isDirectory: true)
            var args = ["music", "separate", try requiredString("audio", in: arguments), "--output-dir", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            args += ["--quiet"]
            return .init(
                command: ["music", "separate"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["stems": directoryOutput(output, contentTypes: ["audio/wav"])],
                streamsEvents: false
            )
        case "music.transcribe":
            let format = arguments["format"]?.stringValue ?? "midi"
            let filename = format == "midi" ? "transcription.mid" : "transcription.\(format)"
            let contentType = format == "midi" ? "audio/midi" : "application/json"
            let output = artifacts.appendingPathComponent(filename)
            var args = [
                "music", "transcribe", try requiredString("audio", in: arguments),
                "--output", output.path,
                "--format", format,
            ]
            appendString("model", flag: "--model", from: arguments, to: &args)
            return .init(
                command: ["music", "transcribe"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["transcription": fileOutput(output, contentTypes: [contentType])],
                streamsEvents: false
            )
        case "speech.diarize":
            let format = arguments["format"]?.stringValue ?? "json"
            let output = artifacts.appendingPathComponent("diarization.\(format)")
            var args = [
                "speech", "diarize", try requiredString("audio", in: arguments),
                "--output", output.path,
                "--format", format,
                "--quiet",
            ]
            appendString("model", flag: "--model", from: arguments, to: &args)
            return .init(
                command: ["speech", "diarize"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: [
                    "segments": fileOutput(
                        output,
                        contentTypes: [format == "json" ? "application/json" : "text/plain"]
                    ),
                ],
                streamsEvents: false
            )
        case "text.embed":
            let output = artifacts.appendingPathComponent("embeddings.json")
            var args = ["text", "embed", try requiredString("text", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendInteger("max_tokens", flag: "--max-tokens", from: arguments, to: &args)
            return .init(
                command: ["text", "embed"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["embeddings": fileOutput(output, contentTypes: ["application/json"])],
                streamsEvents: false
            )
        case "text.anonymize":
            let output = artifacts.appendingPathComponent("anonymized.txt")
            var args = ["text", "anonymize", try requiredString("text", in: arguments), "--output", output.path]
            appendString("model", flag: "--model", from: arguments, to: &args)
            appendString("replacement", flag: "--replacement", from: arguments, to: &args)
            return .init(
                command: ["text", "anonymize"],
                executable: CurrentExecutable.url(),
                preflightArguments: [],
                runArguments: args,
                outputs: ["text": fileOutput(output, contentTypes: ["text/plain"])],
                streamsEvents: false
            )
        default:
            throw ValidationError("Unsupported workflow node kind '\(node.kind)'.")
        }
    }

    private static func pluginInvocation(
        node: WorkflowNode,
        arguments: [String: WorkflowValue],
        nodeDirectory: URL,
        jobID: String
    ) throws -> WorkflowNodeInvocation {
        let provider = try WorkflowGraphProviderRegistry.requireProvider(id: node.resolvedProviderID)
        guard let catalog = provider.nodes.first(where: { $0.kind == node.kind }),
              let executable = PluginProcess.which(provider.executable) else {
            throw ValidationError(
                "Graph provider '\(node.resolvedProviderID)' does not expose node kind '\(node.kind)'."
            )
        }
        var outputs: [String: WorkflowInvocationOutput] = [:]
        for output in catalog.outputs {
            outputs[output.name] = WorkflowInvocationOutput(
                type: output.type,
                path: outputPath(for: output),
                optional: output.optional,
                contentTypes: output.contentTypes
            )
        }
        let request = WorkflowPluginNodeInvocationDocument(
            contractVersion: WorkflowPluginNodeInvocationDocument.contractVersion,
            jobID: jobID,
            nodeID: node.id,
            kind: node.kind,
            arguments: arguments,
            outputs: outputs
        )
        let requestURL = nodeDirectory.appendingPathComponent("invocation.json")
        try WorkflowBundleCodec.write(request, to: requestURL)
        let shared = ["--request", requestURL.path, "--run-dir", nodeDirectory.path]
        return WorkflowNodeInvocation(
            command: [provider.executable, "graph", "execute"],
            executable: executable,
            preflightArguments: ["graph", "preflight"] + shared + ["--json"],
            runArguments: ["graph", "execute"] + shared + ["--json-stream"],
            outputs: outputs,
            streamsEvents: true
        )
    }

    private static func fileOutput(_ url: URL, contentTypes: [String]) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: .asset,
            path: url.path,
            optional: false,
            contentTypes: contentTypes
        )
    }

    private static func directoryOutput(_ url: URL, contentTypes: [String]) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: .assetDirectory,
            path: url.path,
            optional: false,
            contentTypes: contentTypes
        )
    }

    private static func valueOutput(_ type: WorkflowPortType) -> WorkflowInvocationOutput {
        WorkflowInvocationOutput(
            type: type,
            path: nil,
            optional: false,
            contentTypes: []
        )
    }

    private static func outputPath(for output: WorkflowNodeOutput) -> String? {
        switch output.type {
        case .asset:
            let pathExtension = outputExtension(contentTypes: output.contentTypes)
            return "artifacts/\(output.name)\(pathExtension)"
        case .assetDirectory:
            return "artifacts/\(output.name)"
        case .assetCollection, .assetArray:
            return "artifacts/\(output.name).json"
        default:
            return nil
        }
    }

    static func outputExtension(contentTypes: [String]) -> String {
        switch contentTypes.first {
        case "image/png": ".png"
        case "image/jpeg": ".jpg"
        case "image/webp": ".webp"
        case "video/mp4": ".mp4"
        case "audio/wav": ".wav"
        case "image/tiff": ".tif"
        case "application/json": ".json"
        case "application/x-safetensors": ".safetensors"
        default: ""
        }
    }

    private static func requiredString(_ name: String, in arguments: [String: WorkflowValue]) throws -> String {
        guard let value = arguments[name]?.stringValue, !value.isEmpty else {
            throw ValidationError("Workflow node argument '\(name)' did not resolve to a string.")
        }
        return value
    }

    private static func requiredStringArray(
        _ name: String,
        in arguments: [String: WorkflowValue]
    ) throws -> [String] {
        guard case .array(let values)? = arguments[name] else {
            throw ValidationError("Workflow node argument '\(name)' did not resolve to an array.")
        }
        let strings = values.compactMap(\.stringValue)
        guard strings.count == values.count, !strings.isEmpty, strings.allSatisfy({ !$0.isEmpty }) else {
            throw ValidationError("Workflow node argument '\(name)' must contain one or more strings.")
        }
        return strings
    }

    private static func appendString(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.stringValue { output += [flag, value] }
    }

    private static func appendInteger(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.integerValue { output += [flag, String(value)] }
    }

    private static func appendNumber(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if let value = arguments[name]?.numberValue { output += [flag, String(value)] }
    }

    private static func appendFlag(
        _ name: String,
        flag: String,
        from arguments: [String: WorkflowValue],
        to output: inout [String]
    ) {
        if arguments[name]?.booleanValue == true { output.append(flag) }
    }
}
