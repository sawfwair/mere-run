import ArgumentParser
import Foundation
import MLX
import MereRunCore

enum Cosmos3CLIMode: String, CaseIterable, ExpressibleByArgument {
    case textToImage = "text-to-image"
    case imageToImage = "image-to-image"
    case textToVideo = "text-to-video"
    case imageToVideo = "image-to-video"
    case videoToVideo = "video-to-video"
    case policy
    case forwardDynamics = "forward-dynamics"
    case inverseDynamics = "inverse-dynamics"
    case reasoner

    var actionMode: Cosmos3ActionMode? {
        switch self {
        case .policy: .policy
        case .forwardDynamics: .forwardDynamics
        case .inverseDynamics: .inverseDynamics
        default: nil
        }
    }
}

enum Cosmos3CLISchedule: String, CaseIterable, ExpressibleByArgument {
    case nvidia = "nvidia"
    case publishedKarras = "published-karras"

    var value: Cosmos3UniPCSchedule {
        switch self {
        case .nvidia: .nvidiaShiftedFlow
        case .publishedKarras: .publishedKarrasFlow
        }
    }
}

struct VideoCosmos3: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "cosmos3",
        abstract: "Run native NVIDIA Cosmos3 generation and action modes."
    )

    @Argument(help: "Text prompt or action task description.")
    var prompt: String

    @Option(name: [.long], help: "Cosmos3 mode.")
    var mode: Cosmos3CLIMode = .textToVideo

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Cosmos3 model root.")
    var model = Cosmos3Resources.modelID

    @Option(name: [.customShort("o"), .long], help: "Output PNG or MP4 path.")
    var output: String?

    @Option(name: [.customLong("actions-output")], help: "Output JSON path for predicted actions.")
    var actionsOutput: String?

    @Option(name: [.long], help: "Conditioning image path.")
    var image: String?

    @Option(name: [.long], help: "Conditioning video path.")
    var video: String?

    @Option(name: [.customLong("negative-prompt")], help: "Classifier-free negative prompt.")
    var negativePrompt = ""

    @Option(name: [.long], help: "Output width, divisible by 16.")
    var width = 1_280

    @Option(name: [.long], help: "Output height, divisible by 16.")
    var height = 720

    @Option(name: [.customLong("num-frames")], help: "Output frames; video modes require 4n+1.")
    var numFrames = 189

    @Option(name: [.long], help: "Denoising steps; omitted uses the published mode default.")
    var steps: Int?

    @Option(name: [.customLong("guidance-scale")], help: "CFG scale; omitted uses the published mode default.")
    var guidanceScale: Float?

    @Option(name: [.long], help: "Flow schedule shift; omitted uses the published mode default.")
    var shift: Float?

    @Option(name: [.long], help: "UniPC timestep schedule.")
    var schedule: Cosmos3CLISchedule = .nvidia

    @Option(name: [.long], help: "Seed value.")
    var seed: Int = 0

    @Option(name: [.long], help: "Frames per second.")
    var fps: Int?

    @Option(name: [.customLong("condition-latent-frame")], parsing: .upToNextOption, help: "Conditioned VAE latent frame indices for video-to-video.")
    var conditionLatentFrames: [Int] = [0, 1]

    @Flag(name: [.customLong("keep-video-tail")], help: "Use the tail of a long conditioning video.")
    var keepVideoTail = false

    @Option(name: [.customLong("action-domain")], help: "Published action domain name.")
    var actionDomain: String = Cosmos3ActionDomain.cameraPose.rawValue

    @Option(name: [.customLong("action-file")], help: "Forward-dynamics normalized model-space action JSON containing [[Float]].")
    var actionFile: String?

    @Option(name: [.customLong("action-chunk-size")], help: "Number of action steps.")
    var actionChunkSize = 16

    @Option(name: [.customLong("action-resolution")], help: "Action resize tier: 256, 480, 704, or 720.")
    var actionResolution = 480

    @Option(name: [.customLong("action-viewpoint")], help: "ego_view, third_person_view, wrist_view, or concat_view.")
    var actionViewpoint = Cosmos3ActionViewpoint.egoView.rawValue

    @Option(name: [.customLong("max-new-tokens")], help: "Maximum Cosmos3 reasoner response tokens.")
    var maxNewTokens = 256

    @Option(name: [.long], help: "Cosmos3 reasoner sampling temperature; zero is greedy.")
    var temperature: Float = 0.2

    @Option(name: [.customLong("top-p")], help: "Cosmos3 reasoner nucleus sampling threshold.")
    var topP: Float = 0.9

    @Option(name: [.customLong("max-video-frames")], help: "Maximum video frames sent to the reasoner.")
    var maxVideoFrames = 32

    @Flag(name: [.short, .long], help: "Suppress progress diagnostics.")
    var quiet = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else {
            throw ValidationError("Prompt cannot be empty.")
        }
        let imageURL = try existingFile(image, field: "--image")
        let videoURL = try existingFile(video, field: "--video")
        if mode == .reasoner {
            try await runReasoner(
                prompt: trimmedPrompt,
                imageURL: imageURL,
                videoURL: videoURL
            )
            return
        }
        let action = try makeAction(imageURL: imageURL, videoURL: videoURL)
        if let fps, fps <= 0 { throw ValidationError("--fps must be positive.") }
        let resolvedFrames: Int
        switch mode {
        case .textToImage, .imageToImage, .reasoner:
            resolvedFrames = 1
        case .policy, .forwardDynamics, .inverseDynamics:
            resolvedFrames = actionChunkSize + 1
        default:
            resolvedFrames = numFrames
        }
        try validateInputs(
            imageURL: imageURL,
            videoURL: videoURL,
            hasAction: action != nil
        )

        let outputExtension = mode == .textToImage || mode == .imageToImage ? "png" : "mp4"
        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-cosmos3-\(mode.rawValue)",
            defaultExtension: outputExtension
        )
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let root = try await resolveVideoModelRoot(
            explicitModelRoot: nil,
            requestedModel: model,
            variant: .distilled,
            allowAutoDownload: false
        )
        let resources = Cosmos3Resources(rootURL: root)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ValidationError(
                "Cosmos3 resources are incomplete: "
                    + missing.map(\.path).joined(separator: ", ")
            )
        }
        let distilled = try resources.loadDistilledConfiguration()
        if distilled != nil, shift != nil {
            throw ValidationError("The distilled Cosmos3-Super checkpoint does not accept --shift.")
        }
        if distilled != nil, schedule != .nvidia {
            throw ValidationError("The distilled Cosmos3-Super checkpoint does not accept --schedule overrides.")
        }
        let options = try Cosmos3GenerationOptions(
            prompt: trimmedPrompt,
            negativePrompt: negativePrompt,
            outputURL: outputURL,
            imageURL: action == nil ? imageURL : nil,
            videoURL: action == nil ? videoURL : nil,
            action: action,
            conditionedVideoLatentFrames: conditionLatentFrames,
            keepsVideoTail: keepVideoTail,
            width: width,
            height: height,
            numFrames: resolvedFrames,
            steps: steps ?? distilled?.sigmas.count,
            guidanceScale: guidanceScale ?? (distilled == nil ? nil : 1),
            shift: shift,
            schedule: schedule.value,
            seed: UInt64(bitPattern: Int64(seed)),
            fps: fps
        )
        if !quiet {
            CLIStderr.write("Engine: native Cosmos3 MLX\n")
            CLIStderr.write("Model root: \(root.path)\n")
            CLIStderr.write("Mode: \(mode.rawValue)\n")
            if distilled != nil, !negativePrompt.isEmpty {
                CLIStderr.write("Warning: the distilled checkpoint ignores --negative-prompt.\n")
            }
        }
        let reportsProgress = !quiet
        let generator = Cosmos3EdgeGenerator()
        do {
            let result = try await generator.generate(
                options: options,
                resources: resources,
                progressHandler: { progress in
                    guard reportsProgress else { return }
                    if progress.stage == .denoising {
                        CLIStderr.write(
                            "Denoising \(progress.stepIndex + 1)/\(progress.totalSteps)\n"
                        )
                    } else {
                        CLIStderr.write("\(progress.stage.rawValue)\n")
                    }
                }
            )
            if let actions = result.actions {
                let actionURL = actionsOutput.map {
                    URL(fileURLWithPath: $0).standardizedFileURL
                } ?? outputURL.deletingPathExtension().appendingPathExtension("actions.json")
                try writeActions(actions, to: actionURL)
                if !quiet {
                    CLIStderr.write("Actions: \(actionURL.path)\n")
                }
            }
            generator.unload()
        } catch {
            generator.unload()
            throw error
        }
        if !quiet {
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func makeAction(
        imageURL: URL?,
        videoURL: URL?
    ) throws -> Cosmos3ActionCondition? {
        guard let actionMode = mode.actionMode else { return nil }
        guard let domain = Cosmos3ActionDomain(rawValue: actionDomain) else {
            throw ValidationError("Unknown --action-domain: \(actionDomain)")
        }
        guard let tier = Cosmos3ActionResolutionTier(rawValue: actionResolution) else {
            throw ValidationError("--action-resolution must be 256, 480, 704, or 720.")
        }
        guard let viewpoint = Cosmos3ActionViewpoint(rawValue: actionViewpoint) else {
            throw ValidationError("Unknown --action-viewpoint: \(actionViewpoint)")
        }
        let rawActions: [[Float]]?
        if let actionFile {
            let url = try existingFile(actionFile, field: "--action-file")!
            do {
                rawActions = try JSONDecoder().decode(
                    [[Float]].self,
                    from: Data(contentsOf: url)
                )
            } catch {
                throw ValidationError("Invalid action JSON at \(url.path): \(error.localizedDescription)")
            }
        } else {
            rawActions = nil
        }
        do {
            return try Cosmos3ActionCondition(
                mode: actionMode,
                chunkSize: actionChunkSize,
                domain: domain,
                resolutionTier: tier,
                rawActions: rawActions,
                imageURL: imageURL,
                videoURL: videoURL,
                viewpoint: viewpoint
            )
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    private func validateInputs(
        imageURL: URL?,
        videoURL: URL?,
        hasAction: Bool
    ) throws {
        guard !(imageURL != nil && videoURL != nil) else {
            throw ValidationError("Use either --image or --video, not both.")
        }
        switch mode {
        case .textToImage, .textToVideo:
            guard imageURL == nil, videoURL == nil else {
                throw ValidationError("\(mode.rawValue) does not accept conditioning media.")
            }
        case .imageToImage:
            guard imageURL != nil, videoURL == nil else {
                throw ValidationError("image-to-image requires --image.")
            }
        case .imageToVideo:
            guard imageURL != nil, videoURL == nil else {
                throw ValidationError("image-to-video requires --image.")
            }
        case .videoToVideo:
            guard videoURL != nil, imageURL == nil else {
                throw ValidationError("video-to-video requires --video.")
            }
        case .policy, .forwardDynamics, .inverseDynamics:
            guard hasAction else {
                throw ValidationError("\(mode.rawValue) requires action conditioning.")
            }
        case .reasoner:
            break
        }
    }

    private func runReasoner(
        prompt: String,
        imageURL: URL?,
        videoURL: URL?
    ) async throws {
        guard !(imageURL != nil && videoURL != nil) else {
            throw ValidationError("Use either --image or --video, not both.")
        }
        let root = try await resolveVideoModelRoot(
            explicitModelRoot: nil,
            requestedModel: model,
            variant: .distilled,
            allowAutoDownload: false
        )
        let resources = Cosmos3Resources(rootURL: root)
        let missing = resources.validate() + resources.validateReasoner()
        guard missing.isEmpty else {
            throw ValidationError(
                "Cosmos3-Edge reasoner resources are incomplete: "
                    + missing.map(\.path).joined(separator: ", ")
            )
        }
        let request = try Cosmos3ReasonerRequest(
            prompt: prompt,
            imageURL: imageURL,
            videoURL: videoURL,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topP: topP,
            seed: UInt64(bitPattern: Int64(seed)),
            maximumVideoFrames: maxVideoFrames
        )
        if !quiet {
            CLIStderr.write("Engine: native Cosmos3-Edge MLX reasoner\n")
            CLIStderr.write("Model root: \(root.path)\n")
        }
        let reasoner = Cosmos3Reasoner()
        do {
            let reportsProgress = !quiet
            let result = try await reasoner.generate(
                request: request,
                resources: resources,
                progress: { message in
                    if reportsProgress {
                        CLIStderr.write("\(message)\n")
                    }
                }
            )
            if let output {
                let url = URL(fileURLWithPath: output).standardizedFileURL
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try Data(result.text.utf8).write(to: url, options: .atomic)
                if !quiet { CLIStderr.write("Saved: \(url.path)\n") }
            }
            reasoner.unload()
            print(result.text)
        } catch {
            reasoner.unload()
            throw error
        }
    }

    private func existingFile(_ path: String?, field: String) throws -> URL? {
        guard let path, !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("\(field) file not found: \(url.path)")
        }
        return url
    }

    private func writeActions(_ actions: MLXArray, to url: URL) throws {
        let rows = actions.dim(0)
        let columns = actions.dim(1)
        let values = actions.asType(.float32).asArray(Float.self)
        let nested = (0..<rows).map { row in
            Array(values[(row * columns)..<((row + 1) * columns)])
        }
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(nested).write(to: url, options: .atomic)
    }
}
