import ArgumentParser
import Foundation
import MediaIO
import MereRunCore

enum SCAIL2CLIMode: String, CaseIterable, ExpressibleByArgument {
    case animation
    case replacement

    var runtimeMode: SCAIL2Mode {
        switch self {
        case .animation: .animation
        case .replacement: .replacement
        }
    }
}

enum SCAIL2CLITailPolicy: String, CaseIterable, ExpressibleByArgument {
    case drop
    case padTrim = "pad-trim"

    var runtimePolicy: SCAIL2TailPolicy {
        switch self {
        case .drop: .drop
        case .padTrim: .padTrim
        }
    }
}

enum SCAIL2CLIAudioSource: String, CaseIterable, ExpressibleByArgument {
    case none
    case driving
}

enum SCAIL2CLISampler: String, CaseIterable, ExpressibleByArgument {
    case unipc
    case euler

    var runtimeSampler: SCAIL2Sampler {
        switch self {
        case .unipc: .unipc
        case .euler: .euler
        }
    }
}

enum SCAIL2CLIProfile: String, CaseIterable, ExpressibleByArgument {
    case quality
    case fast
}

struct SCAIL2AnimationPreflightReport: Codable, Equatable {
    let status: String
    let model: String
    let modelRoot: String?
    let modelInstalled: Bool
    let missingModelFiles: [String]
    let missingInputFiles: [String]
    let output: String
    let mode: String
    let profile: String
    let width: Int
    let height: Int
    let steps: Int
    let guidanceScale: Float
    let shift: Float
    let sampler: String
    let denoisingSchedule: String
    let distilledAdapter: String?
    let distilledAdapterStrength: Float
    let fps: Int
    let segmentLength: Int
    let segmentOverlap: Int
    let additionalReferenceCount: Int
    let tailPolicy: String
    let audioSource: String

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case modelRoot = "model_root"
        case modelInstalled = "model_installed"
        case missingModelFiles = "missing_model_files"
        case missingInputFiles = "missing_input_files"
        case output
        case mode
        case profile
        case width
        case height
        case steps
        case guidanceScale = "guidance_scale"
        case shift
        case sampler
        case denoisingSchedule = "denoising_schedule"
        case distilledAdapter = "distilled_adapter"
        case distilledAdapterStrength = "distilled_adapter_strength"
        case fps
        case segmentLength = "segment_length"
        case segmentOverlap = "segment_overlap"
        case additionalReferenceCount = "additional_reference_count"
        case tailPolicy = "tail_policy"
        case audioSource = "audio_source"
    }
}

struct VideoAnimate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "animate",
        abstract: "Animate or replace a masked subject with native Swift/MLX SCAIL-2.",
        discussion: """
        SCAIL-2 consumes a reference image and seven-color reference mask, plus
        a driving video and matching seven-color mask video. Animation preserves
        the source scene; replacement moves the reference subject into the
        driving scene. Prints the output MP4 path to stdout.

        Example:
          mere.run video animate "a dancer in a red silk dress" \
            --reference ref.png --reference-mask ref-mask.png \
            --driving-video pose.mp4 --driving-mask pose-mask.mp4 \
            --model-root /path/to/video-scail2-14b-mlx -o result.mp4
        """
    )

    @Argument(help: "Text prompt describing the subject and motion.")
    var prompt: String

    @Option(name: [.customLong("reference")], help: "Reference RGB image path.")
    var reference: String

    @Option(name: [.customLong("reference-mask")], help: "Seven-color reference mask image path.")
    var referenceMask: String

    @Option(name: [.customLong("driving-video")], help: "Driving pose/render video path.")
    var drivingVideo: String

    @Option(name: [.customLong("driving-mask")], help: "Seven-color driving mask video path.")
    var drivingMask: String

    @Option(name: [.customLong("additional-reference")], parsing: .upToNextOption, help: "Additional reference image path; repeat for multiple subjects.")
    var additionalReferences: [String] = []

    @Option(name: [.customLong("additional-reference-mask")], parsing: .upToNextOption, help: "Mask paired by position with each --additional-reference.")
    var additionalReferenceMasks: [String] = []

    @Option(name: [.customShort("o"), .long], help: "Output MP4 path.")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed SCAIL-2 model id or local MLX model root.")
    var model: String = SCAIL2Resources.modelID

    @Option(name: [.customLong("model-root")], help: "Local SCAIL-2 MLX model root; takes precedence over --model.")
    var modelRoot: String?

    @Option(name: [.long], help: "SCAIL-2 task mode: animation or replacement.")
    var mode: SCAIL2CLIMode = .animation

    @Option(name: [.long], help: "Render recipe: fast (default managed four-step adapter) or quality (40-step UniPC).")
    var profile: SCAIL2CLIProfile = .fast

    @Option(name: [.long], help: "Target width; must be divisible by 32.")
    var width: Int = 832

    @Option(name: [.long], help: "Target height; must be divisible by 32.")
    var height: Int = 480

    @Option(name: [.long], help: "Quality-profile denoising steps (default 40; fast is fixed at 4).")
    var steps: Int?

    @Option(name: [.customLong("guidance-scale")], help: "Quality-profile CFG scale (default 5; fast disables CFG).")
    var guidanceScale: Float?

    @Option(name: [.long], help: "Quality-profile flow shift (default 3; fast is fixed at 5).")
    var shift: Float?

    @Option(name: [.long], help: "Quality-profile sampler (default UniPC; fast is fixed to Euler).")
    var sampler: SCAIL2CLISampler?

    @Option(name: [.customLong("distilled-adapter")], help: "Installed adapter catalog id or local distilled Wan 2.1 I2V safetensors path.")
    var distilledAdapter: String?

    @Option(name: [.customLong("distilled-adapter-strength")], help: "Distilled adapter multiplier.")
    var distilledAdapterStrength: Float = 1

    @Option(name: [.long], help: "Random seed.")
    var seed: Int = 42

    @Option(name: [.long], help: "Output frames per second.")
    var fps: Int = 16

    @Option(name: [.customLong("segment-length")], help: "Pixel frames per long-video segment; must equal 1 modulo 4.")
    var segmentLength: Int = 81

    @Option(name: [.customLong("segment-overlap")], help: "Clean-history overlap in pixel frames; must equal 1 modulo 4.")
    var segmentOverlap: Int = 5

    @Option(name: [.customLong("tail-policy")], help: "Incomplete final-window behavior: drop or pad-trim.")
    var tailPolicy: SCAIL2CLITailPolicy = .drop

    @Option(name: [.customLong("audio-source")], help: "Output audio source: none or driving.")
    var audioSource: SCAIL2CLIAudioSource = .none

    @Option(name: [.customLong("negative-prompt")], help: "Optional negative prompt.")
    var negativePrompt: String = ""

    @Flag(name: [.customLong("preflight")], help: "Validate paths and the execution plan without loading models.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "Emit JSON with --preflight.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Suppress diagnostics on stderr.")
    var quiet: Bool = false

    func run() async throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for video animate.")
        }
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { throw ValidationError("Prompt cannot be empty.") }
        guard additionalReferences.count == additionalReferenceMasks.count else {
            throw ValidationError("--additional-reference and --additional-reference-mask counts must match.")
        }

        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-scail2",
            defaultExtension: "mp4"
        )
        let runtimeRoot = try await resolveModelRoot(forPreflight: preflight)
        let generatedVideoURL = audioSource == .driving && !preflight
            ? outputURL.deletingLastPathComponent()
                .appendingPathComponent(".\(outputURL.deletingPathExtension().lastPathComponent)-\(UUID().uuidString).mp4")
            : outputURL
        let options = try makeOptions(
            prompt: trimmedPrompt,
            outputURL: generatedVideoURL,
            requireInstalledAdapter: !preflight
        )

        if preflight {
            let report = makePreflightReport(options: options, modelRootURL: runtimeRoot)
            if json {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                print(String(decoding: try encoder.encode(report), as: UTF8.self))
            } else {
                if !quiet {
                    CLIStderr.write("SCAIL-2 preflight: \(report.status)\n")
                    if let root = report.modelRoot { CLIStderr.write("Model root: \(root)\n") }
                    for path in report.missingInputFiles { CLIStderr.write("Missing input: \(path)\n") }
                    for path in report.missingModelFiles { CLIStderr.write("Missing model file: \(path)\n") }
                }
                print(report.status)
            }
            return
        }

        guard let runtimeRoot else {
            throw ValidationError("Model \(model) is not installed; convert the pinned SCAIL-2 checkpoint or pass --model-root.")
        }
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if !quiet {
            CLIStderr.write("Engine: native Swift/MLX SCAIL-2\n")
            CLIStderr.write("Model root: \(runtimeRoot.path)\n")
            CLIStderr.write("Mode: \(mode.rawValue)\n")
            CLIStderr.write("Profile: \(profile.rawValue)\n")
            if let distilledAdapterURL = options.distilledAdapterURL {
                CLIStderr.write("Distilled adapter: \(distilledAdapterURL.path)\n")
            }
        }
        let reportsProgress = !quiet
        let result = try await SCAIL2Generator().generate(
            options: options,
            resources: SCAIL2Resources(rootURL: runtimeRoot),
            progressHandler: { progress in
                guard reportsProgress else { return }
                if progress.stage == .denoising {
                    CLIStderr.write("Denoising \(progress.stepIndex + 1)/\(progress.totalSteps)\n")
                } else {
                    CLIStderr.write("\(progress.stage.rawValue)\n")
                }
            }
        )
        if generatedVideoURL != outputURL {
            defer { try? FileManager.default.removeItem(at: generatedVideoURL) }
            let drivingURL = URL(fileURLWithPath: drivingVideo).standardizedFileURL
            if MediaVideoIO.hasAudioTrack(drivingURL) {
                try MediaVideoIO.mux(
                    videoURL: generatedVideoURL,
                    audioURL: drivingURL,
                    outputURL: outputURL,
                    audioBitRate: 192_000
                )
            } else {
                if FileManager.default.fileExists(atPath: outputURL.path) {
                    try FileManager.default.removeItem(at: outputURL)
                }
                try FileManager.default.moveItem(at: generatedVideoURL, to: outputURL)
            }
        }
        if !quiet {
            CLIStderr.write("Frames: \(result.frameCount), segments: \(result.segmentCount)\n")
            CLIStderr.write("Saved: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    func makeOptions(
        prompt: String,
        outputURL: URL,
        requireInstalledAdapter: Bool = true,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir
    ) throws -> SCAIL2GenerationOptions {
        let effectiveAdapterReference = profile == .fast
            ? distilledAdapter ?? ManagedAdapterCatalog.scail2LightX2VFourStepID
            : distilledAdapter
        let resolvedDistilledAdapter = try ManagedAdapterArgumentResolver.resolve(
            effectiveAdapterReference,
            baseModelID: SCAIL2Resources.modelID,
            adaptersRoot: adaptersRoot,
            requireInstalled: requireInstalledAdapter
        )
        let effectiveSteps = profile == .fast ? 4 : (steps ?? 40)
        let effectiveGuidanceScale: Float = profile == .fast ? 1 : (guidanceScale ?? 5)
        let effectiveShift: Float = profile == .fast ? 5 : (shift ?? 3)
        let effectiveSampler: SCAIL2Sampler = profile == .fast ? .euler : (sampler?.runtimeSampler ?? .unipc)
        let effectiveSchedule: SCAIL2DenoisingSchedule = profile == .fast
            ? .lightX2VFourStep
            : .standard
        return try SCAIL2GenerationOptions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            reference: SCAIL2ReferenceInput(
                imageURL: URL(fileURLWithPath: reference).standardizedFileURL,
                maskURL: URL(fileURLWithPath: referenceMask).standardizedFileURL
            ),
            additionalReferences: zip(additionalReferences, additionalReferenceMasks).map {
                SCAIL2ReferenceInput(
                    imageURL: URL(fileURLWithPath: $0.0).standardizedFileURL,
                    maskURL: URL(fileURLWithPath: $0.1).standardizedFileURL
                )
            },
            drivingVideoURL: URL(fileURLWithPath: drivingVideo).standardizedFileURL,
            drivingMaskVideoURL: URL(fileURLWithPath: drivingMask).standardizedFileURL,
            outputURL: outputURL,
            mode: mode.runtimeMode,
            width: width,
            height: height,
            steps: effectiveSteps,
            guidanceScale: effectiveGuidanceScale,
            shift: effectiveShift,
            sampler: effectiveSampler,
            denoisingSchedule: effectiveSchedule,
            distilledAdapterURL: resolvedDistilledAdapter.map {
                URL(fileURLWithPath: $0).standardizedFileURL
            },
            distilledAdapterStrength: distilledAdapterStrength,
            seed: UInt64(bitPattern: Int64(seed)),
            fps: fps,
            segmentLength: segmentLength,
            segmentOverlap: segmentOverlap,
            tailPolicy: tailPolicy.runtimePolicy
        )
    }

    func makePreflightReport(
        options: SCAIL2GenerationOptions,
        modelRootURL: URL?,
        adaptersRoot: URL = MereRunModelPaths.adaptersDir
    ) -> SCAIL2AnimationPreflightReport {
        let inputs = [
            options.reference.imageURL,
            options.reference.maskURL,
            options.drivingVideoURL,
            options.drivingMaskVideoURL,
        ] + options.additionalReferences.flatMap { [$0.imageURL, $0.maskURL] }
            + (options.distilledAdapterURL.map { [$0] } ?? [])
        let effectiveAdapterReference = profile == .fast
            ? distilledAdapter ?? ManagedAdapterCatalog.scail2LightX2VFourStepID
            : distilledAdapter
        let managedAdapterSpec = effectiveAdapterReference.flatMap {
            ManagedAdapterCatalog.spec(for: $0)
        }
        let missingInputs = inputs.filter { input in
            if input == options.distilledAdapterURL,
               let managedAdapterSpec {
                return !managedAdapterSpec.isInstalled(adaptersRoot: adaptersRoot)
            }
            return !FileManager.default.fileExists(atPath: input.path)
        }
        let missingModels = modelRootURL.map { SCAIL2Resources(rootURL: $0).validate() } ?? []
        let installed = modelRootURL != nil && missingModels.isEmpty
        return SCAIL2AnimationPreflightReport(
            status: missingInputs.isEmpty && installed ? "ok" : "blocked",
            model: model,
            modelRoot: modelRootURL?.path,
            modelInstalled: installed,
            missingModelFiles: missingModels.map(\.path),
            missingInputFiles: missingInputs.map(\.path),
            output: CLIOutput.resolveOutputURL(
                output,
                defaultPrefix: "mererun-scail2",
                defaultExtension: "mp4"
            ).path,
            mode: mode.rawValue,
            profile: profile.rawValue,
            width: width,
            height: height,
            steps: options.steps,
            guidanceScale: options.guidanceScale,
            shift: options.shift,
            sampler: options.sampler.rawValue,
            denoisingSchedule: options.denoisingSchedule.rawValue,
            distilledAdapter: options.distilledAdapterURL?.path,
            distilledAdapterStrength: options.distilledAdapterStrength,
            fps: fps,
            segmentLength: segmentLength,
            segmentOverlap: segmentOverlap,
            additionalReferenceCount: additionalReferences.count,
            tailPolicy: tailPolicy.rawValue,
            audioSource: audioSource.rawValue
        )
    }

    private func resolveModelRoot(forPreflight: Bool) async throws -> URL? {
        if let modelRoot, !modelRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return URL(fileURLWithPath: modelRoot).standardizedFileURL
        }
        let requested = model.trimmingCharacters(in: .whitespacesAndNewlines)
        let explicit = URL(fileURLWithPath: requested).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicit.path) { return explicit }
        if forPreflight {
            return ManagedModelResolver.resolveInstalledModel(id: requested)
        }
        do {
            return try await ManagedModelResolver.resolveForRuntime(
                requestedModel: requested,
                defaultModelID: SCAIL2Resources.modelID,
                allowAutoDownload: false
            ).url
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }
    }
}
