import Foundation

// Image ▸ Datasets ▸ Run plan reads back what `image run-plan --preflight --json` and
// `--materialize --json` print. A preflight runs `image train-lora --preflight` or `image generate
// --preflight` for the saved plan, so the envelope is `StructuredRunEnvelope` in `MereRunCLI` with a
// `LoRATrainingPreflightResult` or `ImageGenerationPreflightResult`; a materialize prints the run
// directory's files. Studio does not import the CLI, so these types mirror those payloads and read
// only the fields the page shows; a decode failure means the CLI changed.

/// A finished run-plan command, decoded, and the sections the page draws from it.
package struct StudioRunPlanReport: Decodable, Equatable {
    package struct Diagnostic: Decodable, Equatable, Identifiable {
        package let id: String
        package let severity: Severity
        package let title: String
        package let message: String
    }

    package enum Severity: String, Decodable, Equatable {
        case blocker
        case warning
        case note
        case estimate
        /// A severity this build does not know; the report still reads, shown as a note.
        case unknown

        package init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Severity(rawValue: raw) ?? .unknown
        }
    }

    /// What was checked or written, by the command the plan wraps.
    package enum Result: Equatable {
        case training(TrainingPreflight)
        case generation(GenerationPreflight)
        case materialized(Materialization)
    }

    /// `image train-lora --preflight`'s result.
    package struct TrainingPreflight: Decodable, Equatable {
        package struct Dataset: Decodable, Equatable {
            package let directory: String?
            package let mode: String
            package let imageCount: Int
            package let captionCount: Int
            package let usablePairCount: Int
            package let missingCaptionCount: Int
            package let emptyCaptionCount: Int
            package let duplicateCaptionCount: Int
            package let excludedPreviewImageCount: Int
            package let syntheticSampleCount: Int?

            enum CodingKeys: String, CodingKey {
                case directory
                case mode
                case imageCount = "image_count"
                case captionCount = "caption_count"
                case usablePairCount = "usable_pair_count"
                case missingCaptionCount = "missing_caption_count"
                case emptyCaptionCount = "empty_caption_count"
                case duplicateCaptionCount = "duplicate_caption_count"
                case excludedPreviewImageCount = "excluded_preview_image_count"
                case syntheticSampleCount = "synthetic_sample_count"
            }
        }

        /// The resolved recipe values the trainer will use (`LoRATrainingPlanPreflightSummary`).
        package struct Plan: Decodable, Equatable {
            package let recipe: String?
            package let trainingSteps: Int
            package let width: Int
            package let height: Int
            package let rank: Int
            package let alpha: Double?
            package let learningRate: Double
            package let captionDropout: Double
            package let checkpointInterval: Int?
            package let expectedCheckpointCount: Int
            package let maxResolution: Int?
            package let lowRam: Bool
            package let noCompile: Bool
            package let loraTargetPreset: String?
            package let lrWarmupSteps: Int?
            package let useCosineScheduler: Bool?
            package let lrMinFactor: Double?

            enum CodingKeys: String, CodingKey {
                case recipe
                case trainingSteps = "training_steps"
                case width
                case height
                case rank
                case alpha
                case learningRate = "learning_rate"
                case captionDropout = "caption_dropout"
                case checkpointInterval = "checkpoint_interval"
                case expectedCheckpointCount = "expected_checkpoint_count"
                case maxResolution = "max_resolution"
                case lowRam = "low_ram"
                case noCompile = "no_compile"
                case loraTargetPreset = "lora_target_preset"
                case lrWarmupSteps = "lr_warmup_steps"
                case useCosineScheduler = "use_cosine_scheduler"
                case lrMinFactor = "lr_min_factor"
            }
        }

        /// The arguments the saved plan carries (`LoRATrainingRunPlanArguments`), those the page shows.
        package struct Arguments: Decodable, Equatable {
            package let data: String?
            package let output: String
            package let model: String
            package let batchSize: Int
            package let seed: UInt64
            package let schedulerSteps: Int
            package let lite: Bool
            package let progressive: Bool
            package let gradientCheckpointing: Bool
            package let baseQuantizationBits: Int?
            package let resumeFrom: String?
            package let sampleInterval: Int?
            package let samplePrompt: String?
            package let sampleModel: String?

            enum CodingKeys: String, CodingKey {
                case data
                case output
                case model
                case batchSize = "batch_size"
                case seed
                case schedulerSteps = "scheduler_steps"
                case lite
                case progressive
                case gradientCheckpointing = "gradient_checkpointing"
                case baseQuantizationBits = "base_quantization_bits"
                case resumeFrom = "resume_from"
                case sampleInterval = "sample_interval"
                case samplePrompt = "sample_prompt"
                case sampleModel = "sample_model"
            }
        }

        package struct RunPlan: Decodable, Equatable {
            package let kind: String
            package let cwd: String
            package let arguments: Arguments
            package let resolved: Plan
        }

        package let dataset: Dataset
        package let model: StudioRunPlanModelSummary
        package let output: StudioRunPlanOutputSummary
        package let runPlan: RunPlan

        enum CodingKeys: String, CodingKey {
            case dataset
            case model
            case output
            case runPlan = "run_plan"
        }
    }

    /// `image generate --preflight`'s result.
    package struct GenerationPreflight: Decodable, Equatable {
        package struct Plan: Decodable, Equatable {
            package let family: String?
            package let width: Int
            package let height: Int
            package let requestedSteps: Int?
            package let effectiveSteps: Int?
            package let requestedCFGScale: Double?
            package let effectiveCFGScale: Double?
            package let effectiveSigmaShift: Double?
            package let effectiveMaxSequenceLength: Int
            package let inputMode: String

            enum CodingKeys: String, CodingKey {
                case family
                case width
                case height
                case requestedSteps = "requested_steps"
                case effectiveSteps = "effective_steps"
                case requestedCFGScale = "requested_cfg_scale"
                case effectiveCFGScale = "effective_cfg_scale"
                case effectiveSigmaShift = "effective_sigma_shift"
                case effectiveMaxSequenceLength = "effective_max_sequence_length"
                case inputMode = "input_mode"
            }
        }

        package struct Path: Decodable, Equatable {
            package let path: String
            package let exists: Bool
        }

        package struct Inputs: Decodable, Equatable {
            package let inputImage: Path?
            package let maskImage: Path?
            package let referenceImages: [Path]
            package let missingCount: Int

            enum CodingKeys: String, CodingKey {
                case inputImage = "input_image"
                case maskImage = "mask_image"
                case referenceImages = "reference_images"
                case missingCount = "missing_count"
            }
        }

        package struct Adapter: Decodable, Equatable {
            package let requested: String
            package let exists: Bool
            package let scale: Double
        }

        package struct Arguments: Decodable, Equatable {
            package let prompt: String
            package let negativePrompt: String?
            package let model: String
            package let output: String
            package let seed: UInt64?

            enum CodingKeys: String, CodingKey {
                case prompt
                case negativePrompt = "negative_prompt"
                case model
                case output
                case seed
            }
        }

        package struct RunPlan: Decodable, Equatable {
            package let kind: String
            package let cwd: String
            package let arguments: Arguments
        }

        package let model: StudioRunPlanModelSummary
        package let output: StudioRunPlanOutputSummary
        package let inputs: Inputs
        package let loras: [Adapter]
        package let plan: Plan
        package let runPlan: RunPlan

        enum CodingKeys: String, CodingKey {
            case model
            case output
            case inputs
            case loras
            case plan
            case runPlan = "run_plan"
        }
    }

    /// `--materialize`'s result: the run directory and the files written into it.
    package struct Materialization: Decodable, Equatable {
        package let runDirectory: String
        package let planPath: String
        package let actionsPath: String
        package let runManifestPath: String
        package let eventsPath: String
        package let outputPath: String
        package let originalOutputPath: String
        package let structuredPromptOutputPath: String?

        enum CodingKeys: String, CodingKey {
            case runDirectory = "run_directory"
            case planPath = "plan_path"
            case actionsPath = "actions_path"
            case runManifestPath = "run_manifest_path"
            case eventsPath = "events_path"
            case outputPath = "output_path"
            case originalOutputPath = "original_output_path"
            case structuredPromptOutputPath = "structured_prompt_output_path"
        }
    }

    /// "ok", "warning", or "blocked" for a preflight; "ok" for a materialize.
    package let status: String
    package let mode: String
    package let command: [String]
    package let summary: String
    package let diagnostics: [Diagnostic]
    package let result: Result

    enum CodingKeys: String, CodingKey {
        case status
        case mode
        case command
        case summary
        case diagnostics
        case result
    }

    package init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decode(String.self, forKey: .status)
        mode = try container.decode(String.self, forKey: .mode)
        command = try container.decode([String].self, forKey: .command)
        summary = try container.decode(String.self, forKey: .summary)
        diagnostics = try container.decode([Diagnostic].self, forKey: .diagnostics)
        if mode == "materialize" {
            result = .materialized(try container.decode(Materialization.self, forKey: .result))
        } else if command == ["image", "generate"] {
            result = .generation(try container.decode(GenerationPreflight.self, forKey: .result))
        } else {
            result = .training(try container.decode(TrainingPreflight.self, forKey: .result))
        }
    }

    package static func decode(_ data: Data) -> StudioRunPlanReport? {
        try? JSONDecoder().decode(StudioRunPlanReport.self, from: data)
    }

    /// Reads a run's captured output, where stdout comes first and any stderr follows a `STDERR` line.
    package static func decode(outputText: String) -> StudioRunPlanReport? {
        let stdout = outputText.components(separatedBy: "\n\nSTDERR\n").first ?? outputText
        return decode(Data(stdout.utf8))
    }

    // MARK: - Presentation

    /// One labelled group of facts. A row with a `path` is a file or folder the page can reveal.
    package struct Section: Equatable, Identifiable {
        package struct Row: Equatable, Identifiable {
            package let label: String
            package let value: String
            package let path: String?

            package var id: String { label }

            package init(_ label: String, _ value: String, path: String? = nil) {
                self.label = label
                self.value = value
                self.path = path
            }
        }

        package let title: String
        package let rows: [Row]

        package var id: String { title }
    }

    /// "Training plan", "Generation plan", or "Materialized run".
    package var title: String {
        switch result {
        case .training: return "Training plan"
        case .generation: return "Generation plan"
        case .materialized: return "Materialized run"
        }
    }

    /// The plan's facts in the order the page shows them: what will run, on what, to where.
    package var sections: [Section] {
        switch result {
        case .training(let preflight): return Self.sections(for: preflight)
        case .generation(let preflight): return Self.sections(for: preflight)
        case .materialized(let run): return Self.sections(for: run)
        }
    }

    private static func sections(for preflight: TrainingPreflight) -> [Section] {
        let plan = preflight.runPlan.resolved
        let arguments = preflight.runPlan.arguments
        var training: [Section.Row] = [
            .init("Steps", plan.trainingSteps.formatted()),
            .init("Resolution", "\(plan.width) × \(plan.height)"),
            .init("Batch size", arguments.batchSize.formatted()),
            .init("Rank", plan.alpha.map { "\(plan.rank) · alpha \(number($0))" } ?? "\(plan.rank)"),
            .init("Learning rate", number(plan.learningRate)),
            .init("Caption dropout", percent(plan.captionDropout)),
            .init("Seed", String(arguments.seed)),
        ]
        if let recipe = plan.recipe { training.insert(.init("Recipe", recipe), at: 0) }
        if let interval = plan.checkpointInterval, interval > 0 {
            training.append(.init("Checkpoints", "every \(interval.formatted()) steps · \(plan.expectedCheckpointCount) expected"))
        }
        if let interval = arguments.sampleInterval, interval > 0 {
            training.append(.init("Previews", "every \(interval.formatted()) steps"))
        }
        var schedule: [Section.Row] = [.init("Scheduler steps", arguments.schedulerSteps.formatted())]
        if let warmup = plan.lrWarmupSteps { schedule.append(.init("Warmup steps", warmup.formatted())) }
        if let cosine = plan.useCosineScheduler {
            schedule.append(.init("Cosine schedule", cosine ? "On" : "Off"))
        }
        if let factor = plan.lrMinFactor { schedule.append(.init("Minimum rate factor", number(factor))) }
        var memory: [Section.Row] = []
        if let resolution = plan.maxResolution { memory.append(.init("Maximum source resolution", String(resolution))) }
        if let bits = arguments.baseQuantizationBits { memory.append(.init("Frozen-base quantization", "\(bits)-bit")) }
        let switches = [
            ("Progressive resolution", arguments.progressive),
            ("Low RAM latent cache", plan.lowRam),
            ("Gradient checkpointing", arguments.gradientCheckpointing),
            ("Compiled train step", !plan.noCompile),
            ("Lite attention targets", arguments.lite),
        ].filter(\.1).map(\.0)
        if !switches.isEmpty { memory.append(.init("Enabled", switches.joined(separator: ", "))) }
        if let preset = plan.loraTargetPreset { memory.append(.init("Target preset", preset)) }

        let dataset = preflight.dataset
        var datasetRows: [Section.Row] = []
        if let directory = dataset.directory { datasetRows.append(.init("Folder", directory, path: directory)) }
        if let synthetic = dataset.syntheticSampleCount {
            datasetRows.append(.init("Samples", "\(synthetic) synthetic"))
        } else {
            datasetRows.append(.init("Usable pairs", "\(dataset.usablePairCount) of \(dataset.imageCount) images"))
            var issues: [String] = []
            if dataset.missingCaptionCount > 0 { issues.append("\(dataset.missingCaptionCount) missing captions") }
            if dataset.emptyCaptionCount > 0 { issues.append("\(dataset.emptyCaptionCount) empty captions") }
            if dataset.duplicateCaptionCount > 0 { issues.append("\(dataset.duplicateCaptionCount) duplicate captions") }
            if dataset.excludedPreviewImageCount > 0 { issues.append("\(dataset.excludedPreviewImageCount) previews excluded") }
            if !issues.isEmpty { datasetRows.append(.init("Issues", issues.joined(separator: ", "))) }
        }
        if let resume = arguments.resumeFrom { datasetRows.append(.init("Resume from", resume, path: resume)) }

        var sections = [
            Section(title: "Training", rows: training),
            Section(title: "Schedule", rows: schedule),
        ]
        if !memory.isEmpty { sections.append(Section(title: "Memory and targets", rows: memory)) }
        sections.append(Section(title: "Dataset", rows: datasetRows))
        sections.append(Section(title: "Model", rows: modelRows(preflight.model)))
        sections.append(Section(title: "Output", rows: outputRows(preflight.output)))
        return sections
    }

    private static func sections(for preflight: GenerationPreflight) -> [Section] {
        let plan = preflight.plan
        let arguments = preflight.runPlan.arguments
        var generation: [Section.Row] = [
            .init("Prompt", arguments.prompt),
        ]
        if let negative = arguments.negativePrompt, !negative.isEmpty { generation.append(.init("Avoid", negative)) }
        generation.append(.init("Resolution", "\(plan.width) × \(plan.height)"))
        if let steps = plan.effectiveSteps ?? plan.requestedSteps { generation.append(.init("Steps", steps.formatted())) }
        if let cfg = plan.effectiveCFGScale ?? plan.requestedCFGScale { generation.append(.init("Guidance", number(cfg))) }
        if let shift = plan.effectiveSigmaShift { generation.append(.init("Sigma shift", number(shift))) }
        generation.append(.init("Sequence length", plan.effectiveMaxSequenceLength.formatted()))
        if let seed = arguments.seed { generation.append(.init("Seed", String(seed))) }
        generation.append(.init("Input mode", plan.inputMode.replacingOccurrences(of: "_", with: " ")))

        var inputs: [Section.Row] = []
        if let input = preflight.inputs.inputImage {
            inputs.append(.init("Input image", input.exists ? input.path : "Missing: \(input.path)", path: input.path))
        }
        if let mask = preflight.inputs.maskImage {
            inputs.append(.init("Mask", mask.exists ? mask.path : "Missing: \(mask.path)", path: mask.path))
        }
        for (index, reference) in preflight.inputs.referenceImages.enumerated() {
            inputs.append(.init(
                "Reference \(index + 1)",
                reference.exists ? reference.path : "Missing: \(reference.path)",
                path: reference.path
            ))
        }
        for (index, adapter) in preflight.loras.enumerated() {
            inputs.append(.init(
                preflight.loras.count == 1 ? "Adapter" : "Adapter \(index + 1)",
                "\(adapter.requested) · scale \(number(adapter.scale))" + (adapter.exists ? "" : " · missing")
            ))
        }

        var sections = [Section(title: "Generation", rows: generation)]
        if !inputs.isEmpty { sections.append(Section(title: "Inputs", rows: inputs)) }
        sections.append(Section(title: "Model", rows: modelRows(preflight.model)))
        sections.append(Section(title: "Output", rows: outputRows(preflight.output)))
        return sections
    }

    private static func sections(for run: Materialization) -> [Section] {
        var files: [Section.Row] = [
            .init("Run folder", run.runDirectory, path: run.runDirectory),
            .init("Plan", run.planPath, path: run.planPath),
            .init("Actions", run.actionsPath, path: run.actionsPath),
            .init("Run manifest", run.runManifestPath, path: run.runManifestPath),
            .init("Events", run.eventsPath, path: run.eventsPath),
            .init("Output", run.outputPath, path: run.outputPath),
        ]
        if let sidecar = run.structuredPromptOutputPath {
            files.append(.init("Structured prompt", sidecar, path: sidecar))
        }
        return [
            Section(title: "Files", rows: files),
            Section(title: "Before", rows: [.init("Original output", run.originalOutputPath, path: run.originalOutputPath)]),
        ]
    }

    private static func modelRows(_ model: StudioRunPlanModelSummary) -> [Section.Row] {
        var rows: [Section.Row] = [.init("Model", model.requested)]
        if let family = model.family { rows.append(.init("Family", family)) }
        if model.installed {
            rows.append(.init("Installed", model.path ?? "Yes", path: model.path))
        } else if let bytes = model.estimatedDownloadBytes {
            rows.append(.init("Installed", "No · about \(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)) to download"))
        } else {
            rows.append(.init("Installed", "No"))
        }
        return rows
    }

    private static func outputRows(_ output: StudioRunPlanOutputSummary) -> [Section.Row] {
        var rows: [Section.Row] = [.init("File", output.path, path: output.path)]
        var notes: [String] = []
        if output.exists { notes.append("already exists") }
        if !output.extensionValid { notes.append("unexpected extension") }
        if output.parentWillBeCreated { notes.append("folder will be created") }
        if !notes.isEmpty { rows.append(.init("Note", notes.joined(separator: ", "))) }
        return rows
    }

    /// 0.0001 → "0.0001", 16 → "16", 3.5 → "3.5".
    static func number(_ value: Double) -> String {
        value.formatted(.number.precision(.significantDigits(1...6)).grouping(.never))
    }

    /// 0.1 → "10%".
    static func percent(_ value: Double) -> String {
        value.formatted(.percent.precision(.fractionLength(0...1)))
    }
}

/// The model a plan names and whether it is here (`LoRATrainingModelPreflightSummary`,
/// `ImageGenerationModelPreflightSummary`).
package struct StudioRunPlanModelSummary: Decodable, Equatable {
    package let requested: String
    package let installed: Bool
    package let path: String?
    package let family: String?
    package let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case installed
        case path
        case family
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

/// Where the plan writes (`LoRATrainingOutputPreflightSummary`, `ImageGenerationOutputPreflightSummary`).
package struct StudioRunPlanOutputSummary: Decodable, Equatable {
    package let path: String
    package let parentWillBeCreated: Bool
    package let exists: Bool
    package let extensionValid: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case extensionValid = "extension_valid"
    }
}
