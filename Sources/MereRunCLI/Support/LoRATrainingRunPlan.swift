import ArgumentParser
import Foundation

struct LoRATrainingRunPlan: Codable, Equatable {
    static let currentSchemaVersion = 1
    static let kind = "image.train_lora"

    let schemaVersion: Int
    let kind: String
    let command: [String]
    let createdAt: Date
    let cwd: String
    let arguments: LoRATrainingRunPlanArguments
    let resolved: LoRATrainingPlanPreflightSummary

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case command
        case createdAt = "created_at"
        case cwd
        case arguments
        case resolved
    }

    func validateExecutable() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw ValidationError("Unsupported image train-lora plan schema_version \(schemaVersion).")
        }
        guard kind == Self.kind else {
            throw ValidationError("Unsupported image run plan kind '\(kind)'.")
        }
        guard command == ["image", "train-lora"] else {
            throw ValidationError("Unsupported image run plan command: \(command.joined(separator: " ")).")
        }
    }

    static func decode(from url: URL) throws -> LoRATrainingRunPlan {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let plan = try decoder.decode(LoRATrainingRunPlan.self, from: data)
        try plan.validateExecutable()
        return plan
    }

    func relocatingOutput(to output: String) -> LoRATrainingRunPlan {
        LoRATrainingRunPlan(
            schemaVersion: schemaVersion,
            kind: kind,
            command: command,
            createdAt: createdAt,
            cwd: cwd,
            arguments: arguments.relocatingOutput(to: output),
            resolved: resolved
        )
    }
}

struct LoRATrainingRunPlanArguments: Codable, Equatable {
    let data: String?
    let output: String
    let model: String
    let sourceRecipe: String?
    let width: Int
    let height: Int
    let trainingSteps: Int
    let batchSize: Int
    let learningRate: Float
    let rank: Int
    let alpha: Float?
    let maxTextLength: Int
    let schedulerSteps: Int
    let captionDropout: Float
    let seed: UInt64
    let lite: Bool
    let excludePreviewImages: Bool
    let checkpointInterval: Int?
    let maxResolution: Int?
    let progressive: Bool
    let lowRam: Bool
    let noCompile: Bool
    let gradientCheckpointing: Bool
    let benchmarkSteps: Int?
    let benchmarkWarmupSteps: Int
    let sampleInterval: Int?
    let samplePrompt: String?
    let sampleModel: String?
    let sampleSteps: Int
    let sampleGuidanceScale: Double
    let sampleLoRAScale: Double
    let sampleSeed: UInt64?
    let visualize: Bool
    let visualizePort: Int
    let loraTargetRanks: String?
    let loraRankPreset: String?
    let loraTargetPreset: String?
    let loraTargetMode: String?
    let timestepSampling: String?
    let timestepLossWeighting: String?
    let lossWeighting: String?
    let timestepLow: Int?
    let timestepHigh: Int?
    let lrWarmupSteps: Int?
    let noCosineScheduler: Bool
    let lrMinFactor: Float?
    let adamWeightDecay: Float?
    let syntheticSamples: Int?
    let quiet: Bool

    enum CodingKeys: String, CodingKey {
        case data
        case output
        case model
        case sourceRecipe = "source_recipe"
        case width
        case height
        case trainingSteps = "training_steps"
        case batchSize = "batch_size"
        case learningRate = "learning_rate"
        case rank
        case alpha
        case maxTextLength = "max_text_length"
        case schedulerSteps = "scheduler_steps"
        case captionDropout = "caption_dropout"
        case seed
        case lite
        case excludePreviewImages = "exclude_preview_images"
        case checkpointInterval = "checkpoint_interval"
        case maxResolution = "max_resolution"
        case progressive
        case lowRam = "low_ram"
        case noCompile = "no_compile"
        case gradientCheckpointing = "gradient_checkpointing"
        case benchmarkSteps = "benchmark_steps"
        case benchmarkWarmupSteps = "benchmark_warmup_steps"
        case sampleInterval = "sample_interval"
        case samplePrompt = "sample_prompt"
        case sampleModel = "sample_model"
        case sampleSteps = "sample_steps"
        case sampleGuidanceScale = "sample_cfg"
        case sampleLoRAScale = "sample_lora_scale"
        case sampleSeed = "sample_seed"
        case visualize
        case visualizePort = "visualize_port"
        case loraTargetRanks = "lora_target_ranks"
        case loraRankPreset = "lora_rank_preset"
        case loraTargetPreset = "lora_target_preset"
        case loraTargetMode = "lora_target_mode"
        case timestepSampling = "timestep_sampling"
        case timestepLossWeighting = "timestep_loss_weighting"
        case lossWeighting = "loss_weighting"
        case timestepLow = "timestep_low"
        case timestepHigh = "timestep_high"
        case lrWarmupSteps = "lr_warmup_steps"
        case noCosineScheduler = "no_cosine_scheduler"
        case lrMinFactor = "lr_min_factor"
        case adamWeightDecay = "adam_weight_decay"
        case syntheticSamples = "synthetic_samples"
        case quiet
    }

    func executableArgv() -> [String] {
        ["mere.run", "image", "train-lora"] + trainLoRAArguments()
    }

    func relocatingOutput(to output: String) -> LoRATrainingRunPlanArguments {
        LoRATrainingRunPlanArguments(
            data: data,
            output: output,
            model: model,
            sourceRecipe: sourceRecipe,
            width: width,
            height: height,
            trainingSteps: trainingSteps,
            batchSize: batchSize,
            learningRate: learningRate,
            rank: rank,
            alpha: alpha,
            maxTextLength: maxTextLength,
            schedulerSteps: schedulerSteps,
            captionDropout: captionDropout,
            seed: seed,
            lite: lite,
            excludePreviewImages: excludePreviewImages,
            checkpointInterval: checkpointInterval,
            maxResolution: maxResolution,
            progressive: progressive,
            lowRam: lowRam,
            noCompile: noCompile,
            gradientCheckpointing: gradientCheckpointing,
            benchmarkSteps: benchmarkSteps,
            benchmarkWarmupSteps: benchmarkWarmupSteps,
            sampleInterval: sampleInterval,
            samplePrompt: samplePrompt,
            sampleModel: sampleModel,
            sampleSteps: sampleSteps,
            sampleGuidanceScale: sampleGuidanceScale,
            sampleLoRAScale: sampleLoRAScale,
            sampleSeed: sampleSeed,
            visualize: visualize,
            visualizePort: visualizePort,
            loraTargetRanks: loraTargetRanks,
            loraRankPreset: loraRankPreset,
            loraTargetPreset: loraTargetPreset,
            loraTargetMode: loraTargetMode,
            timestepSampling: timestepSampling,
            timestepLossWeighting: timestepLossWeighting,
            lossWeighting: lossWeighting,
            timestepLow: timestepLow,
            timestepHigh: timestepHigh,
            lrWarmupSteps: lrWarmupSteps,
            noCosineScheduler: noCosineScheduler,
            lrMinFactor: lrMinFactor,
            adamWeightDecay: adamWeightDecay,
            syntheticSamples: syntheticSamples,
            quiet: quiet
        )
    }

    func trainLoRAArguments() -> [String] {
        var args: [String] = []
        if let data {
            args += ["--data", data]
        }
        args += [
            "--output", output,
            "--model", model,
            "--width", String(width),
            "--height", String(height),
            "--training-steps", String(trainingSteps),
            "--batch-size", String(batchSize),
            "--learning-rate", String(learningRate),
            "--rank", String(rank),
        ]
        if let alpha {
            args += ["--alpha", String(alpha)]
        }
        args += [
            "--max-text-length", String(maxTextLength),
            "--scheduler-steps", String(schedulerSteps),
            "--caption-dropout", String(captionDropout),
            "--seed", String(seed),
        ]
        appendBoolFlag("--lite", when: lite, to: &args)
        appendBoolFlag("--exclude-preview-images", when: excludePreviewImages, to: &args)
        appendOption("--checkpoint-interval", checkpointInterval, to: &args)
        appendOption("--max-resolution", maxResolution, to: &args)
        appendBoolFlag("--progressive", when: progressive, to: &args)
        appendBoolFlag("--low-ram", when: lowRam, to: &args)
        appendBoolFlag("--no-compile", when: noCompile, to: &args)
        appendBoolFlag("--gradient-checkpointing", when: gradientCheckpointing, to: &args)
        appendOption("--benchmark-steps", benchmarkSteps, to: &args)
        args += ["--benchmark-warmup-steps", String(benchmarkWarmupSteps)]
        appendOption("--sample-interval", sampleInterval, to: &args)
        appendOption("--sample-prompt", samplePrompt, to: &args)
        appendOption("--sample-model", sampleModel, to: &args)
        args += [
            "--sample-steps", String(sampleSteps),
            "--sample-cfg", String(sampleGuidanceScale),
            "--sample-lora-scale", String(sampleLoRAScale),
        ]
        appendOption("--sample-seed", sampleSeed, to: &args)
        appendBoolFlag("--visualize", when: visualize, to: &args)
        if visualize {
            args += ["--visualize-port", String(visualizePort)]
        }
        appendOption("--lora-target-ranks", loraTargetRanks, to: &args)
        appendOption("--lora-rank-preset", loraRankPreset, to: &args)
        appendOption("--lora-target-preset", loraTargetPreset, to: &args)
        appendOption("--lora-target-mode", loraTargetMode, to: &args)
        appendOption("--timestep-sampling", timestepSampling, to: &args)
        appendOption("--timestep-loss-weighting", timestepLossWeighting, to: &args)
        appendOption("--loss-weighting", lossWeighting, to: &args)
        appendOption("--timestep-low", timestepLow, to: &args)
        appendOption("--timestep-high", timestepHigh, to: &args)
        appendOption("--lr-warmup-steps", lrWarmupSteps, to: &args)
        appendBoolFlag("--no-cosine-scheduler", when: noCosineScheduler, to: &args)
        appendOption("--lr-min-factor", lrMinFactor, to: &args)
        appendOption("--adam-weight-decay", adamWeightDecay, to: &args)
        appendOption("--synthetic-samples", syntheticSamples, to: &args)
        appendBoolFlag("--quiet", when: quiet, to: &args)
        return args
    }

    private func appendBoolFlag(_ flag: String, when condition: Bool, to args: inout [String]) {
        if condition {
            args.append(flag)
        }
    }

    private func appendOption<T>(_ flag: String, _ value: T?, to args: inout [String]) {
        if let value {
            args += [flag, String(describing: value)]
        }
    }
}

struct LoRATrainingRunMaterializationRequest: Codable, Equatable {
    let planFile: String
    let runDirectory: String

    enum CodingKeys: String, CodingKey {
        case planFile = "plan_file"
        case runDirectory = "run_directory"
    }
}

struct LoRATrainingRunMaterializationResult: Codable, Equatable {
    let runDirectory: String
    let planPath: String
    let actionsPath: String
    let runManifestPath: String
    let eventsPath: String
    let outputPath: String
    let originalOutputPath: String

    enum CodingKeys: String, CodingKey {
        case runDirectory = "run_directory"
        case planPath = "plan_path"
        case actionsPath = "actions_path"
        case runManifestPath = "run_manifest_path"
        case eventsPath = "events_path"
        case outputPath = "output_path"
        case originalOutputPath = "original_output_path"
    }
}

typealias LoRATrainingRunMaterializationEnvelope = StructuredRunEnvelope<
    LoRATrainingRunMaterializationRequest,
    LoRATrainingRunMaterializationResult
>

extension ImageTrainLoRA {
    func makeRunPlan(
        options: ResolvedLoRATrainingOptions,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> LoRATrainingRunPlan {
        LoRATrainingRunPlan(
            schemaVersion: LoRATrainingRunPlan.currentSchemaVersion,
            kind: LoRATrainingRunPlan.kind,
            command: ["image", "train-lora"],
            createdAt: now(),
            cwd: fileManager.currentDirectoryPath,
            arguments: LoRATrainingRunPlanArguments(
                data: data,
                output: output,
                model: options.model ?? ImageTrainLoRA.defaultManagedModelID.rawValue,
                sourceRecipe: recipe,
                width: options.width,
                height: options.height,
                trainingSteps: options.trainingSteps,
                batchSize: batchSize,
                learningRate: options.learningRate,
                rank: options.rank,
                alpha: options.alpha,
                maxTextLength: maxTextLength,
                schedulerSteps: schedulerSteps,
                captionDropout: options.captionDropout,
                seed: seed,
                lite: lite,
                excludePreviewImages: excludePreviewImages,
                checkpointInterval: options.checkpointInterval,
                maxResolution: options.maxResolution,
                progressive: progressive,
                lowRam: options.lowRam,
                noCompile: options.noCompile,
                gradientCheckpointing: gradientCheckpointing,
                benchmarkSteps: benchmarkSteps,
                benchmarkWarmupSteps: benchmarkWarmupSteps,
                sampleInterval: sampleInterval,
                samplePrompt: samplePrompt,
                sampleModel: sampleModel,
                sampleSteps: sampleSteps,
                sampleGuidanceScale: sampleGuidanceScale,
                sampleLoRAScale: sampleLoRAScale,
                sampleSeed: sampleSeed,
                visualize: visualize,
                visualizePort: visualizePort,
                loraTargetRanks: loraTargetRanks,
                loraRankPreset: loraRankPreset,
                loraTargetPreset: options.loraTargetPreset,
                loraTargetMode: loraTargetMode,
                timestepSampling: timestepSampling,
                timestepLossWeighting: timestepLossWeighting,
                lossWeighting: lossWeighting,
                timestepLow: timestepLow,
                timestepHigh: timestepHigh,
                lrWarmupSteps: options.lrWarmupSteps,
                noCosineScheduler: options.useCosineScheduler == false,
                lrMinFactor: options.lrMinFactor,
                adamWeightDecay: adamWeightDecay,
                syntheticSamples: syntheticSamples,
                quiet: quiet
            ),
            resolved: LoRATrainingPlanPreflightSummary(
                recipe: recipe,
                trainingSteps: options.trainingSteps,
                width: options.width,
                height: options.height,
                rank: options.rank,
                alpha: options.alpha,
                learningRate: options.learningRate,
                captionDropout: options.captionDropout,
                checkpointInterval: options.checkpointInterval,
                expectedCheckpointCount: options.checkpointInterval.map { options.trainingSteps / $0 } ?? 0,
                maxResolution: options.maxResolution,
                lowRam: options.lowRam,
                noCompile: options.noCompile,
                loraTargetPreset: options.loraTargetPreset,
                lrWarmupSteps: options.lrWarmupSteps,
                useCosineScheduler: options.useCosineScheduler,
                lrMinFactor: options.lrMinFactor
            )
        )
    }
}
