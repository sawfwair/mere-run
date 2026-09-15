import ArgumentParser
import Foundation
import MereRunCore

struct TextTrainLoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-lora",
        abstract: "Train a native text or image-conditioned LoRA adapter from chat-style SFT JSONL.",
        discussion: """
        This is the native mere.run text fine-tuning entrypoint. It accepts OpenAI-style chat
        JSONL records with system/user/assistant messages and trains Gemma 4, Laguna XS 2.1,
        Inkling-Small, or the LFM2.5 A1B 8-bit text model. Gemma 4 12B vision
        fine-tuning accepts one dataset-relative local image on each user message.
        It writes a model-family-specific MereRun adapter manifest beside the safetensors file.
        Use --resume-from to continue a matching optimizer-bearing checkpoint toward the
        original total --training-steps. Legacy checkpoints also require --resume-step.
        Use --dry-run to validate data, create manifests, and prepare a reproducible run.
        """
    )

    @Option(name: [.customShort("d"), .long], help: "SFT JSONL dataset path.")
    var data: String

    @Option(name: [.customShort("o"), .long], help: "Output .safetensors adapter path.")
    var output: String

    @Option(name: [.customShort("m"), .long], help: "Base text or Gemma 4 vision model id.")
    var model: String = Gemma4Resources.twelveB4BitModelId

    @Option(name: [.customLong("model-path")], help: "Optional explicit base model directory.")
    var modelPath: String?

    @Option(
        name: [.customLong("eval")],
        help: "Optional held-out SFT JSONL path for before/after assistant-token loss."
    )
    var eval: String?

    @Option(name: [.customLong("adapter-name")], help: "Adapter display name.")
    var adapterName: String = "local-assistant"

    @Option(name: [.customLong("training-steps"), .customLong("steps")], help: "Number of optimizer steps.")
    var trainingSteps: Int = 600

    @Option(name: [.long], help: "Batch size.")
    var batchSize: Int = 1

    @Option(name: [.customLong("learning-rate"), .customLong("lr")], help: "Learning rate.")
    var learningRate: Float = 0.0001

    @Option(name: [.customLong("rank")], help: "LoRA rank.")
    var rank: Int = 16

    @Option(name: [.customLong("alpha")], help: "LoRA alpha. Defaults to rank.")
    var alpha: Float?

    @Option(name: [.customLong("max-sequence-length")], help: "Maximum training sequence length.")
    var maxSequenceLength: Int = 4096

    @Option(
        name: [.customLong("reasoning-effort")],
        help: "Inkling-Small renderer effort from 0 through 0.99. Default: 0.9."
    )
    var reasoningEffort: Double = 0.9

    @Option(name: [.long], help: "Random seed.")
    var seed: UInt64 = 42

    @Option(
        name: [.customLong("resume-from")],
        help: "Resume from a matching text LoRA .safetensors checkpoint with Adam state."
    )
    var resumeFrom: String?

    @Option(
        name: [.customLong("resume-step")],
        help: "Completed global step for a legacy checkpoint without embedded step state."
    )
    var resumeStep: Int?

    @Option(
        name: [.customLong("target-modules")],
        help: "Comma-separated LoRA target suffixes. Defaults to attention for Gemma, Laguna, and LFM2.5; Inkling also includes MLP, expert, and unembedding targets."
    )
    var targetModules: String?

    @Flag(name: [.customLong("dry-run")], help: "Validate data and write manifests without optimizer steps.")
    var dryRun: Bool = false

    @Flag(name: [.customLong("visualize")], help: "Start a loopback LoRA training dashboard for this run.")
    var visualize: Bool = false

    @Option(name: [.customLong("visualize-port")], help: "Loopback port for --visualize.")
    var visualizePort: Int = 8787

    @Flag(name: [.long], help: "Print a machine-readable JSON summary.")
    var json: Bool = false

    func run() async throws {
        try validateOptions()
        _ = try resolvedTrainingFamily()
        if !dryRun {
            try MLXBundleSupport.ensureAvailable(quiet: json)
        }

        let plan: TextLoRATrainingPlan
        do {
            plan = try TextLoRATrainingPlan.resolve(trainingOptions)
        } catch let issue as TextLoRATrainingIssue {
            throw ValidationError(issue.message)
        }
        let dataURL = plan.dataURL
        let outputURL = plan.outputURL
        let summary = plan.dataset.summary
        let evalCount = plan.evalPromptCount
        let visualization = try startVisualizationIfNeeded(
            dataURL: dataURL,
            outputURL: outputURL,
            datasetSummary: summary,
            evalPromptCount: evalCount
        )
        do {
            try await execute(plan, visualization: visualization)
            await visualization?.stop()
        } catch {
            await visualization?.stop()
            throw error
        }
    }

    private func execute(_ plan: TextLoRATrainingPlan, visualization: TextLoRATrainingVisualization?) async throws {
        let dataURL = plan.dataURL
        let outputURL = plan.outputURL
        let summary = plan.dataset.summary
        let evalCount = plan.evalPromptCount
        let outcome: TextLoRATrainingOutcome
        do {
            outcome = try await TextLoRATrainingOperation.execute(
                plan,
                progressHandler: makeChatProgressHandler(eventLogger: visualization?.logger),
                trainingProgressHandler: makeTrainingProgressHandler(eventLogger: visualization?.logger)
            )
        } catch {
            try? visualization?.logger.record(
                type: "run_failed", stage: "failed", message: error.localizedDescription, path: outputURL.path
            )
            throw error
        }
        let report = outcome.report
        let manifest = outcome.manifest
        if !dryRun {
            try recordVisualizationRunManifest(
                dataURL: dataURL,
                outputURL: outputURL,
                datasetSummary: summary,
                evalPromptCount: evalCount,
                status: "trained",
                step: trainingSteps
            )
            try visualization?.logger.record(
                type: "run_finished",
                stage: "finished",
                step: trainingSteps,
                totalSteps: trainingSteps,
                fraction: 1,
                path: outputURL.path
            )
        }

        let result = TextTrainLoRAResult(
            model: model,
            adapterName: adapterName,
            outputPath: outputURL.path,
            manifestPath: TextLoRATrainingManifest.url(nextTo: outputURL).path,
            dataset: summary,
            evalPromptCount: evalCount,
            dryRun: dryRun,
            status: manifest.status,
            trainingReport: report
        )

        if json {
            print(try result.jsonString())
        } else {
            print("Dataset: \(summary.exampleCount) examples, fingerprint \(summary.fingerprint)")
            print("Manifest: \(result.manifestPath)")
            print("Adapter: \(result.outputPath)")
            if let report {
                let finalLoss = report.finalLoss.map { String($0) } ?? "n/a"
                print("Training: \(report.steps) steps, \(report.layerCount) LoRA layers, final loss \(finalLoss)")
                if let initialEvaluationLoss = report.initialEvaluationLoss,
                   let finalEvaluationLoss = report.finalEvaluationLoss {
                    print(
                        "Evaluation: \(report.evaluationExampleCount) examples, "
                            + "\(report.evaluationTargetTokenCount) assistant tokens, "
                            + "loss \(initialEvaluationLoss) -> \(finalEvaluationLoss)"
                    )
                }
            }
        }
    }

    var trainingOptions: TextLoRATrainingOptions {
        TextLoRATrainingOptions(
            data: data,
            output: output,
            model: model,
            modelPath: modelPath,
            eval: eval,
            adapterName: adapterName,
            trainingSteps: trainingSteps,
            batchSize: batchSize,
            learningRate: learningRate,
            rank: rank,
            alpha: alpha,
            maxSequenceLength: maxSequenceLength,
            reasoningEffort: reasoningEffort,
            seed: seed,
            resumeFrom: resumeFrom,
            resumeStep: resumeStep,
            targetModules: targetModules,
            dryRun: dryRun
        )
    }

    private func validateOptions() throws {
        do {
            try trainingOptions.validate()
        } catch let issue as TextLoRATrainingIssue {
            throw ValidationError(issue.message)
        }
        if visualize {
            guard !dryRun else {
                throw ValidationError("--visualize cannot be combined with --dry-run")
            }
            guard (1...65535).contains(visualizePort) else {
                throw ValidationError("--visualize-port must be between 1 and 65535")
            }
        }
    }

    func resolvedTargetModules() -> [String] { trainingOptions.resolvedTargetModules() }

    static func defaultTargetModules(for model: String) -> [String] {
        TextLoRATrainingOptions.defaultTargetModules(for: model)
    }

    private func resolvedTrainingFamily() throws -> TextLoRATrainingFamily {
        do {
            return try trainingOptions.resolvedTrainingFamily()
        } catch let issue as TextLoRATrainingIssue {
            throw ValidationError(issue.message)
        }
    }

    private func startVisualizationIfNeeded(
        dataURL: URL,
        outputURL: URL,
        datasetSummary: TextSFTDatasetSummary,
        evalPromptCount: Int?
    ) throws -> TextLoRATrainingVisualization? {
        guard visualize else { return nil }
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        try recordVisualizationRunManifest(
            dataURL: dataURL,
            outputURL: outputURL,
            datasetSummary: datasetSummary,
            evalPromptCount: evalPromptCount,
            status: "running",
            step: resumeStep ?? 0
        )

        let logger = try LoRATrainingEventLogger(baseOutputURL: outputURL)
        let viewer = LoRATrainingRunViewer(runDirectoryURL: outputURL.deletingLastPathComponent())
        let host = "127.0.0.1"
        let port = visualizePort
        try logger.record(
            type: "run_started",
            stage: "starting",
            message: "Text LoRA training started.",
            step: resumeStep ?? 0,
            totalSteps: trainingSteps,
            fraction: 0,
            path: outputURL.path,
            metadata: [
                "viewer_url": "http://\(host):\(port)",
                "model_id": model,
                "model_path": modelPath ?? "",
                "data": dataURL.path,
                "output": outputURL.path,
                "dataset_fingerprint": datasetSummary.fingerprint,
            ]
        )
        let task = Task {
            do {
                try await viewer.run(host: host, port: port)
            } catch is CancellationError {
                // The training command owns this helper server and cancels it when the run exits.
            } catch {
                CLIStderr.write("[visualize] server stopped: \(error.localizedDescription)\n")
            }
        }
        return TextLoRATrainingVisualization(logger: logger, serverTask: task)
    }

    private func recordVisualizationRunManifest(
        dataURL: URL,
        outputURL: URL,
        datasetSummary: TextSFTDatasetSummary,
        evalPromptCount: Int?,
        status: String,
        step: Int
    ) throws {
        guard visualize else { return }
        let outputDirectory = outputURL.deletingLastPathComponent()
        let manifestURL = TextLoRATrainingManifest.url(nextTo: outputURL)
        let runManifest = LoRATrainingRunManifest(
            format: (try? resolvedTrainingFamily())?.manifestFormat
                ?? TextLoRATrainingManifest.format,
            model: model,
            isEdit: false,
            dataRoot: dataURL.path,
            dataRootRelative: LoRATrainingRunManifest.relativePath(from: outputDirectory, to: dataURL.path),
            dataFingerprint: nil,
            checkpointFiles: [
                "lora_adapter": outputURL.lastPathComponent,
                "manifest": manifestURL.lastPathComponent,
                "loss_csv": outputURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("csv").lastPathComponent,
                "loss_html": outputURL.deletingPathExtension().appendingPathExtension("loss").appendingPathExtension("html").lastPathComponent,
            ],
            step: step,
            totalSteps: trainingSteps,
            seed: seed,
            rngState: nil,
            datasetFingerprint: datasetSummary.fingerprint,
            configFingerprint: LoRATrainingFingerprint.sha256Hex([
                model,
                modelPath ?? "",
                adapterName,
                String(trainingSteps),
                String(batchSize),
                String(learningRate),
                String(rank),
                String(alpha ?? Float(rank)),
                String(maxSequenceLength),
                String(reasoningEffort),
                resolvedTargetModules().joined(separator: ","),
            ].joined(separator: "\n")),
            configSnapshot: [
                "adapter_name": adapterName,
                "batch_size": String(batchSize),
                "dataset_examples": String(datasetSummary.exampleCount),
                "eval_prompt_count": evalPromptCount.map(String.init) ?? "0",
                "learning_rate": String(learningRate),
                "lora_alpha": String(alpha ?? Float(rank)),
                "lora_rank": String(rank),
                "max_sequence_length": String(maxSequenceLength),
                "reasoning_effort": String(reasoningEffort),
                "status": status,
                "target_modules": resolvedTargetModules().joined(separator: ","),
                "training_steps": String(trainingSteps),
            ]
        )
        try runManifest.write(nextTo: outputURL)
    }

    private func makeChatProgressHandler(
        eventLogger: LoRATrainingEventLogger?
    ) -> (@Sendable (ChatProgress) -> Void)? {
        guard !json || eventLogger != nil else { return nil }
        return { progress in
            if !json, let message = progress.message, !message.isEmpty {
                FileHandle.standardError.write(Data("[\(progress.stage.rawValue)] \(message)\n".utf8))
            }
            guard let eventLogger else { return }
            try? eventLogger.record(
                type: "progress",
                stage: progress.stage.rawValue,
                message: progress.message
            )
        }
    }

    private func makeTrainingProgressHandler(
        eventLogger: LoRATrainingEventLogger?
    ) -> (@Sendable (TextLoRATrainingProgress) -> Void)? {
        guard !json || eventLogger != nil else { return nil }
        return { progress in
            switch progress.stage {
            case .training(let step, let total, let loss):
                if !json {
                    CLIStderr.write(String(format: "\rTraining (%d/%d) loss %.6f\n", step, total, loss))
                }
                try? eventLogger?.record(
                    type: "progress",
                    stage: "training",
                    message: "Training.",
                    step: step,
                    totalSteps: total,
                    loss: loss,
                    fraction: progress.fraction
                )
            case .saving:
                if !json {
                    CLIStderr.write("Saving text LoRA artifacts...\n")
                }
                try? eventLogger?.record(
                    type: "progress",
                    stage: "saving",
                    message: "Saving text LoRA artifacts.",
                    fraction: progress.fraction
                )
            }
        }
    }
}

struct TextLoRATrainingVisualization {
    let logger: LoRATrainingEventLogger
    let serverTask: Task<Void, Never>

    func stop() async {
        serverTask.cancel()
        await serverTask.value
    }
}

struct TextTrainLoRAResult: Encodable {
    let model: String
    let adapterName: String
    let outputPath: String
    let manifestPath: String
    let dataset: TextSFTDatasetSummary
    let evalPromptCount: Int?
    let dryRun: Bool
    let status: String
    let trainingReport: TextLoRATrainingReport?

    enum CodingKeys: String, CodingKey {
        case model
        case adapterName = "adapter_name"
        case outputPath = "output_path"
        case manifestPath = "manifest_path"
        case dataset
        case evalPromptCount = "eval_prompt_count"
        case dryRun = "dry_run"
        case status
        case trainingReport = "training_report"
    }

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
}
