import ArgumentParser
import Foundation
import MereRunCore

struct MusicTrainAdapter: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "train-adapter",
        abstract: "Train a native ACE-Step LoRA or LoKr adapter."
    )

    @Option(
        name: [.customShort("m"), .long],
        help: "Managed ACE-Step model id or local checkpoint root."
    )
    var model: String = ModelResolver.ModelID.aceStep.rawValue

    @Option(name: [.long], help: "JSON or JSONL dataset manifest.")
    var dataset: String

    @Option(
        name: [.customShort("o"), .long],
        help: "Output .safetensors adapter path."
    )
    var output: String

    @Option(name: [.customLong("kind")], help: "Adapter kind: lora or lokr.")
    var kind: ACEStepAdapterKind = .lora

    @Option(name: [.long], help: "Adapter rank.")
    var rank: Int = 8

    @Option(name: [.long], help: "Adapter alpha.")
    var alpha: Float = 16

    @Option(
        name: [.long],
        help: "LoKr factorization target; -1 chooses the closest balanced factors."
    )
    var factor: Int = -1

    @Option(name: [.long], help: "Optimizer steps.")
    var steps: Int = 1_000

    @Option(name: [.customLong("learning-rate")], help: "AdamW learning rate.")
    var learningRate: Float = 1e-4

    @Option(name: [.customLong("weight-decay")], help: "AdamW weight decay.")
    var weightDecay: Float = 1e-4

    @Option(name: [.long], help: "Deterministic training seed.")
    var seed: UInt64 = 42

    @Option(
        name: [.customLong("max-duration")],
        help: "Crop every training example to this many seconds."
    )
    var maxDurationSeconds: Float = 30

    @Option(name: [.customLong("checkpoints-root")], help: "ACE-Step checkpoint root.")
    var checkpointsRoot: String?

    @Option(name: [.customLong("decoder-subdirectory")], help: "Decoder subdirectory.")
    var decoderSubdirectory: String = "acestep-v15-turbo"

    @Option(name: [.customLong("vae-subdirectory")], help: "VAE subdirectory.")
    var vaeSubdirectory: String = "vae"

    @Option(name: [.customLong("text-subdirectory")], help: "Text encoder subdirectory.")
    var textSubdirectory: String?

    @Option(name: [.customLong("log-every")], help: "Progress interval.")
    var logEvery: Int = 10

    func run() async throws {
        guard kind == .lora || kind == .lokr else {
            throw ValidationError("--kind must be lora or lokr.")
        }
        guard maxDurationSeconds > 0, maxDurationSeconds <= 600 else {
            throw ValidationError("--max-duration must be in (0, 600].")
        }
        guard logEvery > 0 else {
            throw ValidationError("--log-every must be greater than zero.")
        }

        let manifestURL = ACEStepCLIHelper.resolveUserPath(dataset)
        let records = try Self.loadManifest(from: manifestURL)
        let examples = try records.enumerated().map { index, record in
            let audioURL = Self.resolveAudioURL(
                record.audio,
                relativeTo: manifestURL.deletingLastPathComponent()
            )
            let decoded = try ACEStepCLIHelper.loadAudio48kHz(
                audioURL.path,
                label: "Dataset audio \(index + 1)"
            )
            let maxFrames = max(
                1,
                Int((maxDurationSeconds * 48_000).rounded())
            )
            let audio = decoded.dim(1) > maxFrames
                ? decoded[0..., 0..<maxFrames, 0...]
                : decoded
            return ACEStepAdapterTrainingExample(
                audio48kHz: audio,
                caption: record.caption,
                lyrics: record.lyrics ?? ""
            )
        }

        let root = try await ACEStepCLIHelper.resolveCheckpointsRoot(
            model: model,
            checkpointsRoot: checkpointsRoot,
            turboSubdirectory: decoderSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: nil,
            textSubdirectory: textSubdirectory
        )
        let decoder = try ACEStepCLIHelper.resolveTurboSubdirectory(
            at: root,
            explicit: decoderSubdirectory
        )
        guard let text = try ACEStepCLIHelper.resolveTextSubdirectory(
            at: root,
            explicit: textSubdirectory
        ) else {
            throw ValidationError("ACE-Step text encoder not found.")
        }
        let container = ACEStepModelContainer(
            checkpointsRootURL: root,
            turboSubdirectory: decoder,
            vaeSubdirectory: vaeSubdirectory,
            textEncoderSubdirectory: text
        )
        CLIStderr.write(
            "Loading ACE-Step for \(kind.rawValue) training with "
                + "\(examples.count) example(s)\n"
        )
        let resources = try await container.resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            textEncoderResources: resources.textEncoderResources
        )
        let outputURL = ACEStepCLIHelper.resolveUserPath(output)
        let eventLogger = try LoRATrainingEventLogger(baseOutputURL: outputURL)
        try eventLogger.record(
            type: "run_started",
            stage: "training",
            message: "ACE-Step \(kind.rawValue) training started.",
            step: 0,
            totalSteps: steps,
            fraction: 0,
            path: outputURL.path,
            metadata: ["dataset": manifestURL.path, "model": model]
        )
        let report: ACEStepAdapterTrainingReport
        do {
            report = try pipeline.trainAdapter(
                examples: examples,
                configuration: .init(
                    kind: kind,
                    rank: rank,
                    alpha: alpha,
                    factor: factor,
                    trainingSteps: steps,
                    learningRate: learningRate,
                    weightDecay: weightDecay,
                    seed: seed
                ),
                outputURL: outputURL
            ) { progress in
                try? eventLogger.record(
                    type: "progress",
                    stage: "training",
                    step: progress.step,
                    totalSteps: progress.totalSteps,
                    loss: progress.loss,
                    fraction: Float(progress.step) / Float(max(progress.totalSteps, 1)),
                    path: outputURL.path
                )
                if progress.step == 1
                    || progress.step == progress.totalSteps
                    || progress.step.isMultiple(of: logEvery)
                {
                    CLIStderr.write(
                        String(
                            format: "ACE-Step adapter step %d/%d loss=%.6f\n",
                            progress.step,
                            progress.totalSteps,
                            progress.loss
                        )
                    )
                }
            }
            try eventLogger.record(
                type: "run_finished",
                stage: "finished",
                message: "ACE-Step adapter training finished.",
                step: steps,
                totalSteps: steps,
                loss: report.finalLoss,
                fraction: 1,
                path: outputURL.path,
                metadata: ["sha256": report.outputSHA256]
            )
        } catch {
            try? eventLogger.record(
                type: "run_failed",
                stage: "failed",
                message: error.localizedDescription,
                path: outputURL.path
            )
            throw error
        }
        CLIStderr.write(
            String(
                format: "Saved %d-layer %@ adapter; loss %.6f -> %.6f; sha256=%@\n",
                report.layerCount,
                report.kind.rawValue,
                report.initialLoss,
                report.finalLoss,
                report.outputSHA256
            )
        )
        print(outputURL.path)
    }

    struct ManifestRecord: Codable, Sendable {
        var audio: String
        var caption: String
        var lyrics: String?
    }

    static func loadManifest(from url: URL) throws -> [ManifestRecord] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("Dataset manifest not found: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let records: [ManifestRecord]
        if let decoded = try? decoder.decode([ManifestRecord].self, from: data) {
            records = decoded
        } else {
            let text = String(decoding: data, as: UTF8.self)
            records = try text.split(whereSeparator: \.isNewline)
                .enumerated()
                .map { lineNumber, line in
                    do {
                        return try decoder.decode(
                            ManifestRecord.self,
                            from: Data(line.utf8)
                        )
                    } catch {
                        throw ValidationError(
                            "Invalid dataset JSONL line \(lineNumber + 1): "
                                + error.localizedDescription
                        )
                    }
                }
        }
        guard !records.isEmpty else {
            throw ValidationError("Dataset manifest contains no examples.")
        }
        for (index, record) in records.enumerated() {
            if record.audio.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty {
                throw ValidationError(
                    "Dataset record \(index + 1) has an empty audio path."
                )
            }
            if record.caption.trimmingCharacters(
                in: .whitespacesAndNewlines
            ).isEmpty {
                throw ValidationError(
                    "Dataset record \(index + 1) has an empty caption."
                )
            }
        }
        return records
    }

    private static func resolveAudioURL(
        _ path: String,
        relativeTo directory: URL
    ) -> URL {
        if path.hasPrefix("/") || path.hasPrefix("~") {
            return ACEStepCLIHelper.resolveUserPath(path)
        }
        return directory.appendingPathComponent(path).standardizedFileURL
    }
}
