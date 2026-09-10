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

    func trainingOptions() -> ACEStepAdapterTrainingOptions {
        .init(
            dataset: dataset, output: output, model: model,
            configuration: .init(
                kind: kind, rank: rank, alpha: alpha, factor: factor,
                trainingSteps: steps, learningRate: learningRate,
                weightDecay: weightDecay, seed: seed
            ),
            maxDurationSeconds: maxDurationSeconds, checkpointsRoot: checkpointsRoot,
            decoderSubdirectory: decoderSubdirectory, vaeSubdirectory: vaeSubdirectory,
            textSubdirectory: textSubdirectory
        )
    }

    func run() async throws {
        do {
            let options = trainingOptions()
            try options.validate()
            guard logEvery > 0 else {
                throw ValidationError("--log-every must be greater than zero.")
            }
            let plan = try ACEStepAdapterTrainingPlan.resolve(options)
            CLIStderr.write(
                "Loading ACE-Step for \(kind.rawValue) training with "
                    + "\(plan.records.count) example(s)\n"
            )
            let report = try await ACEStepAdapterTrainingOperation.execute(plan) { progress in
                if progress.step == 1
                    || progress.step == progress.totalSteps
                    || progress.step.isMultiple(of: logEvery)
                {
                    CLIStderr.write(String(
                        format: "ACE-Step adapter step %d/%d loss=%.6f\n",
                        progress.step, progress.totalSteps, progress.loss
                    ))
                }
            }
            CLIStderr.write(String(
                format: "Saved %d-layer %@ adapter; loss %.6f -> %.6f; sha256=%@\n",
                report.layerCount, report.kind.rawValue, report.initialLoss,
                report.finalLoss, report.outputSHA256
            ))
            print(plan.outputURL.path)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }

    typealias ManifestRecord = ACEStepAdapterTrainingPlan.ManifestRecord

    static func loadManifest(from url: URL) throws -> [ManifestRecord] {
        do {
            return try ACEStepAdapterTrainingPlan.loadManifest(from: url)
        } catch let error as ACEStepPreparationIssue {
            throw ValidationError(error.message)
        }
    }
}
