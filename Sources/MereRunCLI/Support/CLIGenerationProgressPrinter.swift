import ArgumentParser
import Foundation
import MereRunCore

enum CLIGenerationProgressPrinter {
    static let flagName = "progress-json"
    static let flagHelpText = "Stream progress to stderr as JSON lines (one object per event) instead of human-readable text. Takes precedence over --quiet for progress output."
    static let flagHelp = ArgumentHelp(flagHelpText)

    /// One NDJSON object per distinct progress event, e.g.
    /// `{"event":"progress","stage":"denoising","step":2,"total_steps":4}`.
    /// `step` is the generator's raw 0-based step index; stages emit a final
    /// event with `step == total_steps` when they finish.
    static func progressJSONLine(_ progress: GenerationProgress) -> String {
        progressJSONLine(stage: progress.stage.rawValue, step: progress.stepIndex, totalSteps: progress.totalSteps)
    }

    /// The same event shape for pipelines that report progress with their own
    /// types. `totalSteps == 0` marks an indeterminate stage (for example token
    /// streaming with no known length); readers should show a spinner for it.
    static func progressJSONLine(stage: String, step: Int, totalSteps: Int) -> String {
        "{\"event\":\"progress\",\"stage\":\"\(stage)\",\"step\":\(step),\"total_steps\":\(totalSteps)}"
    }

    /// Writes one progress line to stderr. Stage names are `rawValue`-style
    /// identifiers without quotes or newlines, so no escaping is needed.
    static func writeJSONProgress(stage: String, step: Int, totalSteps: Int) {
        CLIStderr.write(progressJSONLine(stage: stage, step: step, totalSteps: totalSteps) + "\n")
    }

    /// Machine-readable progress for wrappers (`--progress-json`): one JSON
    /// line per event on stderr, so stdout stays reserved for the output path.
    static func makeJSONProgressHandler(
        write: @escaping @Sendable (String) -> Void = { CLIStderr.write($0) }
    ) -> (@Sendable (GenerationProgress) -> Void) {
        final class State: @unchecked Sendable {
            var last: GenerationProgress?
        }

        let state = State()
        return { progress in
            guard progress != state.last else { return }
            state.last = progress
            write(progressJSONLine(progress) + "\n")
        }
    }

    static func makeProgressHandler() -> (@Sendable (GenerationProgress) -> Void) {
        final class State: @unchecked Sendable {
            var lastStage: GenerationStage?
            var lastStep: Int = -1
        }

        let state = State()
        return { progress in
            let stage = progress.stage
            if stage != state.lastStage {
                if state.lastStage == .denoising || state.lastStage == .loadingModel {
                    CLIStderr.write("\n")
                }
                state.lastStage = stage
                state.lastStep = -1
            }

            switch stage {
            case .loadingModel:
                if progress.totalSteps > 1 {
                    CLIStderr.write("\rDownloading model (\(progress.stepIndex)/\(progress.totalSteps))")
                } else {
                    CLIStderr.write("Loading model…\n")
                }
            case .loadingEncoder:
                if progress.totalSteps > 1 {
                    CLIStderr.write("\rLoading encoder (\(progress.stepIndex)/\(progress.totalSteps))")
                } else {
                    CLIStderr.write("Loading encoder…\n")
                }
            case .encodingText:
                if progress.totalSteps > 1 {
                    CLIStderr.write("\rLoading text encoder (\(progress.stepIndex)/\(progress.totalSteps))")
                } else {
                    CLIStderr.write("Encoding prompt…\n")
                }
            case .encodingReferenceImages:
                if progress.totalSteps > 0 {
                    CLIStderr.write("\rEncoding reference images (\(progress.stepIndex)/\(progress.totalSteps))")
                    if progress.stepIndex >= progress.totalSteps {
                        CLIStderr.write("\n")
                    }
                } else {
                    CLIStderr.write("Encoding reference images…\n")
                }
            case .loadingTransformer:
                if progress.totalSteps > 1 {
                    CLIStderr.write("\rLoading transformer (\(progress.stepIndex)/\(progress.totalSteps))")
                } else {
                    CLIStderr.write("Loading transformer…\n")
                }
            case .loadingVAE:
                CLIStderr.write("Loading VAE…\n")
            case .loadingLoRA:
                if progress.totalSteps > 1 {
                    CLIStderr.write("\rLoading LoRA (\(progress.stepIndex)/\(progress.totalSteps))")
                } else {
                    CLIStderr.write("Loading LoRA…\n")
                }
            case .denoising:
                guard progress.stepIndex != state.lastStep else { return }
                state.lastStep = progress.stepIndex
                let current = min(progress.stepIndex + 1, progress.totalSteps)
                CLIStderr.write("\rGenerating (\(current)/\(progress.totalSteps))")
                if current == progress.totalSteps {
                    CLIStderr.write("\n")
                }
            case .decoding:
                CLIStderr.write("Decoding…\n")
            case .saving:
                CLIStderr.write("Saving…\n")
            }
        }
    }
}

