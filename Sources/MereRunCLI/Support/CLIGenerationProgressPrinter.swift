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
}

/// Normalizes a pipeline's ad-hoc progress callbacks onto the documented
/// `--progress-json` convention before they reach stderr:
///
/// - `step` is 0-based while a stage is in progress;
/// - every determinate stage (`total_steps > 0`) ends with exactly one event
///   whose `step == total_steps`, written when the next stage begins, when the
///   same stage restarts (sliding windows), or by `finish()` at the end of the
///   run at the latest;
/// - indeterminate stages (`total_steps == 0`) carry no terminal event.
///
/// Milestone stages recorded with `mark` (for example MiniMax-H3 sliding
/// windows) interleave with the ordinary stage sequence, so they only close
/// in `finish()`.
final class JSONProgressStream: @unchecked Sendable {
    private let write: @Sendable (String) -> Void
    private let lock = NSLock()
    private var currentStage: String?
    private var currentTotal = 0
    private var lastStep: Int?
    private var milestoneOrder: [String] = []
    private var milestones: [String: (step: Int, total: Int)] = [:]

    init(write: @escaping @Sendable (String) -> Void = { CLIStderr.write($0) }) {
        self.write = write
    }

    /// Reports progress for a sequential stage. `step` is the 0-based index of
    /// the step that is in progress (or just finished when the pipeline only
    /// counts completions; pass `completed - 1` in that case).
    func report(stage: String, step: Int, totalSteps: Int) {
        lock.lock()
        defer { lock.unlock() }
        if stage != currentStage || totalSteps != currentTotal {
            closeCurrentStage()
            currentStage = stage
            currentTotal = totalSteps
            lastStep = nil
        } else if let lastStep, step < lastStep {
            // The stage restarted (for example the next sliding window), so
            // the previous cycle is complete.
            closeCurrentStage()
            self.lastStep = nil
        }
        guard step != lastStep else { return }
        lastStep = step
        emit(stage: stage, step: step, totalSteps: totalSteps)
    }

    /// Records a milestone stage that runs alongside the ordinary stages.
    func mark(stage: String, step: Int, totalSteps: Int) {
        lock.lock()
        defer { lock.unlock() }
        if let existing = milestones[stage], existing.step == step, existing.total == totalSteps { return }
        if milestones[stage] == nil { milestoneOrder.append(stage) }
        milestones[stage] = (step, totalSteps)
        emit(stage: stage, step: step, totalSteps: totalSteps)
    }

    /// Closes every stage that has not reached its total. Call once after the
    /// pipeline returns.
    func finish() {
        lock.lock()
        defer { lock.unlock() }
        closeCurrentStage()
        currentStage = nil
        lastStep = nil
        for stage in milestoneOrder {
            guard let milestone = milestones[stage], milestone.total > 0, milestone.step != milestone.total else { continue }
            emit(stage: stage, step: milestone.total, totalSteps: milestone.total)
        }
        milestoneOrder = []
        milestones = [:]
    }

    private func closeCurrentStage() {
        guard let stage = currentStage, currentTotal > 0, lastStep != currentTotal else { return }
        lastStep = currentTotal
        emit(stage: stage, step: currentTotal, totalSteps: currentTotal)
    }

    private func emit(stage: String, step: Int, totalSteps: Int) {
        write(CLIGenerationProgressPrinter.progressJSONLine(stage: stage, step: step, totalSteps: totalSteps) + "\n")
    }
}

extension CLIGenerationProgressPrinter {
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

