import Foundation
import MereRunCore

enum CLIGenerationProgressPrinter {
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

