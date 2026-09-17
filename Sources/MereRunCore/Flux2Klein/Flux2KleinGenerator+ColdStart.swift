import Foundation
import MereRunImageModels
import MLX

extension Flux2KleinGenerator {
    /// Forces staged first-step evaluation on (`1`) or off (`0`). When unset,
    /// staging is enabled on Linux GPU backends and disabled elsewhere.
    static let stagedFirstStepEnvironmentKey = "MERERUN_FLUX2_STAGED_FIRST_STEP"

    /// Forward stages slower than this are always reported on stderr. A
    /// multi-second block on a 256x256 request means the backend is compiling
    /// kernels or building attention plans rather than running the model.
    static let slowForwardStageThreshold: TimeInterval = 1.0

    static var runsOnLinuxGPU: Bool {
        #if os(Linux)
        return Device.defaultDevice().deviceType == .gpu
        #else
        return false
        #endif
    }

    /// Whether the first denoising step evaluates the transformer one stage at
    /// a time instead of as one lazy graph.
    ///
    /// Linux CUDA compiles quantized matmul and fused elementwise kernels with
    /// NVRTC and builds cuDNN attention plans the first time each shape runs.
    /// Evaluating stage by stage bounds each graph the backend has to compile
    /// and lets the runtime report where a cold start spends its time. Under
    /// `MERERUN_FLUX2_COMPILE` the transformer is traced as one function, so
    /// staging is never used there.
    static func stagedFirstStepEvaluationEnabled(
        environment: [String: String],
        runsOnLinuxGPU: Bool,
        compileEnabled: Bool
    ) -> Bool {
        guard !compileEnabled else { return false }
        if let raw = environment[stagedFirstStepEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased(),
           !raw.isEmpty {
            switch raw {
            case "1", "true", "yes", "on":
                return true
            case "0", "false", "no", "off":
                return false
            default:
                break
            }
        }
        return runsOnLinuxGPU
    }

    /// Builds the handler installed on the transformer during the first
    /// denoising step. It materializes each stage's arrays and reports how
    /// long the stage took. Every stage is reported when `reportEveryStage`
    /// is set; otherwise only stages slower than
    /// ``slowForwardStageThreshold`` are.
    static func makeStagedEvaluationHandler(
        reportEveryStage: Bool,
        clock: @escaping () -> TimeInterval = { CFAbsoluteTimeGetCurrent() },
        report: @escaping (String) -> Void = { line in
            FileHandle.standardError.write(Data((line + "\n").utf8))
        }
    ) -> (Flux2TransformerForwardStage, [MLXArray]) -> Void {
        let start = clock()
        var previous = start
        return { stage, arrays in
            MLX.eval(arrays)
            let now = clock()
            let elapsed = now - previous
            previous = now
            guard reportEveryStage || elapsed >= slowForwardStageThreshold else { return }
            report(Self.stageReportLine(stage: stage, stageSeconds: elapsed, elapsedSeconds: now - start))
        }
    }

    static func stageReportLine(
        stage: Flux2TransformerForwardStage,
        stageSeconds: TimeInterval,
        elapsedSeconds: TimeInterval
    ) -> String {
        let stageText = String(format: "%.3f", stageSeconds)
        let elapsedText = String(format: "%.3f", elapsedSeconds)
        return "[Flux2KleinGenerator] first_step_stage=\(stage.label) stage_s=\(stageText) elapsed_s=\(elapsedText)"
    }

    static func firstStepReportLine(seconds: TimeInterval) -> String {
        let text = String(format: "%.3f", seconds)
        return "[Flux2KleinGenerator] first_step_s=\(text) staged=1"
    }
}
