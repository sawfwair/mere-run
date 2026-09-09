import Foundation
import MLX

/// Opt-in TeaCache controls for full LTX 2.5 generation.
public struct LTXTeaCacheConfiguration: Sendable, Hashable {
    /// Overrides the calibrated threshold for the selected sampler when non-nil.
    public let threshold: Float?
    /// Records full-compute input/output drift samples instead of skipping blocks.
    public let calibrationOutputURL: URL?

    public init(
        threshold: Float? = nil,
        calibrationOutputURL: URL? = nil
    ) {
        self.threshold = threshold
        self.calibrationOutputURL = calibrationOutputURL?.standardizedFileURL
    }
}

package struct LTXTeaCacheCalibration: Sendable, Hashable {
    package let coefficients: [Float]
    package let threshold: Float

    package static func ltx25(
        sampler: LTXSamplerMode,
        key: LTXTeaCacheKey
    ) -> Self {
        precondition(key.pipelineStage == .coarse)
        return switch (sampler, key.stage) {
        // Quadratic fits use five corrected native LTX 2.5 trajectories per
        // sampler and the maximum observed drift across the synchronized
        // conditioned, unconditional, perturbed, and isolated guidance group.
        // Defaults target approximately 15% Euler and 10% Res2S group reuse.
        case (.euler, .primary):
            Self(coefficients: [39.697662, -5.110012, 0.3960171], threshold: 0.235)
        case (.res2s, .primary):
            Self(coefficients: [0.6765311, 0.2155268, 0.24286742], threshold: 0.39)
        case (.res2s, .midpoint):
            Self(coefficients: [0.90598094, 0.008077672, 0.28097582], threshold: 0.39)
        case (.eulerAncestral, _), (.gradientEstimatingEuler, _),
             (.cfgPlusPlus, _), (.euler, .midpoint):
            preconditionFailure("TeaCache calibration is unavailable for this sampler stage.")
        }
    }
}

package enum LTXTeaCacheBranch: String, Codable, Sendable, Hashable {
    case conditioned
    case unconditional
    case perturbed
    case isolated
}

package enum LTXTeaCacheStage: String, Codable, Sendable, Hashable {
    case primary
    case midpoint
}

package enum LTXTeaCachePipelineStage: String, Codable, Sendable, Hashable {
    case coarse
    case detail
}

package struct LTXTeaCacheKey: Sendable, Hashable {
    package let branch: LTXTeaCacheBranch
    package let stage: LTXTeaCacheStage
    package let pipelineStage: LTXTeaCachePipelineStage

    package init(branch: LTXTeaCacheBranch, stage: LTXTeaCacheStage, pipelineStage: LTXTeaCachePipelineStage) {
        self.branch = branch
        self.stage = stage
        self.pipelineStage = pipelineStage
    }
}

package struct LTXTeaCacheRequest {
    package let key: LTXTeaCacheKey
    package let stepIndex: Int
    package let stepCount: Int

    package init(key: LTXTeaCacheKey, stepIndex: Int, stepCount: Int) {
        self.key = key
        self.stepIndex = stepIndex
        self.stepCount = stepCount
    }
}

package enum LTXTeaCacheDecision {
    case compute
    case reuse(videoResidual: MLXArray, audioResidual: MLXArray)
}

package struct LTXTeaCacheCalibrationSample: Codable, Sendable, Hashable {
    package let branch: LTXTeaCacheBranch
    package let stage: LTXTeaCacheStage
    package let pipelineStage: LTXTeaCachePipelineStage
    package let stepIndex: Int
    package let inputRelativeL1: Float
    package let outputRelativeL1: Float
}

package struct LTXTeaCacheCalibrationReport: Codable, Sendable, Hashable {
    package let schemaVersion: Int
    package let sampler: String
    package let samples: [LTXTeaCacheCalibrationSample]
}

package final class LTXTeaCacheMetrics {
    package var decisionSeconds = 0.0
    package var computedBlockStacks = 0
    package var reusedBlockStacks = 0
}

private struct LTXTeaCacheState {
    var previousGate: MLXArray?
    var accumulatedDistance: Float = 0
    var videoResidual: MLXArray?
    var audioResidual: MLXArray?
    var previousVideoResidual: MLXArray?
    var pendingInputRelativeL1: Float?
    var pendingStepIndex: Int?
}

private struct LTXTeaCacheGuidanceGroupKey: Sendable, Hashable {
    let stage: LTXTeaCacheStage
    let pipelineStage: LTXTeaCachePipelineStage
}

private struct LTXTeaCacheGuidanceGroupDecision {
    let stepIndex: Int
    let reusesResiduals: Bool
}

package final class LTXTeaCacheController {
    private let configuration: LTXTeaCacheConfiguration
    private let sampler: LTXSamplerMode
    package private(set) var metrics = LTXTeaCacheMetrics()
    private var states: [LTXTeaCacheKey: LTXTeaCacheState] = [:]
    private var guidanceGroupDecisions: [
        LTXTeaCacheGuidanceGroupKey: LTXTeaCacheGuidanceGroupDecision
    ] = [:]
    private(set) var calibrationSamples: [LTXTeaCacheCalibrationSample] = []

    package init(configuration: LTXTeaCacheConfiguration, sampler: LTXSamplerMode) {
        self.configuration = configuration
        self.sampler = sampler
    }

    package var isCalibrating: Bool {
        configuration.calibrationOutputURL != nil
    }

    package func decide(
        request: LTXTeaCacheRequest,
        gate: MLXArray
    ) -> LTXTeaCacheDecision {
        let started = ltxMonotonicSeconds()
        defer { metrics.decisionSeconds += ltxMonotonicSeconds() - started }

        var state = states[request.key] ?? LTXTeaCacheState()
        let previousGate = state.previousGate
        MLX.eval(gate)
        state.previousGate = gate
        state.pendingInputRelativeL1 = nil
        state.pendingStepIndex = request.stepIndex

        if !isCalibrating, request.key.branch != .conditioned {
            let groupKey = LTXTeaCacheGuidanceGroupKey(
                stage: request.key.stage,
                pipelineStage: request.key.pipelineStage
            )
            guard let groupDecision = guidanceGroupDecisions[groupKey],
                  groupDecision.stepIndex == request.stepIndex else {
                preconditionFailure("TeaCache guidance branches must evaluate conditioned first.")
            }
            if groupDecision.reusesResiduals,
               let videoResidual = state.videoResidual,
               let audioResidual = state.audioResidual {
                states[request.key] = state
                metrics.reusedBlockStacks += 1
                return .reuse(videoResidual: videoResidual, audioResidual: audioResidual)
            }
            states[request.key] = state
            metrics.computedBlockStacks += 1
            return .compute
        }

        let isBoundary = request.stepIndex == 0
            || request.stepIndex == request.stepCount - 1
        guard !isBoundary, let previousGate else {
            state.accumulatedDistance = 0
            states[request.key] = state
            recordGuidanceGroupDecision(request: request, reusesResiduals: false)
            metrics.computedBlockStacks += 1
            return .compute
        }

        let relativeL1 = ltxTeaCacheRelativeL1(current: gate, previous: previousGate)
        state.pendingInputRelativeL1 = relativeL1
        if isCalibrating {
            states[request.key] = state
            metrics.computedBlockStacks += 1
            return .compute
        }

        let calibration = LTXTeaCacheCalibration.ltx25(sampler: sampler, key: request.key)
        state.accumulatedDistance += polynomial(relativeL1, coefficients: calibration.coefficients)
        let threshold = configuration.threshold ?? calibration.threshold
        if state.accumulatedDistance < threshold,
           let videoResidual = state.videoResidual,
           let audioResidual = state.audioResidual {
            states[request.key] = state
            recordGuidanceGroupDecision(request: request, reusesResiduals: true)
            metrics.reusedBlockStacks += 1
            return .reuse(videoResidual: videoResidual, audioResidual: audioResidual)
        }

        state.accumulatedDistance = 0
        states[request.key] = state
        recordGuidanceGroupDecision(request: request, reusesResiduals: false)
        metrics.computedBlockStacks += 1
        return .compute
    }

    package func recordComputedResidual(
        request: LTXTeaCacheRequest,
        videoResidual: MLXArray,
        audioResidual: MLXArray
    ) {
        var state = states[request.key] ?? LTXTeaCacheState()
        MLX.eval(videoResidual, audioResidual)
        if isCalibrating,
           let inputRelativeL1 = state.pendingInputRelativeL1,
           let previousVideoResidual = state.previousVideoResidual,
           let stepIndex = state.pendingStepIndex {
            calibrationSamples.append(
                LTXTeaCacheCalibrationSample(
                    branch: request.key.branch,
                    stage: request.key.stage,
                    pipelineStage: request.key.pipelineStage,
                    stepIndex: stepIndex,
                    inputRelativeL1: inputRelativeL1,
                    outputRelativeL1: ltxTeaCacheRelativeL1(
                        current: videoResidual,
                        previous: previousVideoResidual
                    )
                )
            )
        }
        state.videoResidual = videoResidual
        state.audioResidual = audioResidual
        state.previousVideoResidual = videoResidual
        states[request.key] = state
    }

    package func writeCalibrationReport() throws {
        guard let url = configuration.calibrationOutputURL else { return }
        let report = LTXTeaCacheCalibrationReport(
            schemaVersion: 1,
            sampler: sampler.rawValue,
            samples: calibrationSamples
        )
        let data = try JSONEncoder.ltxTeaCacheEncoder.encode(report)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }

    private func polynomial(_ value: Float, coefficients: [Float]) -> Float {
        coefficients.reduce(Float.zero) { partial, coefficient in
            partial * value + coefficient
        }
    }

    private func recordGuidanceGroupDecision(
        request: LTXTeaCacheRequest,
        reusesResiduals: Bool
    ) {
        guard !isCalibrating, request.key.branch == .conditioned else { return }
        let key = LTXTeaCacheGuidanceGroupKey(
            stage: request.key.stage,
            pipelineStage: request.key.pipelineStage
        )
        guidanceGroupDecisions[key] = LTXTeaCacheGuidanceGroupDecision(
            stepIndex: request.stepIndex,
            reusesResiduals: reusesResiduals
        )
    }
}

private func ltxTeaCacheRelativeL1(current: MLXArray, previous: MLXArray) -> Float {
    let current32 = current.asType(.float32)
    let previous32 = previous.asType(.float32)
    let numerator = MLX.mean(MLX.abs(current32 - previous32))
    let denominator = MLX.maximum(
        MLX.mean(MLX.abs(previous32)),
        MLXArray(Float(1e-12))
    )
    return (numerator / denominator).item(Float.self)
}

private extension JSONEncoder {
    static var ltxTeaCacheEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}
