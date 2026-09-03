import Foundation
import MLX

public enum Cosmos3UniPCSchedule: String, Codable, CaseIterable, Sendable {
    /// NVIDIA cosmos-framework's tested resolution-dependent shifted flow grid.
    case nvidiaShiftedFlow = "nvidia_shifted_flow"
    /// The Karras flow grid published in the Hugging Face scheduler config.
    case publishedKarrasFlow = "published_karras_flow"
}

public struct Cosmos3UniPCScheduler {
    public let schedule: Cosmos3UniPCSchedule
    public let timesteps: [Float]
    public let sigmas: [Float]
    private var solver: Wan2UniPCScheduler

    public init(
        steps: Int,
        shift: Float = 1,
        schedule: Cosmos3UniPCSchedule = .nvidiaShiftedFlow,
        trainTimesteps: Int = 1_000,
        karrasSigmaMinimum: Double = 0.147,
        karrasSigmaMaximum: Double = 200
    ) {
        precondition(steps > 0)
        precondition(shift > 0)
        precondition(trainTimesteps > 1)
        precondition(karrasSigmaMinimum > 0)
        precondition(karrasSigmaMaximum > karrasSigmaMinimum)
        self.schedule = schedule
        switch schedule {
        case .nvidiaShiftedFlow:
            let native = Wan2UniPCScheduler(
                steps: steps,
                shift: shift,
                trainTimesteps: trainTimesteps
            )
            self.timesteps = native.timesteps
            self.sigmas = native.sigmas
            self.solver = native
        case .publishedKarrasFlow:
            let rho = 7.0
            let minimum = pow(karrasSigmaMinimum, 1 / rho)
            let maximum = pow(karrasSigmaMaximum, 1 / rho)
            var flowSigmas: [Float] = []
            flowSigmas.reserveCapacity(steps + 1)
            var scheduledTimesteps: [Float] = []
            scheduledTimesteps.reserveCapacity(steps)
            for index in 0..<steps {
                let ramp = steps == 1 ? 0 : Double(index) / Double(steps - 1)
                let rawSigma = pow(maximum + ramp * (minimum - maximum), rho)
                let flowSigma = rawSigma / (rawSigma + 1)
                flowSigmas.append(Float(flowSigma))
                scheduledTimesteps.append(Float(Int(flowSigma * Double(trainTimesteps))))
            }
            flowSigmas.append(0)
            self.timesteps = scheduledTimesteps
            self.sigmas = flowSigmas
            self.solver = Wan2UniPCScheduler(
                timesteps: scheduledTimesteps,
                sigmas: flowSigmas
            )
        }
    }

    public mutating func step(modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
        solver.step(modelOutput: modelOutput, sample: sample)
    }
}

/// The stochastic four-step Euler sampler published with the distilled
/// Cosmos3-Super text-to-image checkpoint.
public struct Cosmos3DistilledEulerScheduler {
    public let timesteps: [Float]
    public let sigmas: [Float]
    private var stepIndex = 0

    public init(sigmas: [Float], trainTimesteps: Int = 1_000) {
        precondition(!sigmas.isEmpty)
        precondition(trainTimesteps > 0)
        precondition(sigmas.allSatisfy { $0 > 0 && $0 <= 1 })
        precondition(zip(sigmas, sigmas.dropFirst()).allSatisfy { pair in pair.0 > pair.1 })
        self.timesteps = sigmas.map { $0 * Float(trainTimesteps) }
        self.sigmas = sigmas + [0]
    }

    public mutating func step(
        modelOutput: MLXArray,
        sample: MLXArray,
        noise: MLXArray
    ) -> MLXArray {
        precondition(stepIndex < timesteps.count)
        precondition(noise.shape == sample.shape)
        let sigma = sigmas[stepIndex]
        let nextSigma = sigmas[stepIndex + 1]
        stepIndex += 1
        let prediction = sample.asType(.float32) - sigma * modelOutput.asType(.float32)
        let next = (1 - nextSigma) * prediction + nextSigma * noise.asType(.float32)
        return next.asType(modelOutput.dtype)
    }
}
