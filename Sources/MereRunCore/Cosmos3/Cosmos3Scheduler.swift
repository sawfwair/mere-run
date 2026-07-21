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
