import Foundation

public struct ACEStepContinuousScheduler: Sendable, Hashable {
    public let timesteps: [Float]

    public init(
        inferenceSteps: Int,
        shift: Float = 1,
        timesteps: [Float]? = nil
    ) {
        precondition(inferenceSteps > 0, "inferenceSteps must be positive.")
        if var timesteps {
            while timesteps.last == 0 {
                timesteps.removeLast()
            }
            // Caller-supplied timesteps are used in full. The turbo schedule maps custom
            // timesteps onto its 20-entry reference table; the continuous schedule is not
            // quantized that way, so there is nothing here to cap at 20.
            self.timesteps = timesteps.isEmpty
                ? Self.makeTimesteps(inferenceSteps: inferenceSteps, shift: shift)
                : timesteps
        } else {
            self.timesteps = Self.makeTimesteps(
                inferenceSteps: inferenceSteps,
                shift: shift
            )
        }
    }

    public static func makeTimesteps(
        inferenceSteps: Int,
        shift: Float
    ) -> [Float] {
        precondition(inferenceSteps > 0, "inferenceSteps must be positive.")
        return (0..<inferenceSteps).map { index in
            let timestep = Float(1 - Double(index) / Double(inferenceSteps))
            guard shift != 1 else {
                return timestep
            }
            return shift * timestep / (1 + (shift - 1) * timestep)
        }
    }
}
