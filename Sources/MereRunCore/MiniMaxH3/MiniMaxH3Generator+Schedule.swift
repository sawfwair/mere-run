import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3Generator {
    static func scheduleCoefficients(_ schedule: MiniMaxH3Schedule, index: Int) -> MLXArray {
        MLXArray([
            schedule.timesteps[index],
            schedule.sigmas[index + 1] / schedule.sigmas[index],
        ])
    }

    static func advance(
        sample: MLXArray,
        velocity: MLXArray,
        coefficients: MLXArray
    ) -> MLXArray {
        let timestep = coefficients[0]
        let ratio = coefficients[1]
        let denoised = sample + (1 - timestep) * velocity
        return (
            ratio * sample.asType(.float32)
                + (1 - ratio) * denoised.asType(.float32)
        ).asType(sample.dtype)
    }

    static func loadAdaLNCache(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) throws -> MiniMaxH3AdaLNCachePack.Selection? {
        do {
            return try MiniMaxH3AdaLNCachePack.load(
                from: resources.rootURL,
                configuration: .init(configuration),
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: resources.adaLNCacheSourceIdentity()
            )
        } catch {
            guard try !resources.requiresAdaLNCache() else {
                throw error
            }
            return nil
        }
    }

}
