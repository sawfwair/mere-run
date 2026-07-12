import MLX
import XCTest
@testable import MereRunCore

final class Wan2GenerationTests: MereRunCoreTestCase {
    func testCausalDecodeWindowStaysBoundedAfterSecondBlock() {
        XCTAssertEqual(
            Wan2CausalDecodePlan.make(previousLatentFrames: 0, currentLatentFrames: 3),
            Wan2CausalDecodePlan(latentStart: 0, latentCount: 3, transitionStartPixelFrame: 0)
        )
        XCTAssertEqual(
            Wan2CausalDecodePlan.make(previousLatentFrames: 3, currentLatentFrames: 3),
            Wan2CausalDecodePlan(latentStart: 0, latentCount: 6, transitionStartPixelFrame: 8)
        )
        XCTAssertEqual(
            Wan2CausalDecodePlan.make(previousLatentFrames: 36, currentLatentFrames: 3),
            Wan2CausalDecodePlan(latentStart: 33, latentCount: 6, transitionStartPixelFrame: 8)
        )
    }

    func testDreamXColorStabilizerPullsLaterFramesTowardReferencePalette() {
        let values: [UInt8] = [
            90, 110, 130, 100, 120, 140,
            210, 80, 30, 230, 100, 50,
        ]
        let frames = MLXArray(values).reshaped(1, 2, 1, 2, 3)
        let stabilized = Wan2DreamXColorStabilizer.process(frames, strength: 1)
        let output = stabilized.asArray(UInt8.self)
        let originalDistance = zip(values[0..<6], values[6..<12])
            .reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }
        let correctedDistance = zip(output[0..<6], output[6..<12])
            .reduce(0) { $0 + abs(Int($1.0) - Int($1.1)) }
        XCTAssertLessThan(correctedDistance, originalDistance)
    }

    func testNativeResolutionAndLatentGeometry() throws {
        let options = try makeOptions()
        XCTAssertEqual(options.latentShape, [48, 11, 44, 80])
        XCTAssertEqual(options.patchGridShape, [11, 22, 40])
        XCTAssertEqual(options.sequenceLength, 9_680)
    }

    func testRejectsInvalidFrameAndResolutionGeometry() throws {
        XCTAssertThrowsError(try makeOptions(width: 1_279))
        XCTAssertThrowsError(try makeOptions(numFrames: 40))
    }

    func testShiftedFlowScheduleMatchesWanEndpoints() {
        let scheduler = Wan2FlowMatchEulerScheduler(steps: 4, shift: 5)
        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.timesteps.count, 4)
        XCTAssertEqual(scheduler.sigmas.last, 0)
        XCTAssertEqual(scheduler.timesteps[0], 1_000, accuracy: 1e-5)
        XCTAssertEqual(scheduler.timesteps[1], 909.707_15, accuracy: 1e-4)
        XCTAssertEqual(scheduler.timesteps.last!, 24.414_066, accuracy: 1e-4)
        XCTAssertEqual(scheduler.sigmas[0], 1, accuracy: 1e-6)
    }

    func testEulerStepUsesNextMinusCurrentSigma() {
        var scheduler = Wan2FlowMatchEulerScheduler(steps: 2, shift: 1)
        let sample = MLXArray([1, 2], [1, 2])
        let velocity = MLXArray([2, 4], [1, 2])
        let result = scheduler.step(velocity: velocity, sample: sample)
        eval(result)
        let values = result.asArray(Float.self)
        XCTAssertEqual(values.count, 2)
        XCTAssertLessThan(abs(values[0] + 0.998), 1e-5)
        XCTAssertLessThan(abs(values[1] + 1.996), 1e-5)
    }

    func testUniPCScheduleAndFirstOrderStepAreFinite() {
        var scheduler = Wan2UniPCScheduler(steps: 4, shift: 5)
        XCTAssertEqual(scheduler.timesteps.count, 4)
        XCTAssertEqual(scheduler.sigmas.count, 5)
        XCTAssertEqual(scheduler.sigmas.last, 0)
        let sample = MLXArray([Float(1), 2], [1, 2])
        let output = MLXArray([Float(0.25), -0.5], [1, 2])
        var result = sample
        for _ in scheduler.timesteps {
            result = scheduler.step(modelOutput: output, sample: result)
            eval(result)
            XCTAssertTrue(result.asArray(Float.self).allSatisfy(\.isFinite))
        }
    }

    func testCausalForcingScheduleMatchesDreamXWarpedSteps() {
        let scheduler = Wan2CausalForcingScheduler(shift: 5)
        XCTAssertEqual(scheduler.timesteps[0], 1_000, accuracy: 1e-5)
        XCTAssertEqual(scheduler.timesteps[1], 937.5, accuracy: 1e-5)
        XCTAssertEqual(scheduler.timesteps[2], 833.333_3, accuracy: 1e-4)
        XCTAssertEqual(scheduler.timesteps[3], 625, accuracy: 1e-5)
        let sample = MLXArray([Float(1), 2], [1, 2])
        let flow = MLXArray([Float(0.25), -0.5], [1, 2])
        let clean = scheduler.predictClean(flow: flow, sample: sample, timestep: 1_000)
        let noised = scheduler.addNoise(
            clean: clean,
            noise: MLX.zeros([1, 2]),
            timestep: 625
        )
        eval(clean, noised)
        XCTAssertEqual(clean.asArray(Float.self), [0.75, 2.5])
        XCTAssertEqual(noised.asArray(Float.self)[0], 0.28125, accuracy: 1e-6)
        XCTAssertEqual(noised.asArray(Float.self)[1], 0.9375, accuracy: 1e-6)
    }

    func testImageConditioningFreezesFirstLatentFrameAndTokens() {
        let shape = [2, 3, 2, 2]
        let mask = Wan2TI2VConditioning.latentMask(shape: shape)
        let tokenMask = Wan2TI2VConditioning.tokenMask(latentShape: shape, patchSize: [1, 1, 1])
        let image = MLX.ones(shape)
        let noise = MLX.zeros(shape)
        let blended = Wan2TI2VConditioning.blend(imageLatent: image, noise: noise, mask: mask)
        eval(mask, tokenMask, blended)

        XCTAssertEqual(mask[0, 0].asArray(Float.self), [0, 0, 0, 0])
        XCTAssertEqual(mask[0, 1].asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(tokenMask.asArray(Float.self), [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        XCTAssertEqual(blended[0, 0].asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(blended[0, 1].asArray(Float.self), [0, 0, 0, 0])
    }

    private func makeOptions(width: Int = 1_280, numFrames: Int = 41) throws -> Wan2GenerationOptions {
        try Wan2GenerationOptions(
            prompt: "walk forward",
            negativePrompt: "static",
            sourceImageURL: URL(fileURLWithPath: "/tmp/source.png"),
            outputURL: URL(fileURLWithPath: "/tmp/output.mp4"),
            width: width,
            height: 704,
            numFrames: numFrames
        )
    }
}
