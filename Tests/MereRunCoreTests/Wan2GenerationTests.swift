import MLX
import XCTest
@testable import MereRunCore

final class Wan2GenerationTests: MereRunCoreTestCase {
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
        XCTAssertEqual(scheduler.timesteps.last, 624)
        XCTAssertEqual(scheduler.sigmas[0], 4.995 / 4.996, accuracy: 1e-6)
    }

    func testEulerStepUsesNextMinusCurrentSigma() {
        var scheduler = Wan2FlowMatchEulerScheduler(steps: 2, shift: 1)
        let sample = MLXArray([1, 2], [1, 2])
        let velocity = MLXArray([2, 4], [1, 2])
        let result = scheduler.step(velocity: velocity, sample: sample)
        eval(result)
        let values = result.asArray(Float.self)
        XCTAssertEqual(values.count, 2)
        XCTAssertLessThan(abs(values[0] - 0.001), 1e-5)
        XCTAssertLessThan(abs(values[1] - 0.002), 1e-5)
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
