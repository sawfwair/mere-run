import Foundation
import MLX
import MLXRandom
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import MereRunImageModels

final class IdolTests: MLXTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/Idol")
    }
    private func arrays(_ name: String) throws -> [String: MLXArray] { try MLX.loadArrays(url: root.appending(path: name + ".safetensors")) }
    private func config() throws -> IdolConfiguration {
        try JSONDecoder().decode(IdolConfiguration.self, from: Data(contentsOf: root.appending(path: "config.json")))
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 0.002,
                       file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self),
                                 tolerance, file: file, line: line)
    }

    func testDenoiserAndGuidedEulerTrajectoryMatchUpstream() throws {
        let model = try IdolDenoiser(configuration: config(), weights: arrays("denoiser"))
        let input = try arrays("inputs"), expected = try arrays("results")
        let state = IdolFlowState(latent: input["latent"]!, camera: input["camera"]!)
        let condition = IdolConditioning(dino: input["dino"]!, vae: input["vae"]!)
        let prediction = try model.velocity(state, time: 413.725, conditioning: condition)
        close(prediction.latent, expected["velocity_latent"]!)
        close(prediction.camera, expected["velocity_camera"]!)
        let trajectory = try model.sample(noise: state, conditioning: condition, settings: IdolSamplingSettings(steps: 3))
        close(trajectory.latent, expected["trajectory_latent"]!)
        close(trajectory.camera, expected["trajectory_camera"]!)
        XCTAssertThrowsError(try model.velocity(state, time: .nan, conditioning: condition))
    }

    func testBothDecodersAndGaussianActivationMatchUpstream() throws {
        let config = try config(), input = try arrays("inputs"), expected = try arrays("results")
        let octree = try IdolDecoder(configuration: config, gaussian: false, weights: arrays("decoder"))
        close(try octree.predict(points: input["points"]!, latent: input["latent"]!, resolution: 8), expected["octree_logits"]!)
        let gaussian = try IdolDecoder(configuration: config, gaussian: true, weights: arrays("decoder_gs"))
        let features = try gaussian.predict(points: input["points"]!, latent: input["latent"]!)
        close(features, expected["gaussian_features"]!)
        let result = try gaussian.representation(points: input["points"]!, features: expected["gaussian_features"]!)
        close(result.positions, expected["xyz"]!, tolerance: 1e-6)
        close(result.featuresDC, expected["dc"]!, tolerance: 1e-6)
        close(result.scaling, expected["scale"]!, tolerance: 1e-7)
        close(result.rotation, expected["rotation"]!, tolerance: 1e-6)
        close(result.opacity, expected["opacity"]!, tolerance: 1e-6)
        let nativePLY = try result.plyData(), referencePLY = try Data(contentsOf: root.appending(path: "expected.ply"))
        func payload(_ data: Data) throws -> [Float] {
            let marker = Data("end_header\n".utf8)
            let range = try XCTUnwrap(data.range(of: marker))
            let bytes = Array(data[range.upperBound...])
            return stride(from: 0, to: bytes.count, by: 4).map { index in
                let value = UInt32(bytes[index]) | UInt32(bytes[index + 1]) << 8
                    | UInt32(bytes[index + 2]) << 16 | UInt32(bytes[index + 3]) << 24
                return Float(bitPattern: value)
            }
        }
        close(MLXArray(try payload(nativePLY)), MLXArray(try payload(referencePLY)), tolerance: 1e-6)
    }

    func testStochasticVAEPackingMatchesFP32AndBF16Reference() throws {
        let input = try arrays("inputs"), expected = try arrays("results")
        for (dtype, label) in [(DType.float32, "fp32"), (.bfloat16, "bf16")] {
            let tokens = try IdolConditioning.packVAE(input["posterior"]!.asType(dtype), mean: input["running_mean"]!,
                variance: input["running_variance"]!)
            close(tokens, expected["packed_" + label]!, tolerance: dtype == .float32 ? 1e-6 : 0)
        }
    }

    func testSharedDINOv3AndVAEEncodersMatchIndependentLibraries() throws {
        let expected = try arrays("encoder-results")
        for (dtype, label) in [(DType.float32, "fp32"), (.bfloat16, "bf16")] {
            var configuration = DINOv3Config()
            configuration.hiddenSize = 32; configuration.intermediateSize = 64
            configuration.numHiddenLayers = 2; configuration.numAttentionHeads = 4
            configuration.imageSize = 32; configuration.queryBias = true; configuration.valueBias = true
            let dino = DINOv3VisionModel(config: configuration)
            let dinoWeights = try arrays("dino").mapValues { value in
                (value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value).asType(dtype)
            }
            try dino.update(parameters: ModuleParameters.unflattened(dinoWeights), verify: .all)
            let tokens = MLXFast.layerNorm(dino.normalizedTokens(expected["dino_input"]!.asType(dtype)).asType(.float32), eps: 1e-5)
            close(tokens, expected["dino_" + label]!, tolerance: dtype == .float32 ? 2e-5 : 0.03)
            let vae = AutoencoderKL(configuration: VAEConfig(latentChannels: 32, scalingFactor: 1, shiftFactor: 0,
                blockOutChannels: [32, 32], layersPerBlock: 1, useQuantConv: true, usePostQuantConv: true, attentionNormEps: 1e-6))
            let weights = try arrays("vae").filter { !$0.key.hasPrefix("bn.") }.mapValues { value in
                (value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value).asType(dtype)
            }
            try vae.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            close(vae.encode(expected["vae_input"]!.asType(dtype)), expected["vae_" + label]!, tolerance: dtype == .float32 ? 3e-5 : 0.03)
        }
    }

    func testStrictTensorInventoryRejectsMissingUnexpectedAndWrongShape() throws {
        let configuration = try config()
        var weights = try arrays("denoiser")
        XCTAssertEqual(weights.count, IdolWeightSchema.denoiser(configuration.denoiser.args).count)
        weights.removeValue(forKey: "pos_pe")
        XCTAssertThrowsError(try IdolDenoiser(configuration: configuration, weights: weights))
        weights = try arrays("denoiser")
        weights["extra"] = MLXArray(1)
        XCTAssertThrowsError(try IdolDenoiser(configuration: configuration, weights: weights))
        weights = try arrays("denoiser")
        weights["input_layer.weight"] = zeros([1, 1])
        XCTAssertThrowsError(try IdolDenoiser(configuration: configuration, weights: weights))
        weights = try arrays("denoiser").mapValues { $0.asType(.float16) }
        XCTAssertThrowsError(try IdolDenoiser(configuration: configuration, weights: weights))
    }

    func testOctreePackingConservesCountsBoundsAndChunkParity() throws {
        XCTAssertEqual(try IdolDecoder.systematicCounts(probabilities: [0, 0.5, 0, 0.5], count: 8, offset: 0.25), [0, 4, 0, 4])
        XCTAssertEqual(try IdolDecoder.systematicCounts(probabilities: [0, 0, 0, 0], count: 8, offset: 0.25), [2, 2, 2, 2])
        XCTAssertThrowsError(try IdolDecoder.systematicCounts(probabilities: [.nan], count: 1, offset: 0.5))
        let model = try IdolDecoder(configuration: config(), gaussian: false, weights: arrays("decoder"))
        let latent = try arrays("inputs")["latent"]!
        MLXRandom.seed(99)
        let whole = try model.anchors(latent: latent, count: 16, level: 3)
        MLXRandom.seed(99)
        let chunks = try model.anchors(latent: latent, count: 16, level: 3, queryChunkSize: 2)
        XCTAssertEqual(whole.shape, [1, 16, 3])
        XCTAssertTrue(whole.asArray(Float.self).allSatisfy { $0 >= 0 && $0 < 1 })
        close(chunks, whole, tolerance: 0)
    }

    func testScheduleCameraAndSettingsBoundaries() throws {
        let settings = try IdolSamplingSettings(steps: 3)
        XCTAssertEqual(settings.schedule, [1, 6.0 / 7.0, 0.6, 0])
        XCTAssertThrowsError(try IdolSamplingSettings(steps: 0))
        XCTAssertThrowsError(try IdolSamplingSettings(guidance: .infinity))
        XCTAssertThrowsError(try IdolSamplingSettings(gaussianCount: 32769))
        XCTAssertThrowsError(try IdolSamplingSettings(timeShift: 0))
        let perspective = try IdolCamera.decode([0, 1, 0, 0.5, 1])
        XCTAssertEqual(perspective.radius, 2)
        XCTAssertEqual(perspective.yaw, 0)
        XCTAssertEqual(perspective.fov, 53.1301, accuracy: 1e-4)
        let orthographic = try IdolCamera.decode([0, 0, 0, 0, 1])
        XCTAssertEqual(orthographic.radius, 1000)
        XCTAssertEqual(orthographic.pitch, .pi / 2, accuracy: 1e-6)
    }
}
