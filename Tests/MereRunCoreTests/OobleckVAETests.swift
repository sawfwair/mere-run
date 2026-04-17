import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class OobleckVAETests: MereRunCoreTestCase {

    func testDecodeLengthScalingSmallConfig() {
        let config = OobleckVAEConfig(
            audioChannels: 2,
            channelMultiples: [1, 2],
            decoderChannels: 8,
            decoderInputChannels: 4,
            downsamplingRatios: [2, 4],
            encoderHiddenSize: 8,
            samplingRate: 48_000
        )

        let vae = OobleckVAE(config: config)

        let B = 2
        let T = 5
        let latents = MLXRandom.normal([B, T, config.decoderInputChannels]).asType(.float32)

        let audio = vae.decode(latents)

        let factor = config.downsamplingRatios.reduce(1, *)
        XCTAssertEqual(audio.dim(0), B)
        XCTAssertEqual(audio.dim(1), T * factor)
        XCTAssertEqual(audio.dim(2), config.audioChannels)
    }

    func testEncodeLengthScalingSmallConfig() {
        let config = OobleckVAEConfig(
            audioChannels: 2,
            channelMultiples: [1, 2],
            decoderChannels: 8,
            decoderInputChannels: 4,
            downsamplingRatios: [2, 4],
            encoderHiddenSize: 8,
            samplingRate: 48_000
        )

        let vae = OobleckVAE(config: config)

        let B = 2
        let S = 40
        let audio = MLXRandom.normal([B, S, config.audioChannels]).asType(.float32)

        let latents = vae.encode(audio, sample: false)

        let factor = config.downsamplingRatios.reduce(1, *)
        XCTAssertEqual(latents.dim(0), B)
        XCTAssertEqual(latents.dim(1), S / factor)
        XCTAssertEqual(latents.dim(2), config.decoderInputChannels)
    }

    func testTiledDecodeMatchesDecodeSmallConfig() {
        let config = OobleckVAEConfig(
            audioChannels: 2,
            channelMultiples: [1, 2],
            decoderChannels: 8,
            decoderInputChannels: 4,
            downsamplingRatios: [2, 4],
            encoderHiddenSize: 8,
            samplingRate: 48_000
        )

        let vae = OobleckVAE(config: config)

        let B = 1
        let T = 256
        let latents = MLXRandom.normal([B, T, config.decoderInputChannels]).asType(.float32)

        let full = vae.decode(latents)
        let tiled = vae.tiledDecode(latents, chunkSize: 128, overlap: 32)

        XCTAssertEqual(tiled.shape, full.shape)

        let maxDiff = MLX.max(MLX.abs(full - tiled)).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-4)
    }

    func testTiledEncodeProducesValidLatentsSmallConfig() {
        let config = OobleckVAEConfig(
            audioChannels: 2,
            channelMultiples: [1, 2],
            decoderChannels: 8,
            decoderInputChannels: 4,
            downsamplingRatios: [2, 4],
            encoderHiddenSize: 8,
            samplingRate: 48_000
        )

        let vae = OobleckVAE(config: config)

        let B = 1
        let S = 320
        let audio = MLXRandom.normal([B, S, config.audioChannels]).asType(.float32)

        let full = vae.encode(audio, sample: false)
        let tiled = vae.tiledEncode(audio, chunkSize: 128, overlap: 16, sample: false)

        XCTAssertEqual(tiled.shape, full.shape)

        let maxDiff = MLX.max(MLX.abs(full - tiled)).item(Float.self)
        XCTAssertFalse(maxDiff.isNaN)
        XCTAssertFalse(maxDiff.isInfinite)

        let maxAbs = MLX.max(MLX.abs(tiled.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)
    }

    func testLoadVAEWeightsAndDecode() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae to run this test.")
        }

        let resources = OobleckVAEResources(rootURL: URL(fileURLWithPath: root))
        let missing = resources.validate()
        if !missing.isEmpty {
            let list = missing.map { $0.path }.joined(separator: "\n")
            throw XCTSkip("Oobleck VAE checkpoint incomplete:\n\(list)")
        }

        let config = try OobleckVAECheckpointLoader.loadConfig(resources: resources)
        let vae = try OobleckVAECheckpointLoader.loadVAE(resources: resources)

        let B = 1
        let T = 3
        let latents = MLXRandom.normal([B, T, config.decoderInputChannels]).asType(.bfloat16)

        let audio = vae.decode(latents)

        let factor = config.downsamplingRatios.reduce(1, *)
        XCTAssertEqual(audio.dim(0), B)
        XCTAssertEqual(audio.dim(1), T * factor)
        XCTAssertEqual(audio.dim(2), config.audioChannels)

        let maxAbs = MLX.max(MLX.abs(audio.asType(.float32))).item(Float.self)
        XCTAssertFalse(maxAbs.isNaN)
        XCTAssertFalse(maxAbs.isInfinite)
        XCTAssertGreaterThan(maxAbs, 0)

        let samples = factor * 3
        let inputAudio = MLXRandom.normal([B, samples, config.audioChannels]).asType(.bfloat16)
        let encoded = vae.encode(inputAudio, sample: false)
        XCTAssertEqual(encoded.dim(0), B)
        XCTAssertEqual(encoded.dim(1), 3)
        XCTAssertEqual(encoded.dim(2), config.decoderInputChannels)

        let encodedMaxAbs = MLX.max(MLX.abs(encoded.asType(.float32))).item(Float.self)
        XCTAssertFalse(encodedMaxAbs.isNaN)
        XCTAssertFalse(encodedMaxAbs.isInfinite)
        XCTAssertGreaterThan(encodedMaxAbs, 0)
    }
}
