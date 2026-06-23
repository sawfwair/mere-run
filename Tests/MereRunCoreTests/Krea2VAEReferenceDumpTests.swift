import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Krea2VAEReferenceDumpTests: XCTestCase {
    func testQwenConv2DReferenceFixtureWhenRequested() throws {
        let env = ProcessInfo.processInfo.environment
        guard let fixture = env["MERERUN_TEST_KREA2_CONV2D_FIXTURE"], !fixture.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_KREA2_CONV2D_FIXTURE to compare an isolated Qwen conv2d fixture.")
        }

        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let input = try XCTUnwrap(arrays["input_nchw"])
        let weight = try XCTUnwrap(arrays["weight_oihw"])
        let bias = try XCTUnwrap(arrays["bias"])
        let expected = try XCTUnwrap(arrays["expected_nchw"])

        let paddedInput = padded(input.transposed(0, 2, 3, 1), widths: [[0, 0], [1, 1], [1, 1], [0, 0]])
        let kernel = weight.transposed(0, 2, 3, 1)
        let contiguousKernel = kernel.reshaped(-1).reshaped(kernel.shape)
        let contiguousInput = paddedInput.reshaped(-1).reshaped(paddedInput.shape)

        let direct = MLX.conv2d(paddedInput, kernel, stride: .init((1, 1)), padding: 0) + bias
        let kernelContiguous = MLX.conv2d(paddedInput, contiguousKernel, stride: .init((1, 1)), padding: 0) + bias
        let fullyContiguous = MLX.conv2d(contiguousInput, contiguousKernel, stride: .init((1, 1)), padding: 0) + bias

        for (name, channelsLast) in [
            ("direct", direct),
            ("kernel_contiguous", kernelContiguous),
            ("fully_contiguous", fullyContiguous)
        ] {
            let actual = channelsLast.transposed(0, 3, 1, 2).asType(.float32)
            MLX.eval(actual)
            let maxDiff = MLX.max(MLX.abs(actual - expected.asType(.float32))).item(Float.self)
            XCTAssertLessThan(maxDiff, 0.00001, "\(name) should match the PyTorch fixture.")
            let stats = Self.statsNCHW(actual)
            FileHandle.standardError.write(
                "\(name) max_diff=\(maxDiff) mean=\(stats.mean) std=\(stats.std) min=\(stats.min) max=\(stats.max)\n"
                    .data(using: .utf8)!
            )
        }

        if let root = env["MERERUN_TEST_KREA2_ROOT"], !root.isEmpty {
            let resources = Krea2Resources(rootURL: URL(fileURLWithPath: root))
            let configs = try Krea2ModelConfigs.load(from: resources)
            let vae = try Krea2ModelLoader.loadVAE(from: resources, configuration: configs.vae, dtype: .float32)
            let params = Dictionary(uniqueKeysWithValues: vae.underlyingVAE.parameters().flattened())
            let convInWeight = try XCTUnwrap(params["decoder.convIn.weight"] ?? params["decoder.conv_in.weight"])
            let convInBias = try XCTUnwrap(params["decoder.convIn.bias"] ?? params["decoder.conv_in.bias"])
            let loadedSlice = convInWeight[0..., 0..., 2, 0..., 0...]
            let weightDiff = MLX.max(MLX.abs(loadedSlice.asType(.float32) - weight.asType(.float32))).item(Float.self)
            let biasDiff = MLX.max(MLX.abs(convInBias.asType(.float32) - bias.asType(.float32))).item(Float.self)
            XCTAssertLessThan(weightDiff, 0.00001, "Loaded conv_in weight should match the PyTorch fixture.")
            XCTAssertLessThan(biasDiff, 0.00001, "Loaded conv_in bias should match the PyTorch fixture.")
            let loadedKernel = loadedSlice.transposed(0, 2, 3, 1)
            let loadedOut = MLX.conv2d(paddedInput, loadedKernel, stride: .init((1, 1)), padding: 0) + convInBias
            let actual = loadedOut.transposed(0, 3, 1, 2).asType(.float32)
            MLX.eval(actual)
            let outputDiff = MLX.max(MLX.abs(actual - expected.asType(.float32))).item(Float.self)
            XCTAssertLessThan(outputDiff, 0.00001, "Loaded conv_in output should match the PyTorch fixture.")
            let loadedStats = Self.statsNCHW(actual)
            FileHandle.standardError.write(
                "loaded_model weight_diff=\(weightDiff) bias_diff=\(biasDiff) output_diff=\(outputDiff) mean=\(loadedStats.mean) std=\(loadedStats.std) min=\(loadedStats.min) max=\(loadedStats.max)\n"
                    .data(using: .utf8)!
            )
        }
    }

    func testDumpInstalledKreaQwenVAEDecodeWhenRequested() throws {
        let env = ProcessInfo.processInfo.environment
        guard let root = env["MERERUN_TEST_KREA2_ROOT"], !root.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_KREA2_ROOT to dump a Krea/Qwen VAE decode.")
        }

        let output = env["MERERUN_TEST_KREA2_VAE_OUTPUT"] ?? "/tmp/qwen-vae-swift.png"
        let resources = Krea2Resources(rootURL: URL(fileURLWithPath: root))
        let configs = try Krea2ModelConfigs.load(from: resources)
        let dtype: DType = env["MERERUN_TEST_KREA2_VAE_FLOAT32"] == "1" ? .float32 : .bfloat16
        let vae = try Krea2ModelLoader.loadVAE(from: resources, configuration: configs.vae, dtype: dtype)

        var latents = Self.referenceLatents(channels: configs.vae.latentChannels, height: 16, width: 16)
        if let mean = configs.vae.latentsMean, let std = configs.vae.latentsStd {
            let meanTensor = MLXArray(mean).reshaped(1, mean.count, 1, 1).asType(latents.dtype)
            let stdTensor = MLXArray(std).reshaped(1, std.count, 1, 1).asType(latents.dtype)
            latents = latents * stdTensor + meanTensor
        }
        if let shift = configs.vae.shiftFactor, shift != 0 {
            latents = latents - MLXArray(shift).asType(latents.dtype)
        }
        latents = latents * MLXArray(configs.vae.scalingFactor).asType(latents.dtype)

        var decoded = vae.decode(latents.asType(dtype))
        decoded = MLX.clip(decoded.asType(.float32), min: -1.0, max: 1.0)
        let image = MLX.clip(QwenImageIO.denormalizeFromDecoder(decoded), min: 0, max: 1)
        try QwenImageIO.saveImage(array: image, to: URL(fileURLWithPath: output))

        let stats = Self.stats(image)
        FileHandle.standardError.write(
            "swift_vae shape=\(image.shape) mean=\(stats.mean) min=\(stats.min) max=\(stats.max)\n"
                .data(using: .utf8)!
        )
    }

    private static func referenceLatents(channels: Int, height: Int, width: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(channels * height * width)
        for c in 0..<channels {
            for y in 0..<height {
                for x in 0..<width {
                    values.append(sinf(Float(c + 1) * 0.13 + Float(y) * 0.07 + Float(x) * 0.11))
                }
            }
        }
        return MLXArray(values).reshaped(1, channels, height, width)
    }

    private static func stats(_ image: MLXArray) -> (mean: Float, min: Float, max: Float) {
        let evaluated = image.asType(.float32)
        MLX.eval(evaluated)
        return (
            mean: MLX.mean(evaluated).item(Float.self),
            min: MLX.min(evaluated).item(Float.self),
            max: MLX.max(evaluated).item(Float.self)
        )
    }

    private static func statsNCHW(_ image: MLXArray) -> (mean: Float, std: Float, min: Float, max: Float) {
        let evaluated = image.asType(.float32)
        MLX.eval(evaluated)
        return (
            mean: MLX.mean(evaluated).item(Float.self),
            std: MLX.std(evaluated).item(Float.self),
            min: MLX.min(evaluated).item(Float.self),
            max: MLX.max(evaluated).item(Float.self)
        )
    }
}
