import MLX
@testable import MereRunCore
import XCTest

final class LTXVocoderBWETests: XCTestCase {
    func testDistilledGeneratorRejectsLTX23SplitModelBeforeLegacyLayoutChecks() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try #"{"model_version":"2.3.0"}"#.write(
            to: root.appendingPathComponent("config.json", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("split_model.json").path,
            contents: Data()
        ))
        XCTAssertTrue(FileManager.default.createFile(
            atPath: root.appendingPathComponent("transformer-distilled.safetensors").path,
            contents: Data()
        ))

        let generator = LTXDistilledLatentGenerator()
        do {
            try await generator.load(modelRoot: root)
            XCTFail("Expected the LTX 2.3 split model guard to reject this root.")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("LTX 2.3 split MLX model"))
            XCTAssertFalse(error.localizedDescription.contains("ltx-2-19b-distilled.safetensors"))
        }
    }

    func testDetectsLegacyFlatVocoderKeys() {
        let keys = [
            "vocoder.conv_pre.weight",
            "vocoder.resblocks.0.convs1.0.weight",
            "vocoder.ups.0.weight",
        ]

        XCTAssertEqual(detectLTXVocoderFlavor(keys: keys), .legacy)
    }

    func testDetectsBandwidthExtensionVocoderKeys() {
        let keys = [
            "vocoder.conv_pre.weight",
            "vocoder.bwe_generator.conv_pre.weight",
            "vocoder.mel_stft.mel_basis",
        ]

        XCTAssertEqual(detectLTXVocoderFlavor(keys: keys), .bandwidthExtension)
    }

    func testVocoderWeightMapperRoutesDirectBWEBaseWeightsIntoNestedVocoder() {
        let weight = MLXArray.zeros([2, 3, 5])
        let mapped = mapVocoderWeight(
            key: "vocoder.conv_pre.weight",
            value: weight,
            dtype: .float32,
            targetFlavor: .bandwidthExtension
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "vocoder.conv_pre.weight")
        XCTAssertEqual(mapped[0].1.shape, [2, 5, 3])
    }

    func testVocoderWeightMapperNormalizesNestedBWEBasePrefix() {
        let weight = MLXArray.zeros([2, 3, 5])
        let mapped = mapVocoderWeight(
            key: "vocoder.vocoder.conv_pre.weight",
            value: weight,
            dtype: .float32,
            targetFlavor: .bandwidthExtension
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "vocoder.conv_pre.weight")
        XCTAssertEqual(mapped[0].1.shape, [2, 5, 3])
    }

    func testVocoderWeightMapperTransposesBWEFilterBuffers() {
        let basis = MLXArray.zeros([4, 1, 8])
        let mapped = mapVocoderWeight(
            key: "vocoder.mel_stft.stft_fn.forward_basis",
            value: basis,
            dtype: .float32,
            targetFlavor: .bandwidthExtension
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "mel_stft.stft_fn.forward_basis")
        XCTAssertEqual(mapped[0].1.shape, [4, 8, 1])
    }

    func testVocoderWeightMapperPreservesAlreadyMLXWeights() {
        let weight = MLXArray.zeros([1536, 7, 128])
        let mapped = mapVocoderWeight(
            key: "vocoder.conv_pre.weight",
            value: weight,
            dtype: .float32,
            sourceLayout: .mlx
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "conv_pre.weight")
        XCTAssertEqual(mapped[0].1.shape, [1536, 7, 128])
    }

    func testVocoderWeightMapperNormalizesActivationDownsampleLowpassFilterPath() {
        let filter = MLXArray.zeros([1, 12, 1])
        let mapped = mapVocoderWeight(
            key: "vocoder.bwe_generator.resblocks.0.acts1.0.downsample.lowpass.filter",
            value: filter,
            dtype: .float32,
            sourceLayout: .mlx,
            targetFlavor: .bandwidthExtension
        )

        XCTAssertEqual(mapped.count, 1)
        XCTAssertEqual(mapped[0].0, "bwe_generator.resblocks.0.acts1.0.downsample.filter")
        XCTAssertEqual(mapped[0].1.shape, [1, 12, 1])
    }

    func testMatchesShortWaveformToVideoDurationByPadding() {
        let waveform = MLXArray.ones([1, 2, 52_320], dtype: .float32)
        let matched = matchLTXAudioWaveformDuration(waveform, videoFrames: 9, fps: 8, sampleRate: 48_000)

        XCTAssertEqual(matched.shape, [1, 2, 54_000])
        XCTAssertEqual(matched[0, 0, 52_319].item(Float.self), 1.0, accuracy: 0.0001)
        XCTAssertEqual(matched[0, 0, 53_999].item(Float.self), 0.0, accuracy: 0.0001)
    }

    func testMatchesLongWaveformToVideoDurationByTrimming() {
        let waveform = MLXArray.ones([1, 2, 54_010], dtype: .float32)
        let matched = matchLTXAudioWaveformDuration(waveform, videoFrames: 9, fps: 8, sampleRate: 48_000)

        XCTAssertEqual(matched.shape, [1, 2, 54_000])
    }

    func testHannSincUpsamplerMatchesReferenceZeroInsertConvolution() {
        let inputValues: [Float] = [0.0, 0.25, -0.5, 0.75, -1.0]
        let input = MLXArray(inputValues, [1, inputValues.count, 1])
        let upsampler = LTXSincUpsample1d(ratio: 3, windowType: "hann")
        let output = upsampler(input)

        XCTAssertEqual(output.shape, [1, inputValues.count * 3, 1])
        let actual = output.asArray(Float.self)
        let expected = referenceHannSincUpsample(inputValues, ratio: 3)
        XCTAssertEqual(actual.count, expected.count)
        for index in actual.indices {
            XCTAssertEqual(actual[index], expected[index], accuracy: 1e-5, "Mismatch at sample \(index)")
        }
    }

    func testLoadsNestedBWEVocoderConfig() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let vocoderRoot = root.appendingPathComponent("vocoder", isDirectory: true)
        try FileManager.default.createDirectory(at: vocoderRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let config = """
        {
          "vocoder": {
            "bwe": {
              "input_sampling_rate": 24000,
              "output_sampling_rate": 48000,
              "hop_length": 60,
              "n_fft": 1024,
              "num_mels": 128
            }
          }
        }
        """
        try config.write(
            to: vocoderRoot.appendingPathComponent("config.json", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let loaded = try XCTUnwrap(loadLTXBWEVocoderConfig(modelRoot: root))
        XCTAssertEqual(loaded.inputSamplingRate, 24_000)
        XCTAssertEqual(loaded.outputSamplingRate, 48_000)
        XCTAssertEqual(loaded.hopLength, 60)
        XCTAssertEqual(loaded.filterLength, 1024)
        XCTAssertEqual(loaded.melChannels, 128)
    }

    func testLoadsEmbeddedLTX23BWEVocoderArchitectureConfig() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let config = """
        {
          "vocoder": {
            "vocoder": {
              "upsample_initial_channel": 1536,
              "resblock": "AMP1",
              "upsample_rates": [5, 2, 2, 2, 2, 2],
              "resblock_kernel_sizes": [3, 7, 11],
              "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
              "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
              "use_tanh_at_final": false,
              "activation": "snakebeta",
              "use_bias_at_final": false
            },
            "bwe": {
              "upsample_initial_channel": 512,
              "resblock": "AMP1",
              "upsample_rates": [6, 5, 2, 2, 2],
              "resblock_kernel_sizes": [3, 7, 11],
              "upsample_kernel_sizes": [12, 11, 4, 4, 4],
              "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
              "use_tanh_at_final": false,
              "activation": "snakebeta",
              "use_bias_at_final": false,
              "apply_final_activation": false,
              "input_sampling_rate": 16000,
              "output_sampling_rate": 48000,
              "hop_length": 80,
              "n_fft": 512,
              "win_size": 512,
              "num_mels": 64
            }
          }
        }
        """
        try config.write(
            to: root.appendingPathComponent("embedded_config.json", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let loaded = try XCTUnwrap(loadLTXBWEVocoderConfig(modelRoot: root))
        XCTAssertEqual(loaded.inputSamplingRate, 16_000)
        XCTAssertEqual(loaded.outputSamplingRate, 48_000)
        XCTAssertEqual(loaded.hopLength, 80)
        XCTAssertEqual(loaded.filterLength, 512)
        XCTAssertEqual(loaded.melChannels, 64)
        XCTAssertEqual(loaded.baseVocoder.upsampleInitialChannels, 1536)
        XCTAssertEqual(loaded.baseVocoder.upsampleRates, [5, 2, 2, 2, 2, 2])
        XCTAssertEqual(loaded.baseVocoder.upsampleKernelSizes, [11, 4, 4, 4, 4, 4])
        XCTAssertEqual(loaded.baseVocoder.useTanhAtFinal, false)
        XCTAssertEqual(loaded.baseVocoder.useBiasAtFinal, false)
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.upsampleInitialChannels, 512)
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.upsampleKernelSizes, [12, 11, 4, 4, 4])
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.applyFinalActivation, false)
    }

    func testLoadsTopLevelLTX23BWEVocoderArchitectureConfig() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let config = """
        {
          "vocoder": {
            "upsample_initial_channel": 1536,
            "resblock": "AMP1",
            "upsample_rates": [5, 2, 2, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_tanh_at_final": false,
            "activation": "snakebeta",
            "use_bias_at_final": false
          },
          "bwe": {
            "upsample_initial_channel": 512,
            "resblock": "AMP1",
            "upsample_rates": [6, 5, 2, 2, 2],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_kernel_sizes": [12, 11, 4, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_tanh_at_final": false,
            "activation": "snakebeta",
            "use_bias_at_final": false,
            "apply_final_activation": false,
            "input_sampling_rate": 16000,
            "output_sampling_rate": 48000,
            "hop_length": 80,
            "n_fft": 512,
            "win_size": 512,
            "num_mels": 64
          }
        }
        """
        try config.write(
            to: root.appendingPathComponent("embedded_config.json", isDirectory: false),
            atomically: true,
            encoding: .utf8
        )

        let loaded = try XCTUnwrap(loadLTXBWEVocoderConfig(modelRoot: root))
        XCTAssertEqual(loaded.inputSamplingRate, 16_000)
        XCTAssertEqual(loaded.outputSamplingRate, 48_000)
        XCTAssertEqual(loaded.hopLength, 80)
        XCTAssertEqual(loaded.filterLength, 512)
        XCTAssertEqual(loaded.melChannels, 64)
        XCTAssertEqual(loaded.baseVocoder.upsampleInitialChannels, 1536)
        XCTAssertEqual(loaded.baseVocoder.upsampleRates, [5, 2, 2, 2, 2, 2])
        XCTAssertEqual(loaded.baseVocoder.upsampleKernelSizes, [11, 4, 4, 4, 4, 4])
        XCTAssertEqual(loaded.baseVocoder.useTanhAtFinal, false)
        XCTAssertEqual(loaded.baseVocoder.useBiasAtFinal, false)
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.upsampleInitialChannels, 512)
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.upsampleKernelSizes, [12, 11, 4, 4, 4])
        XCTAssertEqual(loaded.bandwidthExtensionVocoder.applyFinalActivation, false)
    }
}

private func referenceHannSincUpsample(_ input: [Float], ratio: Int) -> [Float] {
    let rolloff: Float = 0.99
    let lowpassFilterWidth: Float = 6.0
    let width = Int(ceil(lowpassFilterWidth / rolloff))
    let kernelSize = 2 * width * ratio + 1
    let pad = width
    let padLeft = 2 * width * ratio
    let padRight = kernelSize - ratio

    let kernel: [Float] = (0..<kernelSize).map { index in
        let timeAxis = (Float(index) / Float(ratio) - Float(width)) * rolloff
        let clamped = min(lowpassFilterWidth, max(-lowpassFilterWidth, timeAxis))
        let window = pow(cos(clamped * Float.pi / lowpassFilterWidth / 2), 2)
        return sinc(timeAxis) * window * rolloff / Float(ratio)
    }

    let left = Array(repeating: input.first ?? 0, count: pad)
    let right = Array(repeating: input.last ?? 0, count: pad)
    let padded = left + input + right
    var zeroInserted = [Float](repeating: 0, count: (padded.count - 1) * ratio + 1)
    for (index, value) in padded.enumerated() {
        zeroInserted[index * ratio] = value
    }

    let convInput = Array(repeating: Float(0), count: kernelSize - 1)
        + zeroInserted
        + Array(repeating: Float(0), count: kernelSize - 1)
    var filtered = [Float]()
    filtered.reserveCapacity(zeroInserted.count + kernelSize - 1)
    for start in 0..<(zeroInserted.count + kernelSize - 1) {
        var sum: Float = 0
        for offset in 0..<kernelSize {
            sum += convInput[start + offset] * kernel[offset]
        }
        filtered.append(sum * Float(ratio))
    }

    return Array(filtered[padLeft..<(filtered.count - padRight)])
}

private func sinc(_ value: Float) -> Float {
    if abs(value) < 1e-8 {
        return 1
    }
    return sin(Float.pi * value) / (Float.pi * value)
}
