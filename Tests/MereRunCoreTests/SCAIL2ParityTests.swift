import Foundation
import MediaIO
import MLX
@testable import MereRunCore
import XCTest

final class SCAIL2ParityTests: MereRunCoreTestCase {
    func testOfficialExamplePreprocessingParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        try assertExamplePreprocessing(prefix: "animation", arrays: arrays)
        try assertExamplePreprocessing(prefix: "replace", arrays: arrays)
    }

    func testOfficialFlowUniPCTrajectoryParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let expectedTimesteps = try XCTUnwrap(arrays["scheduler_timesteps"])
        let expectedSigmas = try XCTUnwrap(arrays["scheduler_sigmas"])
        var scheduler = Wan2UniPCScheduler(steps: expectedTimesteps.dim(0), shift: 3)
        assertClose(
            MLXArray(scheduler.timesteps),
            expectedTimesteps,
            maxTolerance: 0,
            meanTolerance: 0,
            name: "UniPC timesteps"
        )
        assertClose(
            MLXArray(scheduler.sigmas),
            expectedSigmas,
            maxTolerance: 0,
            meanTolerance: 0,
            name: "UniPC sigmas"
        )
        var sample = try XCTUnwrap(arrays["scheduler_sample_00"])
        for index in 0..<expectedTimesteps.dim(0) {
            sample = scheduler.step(
                modelOutput: try XCTUnwrap(arrays[String(format: "scheduler_model_output_%02d", index)]),
                sample: sample
            )
            let expected = try XCTUnwrap(
                arrays[String(format: "scheduler_sample_%02d", index + 1)]
            )
            assertClose(
                sample,
                expected,
                maxTolerance: 2e-5,
                meanTolerance: 2e-6,
                name: "UniPC sample (index + 1)"
            )
        }
    }

    func testOfficialMixedRoPEParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let expectedFrequency = MLX.stacked([
            try XCTUnwrap(arrays["rope_frequencies_real"]),
            try XCTUnwrap(arrays["rope_frequencies_imag"]),
        ], axis: -1)
        let frequencies = Wan2RoPE.frequencies(maxSequence: 256, dimensions: [4, 4, 4])
        assertClose(
            frequencies,
            expectedFrequency,
            maxTolerance: 1e-5,
            meanTolerance: 1e-6,
            name: "RoPE frequencies"
        )
        let input = try XCTUnwrap(arrays["rope_input"])
        let layout = SCAIL2TokenLayout(
            additionalReferenceGrid: Wan2GridSize(frames: 1, height: 4, width: 4),
            referenceGrid: Wan2GridSize(frames: 1, height: 4, width: 4),
            videoGrid: Wan2GridSize(frames: 3, height: 4, width: 4),
            drivingGrid: Wan2GridSize(frames: 3, height: 2, width: 2)
        )
        for (mode, key) in [
            (SCAIL2Mode.animation, "rope_animation_output"),
            (SCAIL2Mode.replacement, "rope_replacement_output"),
        ] {
            let cache = SCAIL2RoPE.prepare(
                layout: layout,
                frequencies: frequencies,
                mode: mode,
                poseWidthShift: 120,
                replacementReferenceHeightShift: 120
            )
            assertClose(
                SCAIL2RoPE.apply(input, cache: cache),
                try XCTUnwrap(arrays[key]),
                maxTolerance: 2e-5,
                meanTolerance: 2e-6,
                name: "(mode) mixed RoPE"
            )
        }
    }

    func testOfficialOneBlockTransformerParityWhenTraceIsAvailable() throws {
        let arrays = try transformerTraceArrays()
        let rootPath = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_ROOT"] ?? ""
        try XCTSkipIf(
            rootPath.isEmpty || !FileManager.default.fileExists(atPath: rootPath),
            "Set MERERUN_TEST_SCAIL2_ROOT to the installed SCAIL-2 model root."
        )
        let resources = SCAIL2Resources(rootURL: URL(fileURLWithPath: rootPath))
        let model = SCAIL2TransformerModel(
            configuration: SCAIL2TransformerConfiguration(layerCount: 1)
        )
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerIndexURL,
            singleURL: resources.transformerURL,
            to: model,
            dtype: nil,
            verify: .none,
            mapper: { key, value in
                guard !key.hasPrefix("blocks.") || key.hasPrefix("blocks.0.") else {
                    return []
                }
                return [(key, value.asType(SCAIL2ModelLoader.transformerDType(for: key)))]
            }
        )
        MLX.eval(model.parameters().flattened().map(\.1))
        let common = (
            videoLatent: try XCTUnwrap(arrays["transformer_video"]),
            referenceLatent: try XCTUnwrap(arrays["transformer_reference"]),
            referenceMask: try XCTUnwrap(arrays["transformer_reference_mask"]),
            drivingLatent: try XCTUnwrap(arrays["transformer_driving"]),
            drivingMask: try XCTUnwrap(arrays["transformer_driving_mask"]),
            textEmbeddings: try XCTUnwrap(arrays["transformer_text"]),
            imageEmbeddings: try XCTUnwrap(arrays["transformer_image"]),
            timestep: try XCTUnwrap(arrays["transformer_timestep"])
        )
        for (mode, key) in [
            (SCAIL2Mode.animation, "transformer_animation_output"),
            (SCAIL2Mode.replacement, "transformer_replacement_output"),
        ] {
            let expectedAssembled = try XCTUnwrap(
                arrays["transformer_\(mode.rawValue)_assembled_tokens"]
            )
            let trace = model.parityTrace(SCAIL2TransformerInput(
                videoLatent: common.videoLatent,
                referenceLatent: common.referenceLatent,
                referenceMask: common.referenceMask,
                drivingLatent: common.drivingLatent,
                drivingMask: common.drivingMask,
                textEmbeddings: common.textEmbeddings,
                imageEmbeddings: common.imageEmbeddings,
                timestep: common.timestep,
                mode: mode
            ), assembledTokensOverride: expectedAssembled)
            let stageTolerances: [(String, Float, Float)] = [
                ("timestep_embedding", 0.02, 0.002),
                ("modulation", 0.05, 0.005),
                ("text_conditioning", 0.08, 0.008),
                ("image_conditioning", 0.08, 0.008),
                ("assembled_tokens", 0.25, 0.012),
                ("self_input", 0.12, 0.015),
                ("self_output", 0.55, 0.016),
                ("post_self", 0.12, 0.015),
                ("cross_input", 0.12, 0.015),
                ("cross_output", 0.12, 0.015),
                ("post_cross", 0.12, 0.015),
                ("ffn_input", 0.12, 0.015),
                ("ffn_output", 0.3, 0.015),
                ("block_0_output", 0.8, 0.015),
                ("head_patches", 0.12, 0.015),
            ]
            for (stage, maximum, mean) in stageTolerances {
                assertClose(
                    try XCTUnwrap(trace[stage]),
                    try XCTUnwrap(arrays["transformer_\(mode.rawValue)_\(stage)"]),
                    maxTolerance: maximum,
                    meanTolerance: mean,
                    name: "\(mode) \(stage)"
                )
            }
            assertClose(
                try XCTUnwrap(trace["output"]),
                try XCTUnwrap(arrays[key]),
                maxTolerance: 0.12,
                meanTolerance: 0.015,
                name: "\(mode) learned transformer block"
            )
        }
    }

    func testOfficialFortyBlockTransformerParityWhenTraceIsAvailable() throws {
        let arrays = try fullTransformerTraceArrays()
        let rootPath = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_ROOT"] ?? ""
        try XCTSkipIf(
            rootPath.isEmpty || !FileManager.default.fileExists(atPath: rootPath),
            "Set MERERUN_TEST_SCAIL2_ROOT to the installed SCAIL-2 model root."
        )
        let resources = SCAIL2Resources(rootURL: URL(fileURLWithPath: rootPath))
        let model = SCAIL2TransformerModel()
        try ModelWeightsLoader.applyHFSafetensors(
            indexURL: resources.transformerIndexURL,
            singleURL: resources.transformerURL,
            to: model,
            dtype: nil,
            verify: .none,
            mapper: { key, value in
                [(key, value.asType(SCAIL2ModelLoader.transformerDType(for: key)))]
            }
        )
        MLX.eval(model.parameters().flattened().map(\.1))
        let common = (
            videoLatent: try XCTUnwrap(arrays["transformer_video"]),
            referenceLatent: try XCTUnwrap(arrays["transformer_reference"]),
            referenceMask: try XCTUnwrap(arrays["transformer_reference_mask"]),
            drivingLatent: try XCTUnwrap(arrays["transformer_driving"]),
            drivingMask: try XCTUnwrap(arrays["transformer_driving_mask"]),
            textEmbeddings: try XCTUnwrap(arrays["transformer_text"]),
            imageEmbeddings: try XCTUnwrap(arrays["transformer_image"]),
            timestep: try XCTUnwrap(arrays["transformer_timestep"])
        )
        for mode in [SCAIL2Mode.animation, .replacement] {
            let prefix = "transformer_\(mode.rawValue)"
            let blockInputs = try (0..<40).map {
                let key = $0 == 0
                    ? "\(prefix)_assembled_tokens"
                    : "\(prefix)_block_\($0)_input"
                return try XCTUnwrap(arrays[key])
            }
            let trace = model.parityFullTrace(
                SCAIL2TransformerInput(
                    videoLatent: common.videoLatent,
                    referenceLatent: common.referenceLatent,
                    referenceMask: common.referenceMask,
                    drivingLatent: common.drivingLatent,
                    drivingMask: common.drivingMask,
                    textEmbeddings: common.textEmbeddings,
                    imageEmbeddings: common.imageEmbeddings,
                    timestep: common.timestep,
                    mode: mode
                ),
                blockInputs: blockInputs,
                finalHidden: try XCTUnwrap(arrays["\(prefix)_block_39_output"])
            )
            assertClose(
                try XCTUnwrap(trace["assembled_tokens"]),
                try XCTUnwrap(arrays["\(prefix)_assembled_tokens"]),
                maxTolerance: 0.25,
                meanTolerance: 0.012,
                name: "\(mode) assembled tokens"
            )
            for index in 0..<40 {
                assertClose(
                    try XCTUnwrap(trace["block_\(index)_output"]),
                    try XCTUnwrap(arrays["\(prefix)_block_\(index)_output"]),
                    maxTolerance: 1.05,
                    meanTolerance: 0.02,
                    name: "\(mode) block \(index)"
                )
            }
            assertClose(
                try XCTUnwrap(trace["head_patches"]),
                try XCTUnwrap(arrays["\(prefix)_head_patches"]),
                maxTolerance: 0.12,
                meanTolerance: 0.015,
                name: "\(mode) output head"
            )
            assertClose(
                try XCTUnwrap(trace["output"]),
                try XCTUnwrap(arrays["\(prefix)_output"]),
                maxTolerance: 0.12,
                meanTolerance: 0.015,
                name: "\(mode) full transformer output"
            )
        }
    }

    func testOfficialWan21VAEParityWhenTraceIsAvailable() throws {
        let arrays = try vaeTraceArrays()
        let rootPath = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_ROOT"] ?? ""
        try XCTSkipIf(
            rootPath.isEmpty || !FileManager.default.fileExists(atPath: rootPath),
            "Set MERERUN_TEST_SCAIL2_ROOT to the installed SCAIL-2 model root."
        )
        let resources = SCAIL2Resources(rootURL: URL(fileURLWithPath: rootPath))
        let vae = try SCAIL2ModelLoader.loadVAE(resources: resources)
        let video = try XCTUnwrap(arrays["vae_video_nthwc"])
        let expectedEncoded = try XCTUnwrap(arrays["vae_encoded_nthwc"])
        let encoded = vae.encodeVideo(video)
        assertClose(
            encoded,
            expectedEncoded,
            maxTolerance: 0.03,
            meanTolerance: 0.003,
            name: "Wan 2.1 VAE encode"
        )
        assertClose(
            vae.decode(expectedEncoded),
            try XCTUnwrap(arrays["vae_decoded_nthwc"]),
            maxTolerance: 0.03,
            meanTolerance: 0.003,
            name: "Wan 2.1 VAE decode"
        )
    }

    func testOfficial31BlockCLIPParityWhenTraceIsAvailable() throws {
        let arrays = try clipTraceArrays()
        let rootPath = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_ROOT"] ?? ""
        try XCTSkipIf(
            rootPath.isEmpty || !FileManager.default.fileExists(atPath: rootPath),
            "Set MERERUN_TEST_SCAIL2_ROOT to the installed SCAIL-2 model root."
        )
        let resources = SCAIL2Resources(rootURL: URL(fileURLWithPath: rootPath))
        let clip = try SCAIL2ModelLoader.loadCLIP(resources: resources)
        let input = try XCTUnwrap(arrays["clip_input_nhwc"])
        var hidden = clip.patchEmbedding(input.asType(clip.patchEmbedding.weight.dtype))
        let batch = hidden.dim(0)
        hidden = hidden.reshaped(batch, -1, clip.configuration.hiddenSize)
        hidden = clip.inputNorm(
            MLX.concatenated([
                MLX.broadcast(
                    clip.classEmbedding.asType(hidden.dtype),
                    to: [batch, 1, clip.configuration.hiddenSize]
                ),
                hidden,
            ], axis: 1) + clip.positionEmbedding.asType(hidden.dtype)
        )
        assertClose(
            hidden,
            try XCTUnwrap(arrays["clip_pre_norm_ntc"]),
            maxTolerance: 0.02,
            meanTolerance: 0.002,
            name: "SCAIL-2 CLIP pre-norm"
        )
        for (index, block) in clip.blocks.enumerated() {
            hidden = block(hidden)
            // FP16 attention/GEMM rounding accumulates across the 31-block
            // PyTorch-MPS versus MLX path. The first 14 blocks stay under
            // 0.08 max; later blocks keep the stricter mean-error gate while
            // allowing isolated outliers in otherwise matching activations.
            let maximumTolerance: Float = index < 14 ? 0.08 : 0.7
            assertClose(
                hidden,
                try XCTUnwrap(arrays[String(format: "clip_block_%02d_ntc", index)]),
                maxTolerance: maximumTolerance,
                meanTolerance: 0.008,
                name: "SCAIL-2 CLIP block \(index)"
            )
        }
        assertClose(
            hidden,
            try XCTUnwrap(arrays["clip_output_ntc"]),
            maxTolerance: 0.7,
            meanTolerance: 0.008,
            name: "SCAIL-2 31-block CLIP visual"
        )
    }

    func testOfficialPromptUMT5ParityWhenTraceIsAvailable() throws {
        let arrays = try t5TraceArrays()
        let rootPath = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_ROOT"] ?? ""
        try XCTSkipIf(
            rootPath.isEmpty || !FileManager.default.fileExists(atPath: rootPath),
            "Set MERERUN_TEST_SCAIL2_ROOT to the installed SCAIL-2 model root."
        )
        let resources = SCAIL2Resources(rootURL: URL(fileURLWithPath: rootPath))
        let tokenizer = try SCAIL2ModelLoader.loadTokenizer(resources: resources)
        let tokenIDs = try XCTUnwrap(arrays["t5_token_ids"])
        let mask = try XCTUnwrap(arrays["t5_mask"])
        let prompts = [
            "The girl is dancing",
            "A blond white male wearing a black suit, trousers, and leather shoes is playing the violin on the street while pedestrians walk past him.",
            "",
        ]
        for (index, prompt) in prompts.enumerated() {
            let encoded = tokenizer.encode(prompt)
            XCTAssertEqual(encoded.tokenIDs, tokenIDs[index].asArray(Int32.self).map(Int.init))
            XCTAssertEqual(encoded.mask, mask[index].asArray(Int32.self).map(Int.init))
        }
        let encoder = try SCAIL2ModelLoader.loadTextEncoder(resources: resources)
        assertClose(
            encoder(tokenIDs: tokenIDs, mask: mask),
            try XCTUnwrap(arrays["t5_output_ntc"]),
            maxTolerance: 0.3,
            meanTolerance: 0.02,
            name: "SCAIL-2 official-prompt UMT5-XXL"
        )
    }

    private func assertExamplePreprocessing(
        prefix: String,
        arrays: [String: MLXArray]
    ) throws {
        let source = try mediaImage(try XCTUnwrap(arrays["\(prefix)_source_rgb_hwc_u8"]))
        let maskSource = try mediaImage(
            try XCTUnwrap(arrays["\(prefix)_mask_source_rgb_hwc_u8"])
        )
        let expected = try XCTUnwrap(arrays["\(prefix)_cropped_nhwc"])
        let expectedMask = try XCTUnwrap(arrays["\(prefix)_mask_cropped_nhwc"])
        let actual = SCAIL2InputPreprocessor.centerCroppedTensor(
            image: source,
            width: expected.dim(2),
            height: expected.dim(1)
        )
        let actualMask = SCAIL2InputPreprocessor.centerCroppedTensor(
            image: maskSource,
            width: expectedMask.dim(2),
            height: expectedMask.dim(1)
        )
        assertClose(
            actual,
            expected,
            maxTolerance: 0.02,
            meanTolerance: 0.001,
            name: "(prefix) reference crop"
        )
        assertClose(
            actualMask,
            expectedMask,
            maxTolerance: 0.02,
            meanTolerance: 0.001,
            name: "(prefix) mask crop"
        )
        assertClose(
            SCAIL2InputPreprocessor.halfResolutionBilinear(expected),
            try XCTUnwrap(arrays["\(prefix)_half_nhwc"]),
            maxTolerance: 1e-6,
            meanTolerance: 1e-7,
            name: "(prefix) half-resolution bilinear"
        )
        assertClose(
            SCAIL2MaskEncoder.encode(expectedMask),
            try XCTUnwrap(arrays["\(prefix)_mask_encoded_cthw"]),
            maxTolerance: 1e-6,
            meanTolerance: 1e-7,
            name: "(prefix) mask packing"
        )
        assertClose(
            SCAIL2CLIPPreprocessor.normalizedNHWC(croppedImages: expected),
            try XCTUnwrap(arrays["\(prefix)_clip_nhwc"]),
            maxTolerance: 2e-4,
            meanTolerance: 2e-5,
            name: "(prefix) CLIP preprocessing"
        )
    }

    private func mediaImage(_ rgb: MLXArray) throws -> MediaImage {
        XCTAssertEqual(rgb.ndim, 3)
        XCTAssertEqual(rgb.dim(2), 3)
        let height = rgb.dim(0)
        let width = rgb.dim(1)
        let values = rgb.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: height * width * 4)
        for pixel in 0..<(height * width) {
            rgba[pixel * 4] = values[pixel * 3]
            rgba[pixel * 4 + 1] = values[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = values[pixel * 3 + 2]
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    private func traceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_TRACE to export_scail2_trace.py output."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func transformerTraceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_TRANSFORMER_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_TRANSFORMER_TRACE to the learned PyTorch trace."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func fullTransformerTraceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_TRANSFORMER_FULL_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_TRANSFORMER_FULL_TRACE to the 40-block PyTorch trace."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func vaeTraceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_VAE_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_VAE_TRACE to the learned PyTorch VAE trace."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func clipTraceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_CLIP_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_CLIP_TRACE to the learned PyTorch CLIP trace."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func t5TraceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_SCAIL2_T5_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_SCAIL2_T5_TRACE to the learned PyTorch UMT5 trace."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func assertClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        maxTolerance: Float,
        meanTolerance: Float,
        name: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "\(name) shape", file: file, line: line)
        guard actual.shape == expected.shape else { return }
        let difference = MLX.abs(actual.asType(.float32) - expected.asType(.float32))
        MLX.eval(difference)
        let maximum = MLX.max(difference).item(Float.self)
        let mean = MLX.mean(difference).item(Float.self)
        XCTAssertLessThanOrEqual(
            maximum,
            maxTolerance,
            "\(name) max difference \(maximum)",
            file: file,
            line: line
        )
        XCTAssertLessThanOrEqual(
            mean,
            meanTolerance,
            "\(name) mean difference \(mean)",
            file: file,
            line: line
        )
    }
}
