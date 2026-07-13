import Foundation
import MediaIO
import MLX
import MLXNN
@testable import MereRunCore
import XCTest

final class MMAudioParityTests: XCTestCase {
    func testTorchvisionPreprocessingParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let source = try XCTUnwrap(arrays["preprocess_rgb_hwc_u8"])
        let expectedCLIP = try XCTUnwrap(arrays["preprocess_clip_nhwc"])
        let expectedSync = try XCTUnwrap(arrays["preprocess_sync_chw"])
        let height = source.dim(0)
        let width = source.dim(1)
        let rgb = source.asArray(UInt8.self)
        var rgba = [UInt8](repeating: 255, count: height * width * 4)
        for pixel in 0..<(height * width) {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        let image = try MediaImage(width: width, height: height, rgba8: rgba)

        // Accelerate and Torchvision use different bicubic integer kernels;
        // these bounds are below one source pixel after CLIP normalization.
        assertClose(
            MMAudioVideoPreprocessor.normalizedCLIPFrame(image),
            expectedCLIP,
            maxTolerance: 0.08,
            meanTolerance: 0.0035,
            name: "CLIP preprocessing"
        )
        let actualSync = MLXArray(
            MMAudioVideoPreprocessor.normalizedSynchformerFrameCHW(image)
        ).reshaped(expectedSync.shape)
        assertClose(
            actualSync,
            expectedSync,
            maxTolerance: 0.008,
            meanTolerance: 0.0018,
            name: "Synchformer preprocessing"
        )
    }

    func testCLIPTextAndImageParityWhenTraceIsAvailable() async throws {
        let arrays = try traceArrays()
        let resources = try modelResources()
        let conditioner = try await MMAudioCLIPConditioner.load(resources: resources)
        let expectedTokenIDs = try XCTUnwrap(arrays["clip_token_ids"])
        XCTAssertEqual(
            conditioner.tokenIDs(for: ["a wooden door slamming shut"]),
            expectedTokenIDs.asArray(Int32.self)
        )
        assertClose(
            conditioner.encodeText(["a wooden door slamming shut"]),
            try XCTUnwrap(arrays["clip_text_hidden"]),
            maxTolerance: 0.025,
            meanTolerance: 0.001,
            name: "CLIP text"
        )
        assertClose(
            conditioner.encodeImages(try XCTUnwrap(arrays["clip_image_nhwc"])),
            try XCTUnwrap(arrays["clip_image_features"]),
            maxTolerance: 0.025,
            meanTolerance: 0.001,
            name: "CLIP image"
        )
    }

    func testMMDiTFlowParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let network = try MMAudioNetwork.load(resources: modelResources())
        let parameters = Dictionary(uniqueKeysWithValues: network.parameters().flattened())
        let expectedFinalWeight = try XCTUnwrap(arrays["network_final_conv_weight_oik"])
            .transposed(0, 2, 1)
        assertClose(
            try XCTUnwrap(parameters["final_layer.conv.weight"]),
            expectedFinalWeight,
            maxTolerance: 0,
            meanTolerance: 0,
            name: "final convolution weight"
        )
        assertClose(
            try XCTUnwrap(parameters["final_layer.conv.bias"]),
            try XCTUnwrap(arrays["network_final_conv_bias"]),
            maxTolerance: 0,
            meanTolerance: 0,
            name: "final convolution bias"
        )
        assertClose(
            try XCTUnwrap(parameters["final_layer.adaLN_modulation.linear.weight"]),
            try XCTUnwrap(arrays["network_final_adaln_weight"]),
            maxTolerance: 0,
            meanTolerance: 0,
            name: "final AdaLN weight"
        )
        assertClose(
            try XCTUnwrap(parameters["final_layer.adaLN_modulation.linear.bias"]),
            try XCTUnwrap(arrays["network_final_adaln_bias"]),
            maxTolerance: 0,
            meanTolerance: 0,
            name: "final AdaLN bias"
        )
        let config = MMAudioGenerationConfig(durationSeconds: 1, steps: 1, guidanceScale: 0)
        let conditions = MMAudioConditionFeatures(
            clip: try XCTUnwrap(arrays["network_clip_ntc"]),
            sync: try XCTUnwrap(arrays["network_sync_ntc"]),
            text: try XCTUnwrap(arrays["network_text_ntc"])
        )
        let projected = network.preprocess(conditions, config: config)
        assertClose(projected.clip, try XCTUnwrap(arrays["network_projected_clip"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "projected CLIP")
        assertClose(projected.sync, try XCTUnwrap(arrays["network_projected_sync"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "projected sync")
        assertClose(projected.text, try XCTUnwrap(arrays["network_projected_text"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "projected text")
        assertClose(projected.clipGlobal, try XCTUnwrap(arrays["network_clip_global"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "CLIP global")
        assertClose(projected.textGlobal, try XCTUnwrap(arrays["network_text_global"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "text global")
        let latent = try XCTUnwrap(arrays["network_latent_ntc"]).asType(.float16)
        let timestep = try XCTUnwrap(arrays["network_timestep"]).asType(.float16)
        let audioProjected = network.audioInputProjection(latent)
        assertClose(audioProjected, try XCTUnwrap(arrays["network_audio_projected"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "audio projection")
        let timestepEmbedding = network.timestepEmbedder(timestep)
        assertClose(timestepEmbedding, try XCTUnwrap(arrays["network_timestep_embedding"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "timestep embedding")
        let globalBase = network.globalConditionMLP(projected.clipGlobal + projected.textGlobal)
        let global = timestepEmbedding.expandedDimensions(axis: 1) + globalBase.expandedDimensions(axis: 1)
        let extended = global + projected.sync
        assertClose(global, try XCTUnwrap(arrays["network_global_condition"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "global condition")
        assertClose(extended, try XCTUnwrap(arrays["network_extended_condition"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "extended condition")
        let rope = MMAudioRoPE.make(length: config.latentSequenceLength, dimensions: 64, frequencyScaling: 1)
        let projection = network.jointBlocks[0].latentBlock.projectAttention(
            audioProjected,
            condition: extended,
            rope: rope
        )
        assertClose(projection.query, try XCTUnwrap(arrays["network_block0_latent_query"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "block 0 query")
        assertClose(projection.key, try XCTUnwrap(arrays["network_block0_latent_key"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "block 0 key")
        assertClose(projection.value, try XCTUnwrap(arrays["network_block0_latent_value"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "block 0 value")
        let clipRoPE = MMAudioRoPE.make(
            length: config.clipSequenceLength,
            dimensions: 64,
            frequencyScaling: Float(config.latentSequenceLength) / Float(config.clipSequenceLength)
        )
        var latentHidden = audioProjected
        var clipHidden = projected.clip
        var textHidden = projected.text
        for (index, block) in network.jointBlocks.enumerated() {
            (latentHidden, clipHidden, textHidden) = block(
                latent: latentHidden,
                clip: clipHidden,
                text: textHidden,
                globalCondition: global,
                extendedCondition: extended,
                latentRoPE: rope,
                clipRoPE: clipRoPE
            )
            assertClose(latentHidden, try XCTUnwrap(arrays["network_joint_\(index)_latent"]), maxTolerance: 0.4, meanTolerance: 0.011, name: "joint \(index) latent")
            assertClose(clipHidden, try XCTUnwrap(arrays["network_joint_\(index)_clip"]), maxTolerance: 0.4, meanTolerance: 0.011, name: "joint \(index) CLIP")
            assertClose(textHidden, try XCTUnwrap(arrays["network_joint_\(index)_text"]), maxTolerance: 0.4, meanTolerance: 0.011, name: "joint \(index) text")
        }
        for (index, block) in network.fusedBlocks.enumerated() {
            latentHidden = block(latentHidden, condition: extended, rope: rope)
            assertClose(latentHidden, try XCTUnwrap(arrays["network_fused_\(index)_latent"]), maxTolerance: 2.1, meanTolerance: 0.06, name: "fused \(index) latent")
        }
        let referenceFinalInput = try XCTUnwrap(arrays["network_fused_13_latent"]).asType(.float16)
        let referenceGlobal = try XCTUnwrap(arrays["network_global_condition"]).asType(.float16)
        assertClose(
            network.finalLayer.adaLN.linear(MLXNN.silu(referenceGlobal)),
            try XCTUnwrap(arrays["network_final_modulation_raw"]),
            maxTolerance: 0.02,
            meanTolerance: 0.002,
            name: "final modulation projection"
        )
        let finalModulation = network.finalLayer.adaLN(referenceGlobal)
        assertClose(finalModulation[0], try XCTUnwrap(arrays["network_final_shift"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "final shift")
        assertClose(finalModulation[1], try XCTUnwrap(arrays["network_final_scale"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "final scale")
        let finalNormalized = MMAudioTensorOps.layerNorm(referenceFinalInput)
        assertClose(finalNormalized, try XCTUnwrap(arrays["network_final_normalized"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "final normalization")
        let finalModulated = finalNormalized * (1 + finalModulation[1]) + finalModulation[0]
        assertClose(finalModulated, try XCTUnwrap(arrays["network_final_modulated"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "final modulation")
        assertClose(network.finalLayer.conv(finalModulated), try XCTUnwrap(arrays["network_final_from_blocks"]), maxTolerance: 0.08, meanTolerance: 0.008, name: "isolated final convolution")
        assertClose(
            network.finalLayer(referenceFinalInput, condition: referenceGlobal),
            try XCTUnwrap(arrays["network_final_from_blocks"]),
            maxTolerance: 0.08,
            meanTolerance: 0.008,
            name: "isolated final block"
        )
        assertClose(network.finalLayer(latentHidden, condition: global), try XCTUnwrap(arrays["network_final_from_blocks"]), maxTolerance: 0.08, meanTolerance: 0.008, name: "final block")
        let actual = network.predictFlow(
            latent,
            timestep: timestep,
            conditions: projected,
            config: config
        )
        assertClose(
            actual,
            try XCTUnwrap(arrays["network_flow_ntc"]),
            maxTolerance: 0.08,
            meanTolerance: 0.008,
            name: "MMDiT flow"
        )
    }

    func testVAEParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let vae = try MMAudioVAE.load(resources: modelResources())
        let latent = try XCTUnwrap(arrays["vae_latent_ntc"])
        let actual = vae.decode(latent)
        let expected = try XCTUnwrap(arrays["vae_spectrogram_ntc"])
        let stages = vae.parityStages(latent)
        assertClose(stages.rawInputWeight, try XCTUnwrap(arrays["vae_input_weight_raw_oik"]).transposed(0, 2, 1), maxTolerance: 0, meanTolerance: 0, name: "VAE raw input weight")
        assertClose(stages.effectiveInputWeight, try XCTUnwrap(arrays["vae_input_weight_effective_oik"]).transposed(0, 2, 1), maxTolerance: 0.000_2, meanTolerance: 0.000_02, name: "VAE effective input weight")
        assertClose(stages.inputProjection, try XCTUnwrap(arrays["vae_input_projection_ntc"]), maxTolerance: 0.02, meanTolerance: 0.002, name: "VAE input projection")
        assertClose(stages.middleFirst, try XCTUnwrap(arrays["vae_middle_first_ntc"]), maxTolerance: 0.03, meanTolerance: 0.003, name: "VAE middle first")
        assertClose(stages.middleAttention, try XCTUnwrap(arrays["vae_middle_attention_ntc"]), maxTolerance: 0.04, meanTolerance: 0.004, name: "VAE middle attention")
        assertClose(stages.middleSecond, try XCTUnwrap(arrays["vae_middle_second_ntc"]), maxTolerance: 0.05, meanTolerance: 0.005, name: "VAE middle second")
        for index in stages.upLevels.indices {
            assertClose(stages.upLevels[index], try XCTUnwrap(arrays["vae_up_\(index)_ntc"]), maxTolerance: 0.08, meanTolerance: 0.008, name: "VAE up \(index)")
        }
        assertClose(stages.preOutput, try XCTUnwrap(arrays["vae_pre_output_ntc"]), maxTolerance: 0.08, meanTolerance: 0.008, name: "VAE pre-output")
        assertClose(stages.rawOutput, try XCTUnwrap(arrays["vae_raw_spectrogram_ntc"]), maxTolerance: 0.08, meanTolerance: 0.008, name: "VAE raw output")
        assertClose(
            actual,
            expected,
            maxTolerance: 0.08,
            meanTolerance: 0.008,
            name: "VAE decode"
        )
    }

    func testBigVGANParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let vocoder = try MMAudioBigVGAN.load(resources: modelResources())
        let input = try XCTUnwrap(arrays["bigvgan_spectrogram_nct"]).transposed(0, 2, 1)
        let expected = try XCTUnwrap(arrays["bigvgan_waveform_nct"]).transposed(0, 2, 1)
        assertClose(
            vocoder(input),
            expected,
            maxTolerance: 0.08,
            meanTolerance: 0.008,
            name: "BigVGAN"
        )
    }

    func testSynchformerParityWhenTraceIsAvailable() throws {
        let arrays = try traceArrays()
        let frames = try XCTUnwrap(arrays["synchformer_frames_tchw"])
        let synchformer = try WooshSynchformer(
            resources: WooshSynchformerResources(rootURL: modelResources().rootURL)
        )
        let actual = try synchformer.extractMMAudioFeatures(
            framesCHW: frames.asArray(Float.self),
            frameCount: frames.dim(0)
        )
        assertClose(
            actual,
            try XCTUnwrap(arrays["synchformer_features_ntc"]),
            maxTolerance: 0.03,
            meanTolerance: 0.003,
            name: "Synchformer"
        )
    }

    private func traceArrays() throws -> [String: MLXArray] {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_MMAUDIO_TRACE"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_MMAUDIO_TRACE to export_mmaudio_trace.py output."
        )
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    private func modelResources() throws -> MMAudioModelResources {
        let path = ProcessInfo.processInfo.environment["MERERUN_TEST_MMAUDIO_ROOT"] ?? ""
        try XCTSkipIf(
            path.isEmpty || !FileManager.default.fileExists(atPath: path),
            "Set MERERUN_TEST_MMAUDIO_ROOT to the installed MMAudio root."
        )
        return MMAudioModelResources(rootURL: URL(fileURLWithPath: path))
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
