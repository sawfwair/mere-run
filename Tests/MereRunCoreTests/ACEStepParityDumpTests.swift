import AudioCodecs
import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class ACEStepParityDumpTests: MereRunCoreTestCase {
    func testDumpPromptConditionFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let decoderRoot = env["MERERUN_TEST_ACESTEP_DECODER_ROOT"], !decoderRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_DECODER_ROOT=/path/to/acestep-v15-turbo.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/vae.")
        }
        guard let textRoot = env["MERERUN_TEST_ACESTEP_TEXT_ROOT"], !textRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_TEXT_ROOT=/path/to/Qwen3-Embedding-0.6B.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
        let sourceURL = outputURL.appendingPathComponent("condition_source_latents_ntc_f32.raw")
        let noiseURL = outputURL.appendingPathComponent("condition_noise_ntc_f32.raw")
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw XCTSkip("Missing source latent fixture: \(sourceURL.path)")
        }
        guard FileManager.default.fileExists(atPath: noiseURL.path) else {
            throw XCTSkip("Missing noise fixture: \(noiseURL.path)")
        }

        let sourceValues = try readFloats(from: sourceURL)
        XCTAssertEqual(sourceValues.count % 64, 0)
        let frames = sourceValues.count / 64
        let noiseValues = try readFloats(from: noiseURL)
        XCTAssertEqual(noiseValues.count, sourceValues.count)

        let pipeline = try ACEStepPipeline(
            decoderResources: ACEStepResources(rootURL: URL(fileURLWithPath: decoderRoot)),
            vaeResources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            textEncoderResources: ACEStep5HzLMResources(rootURL: URL(fileURLWithPath: textRoot)),
            dtype: .float32
        )

        let caption = env["MERERUN_PARITY_CAPTION"] ?? "faithful dream-pop cover with soft vocals, warm guitars, gentle drums, and the original song structure"
        let lyrics = try env["MERERUN_PARITY_LYRICS_FILE"].map {
            try String(contentsOf: URL(fileURLWithPath: $0), encoding: .utf8)
        } ?? "The woman was a dream I had"
        let metadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: nil,
            caption: caption,
            duration: env["MERERUN_PARITY_METADATA_DURATION"] ?? "\(max(1, frames / 25)) seconds",
            keyscale: nil,
            language: nil,
            timesignature: nil
        )
        let instruction = env["MERERUN_PARITY_INSTRUCTION"] ?? "Generate audio semantic tokens based on the given conditions:"
        let sourceLatents = MLXArray(sourceValues, [1, frames, 64]).asType(.float32)
        let noise = MLXArray(noiseValues, [1, frames, 64]).asType(.float32)

        let metas = pipeline.makePromptMetasString(metadata)
        let captionPrompt = pipeline.makeCaptionPrompt(
            instruction: pipeline.formatInstruction(instruction),
            caption: caption,
            metas: metas
        )
        let lyricsPrompt = pipeline.makeLyricsPrompt(lyrics: lyrics, language: ACEStepPipeline.defaultVocalLanguage)
        try writeString(captionPrompt, to: outputURL.appendingPathComponent("swift_condition_caption_prompt.txt"))
        try writeString(lyricsPrompt, to: outputURL.appendingPathComponent("swift_condition_lyrics_prompt.txt"))

        guard let tokenizer = pipeline.conditionTextTokenizer else {
            throw XCTSkip("Pipeline was initialized without condition text tokenizer.")
        }
        let captionTokens = pipeline.tokenizePrompt(captionPrompt, maxTokens: 256, tokenizer: tokenizer)
        let lyricTokens = pipeline.tokenizePrompt(lyricsPrompt, maxTokens: 2048, tokenizer: tokenizer)
        try writeInt32(captionTokens.ids, to: outputURL.appendingPathComponent("swift_condition_caption_token_ids_i32.raw"))
        try writeInt32(captionTokens.mask, to: outputURL.appendingPathComponent("swift_condition_caption_mask_i32.raw"))
        try writeInt32(lyricTokens.ids, to: outputURL.appendingPathComponent("swift_condition_lyric_token_ids_i32.raw"))
        try writeInt32(lyricTokens.mask, to: outputURL.appendingPathComponent("swift_condition_lyric_mask_i32.raw"))

        let inputs = try pipeline.preparePromptConditionInputs(
            caption: caption,
            lyrics: lyrics,
            srcLatents: sourceLatents,
            chunkChannels: pipeline.chunkChannelsForPromptConditioning(),
            lmUserMetadata: metadata,
            audioCoverStrength: 0.85,
            vocalLanguage: ACEStepPipeline.defaultVocalLanguage,
            instruction: instruction,
            isCover: true
        )
        try dumpTensor(inputs.textHiddenStates, name: "swift_condition_text_hidden_bld_f32", to: outputURL)
        try dumpTensor(inputs.textAttentionMask.asType(DType.float32), name: "swift_condition_text_mask_f32", to: outputURL)
        try dumpTensor(inputs.lyricHiddenStates, name: "swift_condition_lyric_hidden_bld_f32", to: outputURL)
        try dumpTensor(inputs.lyricAttentionMask.asType(DType.float32), name: "swift_condition_lyric_mask_f32", to: outputURL)
        try dumpTensor(inputs.referAudioAcousticHiddenStatesPacked, name: "swift_condition_refer_latents_ntc_f32", to: outputURL)
        try dumpTensor(inputs.referAudioOrderMask.asType(DType.float32), name: "swift_condition_refer_order_f32", to: outputURL)
        try dumpTensor(inputs.srcLatents, name: "swift_condition_src_latents_ntc_f32", to: outputURL)
        try dumpTensor(inputs.chunkMasks, name: "swift_condition_chunk_masks_ntc_f32", to: outputURL)

        let tokenized = pipeline.tokenizeForLMHints(
            hiddenStates: inputs.hiddenStates ?? inputs.srcLatents,
            attentionMask: inputs.attentionMask ?? MLXArray.ones([1, frames], dtype: .int32),
            silenceLatent: inputs.silenceLatent
        )
        try dumpTensor(tokenized.quantized5Hz, name: "swift_condition_lm_hints_5hz_btd_f32", to: outputURL)
        try dumpTensor(tokenized.indices.asType(DType.float32), name: "swift_condition_audio_code_indices_f32", to: outputURL)
        try dumpTensor(tokenized.pooledMask5Hz.asType(DType.float32), name: "swift_condition_pooled_mask_5hz_f32", to: outputURL)
        let lmHints25Hz = pipeline.detokenizer(tokenized.quantized5Hz)
        try dumpTensor(lmHints25Hz, name: "swift_condition_lm_hints_25hz_ntc_f32", to: outputURL)

        let prepared = pipeline.prepareCondition(
            textHiddenStates: inputs.textHiddenStates,
            textAttentionMask: inputs.textAttentionMask,
            lyricHiddenStates: inputs.lyricHiddenStates,
            lyricAttentionMask: inputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked: inputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: inputs.referAudioOrderMask,
            hiddenStates: inputs.hiddenStates ?? inputs.srcLatents,
            attentionMask: inputs.attentionMask ?? MLXArray.ones([1, frames], dtype: .int32),
            silenceLatent: inputs.silenceLatent,
            srcLatents: inputs.srcLatents,
            chunkMasks: inputs.chunkMasks,
            isCovers: inputs.isCovers
        )
        try dumpTensor(prepared.encoderHiddenStates, name: "swift_condition_encoder_hidden_bld_f32", to: outputURL)
        try dumpTensor(prepared.encoderAttentionMask.asType(DType.float32), name: "swift_condition_encoder_mask_f32", to: outputURL)
        try dumpTensor(prepared.contextLatents, name: "swift_condition_context_latents_ntc_f32", to: outputURL)
        try dumpTensor(prepared.encoderHiddenStates, name: "swift_full_cover_encoder_hidden_bld_f32", to: outputURL)
        try dumpTensor(prepared.contextLatents, name: "swift_full_cover_context_latents_ntc_f32", to: outputURL)

        let nonCoverPrepared = pipeline.prepareNonCoverConditionIfNeeded(conditionInputs: inputs)
        if let nonCoverTextHiddenStates = inputs.nonCoverTextHiddenStates {
            try dumpTensor(nonCoverTextHiddenStates, name: "swift_full_noncover_text_hidden_bld_f32", to: outputURL)
        }
        if let nonCoverPrepared {
            try dumpTensor(nonCoverPrepared.encoderHiddenStates, name: "swift_full_noncover_encoder_hidden_bld_f32", to: outputURL)
            try dumpTensor(nonCoverPrepared.contextLatents, name: "swift_full_noncover_context_latents_ntc_f32", to: outputURL)
        }

        let timestep = MLXArray([Float(1.0)]).asType(.float32)
        let vt = pipeline.decoder(
            hiddenStates: noise,
            timestep: timestep,
            timestepR: timestep,
            encoderHiddenStates: prepared.encoderHiddenStates,
            encoderAttentionMask: prepared.encoderAttentionMask,
            contextLatents: prepared.contextLatents
        )
        try dumpTensor(vt, name: "swift_condition_first_vt_ntc_f32", to: outputURL)

        let fullLatents = pipeline.denoiseTurbo(
            noise: noise,
            timesteps: ACEStepTurboScheduler(fixNFE: 8, shift: 1.0, timesteps: nil).timesteps,
            inferMethod: .ode,
            encoderHiddenStates: prepared.encoderHiddenStates,
            encoderAttentionMask: prepared.encoderAttentionMask,
            contextLatents: prepared.contextLatents,
            nonCoverEncoderHiddenStates: nonCoverPrepared?.encoderHiddenStates,
            nonCoverEncoderAttentionMask: nonCoverPrepared?.encoderAttentionMask,
            nonCoverContextLatents: nonCoverPrepared?.contextLatents,
            audioCoverStrength: 0.85,
            dcwEnabled: true,
            dcwMode: .double,
            dcwScaler: 0.05,
            dcwHighScaler: 0.02
        )
        try dumpTensor(fullLatents, name: "swift_full_final_latents_ntc_f32", to: outputURL)
        let decodedAudio = pipeline.vae.decode(fullLatents).asType(.float32)
        try dumpTensor(decodedAudio, name: "swift_full_decoded_audio_ntc_f32", to: outputURL)
    }

    func testDumpSmallVAEDecodeFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        let fixtureURL = outputURL.appendingPathComponent("vae_decode_fixture_latents_ntc_f32.raw")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("Missing VAE decode fixture: \(fixtureURL.path)")
        }

        let latents = MLXArray(try readFloats(from: fixtureURL), [1, 3, 64]).asType(.float32)
        let vae = try OobleckVAECheckpointLoader.loadVAE(
            resources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            dtype: .float32
        )
        let decoded = vae.decode(latents).asType(.float32)
        try dumpTensor(decoded, name: "swift_vae_decode_fixture_audio_ntc_f32", to: outputURL)
    }

    func testDumpVAEDecodeLatentsFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae.")
        }
        guard let latentsPath = env["MERERUN_PARITY_DECODE_LATENTS"], !latentsPath.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DECODE_LATENTS=/path/to/latents_ntc_f32.raw.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        let latentsURL = URL(fileURLWithPath: latentsPath)
        let values = try readFloats(from: latentsURL)
        XCTAssertEqual(values.count % 64, 0)

        let frames = values.count / 64
        let latents = MLXArray(values, [1, frames, 64]).asType(.float32)
        let vae = try OobleckVAECheckpointLoader.loadVAE(
            resources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            dtype: .float32
        )
        let decoded = vae.tiledDecode(latents).asType(.float32)
        try dumpTensor(decoded, name: "swift_vae_decode_latents_fixture_audio_ntc_f32", to: outputURL)
    }

    func testDumpSmallVAEEncodeFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        let fixtureURL = outputURL.appendingPathComponent("vae_encode_fixture_audio_ntc_f32.raw")
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("Missing VAE encode fixture: \(fixtureURL.path)")
        }

        let audio = MLXArray(try readFloats(from: fixtureURL), [1, 5_760, 2]).asType(.float32)
        let vae = try OobleckVAECheckpointLoader.loadVAE(
            resources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            dtype: .float32
        )
        let latents = vae.encode(audio, sample: false).asType(.float32)
        try dumpTensor(latents, name: "swift_vae_encode_fixture_mean_ntc_f32", to: outputURL)
    }

    func testDumpSmallDiTFixture() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let decoderRoot = env["MERERUN_TEST_ACESTEP_DECODER_ROOT"], !decoderRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_DECODER_ROOT=/path/to/acestep-v15-turbo.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        let hiddenURL = outputURL.appendingPathComponent("dit_fixture_hidden_btc_f32.raw")
        let contextURL = outputURL.appendingPathComponent("dit_fixture_context_btc_f32.raw")
        let encoderURL = outputURL.appendingPathComponent("dit_fixture_encoder_bld_f32.raw")
        let timestepURL = outputURL.appendingPathComponent("dit_fixture_timestep_f32.raw")
        for fixtureURL in [hiddenURL, contextURL, encoderURL, timestepURL] {
            guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
                throw XCTSkip("Missing DiT fixture: \(fixtureURL.path)")
            }
        }

        let bundle = try ACEStepCheckpointLoader.loadDecoderBundle(
            resources: ACEStepResources(rootURL: URL(fileURLWithPath: decoderRoot)),
            dtype: .float32
        )
        let hidden = MLXArray(try readFloats(from: hiddenURL), [1, 4, 64]).asType(.float32)
        let context = MLXArray(try readFloats(from: contextURL), [1, 4, 128]).asType(.float32)
        let encoder = MLXArray(try readFloats(from: encoderURL), [1, 7, 2_048]).asType(.float32)
        let timestep = MLXArray(try readFloats(from: timestepURL), [1]).asType(.float32)
        let output = bundle.decoder(
            hiddenStates: hidden,
            timestep: timestep,
            timestepR: timestep,
            encoderHiddenStates: encoder,
            encoderAttentionMask: nil,
            contextLatents: context
        ).asType(.float32)

        try dumpTensor(output, name: "swift_dit_fixture_output_btc_f32", to: outputURL)
    }

    func testDumpSourceAudioAndVAELatents() throws {
        let env = ProcessInfo.processInfo.environment
        guard let outputRoot = env["MERERUN_PARITY_DUMP_DIR"], !outputRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_DUMP_DIR=/tmp/acestep-parity to dump ACE-Step parity tensors.")
        }
        guard let sourceAudio = env["MERERUN_PARITY_SOURCE_AUDIO"], !sourceAudio.isEmpty else {
            throw XCTSkip("Set MERERUN_PARITY_SOURCE_AUDIO=/path/to/source.mp3 to dump ACE-Step parity tensors.")
        }
        guard let vaeRoot = env["MERERUN_TEST_ACESTEP_VAE_ROOT"], !vaeRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ACESTEP_VAE_ROOT=/path/to/ACE-Step-1.5/checkpoints/vae.")
        }

        let outputURL = URL(fileURLWithPath: outputRoot)
        try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let sourceURL = URL(fileURLWithPath: sourceAudio).standardizedFileURL
        let buffer = try AudioReader.readAudioBuffer(from: sourceURL, sampleRate: 48_000, channels: 2)
        XCTAssertTrue(buffer.isInterleaved)
        XCTAssertEqual(buffer.channelCount, 2)

        let frames = buffer.samples.count / buffer.channelCount
        let trimmedSamples = Array(buffer.samples.prefix(frames * buffer.channelCount))
        let interleaved = trimmedSamples.map { sample -> Float in
            guard sample.isFinite else {
                return 0
            }
            return max(-1, min(1, sample))
        }

        var channelsFirst = [Float](repeating: 0, count: frames * buffer.channelCount)
        for frame in 0..<frames {
            channelsFirst[frame] = interleaved[frame * 2]
            channelsFirst[frames + frame] = interleaved[frame * 2 + 1]
        }
        try writeFloats(channelsFirst, to: outputURL.appendingPathComponent("swift_source_audio_csf32.raw"))

        let vaeSeconds = Float(env["MERERUN_PARITY_VAE_SECONDS"] ?? "30") ?? 30
        let vaeFrames = max(1, min(frames, Int((vaeSeconds * 48_000).rounded())))
        let vaeSamples = Array(interleaved.prefix(vaeFrames * buffer.channelCount))
        let audio = MLXArray(vaeSamples, [1, vaeFrames, buffer.channelCount]).asType(.float32)
        let sourceSuffix = "\(Int(vaeSeconds.rounded()))s"
        let vae = try OobleckVAECheckpointLoader.loadVAE(
            resources: OobleckVAEResources(rootURL: URL(fileURLWithPath: vaeRoot)),
            dtype: .float32
        )
        let meanF32 = vae.tiledEncode(audio, sample: false).asType(.float32)
        try dumpTensor(meanF32, name: "swift_source_latents_mean_f32_input_\(sourceSuffix)_ntc_f32", to: outputURL)

        let shouldSkipBF16 = env["MERERUN_PARITY_SKIP_BF16_SOURCE_ENCODE"] == "1"
        let meanBF16Input: MLXArray?
        if shouldSkipBF16 {
            meanBF16Input = nil
        } else {
            let encoded = vae.tiledEncode(audio.asType(.bfloat16), sample: false).asType(.float32)
            try dumpTensor(encoded, name: "swift_source_latents_mean_bf16_input_\(sourceSuffix)_ntc_f32", to: outputURL)
            meanBF16Input = encoded
        }

        let stats = DumpStats(
            sourceAudio: stats(for: channelsFirst, shape: [buffer.channelCount, frames]),
            meanF32Input: stats(for: meanF32),
            meanBF16Input: meanBF16Input.map(stats(for:))
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(stats).write(to: outputURL.appendingPathComponent("swift_dump_stats.json"), options: [.atomic])
    }

    private func dumpTensor(_ tensor: MLXArray, name: String, to outputURL: URL) throws {
        let values = tensor.asType(.float32).reshaped(-1).asArray(Float.self)
        try writeFloats(values, to: outputURL.appendingPathComponent("\(name).raw"))
    }

    private func writeFloats(_ values: [Float], to url: URL) throws {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url, options: [.atomic])
    }

    private func writeInt32(_ values: [Int32], to url: URL) throws {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url, options: [.atomic])
    }

    private func writeString(_ value: String, to url: URL) throws {
        try value.write(to: url, atomically: true, encoding: .utf8)
    }

    private func readFloats(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }

    private func stats(for tensor: MLXArray) -> TensorStats {
        stats(for: tensor.asType(.float32).reshaped(-1).asArray(Float.self), shape: tensor.shape)
    }

    private func stats(for values: [Float], shape: [Int]) -> TensorStats {
        var sum: Double = 0
        var sumSquares: Double = 0
        var minValue = Float.infinity
        var maxValue = -Float.infinity
        for value in values {
            sum += Double(value)
            sumSquares += Double(value * value)
            minValue = min(minValue, value)
            maxValue = max(maxValue, value)
        }
        let count = max(1, values.count)
        let mean = sum / Double(count)
        let rms = sqrt(sumSquares / Double(count))
        let variance = max(0, (sumSquares / Double(count)) - mean * mean)
        return TensorStats(
            shape: shape,
            mean: mean,
            std: sqrt(variance),
            rms: rms,
            min: Double(minValue),
            max: Double(maxValue)
        )
    }

    private struct DumpStats: Codable {
        var sourceAudio: TensorStats
        var meanF32Input: TensorStats
        var meanBF16Input: TensorStats?
    }

    private struct TensorStats: Codable {
        var shape: [Int]
        var mean: Double
        var std: Double
        var rms: Double
        var min: Double
        var max: Double
    }
}
