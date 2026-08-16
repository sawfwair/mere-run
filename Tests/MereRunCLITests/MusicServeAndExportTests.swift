import ArgumentParser
import Foundation
import MLX
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class MusicServeAndExportTests: XCTestCase {
    func testMusicCommandExposesResidentServe() {
        let names = Set(Music.configuration.subcommands.map {
            $0.configuration.commandName
        })
        XCTAssertTrue(names.contains("serve"))
        XCTAssertTrue(names.contains("train-adapter"))
    }

    func testMusicServeParsesResidentAdapters() throws {
        let command = try MusicServe.parse([
            "--adapter", "/tmp/style.safetensors", "/tmp/voice.safetensors",
            "--adapter-kind", "lora",
            "--adapter-scale", "0.8", "0.5",
        ])

        XCTAssertEqual(
            command.adapters,
            ["/tmp/style.safetensors", "/tmp/voice.safetensors"]
        )
        XCTAssertEqual(command.adapterKind, .lora)
        XCTAssertEqual(command.adapterScales, [0.8, 0.5])
    }

    func testMusicServeParsesMiniMaxResidentSpeechAPI() throws {
        let command = try MusicServe.parse([
            "--model", MiniMaxMusic3Resources.modelID,
            "--memory-mode", "resident",
            "--performance-mode", "q4",
            "--port", "8091",
        ])

        XCTAssertEqual(command.model, MiniMaxMusic3Resources.modelID)
        XCTAssertEqual(command.miniMaxLoadingStrategy, .resident)
        XCTAssertEqual(command.miniMaxPerformanceMode, .q4)
        XCTAssertEqual(command.port, 8_091)
    }

    func testMiniMaxSpeechRequestDecodesReferenceAndNativeControls() throws {
        let request = try JSONDecoder().decode(
            MiniMaxMusic3SpeechRequest.self,
            from: Data(
                """
                {
                  "model": "MiniMaxAI/MiniMax-Music3",
                  "input": "[Verse]\\nwe glow",
                  "instructions": "cinematic synth-pop",
                  "response_format": "wav",
                  "seed": 7,
                  "max_new_tokens": 750,
                  "min_new_tokens": 500,
                  "stream": false,
                  "audio_duration": 30,
                  "minimum_audio_duration": 20,
                  "sampling_tier": "draft",
                  "flow_strategy": "overlap-average",
                  "seed_strategy": "stage-separated-v1",
                  "num_inference_steps": 24,
                  "guidance_scale": 1.7,
                  "sample_rate": 44100
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.maxNewTokens, 750)
        XCTAssertEqual(request.minNewTokens, 500)
        XCTAssertEqual(request.audioDuration, 30)
        XCTAssertEqual(request.minimumAudioDuration, 20)
        XCTAssertEqual(request.samplingTier, .draft)
        XCTAssertEqual(request.flowStrategy, .overlapAverage)
        XCTAssertEqual(request.seedStrategy, .stageSeparatedV1)
        XCTAssertEqual(request.numInferenceSteps, 24)
        XCTAssertEqual(request.guidanceScale, 1.7)
        XCTAssertEqual(request.sampleRate, 44_100)
        XCTAssertEqual(request.stream, false)
    }

    func testMusicTrainAdapterParsesAndLoadsJSONLManifest() throws {
        let command = try MusicTrainAdapter.parse([
            "--dataset", "/tmp/music.jsonl",
            "--output", "/tmp/style.safetensors",
            "--kind", "lokr",
            "--rank", "16",
            "--alpha", "32",
            "--factor", "8",
            "--steps", "25",
            "--learning-rate", "0.0002",
            "--max-duration", "12",
        ])
        XCTAssertEqual(command.kind, .lokr)
        XCTAssertEqual(command.rank, 16)
        XCTAssertEqual(command.alpha, 32)
        XCTAssertEqual(command.factor, 8)
        XCTAssertEqual(command.steps, 25)
        XCTAssertEqual(command.learningRate, 0.0002)
        XCTAssertEqual(command.maxDurationSeconds, 12)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let manifestURL = directory.appendingPathComponent("music.jsonl")
        try """
        {"audio":"one.wav","caption":"glassy dream pop","lyrics":"hello"}
        {"audio":"two.wav","caption":"heavy breakbeat"}
        """.write(to: manifestURL, atomically: true, encoding: .utf8)

        let records = try MusicTrainAdapter.loadManifest(from: manifestURL)
        XCTAssertEqual(records.count, 2)
        XCTAssertEqual(records[0].lyrics, "hello")
        XCTAssertEqual(records[1].audio, "two.wav")
    }

    func testListeningRegressionFixturePinsDiverseCases() throws {
        let fixtureURL = try XCTUnwrap(Bundle.module.resourceURL)
            .appendingPathComponent(
                "Fixtures/ACEStep/listening-regression.json"
            )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: fixtureURL)
            ) as? [String: Any]
        )
        XCTAssertEqual(object["schema_version"] as? Int, 1)
        XCTAssertEqual(
            object["model"] as? String,
            "music-acestep-xl-turbo-lm4b"
        )
        let cases = try XCTUnwrap(object["cases"] as? [[String: Any]])
        XCTAssertEqual(cases.count, 5)
        XCTAssertEqual(Set(cases.compactMap { $0["seed"] as? Int }).count, 5)
        XCTAssertTrue(
            cases.allSatisfy {
                ($0["listen_for"] as? String)?.isEmpty == false
            }
        )
    }

    func testRecipeSchemaPersistsEffectiveConditioningMetadata() throws {
        let metadata = ACEStepRecipeConditioningMetadata(
            .init(
                bpm: "118",
                caption: "not duplicated here",
                duration: "12",
                keyscale: "D major",
                language: "en",
                timesignature: "4"
            )
        )
        let encoded = try JSONEncoder().encode(metadata)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: String]
        )

        XCTAssertEqual(ACEStepGenerationRecipe.currentSchemaVersion, 5)
        XCTAssertEqual(object["bpm"], "118")
        XCTAssertEqual(object["duration"], "12")
        XCTAssertEqual(object["keyscale"], "D major")
        XCTAssertEqual(object["language"], "en")
        XCTAssertEqual(object["timesignature"], "4")
        XCTAssertNil(object["caption"])
    }

    func testRecipeSchemaPersistsLanguageModelSampling() throws {
        let sampling = ACEStepRecipeLMSampling(
            temperature: 0.7,
            topK: 32,
            topP: 0.85,
            repetitionPenalty: 1.08,
            cfgScale: 2,
            negativePrompt: "NO USER INPUT",
            useCotCaption: false
        )
        let encoded = try JSONEncoder().encode(sampling)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )

        XCTAssertEqual(
            try XCTUnwrap(object["temperature"] as? Double),
            0.7,
            accuracy: 0.0001
        )
        XCTAssertEqual(object["top_k"] as? Int, 32)
        XCTAssertEqual(
            try XCTUnwrap(object["top_p"] as? Double),
            0.85,
            accuracy: 0.0001
        )
        XCTAssertEqual(
            try XCTUnwrap(object["cfg_scale"] as? Double),
            2,
            accuracy: 0.0001
        )
        XCTAssertEqual(object["negative_prompt"] as? String, "NO USER INPUT")
        XCTAssertEqual(object["use_cot_caption"] as? Bool, false)
        XCTAssertEqual(
            try XCTUnwrap(object["repetition_penalty"] as? Double),
            1.08,
            accuracy: 0.0001
        )
    }

    func testRecipePlannerProvenanceFindsBundledAndIndependentModels() throws {
        let bundled = ACEStepGenerationRecipe.languageModelProvenance(
            source: "acestep-5Hz-lm-4B",
            subdirectory: "acestep-5Hz-lm-4B",
            checkpointModelID: ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
            checkpointManifest: nil
        )
        XCTAssertEqual(bundled.count, 1)
        XCTAssertEqual(bundled[0].repository, "ACE-Step/acestep-5Hz-lm-4B")
        XCTAssertEqual(bundled[0].destinationPath, "acestep-5Hz-lm-4B")

        let independent = ACEStepGenerationRecipe.languageModelProvenance(
            source: ModelResolver.ModelID.aceStepLM17B.rawValue,
            subdirectory: "acestep-5Hz-lm-1.7B",
            checkpointModelID: ModelResolver.ModelID.aceStepXLSFT.rawValue,
            checkpointManifest: nil
        )
        XCTAssertEqual(independent.count, 1)
        XCTAssertEqual(independent[0].repository, "ACE-Step/Ace-Step1.5")
        XCTAssertNil(independent[0].destinationPath)
    }

    func testMusicServeParsesSecureRuntimeOptions() throws {
        let command = try MusicServe.parse([
            "--host", "0.0.0.0",
            "--port", "8089",
            "--model", "music-acestep-xl-sft",
            "--api-key", "secret",
        ])
        XCTAssertEqual(command.host, "0.0.0.0")
        XCTAssertEqual(command.port, 8_089)
        XCTAssertEqual(command.model, "music-acestep-xl-sft")
        XCTAssertNil(command.lmModel)
        XCTAssertEqual(command.apiKey, "secret")
        XCTAssertTrue(MusicServe.isLoopback("127.0.0.1"))
        XCTAssertTrue(MusicServe.isLoopback("::1"))
        XCTAssertFalse(MusicServe.isLoopback("0.0.0.0"))
        XCTAssertEqual(
            MusicServe.apiErrorMessage(
                ValidationError("resident model mismatch")
            ),
            "resident model mismatch"
        )
    }

    func testMusicServeParsesIndependentPlannerModel() throws {
        let command = try MusicServe.parse([
            "--model", ModelResolver.ModelID.aceStepXLSFT.rawValue,
            "--lm-model", ModelResolver.ModelID.aceStepLM17B.rawValue,
        ])

        XCTAssertEqual(command.model, ModelResolver.ModelID.aceStepXLSFT.rawValue)
        XCTAssertEqual(command.lmModel, ModelResolver.ModelID.aceStepLM17B.rawValue)
    }

    func testMusicAPIRequestDecodesSnakeCaseAdvancedControls() throws {
        let request = try JSONDecoder().decode(
            MusicAPIGenerationRequest.self,
            from: Data(
                """
                {
                  "prompt": "glassy synth pop",
                  "instruction": "custom semantic instruction",
                  "duration_seconds": 12,
                  "quality": "final",
                  "task": "text2music",
                  "seed": 42,
                  "candidates": 3,
                  "shift": 1.5,
                  "infer_method": "sde",
                  "guidance_scale": 6.5,
                  "guidance_mode": "apg",
                  "cfg_interval_start": 0.2,
                  "cfg_interval_end": 0.8,
                  "velocity_norm_threshold": 2.5,
                  "velocity_ema_factor": 0.15,
                  "sampler": "heun",
                  "use_lm": true,
                  "lm_top_k": 50,
                  "lm_top_p": 0.85,
                  "lm_temperature": 0.7,
                  "lm_repetition_penalty": 1.08,
                  "lm_cfg_scale": 2.5,
                  "lm_negative_prompt": "muddy mix",
                  "instrumental": true,
                  "bpm": 118,
                  "keyscale": "D major",
                  "metadata_language": "en",
                  "time_signature": "4",
                  "vocal_language": "en",
                  "reference_audio_paths": ["/tmp/ref.wav"],
                  "audio_cover_strength": 0.75,
                  "cover_noise_strength": 0.1,
                  "complete_track_classes": ["Drums", "Bass"],
                  "chunk_mask_mode": "explicit",
                  "repaint_mode": "conservative",
                  "repaint_strength": 0.25,
                  "use_tiled_vae_decode": true,
                  "vae_chunk_size": 256,
                  "vae_overlap": 32,
                  "response_format": "wav"
                }
                """.utf8
            )
        )

        XCTAssertEqual(request.instruction, "custom semantic instruction")
        XCTAssertEqual(request.durationSeconds, 12)
        XCTAssertEqual(request.quality, .final)
        XCTAssertEqual(request.task, .textToMusic)
        XCTAssertEqual(request.seed, 42)
        XCTAssertEqual(request.candidates, 3)
        XCTAssertEqual(request.shift, 1.5)
        XCTAssertEqual(request.inferMethod, .sde)
        XCTAssertEqual(request.guidanceScale, 6.5)
        XCTAssertEqual(request.guidanceMode, .apg)
        XCTAssertEqual(request.cfgIntervalStart, 0.2)
        XCTAssertEqual(request.cfgIntervalEnd, 0.8)
        XCTAssertEqual(request.velocityNormThreshold, 2.5)
        XCTAssertEqual(request.velocityEMAFactor, 0.15)
        XCTAssertEqual(request.sampler, .heun)
        XCTAssertEqual(request.useLanguageModel, true)
        XCTAssertEqual(request.lmTopK, 50)
        XCTAssertEqual(request.lmTopP, 0.85)
        XCTAssertEqual(request.lmTemperature, 0.7)
        XCTAssertEqual(request.lmRepetitionPenalty, 1.08)
        XCTAssertEqual(request.lmCFGScale, 2.5)
        XCTAssertEqual(request.lmNegativePrompt, "muddy mix")
        XCTAssertEqual(request.instrumental, true)
        XCTAssertEqual(request.bpm, 118)
        XCTAssertEqual(request.keyscale, "D major")
        XCTAssertEqual(request.metadataLanguage, "en")
        XCTAssertEqual(request.timeSignature, "4")
        XCTAssertEqual(request.vocalLanguage, "en")
        XCTAssertEqual(request.referenceAudioPaths, ["/tmp/ref.wav"])
        XCTAssertEqual(request.audioCoverStrength, 0.75)
        XCTAssertEqual(request.coverNoiseStrength, 0.1)
        XCTAssertEqual(request.completeTrackClasses, ["Drums", "Bass"])
        XCTAssertEqual(request.chunkMaskMode, .explicit)
        XCTAssertEqual(request.repaintMode, .conservative)
        XCTAssertEqual(request.repaintStrength, 0.25)
        XCTAssertEqual(request.useTiledVAEDecode, true)
        XCTAssertEqual(request.vaeChunkSize, 256)
        XCTAssertEqual(request.vaeOverlap, 32)
        XCTAssertEqual(request.responseFormat, "wav")
    }

    func testHighQualityWAVFormatsHaveTruthfulHeadersAndSizes() throws {
        let audio = MLXArray([
            Float(0.25), Float(-0.25),
            Float(0.5), Float(-0.5),
            Float(0.75), Float(-0.75),
        ], [1, 3, 2])

        for (format, bits, bytesPerSample) in [
            (ACEStepAudioFormat.pcm16, UInt16(16), 2),
            (.pcm24, UInt16(24), 3),
            (.float32, UInt16(32), 4),
        ] {
            let data = try ACEStepWAVWriter.wavData(
                audio,
                sampleRate: 48_000,
                options: .init(
                    format: format,
                    normalization: .none,
                    targetPeakDB: -1,
                    fadeInMilliseconds: 0,
                    fadeOutMilliseconds: 0,
                    dither: false
                )
            )
            XCTAssertEqual(String(data: data[0..<4], encoding: .utf8), "RIFF")
            XCTAssertEqual(readUInt16(data, offset: 34), bits)
            XCTAssertEqual(data.count, 44 + 6 * bytesPerSample)
        }
    }

    func testStereoResamplingPreservesChannelOrderAndDuration() throws {
        let audio = MLXArray(
            [Float(0), 10, 1, 11, 2, 12, 3, 13],
            [1, 4, 2]
        )
        let resampled = try ACEStepWAVWriter.resample(
            audio,
            from: 4,
            to: 2
        )

        XCTAssertEqual(resampled.shape, [1, 2, 2])
        XCTAssertEqual(
            resampled.reshaped(-1).asArray(Float.self),
            [0, 10, 2, 12]
        )
    }

    func testDAWBundleContainsPortableAudioRecipeMarkersAndProject() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        let mixURL = root.appendingPathComponent("source.wav")
        let recipeURL = root.appendingPathComponent("source.recipe.json")
        let lrcURL = root.appendingPathComponent("source.lrc")
        let audio = MLXArray(
            [Float](repeating: 0.1, count: 960),
            [1, 480, 2]
        )
        try ACEStepWAVWriter.writeWAV(
            audio,
            to: mixURL,
            sampleRate: 48_000
        )
        try Data("{}".utf8).write(to: recipeURL)
        let lrc = ACEStepLRCDocument(
            lines: [.init(timestampSeconds: 0, text: "hello")]
        )
        try lrc.rendered().write(
            to: lrcURL,
            atomically: true,
            encoding: .utf8
        )
        let bundleURL = root.appendingPathComponent("bundle", isDirectory: true)

        try ACEStepDAWBundleWriter.write(
            directory: bundleURL,
            mixURL: mixURL,
            recipeURL: recipeURL,
            lrcURL: lrcURL,
            candidates: [.init(name: "Candidate 1", audio: audio)],
            stems: [.init(name: "Drums", audio: audio)],
            lrc: lrc,
            exportOptions: .init()
        )

        for path in [
            "audio/mix.wav",
            "audio/candidate-1-candidate-1.wav",
            "audio/stem-1-drums.wav",
            "recipe.json",
            "lyrics.lrc",
            "markers.csv",
            "mere-run-session.rpp",
            "README.md",
        ] {
            XCTAssertTrue(
                FileManager.default.fileExists(
                    atPath: bundleURL.appendingPathComponent(path).path
                ),
                path
            )
        }
    }

    private func readUInt16(_ data: Data, offset: Int) -> UInt16 {
        UInt16(data[offset]) | UInt16(data[offset + 1]) << 8
    }
}
