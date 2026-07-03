import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageTrainLoRACommandParsingTests: XCTestCase {
    func testImageCommandExposesTrainLoRA() {
        let commandNames = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("train-lora"))
        XCTAssertTrue(commandNames.contains("visualize-run"))
    }

    func testTrainLoRAParsesDefaults() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/krea-dataset",
            "--output", "/tmp/krea-style.safetensors",
        ])

        XCTAssertEqual(cmd.data, "/tmp/krea-dataset")
        XCTAssertEqual(cmd.output, "/tmp/krea-style.safetensors")
        XCTAssertNil(cmd.model)
        XCTAssertEqual(cmd.width, 1024)
        XCTAssertEqual(cmd.height, 1024)
        XCTAssertEqual(cmd.trainingSteps, 1000)
        XCTAssertEqual(cmd.batchSize, 1)
        XCTAssertEqual(cmd.learningRate, 1e-4)
        XCTAssertEqual(cmd.rank, 16)
        XCTAssertNil(cmd.alpha)
        XCTAssertEqual(cmd.maxTextLength, 512)
        XCTAssertEqual(cmd.schedulerSteps, 1000)
        XCTAssertEqual(cmd.captionDropout, 0.05)
        XCTAssertEqual(cmd.seed, 0)
        XCTAssertFalse(cmd.lite)
        XCTAssertFalse(cmd.excludePreviewImages)
        XCTAssertNil(cmd.checkpointInterval)
        XCTAssertNil(cmd.maxResolution)
        XCTAssertFalse(cmd.progressive)
        XCTAssertFalse(cmd.lowRam)
        XCTAssertFalse(cmd.noCompile)
        XCTAssertFalse(cmd.gradientCheckpointing)
        XCTAssertNil(cmd.recipe)
        XCTAssertNil(cmd.benchmarkSteps)
        XCTAssertEqual(cmd.benchmarkWarmupSteps, 5)
        XCTAssertNil(cmd.sampleInterval)
        XCTAssertNil(cmd.samplePrompt)
        XCTAssertNil(cmd.sampleModel)
        XCTAssertEqual(cmd.sampleSteps, 8)
        XCTAssertEqual(cmd.sampleGuidanceScale, 1.0)
        XCTAssertEqual(cmd.sampleLoRAScale, 1.0)
        XCTAssertNil(cmd.sampleSeed)
        XCTAssertFalse(cmd.visualize)
        XCTAssertEqual(cmd.visualizePort, 8787)
        XCTAssertFalse(cmd.preflight)
        XCTAssertFalse(cmd.json)
        XCTAssertNil(cmd.loraTargetRanks)
        XCTAssertNil(cmd.loraRankPreset)
        XCTAssertNil(cmd.loraTargetPreset)
        XCTAssertNil(cmd.loraTargetMode)
        XCTAssertNil(cmd.timestepSampling)
        XCTAssertNil(cmd.timestepLossWeighting)
        XCTAssertNil(cmd.lossWeighting)
        XCTAssertNil(cmd.timestepLow)
        XCTAssertNil(cmd.timestepHigh)
        XCTAssertNil(cmd.lrWarmupSteps)
        XCTAssertFalse(cmd.noCosineScheduler)
        XCTAssertNil(cmd.lrMinFactor)
        XCTAssertNil(cmd.adamWeightDecay)
        XCTAssertNil(cmd.syntheticSamples)
    }

    func testTrainLoRAParsesOverrides() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/data",
            "--output", "/tmp/out.safetensors",
            "--model", "image-krea2-raw",
            "--width", "768",
            "--height", "1024",
            "--steps", "25",
            "--batch-size", "2",
            "--learning-rate", "0.0002",
            "--rank", "8",
            "--alpha", "4",
            "--max-text-length", "384",
            "--scheduler-steps", "500",
            "--caption-dropout", "0.1",
            "--seed", "7",
            "--lite",
            "--exclude-preview-images",
            "--checkpoint-interval", "250",
            "--max-resolution", "1536",
            "--low-ram",
            "--no-compile",
            "--gradient-checkpointing",
            "--benchmark-steps", "7",
            "--benchmark-warmup-steps", "3",
            "--sample-interval", "150",
            "--sample-prompt", "esfakira portrait in a diner",
            "--sample-model", "image-klein-9b",
            "--sample-steps", "10",
            "--sample-cfg", "1.2",
            "--sample-lora-scale", "0.75",
            "--sample-seed", "99",
            "--visualize",
            "--visualize-port", "8899",
            "--preflight",
            "--json",
            "--lora-target-ranks", ".attn.to_q=128,.ff.linear_in=64",
            "--timestep-sampling", "shift",
            "--timestep-loss-weighting", "weighted",
            "--loss-weighting", "minSNR",
            "--timestep-low", "50",
            "--timestep-high", "450",
            "--lr-warmup-steps", "100",
            "--no-cosine-scheduler",
            "--lr-min-factor", "0.2",
            "--adam-weight-decay", "0.0001",
        ])

        XCTAssertEqual(cmd.model, "image-krea2-raw")
        XCTAssertEqual(cmd.width, 768)
        XCTAssertEqual(cmd.height, 1024)
        XCTAssertEqual(cmd.trainingSteps, 25)
        XCTAssertEqual(cmd.batchSize, 2)
        XCTAssertEqual(cmd.learningRate, 0.0002)
        XCTAssertEqual(cmd.rank, 8)
        XCTAssertEqual(cmd.alpha, 4)
        XCTAssertEqual(cmd.maxTextLength, 384)
        XCTAssertEqual(cmd.schedulerSteps, 500)
        XCTAssertEqual(cmd.captionDropout, 0.1)
        XCTAssertEqual(cmd.seed, 7)
        XCTAssertTrue(cmd.lite)
        XCTAssertTrue(cmd.excludePreviewImages)
        XCTAssertEqual(cmd.checkpointInterval, 250)
        XCTAssertEqual(cmd.maxResolution, 1536)
        XCTAssertFalse(cmd.progressive)
        XCTAssertTrue(cmd.lowRam)
        XCTAssertTrue(cmd.noCompile)
        XCTAssertTrue(cmd.gradientCheckpointing)
        XCTAssertNil(cmd.recipe)
        XCTAssertEqual(cmd.benchmarkSteps, 7)
        XCTAssertEqual(cmd.benchmarkWarmupSteps, 3)
        XCTAssertEqual(cmd.sampleInterval, 150)
        XCTAssertEqual(cmd.samplePrompt, "esfakira portrait in a diner")
        XCTAssertEqual(cmd.sampleModel, "image-klein-9b")
        XCTAssertEqual(cmd.sampleSteps, 10)
        XCTAssertEqual(cmd.sampleGuidanceScale, 1.2)
        XCTAssertEqual(cmd.sampleLoRAScale, 0.75)
        XCTAssertEqual(cmd.sampleSeed, 99)
        XCTAssertTrue(cmd.visualize)
        XCTAssertEqual(cmd.visualizePort, 8899)
        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
        XCTAssertEqual(cmd.loraTargetRanks, ".attn.to_q=128,.ff.linear_in=64")
        XCTAssertNil(cmd.loraRankPreset)
        XCTAssertNil(cmd.loraTargetPreset)
        XCTAssertNil(cmd.loraTargetMode)
        XCTAssertEqual(cmd.timestepSampling, "shift")
        XCTAssertEqual(cmd.timestepLossWeighting, "weighted")
        XCTAssertEqual(cmd.lossWeighting, "minSNR")
        XCTAssertEqual(cmd.timestepLow, 50)
        XCTAssertEqual(cmd.timestepHigh, 450)
        XCTAssertEqual(cmd.lrWarmupSteps, 100)
        XCTAssertTrue(cmd.noCosineScheduler)
        XCTAssertEqual(cmd.lrMinFactor, 0.2)
        XCTAssertEqual(cmd.adamWeightDecay, 0.0001)
    }

    func testTrainLoRAParsesSyntheticSamples() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--output", "/tmp/smoke.safetensors",
            "--synthetic-samples", "2",
        ])

        XCTAssertNil(cmd.data)
        XCTAssertEqual(cmd.syntheticSamples, 2)
    }

    func testImageVisualizeRunParsesDefaults() throws {
        let cmd = try ImageVisualizeRun.parse(["/tmp/lora-run"])

        XCTAssertEqual(cmd.runDirectory, "/tmp/lora-run")
        XCTAssertEqual(cmd.port, 8787)
    }

    func testImageVisualizeRunParsesPort() throws {
        let cmd = try ImageVisualizeRun.parse([
            "/tmp/lora-run",
            "--port", "8899",
        ])

        XCTAssertEqual(cmd.runDirectory, "/tmp/lora-run")
        XCTAssertEqual(cmd.port, 8899)
    }

    func testLoRATrainingRunViewerSnapshotReadsRunArtifacts() throws {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-viewer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        let outputURL = temp.appendingPathComponent("demo.safetensors")
        try Data("adapter".utf8).write(to: outputURL)

        let metrics = try LoRATrainingMetricsLogger(baseOutputURL: outputURL, resumeExisting: false)
        try metrics.record(step: 10, loss: 0.75)
        try metrics.record(step: 20, loss: 0.5)

        let logger = try LoRATrainingEventLogger(baseOutputURL: outputURL)
        try logger.record(type: "run_started", stage: "starting", step: 0, totalSteps: 20)
        try logger.record(type: "progress", stage: "training", step: 20, totalSteps: 20, loss: 0.5, fraction: 1)

        let sampleDir = temp.appendingPathComponent("samples", isDirectory: true)
        try FileManager.default.createDirectory(at: sampleDir, withIntermediateDirectories: true)
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: sampleDir.appendingPathComponent("demo-step20-sample.png"))

        let viewer = LoRATrainingRunViewer(runDirectoryURL: temp)
        let snapshot = try viewer.snapshot()

        XCTAssertEqual(snapshot.status, "running")
        XCTAssertEqual(snapshot.lossPoints.map(\.step), [10, 20])
        XCTAssertEqual(snapshot.events.map(\.type), ["run_started", "progress"])
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "root" && $0.name == "demo.safetensors" })
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "samples" && $0.isImage })
    }

    func testTrainLoRAParsesKleinModelPath() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/gpk-dataset",
            "--output", "/tmp/gpk-klein.safetensors",
            "--model", "/tmp/image-klein-base-9b-8bit",
            "--width", "240",
            "--height", "336",
            "--steps", "2",
            "--rank", "4",
        ])

        XCTAssertEqual(cmd.data, "/tmp/gpk-dataset")
        XCTAssertEqual(cmd.output, "/tmp/gpk-klein.safetensors")
        XCTAssertEqual(cmd.model, "/tmp/image-klein-base-9b-8bit")
        XCTAssertEqual(cmd.width, 240)
        XCTAssertEqual(cmd.height, 336)
        XCTAssertEqual(cmd.trainingSteps, 2)
        XCTAssertEqual(cmd.rank, 4)
    }

    func testTrainLoRAParsesProgressiveAndRankPreset() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/esfakira",
            "--output", "/tmp/esfakira.safetensors",
            "--model", "image-klein-base-9b",
            "--progressive",
            "--lora-rank-preset", "flux2-style-128",
        ])

        XCTAssertTrue(cmd.progressive)
        XCTAssertEqual(cmd.loraRankPreset, "flux2-style-128")
        XCTAssertNil(cmd.loraTargetRanks)
    }

    func testTrainLoRAParsesKleinLoRATargetMode() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/br2049",
            "--output", "/tmp/br2049.safetensors",
            "--model", "image-klein-base-9b",
            "--lora-target-mode", "transformer-linear-walk",
        ])

        XCTAssertEqual(cmd.loraTargetMode, "transformer-linear-walk")
    }

    func testTrainLoRAParsesFALKleinFastTargetPreset() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/ffmrfox",
            "--output", "/tmp/ffmrfox.safetensors",
            "--model", "image-klein-base-9b",
            "--lora-target-preset", "fal-klein-fast",
        ])

        XCTAssertEqual(cmd.loraTargetPreset, "fal-klein-fast")

        let ranks = try XCTUnwrap(ImageTrainLoRA.resolveKleinTargetPreset("fal-klein-fast", rank: 16))
        XCTAssertEqual(ranks.count, 120)
        XCTAssertEqual(ranks["x_embedder"], 16)
        XCTAssertEqual(ranks["context_embedder"], 16)
        XCTAssertEqual(ranks["time_guidance_embed.timestep_embedder.linear_1"], 16)
        XCTAssertEqual(ranks["double_stream_modulation_img.linear"], 16)
        XCTAssertEqual(ranks["transformer_blocks.0.attn.to_q"], 16)
        XCTAssertEqual(ranks["transformer_blocks.7.attn.to_add_out"], 16)
        XCTAssertEqual(ranks["single_transformer_blocks.0.attn.to_qkv_mlp_proj"], 16)
        XCTAssertEqual(ranks["single_transformer_blocks.23.attn.to_out"], 16)
        XCTAssertNil(ranks["transformer_blocks.8.attn.to_q"])
        XCTAssertNil(ranks["single_transformer_blocks.24.attn.to_out"])
    }

    func testTrainLoRAResolvesKleinFastStyleRecipe() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/spirited",
            "--output", "/tmp/spirited.safetensors",
            "--recipe", "klein-fast-style",
        ])

        XCTAssertEqual(cmd.recipe, "klein-fast-style")

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.model, "image-klein-base-9b")
        XCTAssertEqual(options.width, 1024)
        XCTAssertEqual(options.height, 1024)
        XCTAssertEqual(options.trainingSteps, 1000)
        XCTAssertEqual(options.learningRate, 0.00005)
        XCTAssertEqual(options.rank, 16)
        XCTAssertNil(options.alpha)
        XCTAssertEqual(options.captionDropout, 0.05)
        XCTAssertEqual(options.checkpointInterval, 250)
        XCTAssertEqual(options.maxResolution, 512)
        XCTAssertTrue(options.lowRam)
        XCTAssertTrue(options.noCompile)
        XCTAssertEqual(options.loraTargetPreset, "fal-klein-fast")
        XCTAssertNil(options.lrWarmupSteps)
        XCTAssertNil(options.useCosineScheduler)
        XCTAssertNil(options.lrMinFactor)

        let ranks = try XCTUnwrap(ImageTrainLoRA.resolveKleinTargetPreset(options.loraTargetPreset, rank: 16))
        XCTAssertEqual(ranks.count, 120)
    }

    func testTrainLoRAResolvesKreaFastStyleRecipe() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/ffmrfox",
            "--output", "/tmp/ffmrfox.safetensors",
            "--recipe", "krea-fast-style",
        ])

        XCTAssertEqual(cmd.recipe, "krea-fast-style")

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.model, "image-krea2-raw")
        XCTAssertEqual(options.width, 768)
        XCTAssertEqual(options.height, 768)
        XCTAssertEqual(options.trainingSteps, 100)
        XCTAssertEqual(options.learningRate, 0.0005)
        XCTAssertEqual(options.rank, 32)
        XCTAssertEqual(options.alpha, 32)
        XCTAssertEqual(options.captionDropout, 0.05)
        XCTAssertNil(options.checkpointInterval)
        XCTAssertNil(options.maxResolution)
        XCTAssertFalse(options.lowRam)
        XCTAssertFalse(options.noCompile)
        XCTAssertNil(options.loraTargetPreset)
        XCTAssertEqual(options.lrWarmupSteps, 10)
        XCTAssertEqual(options.useCosineScheduler, true)
        XCTAssertEqual(options.lrMinFactor, 0)
    }

    func testTrainLoRAKreaFastStyleRecipeAllowsExplicitSafeOverrides() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/ffmrfox",
            "--output", "/tmp/ffmrfox.safetensors",
            "--recipe", "krea-fast-style",
            "--model", "/tmp/custom-krea-raw",
            "--width", "1024",
            "--height", "768",
            "--steps", "300",
            "--learning-rate", "0.0003",
            "--rank", "48",
            "--alpha", "24",
            "--caption-dropout", "0",
            "--lr-warmup-steps", "25",
            "--lr-min-factor", "0.1",
        ])

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.model, "/tmp/custom-krea-raw")
        XCTAssertEqual(options.width, 1024)
        XCTAssertEqual(options.height, 768)
        XCTAssertEqual(options.trainingSteps, 300)
        XCTAssertEqual(options.learningRate, 0.0003)
        XCTAssertEqual(options.rank, 48)
        XCTAssertEqual(options.alpha, 24)
        XCTAssertEqual(options.captionDropout, 0)
        XCTAssertNil(options.loraTargetPreset)
        XCTAssertEqual(options.lrWarmupSteps, 25)
        XCTAssertEqual(options.useCosineScheduler, true)
        XCTAssertEqual(options.lrMinFactor, 0.1)
    }

    func testTrainLoRAResolvesKreaCinematicStyleRecipe() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/neon",
            "--output", "/tmp/neon.safetensors",
            "--recipe", "krea-cinematic-style",
        ])

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.model, "image-krea2-raw")
        XCTAssertEqual(options.width, 768)
        XCTAssertEqual(options.height, 416)
        XCTAssertEqual(options.trainingSteps, 200)
        XCTAssertEqual(options.learningRate, 0.0001)
        XCTAssertEqual(options.rank, 32)
        XCTAssertEqual(options.alpha, 32)
        XCTAssertEqual(options.captionDropout, 0.05)
        XCTAssertFalse(options.lowRam)
        XCTAssertTrue(options.noCompile)
        XCTAssertNil(options.loraTargetPreset)
        XCTAssertEqual(options.lrWarmupSteps, 20)
        XCTAssertEqual(options.useCosineScheduler, true)
        XCTAssertEqual(options.lrMinFactor, 0)
    }

    func testTrainLoRAKreaSchedulerCanBeDisabled() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/neon",
            "--output", "/tmp/neon.safetensors",
            "--recipe", "krea-fast-style",
            "--no-cosine-scheduler",
        ])

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.lrWarmupSteps, 10)
        XCTAssertEqual(options.useCosineScheduler, false)
        XCTAssertEqual(options.lrMinFactor, 0)
    }

    func testTrainLoRARecipeAllowsExplicitSafeOverrides() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/spirited",
            "--output", "/tmp/spirited.safetensors",
            "--recipe", "klein-fast-style",
            "--model", "/tmp/custom-klein-base",
            "--steps", "1200",
            "--learning-rate", "0.00007",
            "--max-resolution", "768",
            "--checkpoint-interval", "300",
        ])

        let options = try cmd.resolvedTrainingOptions()
        XCTAssertEqual(options.model, "/tmp/custom-klein-base")
        XCTAssertEqual(options.width, 1024)
        XCTAssertEqual(options.height, 1024)
        XCTAssertEqual(options.trainingSteps, 1200)
        XCTAssertEqual(options.learningRate, 0.00007)
        XCTAssertEqual(options.rank, 16)
        XCTAssertNil(options.alpha)
        XCTAssertEqual(options.captionDropout, 0.05)
        XCTAssertEqual(options.maxResolution, 768)
        XCTAssertEqual(options.checkpointInterval, 300)
        XCTAssertTrue(options.lowRam)
        XCTAssertTrue(options.noCompile)
        XCTAssertEqual(options.loraTargetPreset, "fal-klein-fast")
    }

    func testTrainLoRARejectsUnknownRecipeWhenResolved() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--data", "/tmp/spirited",
            "--output", "/tmp/spirited.safetensors",
            "--recipe", "mystery",
        ])

        XCTAssertThrowsError(try cmd.resolvedTrainingOptions())
    }

    func testTrainLoRAPreflightReportsMissingCaptionsAndDisablesTraining() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let dataset = temp.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data("not a real png".utf8).write(to: dataset.appendingPathComponent("frame-001.png"))

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-krea", family: .krea, to: model)

        let cmd = try ImageTrainLoRA.parse([
            "--data", dataset.path,
            "--output", temp.appendingPathComponent("style.safetensors").path,
            "--model", model.path,
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            options: try cmd.resolvedTrainingOptions(),
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertEqual(envelope.result.dataset.imageCount, 1)
        XCTAssertEqual(envelope.result.dataset.missingCaptionCount, 1)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "missing_captions" && $0.severity == .blocker })

        let startTraining = try XCTUnwrap(envelope.actions.first { $0.id == "start-training" })
        XCTAssertFalse(startTraining.enabled)
        XCTAssertEqual(startTraining.command?.argv.prefix(3), ["mere.run", "image", "train-lora"])

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(LoRATrainingPreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded.status, .blocked)
        XCTAssertEqual(decoded.command, ["image", "train-lora"])
    }

    func testTrainLoRAPreflightReportsWarningsAndEnabledTrainingAction() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let dataset = temp.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data("image 1".utf8).write(to: dataset.appendingPathComponent("frame-001.png"))
        try Data("image 2".utf8).write(to: dataset.appendingPathComponent("frame-002.png"))
        try Data("same caption".utf8).write(to: dataset.appendingPathComponent("frame-001.txt"))
        try Data("same caption".utf8).write(to: dataset.appendingPathComponent("frame-002.txt"))

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-klein", family: .klein, to: model)

        let cmd = try ImageTrainLoRA.parse([
            "--data", dataset.path,
            "--output", temp.appendingPathComponent("style.safetensors").path,
            "--model", model.path,
            "--steps", "10",
            "--preflight",
            "--json",
        ])
        let envelope = cmd.makePreflightEnvelope(
            options: try cmd.resolvedTrainingOptions(),
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .warning)
        XCTAssertEqual(envelope.result.dataset.usablePairCount, 2)
        XCTAssertEqual(envelope.result.dataset.duplicateCaptionGroupCount, 1)
        XCTAssertEqual(envelope.result.model.family, "klein")
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "duplicate_captions" && $0.severity == .warning })

        let startTraining = try XCTUnwrap(envelope.actions.first { $0.id == "start-training" })
        XCTAssertTrue(startTraining.enabled)
        XCTAssertEqual(startTraining.command?.argv, [
            "mere.run",
            "image",
            "train-lora",
            "--data",
            dataset.path,
            "--output",
            temp.appendingPathComponent("style.safetensors").path,
            "--model",
            model.path,
            "--training-steps",
            "10",
        ])
    }

    private func makeTemporaryDirectory() throws -> URL {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-lora-preflight-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        return temp
    }

    private func writeManifest(
        id: String,
        family: MereRunModelManifest.Family,
        to directory: URL
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let manifest = MereRunModelManifest(
            id: id,
            family: family,
            variant: .base,
            precision: .bf16,
            supports: [.loraTraining]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest)
            .write(to: directory.appendingPathComponent(MereRunModelManifest.filename))
    }
}
