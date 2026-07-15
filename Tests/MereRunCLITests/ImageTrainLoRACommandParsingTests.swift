import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageTrainLoRACommandParsingTests: XCTestCase {
    func testImageCommandExposesTrainLoRA() {
        let commandNames = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("dataset"))
        XCTAssertTrue(commandNames.contains("run-plan"))
        XCTAssertTrue(commandNames.contains("train-lora"))
        XCTAssertTrue(commandNames.contains("visualize-run"))
    }

    func testImageDatasetDiscoverFindsNestedTrainableLeaves() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("data", isDirectory: true)
        let trainable = root
            .appendingPathComponent("lora-datasets", isDirectory: true)
            .appendingPathComponent("esf", isDirectory: true)
            .appendingPathComponent("amelie", isDirectory: true)
            .appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: trainable, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: trainable.appendingPathComponent("frame-001.jpg"))
        try Data("warm cafe frame".utf8).write(to: trainable.appendingPathComponent("frame-001.txt"))

        let needsReview = root
            .appendingPathComponent("lora-datasets", isDirectory: true)
            .appendingPathComponent("esf", isDirectory: true)
            .appendingPathComponent("missing-captions", isDirectory: true)
            .appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: needsReview, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: needsReview.appendingPathComponent("frame-001.png"))
        try Data("".utf8).write(to: needsReview.appendingPathComponent("frame-001.txt"))

        let cmd = try ImageDatasetDiscover.parse([
            "--root", root.path,
            "--json",
        ])
        let envelope = try cmd.makeEnvelope(now: { Date(timeIntervalSince1970: 0) })

        XCTAssertEqual(envelope.status, .warning)
        XCTAssertEqual(envelope.command, ["image", "dataset", "discover"])
        XCTAssertEqual(envelope.result.candidateCount, 2)
        XCTAssertEqual(envelope.result.trainableCandidateCount, 1)
        XCTAssertTrue(envelope.result.candidates.contains {
            $0.relativePath == "lora-datasets/esf/amelie/dataset" && $0.trainable
        })
        XCTAssertTrue(envelope.result.candidates.contains {
            $0.relativePath == "lora-datasets/esf/missing-captions/dataset" && !$0.trainable
        })
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "dataset_candidates_need_review" })

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(LoRATrainingDatasetDiscoveryEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded.result.trainableCandidateCount, 1)
        let choose = try XCTUnwrap(decoded.actions.first { $0.id == "choose-dataset" })
        XCTAssertEqual(choose.candidates.count, 1)
        XCTAssertEqual(choose.candidates.first?.patches.map(\.path), ["request.data", "run_plan.arguments.data"])
        XCTAssertNil(choose.candidates.first?.command)
    }

    func testImageDatasetDiscoverEmitsTrainingPreflightCommands() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("data", isDirectory: true)
        let trainable = root
            .appendingPathComponent("lora-datasets", isDirectory: true)
            .appendingPathComponent("esf", isDirectory: true)
            .appendingPathComponent("amelie", isDirectory: true)
            .appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: trainable, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: trainable.appendingPathComponent("frame-001.jpg"))
        try Data("warm cafe frame".utf8).write(to: trainable.appendingPathComponent("frame-001.txt"))

        let outputRoot = temp.appendingPathComponent("lora-output", isDirectory: true)
        let cmd = try ImageDatasetDiscover.parse([
            "--root", root.path,
            "--training-output-root", outputRoot.path,
            "--training-model", "image-krea2-raw",
            "--training-recipe", "krea-fast-style",
            "--json",
        ])
        let envelope = try cmd.makeEnvelope(now: { Date(timeIntervalSince1970: 0) })

        XCTAssertEqual(envelope.status, .ok)
        let standardizedOutputRoot = outputRoot.standardizedFileURL
        XCTAssertEqual(envelope.request.trainingOutputRoot, standardizedOutputRoot.path)
        XCTAssertEqual(envelope.request.trainingModel, "image-krea2-raw")
        XCTAssertEqual(envelope.request.trainingRecipe, "krea-fast-style")

        let candidate = try XCTUnwrap(envelope.actions.first { $0.id == "choose-dataset" }?.candidates.first)
        let command = try XCTUnwrap(candidate.command)
        let discoveredDataset = try XCTUnwrap(envelope.result.candidates.first?.path)
        XCTAssertEqual(command.commandPath, ["image", "train-lora"])
        XCTAssertEqual(command.argv, [
            "mere.run",
            "image",
            "train-lora",
            "--data",
            discoveredDataset,
            "--output",
            standardizedOutputRoot.appendingPathComponent("lora-datasets-esf-amelie-dataset.safetensors").path,
            "--model",
            "image-krea2-raw",
            "--recipe",
            "krea-fast-style",
            "--preflight",
            "--json",
        ])
        XCTAssertEqual(candidate.patches.map(\.path), ["request.data", "run_plan.arguments.data"])
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
        XCTAssertNil(cmd.baseQuantizationBits)
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
            "--base-quantization-bits", "4",
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
        XCTAssertEqual(cmd.baseQuantizationBits, 4)
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

        let plan = cmd.makeRunPlan(options: options, now: { Date(timeIntervalSince1970: 0) })
        XCTAssertEqual(plan.arguments.sourceRecipe, "krea-cinematic-style")
        XCTAssertEqual(plan.arguments.model, "image-krea2-raw")
        XCTAssertEqual(plan.arguments.width, 768)
        XCTAssertEqual(plan.arguments.height, 416)
        XCTAssertTrue(plan.arguments.noCompile)
        XCTAssertFalse(plan.arguments.trainLoRAArguments().contains("--recipe"))
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

    func testTrainLoRAPreflightDiscoversChildDatasetsForRootFolder() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("animatic-data", isDirectory: true)
        let dataset = root
            .appendingPathComponent("lora-datasets", isDirectory: true)
            .appendingPathComponent("esf", isDirectory: true)
            .appendingPathComponent("spirited-away", isDirectory: true)
            .appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: dataset.appendingPathComponent("frame-001.jpg"))
        try Data("bathhouse bridge at dusk".utf8).write(to: dataset.appendingPathComponent("frame-001.txt"))

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-krea", family: .krea, to: model)

        let cmd = try ImageTrainLoRA.parse([
            "--data", root.path,
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
        XCTAssertEqual(envelope.result.dataset.imageCount, 0)
        XCTAssertEqual(envelope.result.datasetDiscovery?.trainableCandidateCount, 1)
        XCTAssertEqual(
            envelope.result.datasetDiscovery?.candidates.first?.relativePath,
            "lora-datasets/esf/spirited-away/dataset"
        )

        let noImages = try XCTUnwrap(envelope.diagnostics.first { $0.id == "no_training_images" })
        XCTAssertTrue(noImages.suggestedActionIDs.contains("discover-datasets"))
        XCTAssertTrue(noImages.suggestedActionIDs.contains("choose-dataset"))

        let discover = try XCTUnwrap(envelope.actions.first { $0.id == "discover-datasets" })
        XCTAssertEqual(discover.command?.argv, [
            "mere.run",
            "image",
            "dataset",
            "discover",
            "--root",
            root.resolvingSymlinksInPath().path,
            "--json",
        ])

        let choose = try XCTUnwrap(envelope.actions.first { $0.id == "choose-dataset" })
        XCTAssertEqual(choose.kind, .select)
        XCTAssertEqual(choose.candidates.first?.label, "lora-datasets/esf/spirited-away/dataset")
        XCTAssertEqual(choose.candidates.first?.patches.map(\.path), ["request.data", "run_plan.arguments.data"])
        XCTAssertEqual(choose.candidates.first?.patches.first?.value, envelope.result.datasetDiscovery?.candidates.first?.path)
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

        let plan = envelope.result.runPlan
        XCTAssertEqual(plan.kind, LoRATrainingRunPlan.kind)
        XCTAssertEqual(plan.command, ["image", "train-lora"])
        XCTAssertEqual(plan.arguments.data, dataset.path)
        XCTAssertEqual(plan.arguments.output, temp.appendingPathComponent("style.safetensors").path)
        XCTAssertEqual(plan.arguments.model, model.path)
        XCTAssertEqual(plan.arguments.trainingSteps, 10)
        XCTAssertEqual(plan.arguments.executableArgv().prefix(3), ["mere.run", "image", "train-lora"])
        XCTAssertTrue(plan.arguments.executableArgv().contains("--model"))

        let encodedPlan = try StructuredRunOutput.encode(plan)
        let planURL = temp.appendingPathComponent("style.plan.json")
        try Data(encodedPlan.utf8).write(to: planURL)

        let runPlan = try ImageRunPlan.parse([
            planURL.path,
            "--preflight",
            "--json",
        ])
        let loadedPlan = try runPlan.loadPlan()
        XCTAssertEqual(loadedPlan, plan)

        let commandFromPlan = try runPlan.makeTrainLoRACommand(from: loadedPlan)
        XCTAssertEqual(commandFromPlan.data, dataset.path)
        XCTAssertEqual(commandFromPlan.output, temp.appendingPathComponent("style.safetensors").path)
        XCTAssertEqual(commandFromPlan.model, model.path)
        XCTAssertEqual(commandFromPlan.trainingSteps, 10)
        XCTAssertFalse(commandFromPlan.preflight)
        XCTAssertFalse(commandFromPlan.json)
    }

    func testTrainLoRAPreflightBlocksKleinPreviewOptionsForKreaModel() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let dataset = temp.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: dataset.appendingPathComponent("frame.png"))
        try Data("caption".utf8).write(to: dataset.appendingPathComponent("frame.txt"))
        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-krea", family: .krea, to: model)
        let cmd = try ImageTrainLoRA.parse([
            "--data", dataset.path,
            "--output", temp.appendingPathComponent("style.safetensors").path,
            "--model", model.path,
            "--sample-prompt", "preview",
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            options: try cmd.resolvedTrainingOptions(),
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains {
            $0.id == "klein_training_options_require_klein_model" && $0.severity == .blocker
        })
        XCTAssertEqual(envelope.actions.first { $0.id == "start-training" }?.enabled, false)
    }

    func testImageRunPlanMaterializesDurableRunDirectory() throws {
        let temp = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }

        let dataset = temp.appendingPathComponent("dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        try Data("image".utf8).write(to: dataset.appendingPathComponent("frame-001.png"))
        try Data("materialized frame".utf8).write(to: dataset.appendingPathComponent("frame-001.txt"))

        let model = temp.appendingPathComponent("model", isDirectory: true)
        try writeManifest(id: "local-krea", family: .krea, to: model)

        let originalOutput = temp.appendingPathComponent("outside.safetensors")
        let train = try ImageTrainLoRA.parse([
            "--data", dataset.path,
            "--output", originalOutput.path,
            "--model", model.path,
            "--steps", "10",
        ])
        let plan = train.makeRunPlan(
            options: try train.resolvedTrainingOptions(),
            now: { Date(timeIntervalSince1970: 0) }
        )
        let sourcePlanURL = temp.appendingPathComponent("source.plan.json")
        try StructuredRunOutput.encode(plan).write(to: sourcePlanURL, atomically: true, encoding: .utf8)

        let runDirectory = temp.appendingPathComponent("runs", isDirectory: true)
            .appendingPathComponent("style", isDirectory: true)
        let command = try ImageRunPlan.parse([
            sourcePlanURL.path,
            "--materialize",
            runDirectory.path,
            "--json",
        ])
        let envelope = try command.materializeEnvelope(
            plan: try command.loadPlan(),
            runDirectory: runDirectory.path,
            now: { Date(timeIntervalSince1970: 10) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.mode, .materialize)
        XCTAssertEqual(URL(fileURLWithPath: envelope.request.planFile).lastPathComponent, "source.plan.json")
        XCTAssertEqual(URL(fileURLWithPath: envelope.result.runDirectory).lastPathComponent, "style")
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-training" })
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "output_relocated" })

        let materializedPlanURL = URL(fileURLWithPath: envelope.result.planPath)
        let materializedPlan = try LoRATrainingRunPlan.decode(from: materializedPlanURL)
        XCTAssertEqual(materializedPlan.arguments.output, envelope.result.outputPath)
        XCTAssertEqual(materializedPlan.arguments.data, dataset.path)

        let commandFromPlan = try command.makeTrainLoRACommand(from: materializedPlan)
        XCTAssertEqual(commandFromPlan.output, envelope.result.outputPath)
        XCTAssertEqual(commandFromPlan.data, dataset.path)

        let decoder = JSONDecoder()
        let actionsURL = URL(fileURLWithPath: envelope.result.actionsPath)
        let actions = try decoder.decode([DeclarativeAction].self, from: Data(contentsOf: actionsURL))
        let startTraining = try XCTUnwrap(actions.first { $0.id == "start-training" })
        XCTAssertEqual(startTraining.command?.argv, ["mere.run", "image", "run-plan", materializedPlanURL.path])

        let manifestURL = URL(fileURLWithPath: envelope.result.runManifestPath)
        let manifest = try LoRATrainingRunManifest.decode(from: Data(contentsOf: manifestURL))
        XCTAssertEqual(manifest.format, LoRATrainingRunPlan.kind)
        XCTAssertEqual(manifest.step, 0)
        XCTAssertEqual(manifest.totalSteps, 10)
        XCTAssertEqual(manifest.checkpointFiles["plan"], "plan.json")
        XCTAssertEqual(manifest.checkpointFiles["actions"], "actions.json")

        let eventsURL = URL(fileURLWithPath: envelope.result.eventsPath)
        let events = try LoRATrainingRunEvent.load(from: eventsURL)
        XCTAssertEqual(events.first?.type, "run_planned")
        XCTAssertEqual(events.first?.path, envelope.result.outputPath)

        let viewer = LoRATrainingRunViewer(runDirectoryURL: URL(fileURLWithPath: envelope.result.runDirectory))
        let snapshot = try viewer.snapshot()
        XCTAssertEqual(snapshot.status, "planned")
        XCTAssertEqual(snapshot.events.map(\.type), ["run_planned"])
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "root" && $0.name == "plan.json" })
        XCTAssertTrue(snapshot.artifacts.contains { $0.kind == "root" && $0.name == "actions.json" })

        let eventLogger = try XCTUnwrap(
            commandFromPlan.makeRunEventLoggerIfNeeded(
                outputURL: URL(fileURLWithPath: envelope.result.outputPath),
                modelRoot: model,
                modelManifest: MereRunModelManifest(
                    id: "local-krea",
                    family: .krea,
                    variant: .base,
                    precision: .bf16,
                    supports: [.loraTraining]
                ),
                options: try commandFromPlan.resolvedTrainingOptions()
            )
        )
        XCTAssertEqual(try LoRATrainingRunEvent.load(from: eventsURL).map(\.type), ["run_planned", "run_started"])
        XCTAssertEqual(try viewer.snapshot().status, "running")

        try eventLogger.record(
            type: "run_finished",
            stage: "finished",
            step: 10,
            totalSteps: 10,
            fraction: 1,
            path: envelope.result.outputPath
        )
        XCTAssertEqual(try viewer.snapshot().status, "finished")
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
