import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageTrainLoRACommandParsingTests: XCTestCase {
    func testImageCommandExposesTrainLoRA() {
        let commandNames = Set(Image.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("train-lora"))
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
        XCTAssertNil(cmd.sampleInterval)
        XCTAssertNil(cmd.samplePrompt)
        XCTAssertNil(cmd.sampleModel)
        XCTAssertEqual(cmd.sampleSteps, 8)
        XCTAssertEqual(cmd.sampleGuidanceScale, 1.0)
        XCTAssertEqual(cmd.sampleLoRAScale, 1.0)
        XCTAssertNil(cmd.sampleSeed)
        XCTAssertNil(cmd.loraTargetRanks)
        XCTAssertNil(cmd.loraRankPreset)
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
            "--sample-interval", "150",
            "--sample-prompt", "esfakira portrait in a diner",
            "--sample-model", "image-klein-9b",
            "--sample-steps", "10",
            "--sample-cfg", "1.2",
            "--sample-lora-scale", "0.75",
            "--sample-seed", "99",
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
        XCTAssertEqual(cmd.sampleInterval, 150)
        XCTAssertEqual(cmd.samplePrompt, "esfakira portrait in a diner")
        XCTAssertEqual(cmd.sampleModel, "image-klein-9b")
        XCTAssertEqual(cmd.sampleSteps, 10)
        XCTAssertEqual(cmd.sampleGuidanceScale, 1.2)
        XCTAssertEqual(cmd.sampleLoRAScale, 0.75)
        XCTAssertEqual(cmd.sampleSeed, 99)
        XCTAssertEqual(cmd.loraTargetRanks, ".attn.to_q=128,.ff.linear_in=64")
        XCTAssertNil(cmd.loraRankPreset)
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
}
