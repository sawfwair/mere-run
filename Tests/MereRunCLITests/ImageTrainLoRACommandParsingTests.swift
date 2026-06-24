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
    }

    func testTrainLoRAParsesSyntheticSamples() throws {
        let cmd = try ImageTrainLoRA.parse([
            "--output", "/tmp/smoke.safetensors",
            "--synthetic-samples", "2",
        ])

        XCTAssertNil(cmd.data)
        XCTAssertEqual(cmd.syntheticSamples, 2)
    }
}
