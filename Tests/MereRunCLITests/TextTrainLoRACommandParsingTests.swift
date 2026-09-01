import Foundation
import MediaIO
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextTrainLoRACommandParsingTests: XCTestCase {
    func testTextCommandExposesTrainLoRA() {
        let commandNames = Set(Text.configuration.subcommands.map { $0.configuration.commandName })

        XCTAssertTrue(commandNames.contains("train-lora"))
    }

    func testTrainLoRAParsesDefaults() throws {
        let cmd = try TextTrainLoRA.parse([
            "--data", "pairs.jsonl",
            "--output", "local-assistant.safetensors",
        ])

        XCTAssertEqual(cmd.model, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(cmd.adapterName, "local-assistant")
        XCTAssertEqual(cmd.trainingSteps, 600)
        XCTAssertEqual(cmd.batchSize, 1)
        XCTAssertEqual(cmd.rank, 16)
        XCTAssertEqual(cmd.maxSequenceLength, 4096)
        XCTAssertEqual(cmd.reasoningEffort, 0.9)
        XCTAssertFalse(cmd.dryRun)
        XCTAssertFalse(cmd.visualize)
        XCTAssertEqual(cmd.visualizePort, 8787)
    }

    func testTrainLoRAParsesOverrides() throws {
        let cmd = try TextTrainLoRA.parse([
            "--data", "pairs.jsonl",
            "--output", "adapter.safetensors",
            "--model", Gemma4Resources.turboModelId,
            "--model-path", "/tmp/gemma4",
            "--eval", "eval.jsonl",
            "--adapter-name", "support-draft",
            "--steps", "20",
            "--batch-size", "2",
            "--lr", "0.00005",
            "--rank", "8",
            "--alpha", "16",
            "--max-sequence-length", "2048",
            "--reasoning-effort", "0.2",
            "--target-modules", "q_proj,v_proj",
            "--visualize",
            "--visualize-port", "8788",
            "--dry-run",
            "--json",
        ])

        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertEqual(cmd.modelPath, "/tmp/gemma4")
        XCTAssertEqual(cmd.eval, "eval.jsonl")
        XCTAssertEqual(cmd.adapterName, "support-draft")
        XCTAssertEqual(cmd.trainingSteps, 20)
        XCTAssertEqual(cmd.batchSize, 2)
        XCTAssertEqual(cmd.learningRate, 0.00005)
        XCTAssertEqual(cmd.rank, 8)
        XCTAssertEqual(cmd.alpha, 16)
        XCTAssertEqual(cmd.maxSequenceLength, 2048)
        XCTAssertEqual(cmd.reasoningEffort, 0.2)
        XCTAssertTrue(cmd.visualize)
        XCTAssertEqual(cmd.visualizePort, 8788)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.json)
    }

    func testTrainLoRAParsesLagunaXSModel() throws {
        let cmd = try TextTrainLoRA.parse([
            "--data", "pairs.jsonl",
            "--output", "laguna-support.safetensors",
            "--model", LagunaResources.xsModelID,
        ])

        XCTAssertEqual(cmd.model, LagunaResources.xsModelID)
        XCTAssertEqual(cmd.resolvedTargetModules(), ["q_proj", "k_proj", "v_proj", "o_proj"])
    }

    func testTrainLoRAParsesInklingSmallModel() throws {
        let cmd = try TextTrainLoRA.parse([
            "--data", "pairs.jsonl",
            "--output", "inkling-support.safetensors",
            "--model", InklingResources.modelID,
        ])

        XCTAssertEqual(cmd.model, InklingResources.modelID)
        XCTAssertEqual(
            cmd.resolvedTargetModules(),
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
        )
    }

    func testTrainLoRAParsesLFM25A1BModel() throws {
        let command = try TextTrainLoRA.parse([
            "--data", "pairs.jsonl",
            "--output", "lfm25-support.safetensors",
            "--model", LFM2Resources.defaultModelId,
        ])

        XCTAssertEqual(command.model, LFM2Resources.defaultModelId)
        XCTAssertEqual(
            command.resolvedTargetModules(),
            ["q_proj", "k_proj", "v_proj", "out_proj"]
        )
    }

    func testGemma4VisionDryRunWritesVLMManifestAndImageProvenance() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        let imageDirectory = directory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imageDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let imageURL = imageDirectory.appendingPathComponent("frame.png")
        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 96, count: 16)),
            to: imageURL
        )
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")
        let example = TextSFTExample(
            id: "vlm-dry-run",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Describe visible evidence."),
                ChatMessage(
                    role: .user,
                    content: "What is visible in this image?",
                    imageUrl: "images/frame.png"
                ),
                ChatMessage(role: .assistant, content: "A test frame is visible."),
            ]
        )
        let encodedExample = try JSONEncoder().encode(example)
        try Data(encodedExample + Data("\n".utf8)).write(to: datasetURL)
        let outputURL = directory.appendingPathComponent("adapter.safetensors")
        let command = try TextTrainLoRA.parse([
            "--model", Gemma4Resources.visionTwelveBModelId,
            "--data", datasetURL.path,
            "--output", outputURL.path,
            "--dry-run",
            "--json",
        ])

        try await command.run()

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let manifest = try decoder.decode(
            TextLoRATrainingManifest.self,
            from: Data(contentsOf: TextLoRATrainingManifest.url(nextTo: outputURL))
        )
        XCTAssertEqual(manifest.format, TextLoRATrainingManifest.gemma4VLMFormat)
        XCTAssertEqual(manifest.modality, "image")
        XCTAssertEqual(manifest.trainingScope, "language_attention")
        XCTAssertEqual(manifest.training.dataset.imageReferenceCount, 1)
        XCTAssertEqual(manifest.training.dataset.uniqueImageCount, 1)
        XCTAssertEqual(manifest.training.dataset.imageFingerprint?.count, 64)
    }

    func testGemma4VisionRejectsBatchSizeGreaterThanOne() async throws {
        let command = try TextTrainLoRA.parse([
            "--model", Gemma4Resources.visionTwelveBModelId,
            "--data", "/tmp/missing-vlm.jsonl",
            "--output", "/tmp/missing-vlm.safetensors",
            "--batch-size", "2",
            "--dry-run",
        ])

        do {
            try await command.run()
            XCTFail("Expected Gemma 4 VLM batch-size validation to fail.")
        } catch {
            XCTAssertEqual(
                String(describing: error),
                "Gemma 4 VLM LoRA training currently requires --batch-size 1"
            )
        }
    }

    func testInklingReceptivityFixturesAreHeldOutParaphrases() throws {
        let fixtureRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/Inkling")
        let training = try TextSFTDataset.load(
            from: fixtureRoot.appendingPathComponent("receptivity-train.jsonl")
        )
        let evaluation = try TextSFTDataset.load(
            from: fixtureRoot.appendingPathComponent("receptivity-eval.jsonl")
        )
        let trainingPrompts = Set(training.compactMap {
            $0.messages.first(where: { $0.role == .user })?.content
        })
        let evaluationPrompts = Set(evaluation.compactMap {
            $0.messages.first(where: { $0.role == .user })?.content
        })

        XCTAssertEqual(training.count, 32)
        XCTAssertEqual(evaluation.count, 4)
        XCTAssertTrue(trainingPrompts.isDisjoint(with: evaluationPrompts))
        XCTAssertEqual(
            Set(evaluation.compactMap { $0.messages.last?.content }),
            ["KITE-731", "MOSS-284", "LARK-956", "PINE-407"]
        )
    }
}
