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
