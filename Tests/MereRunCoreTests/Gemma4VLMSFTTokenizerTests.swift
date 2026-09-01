import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class Gemma4VLMSFTTokenizerTests: XCTestCase {
    func testOfficialUnifiedTokenizerExpandsImageAndMasksAssistantWhenAvailable() async throws {
        guard let path = ProcessInfo.processInfo.environment[
            "MERERUN_GEMMA4_VLM_MODEL_ROOT"
        ] else {
            throw XCTSkip(
                "Set MERERUN_GEMMA4_VLM_MODEL_ROOT to run the official Gemma 4 VLM SFT test."
            )
        }
        let rootURL = URL(fileURLWithPath: path).standardizedFileURL
        let config = try JSONDecoder().decode(
            Gemma4Config.self,
            from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        let tokenizer = try await Gemma4TokenizerAndTemplate.load(
            from: rootURL,
            maxLengthOverride: 768
        )
        let imageURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("png")
        defer { try? FileManager.default.removeItem(at: imageURL) }
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 64,
                height: 48,
                rgba8: Array(repeating: 128, count: 64 * 48 * 4)
            ),
            to: imageURL
        )
        let example = TextSFTExample(
            id: "official-vlm-tokenizer",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Describe visible evidence precisely."),
                ChatMessage(
                    role: .user,
                    content: "What is visible in this image?",
                    imageUrl: imageURL.path
                ),
                ChatMessage(role: .assistant, content: "A uniform gray test image is visible."),
            ]
        )

        let tokenized = try XCTUnwrap(Gemma4VLMSFTTokenizer.tokenize(
            [example],
            tokenizerAndTemplate: tokenizer,
            config: config,
            maxSequenceLength: 768
        ).first)
        let multimodal = try XCTUnwrap(tokenized.multimodalInputs)
        let imageTokenId = try XCTUnwrap(config.imageTokenId)

        XCTAssertEqual(multimodal.imageReferences, [imageURL.path])
        XCTAssertEqual(
            tokenized.inputTokenIds.filter { $0 == imageTokenId }.count,
            multimodal.softTokenCounts.reduce(0, +)
        )
        XCTAssertEqual(multimodal.mmTokenTypeShape, [1, tokenized.inputTokenIds.count])
        XCTAssertTrue(tokenized.lossMask.contains(where: { $0 > 0 }))
        XCTAssertEqual(
            zip(tokenized.lossMask, multimodal.mmTokenTypeIds)
                .filter { mask, tokenType in mask > 0 && tokenType == 1 }
                .count,
            0
        )
    }

    func testOfficialTwelveBCheckpointCompletesOneLoRAStepWhenEnabled() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["MERERUN_RUN_GEMMA4_VLM_LORA_E2E"] == "1" else {
            throw XCTSkip(
                "Set MERERUN_RUN_GEMMA4_VLM_LORA_E2E=1 to run the 12B VLM LoRA canary."
            )
        }
        guard let path = environment["MERERUN_GEMMA4_VLM_MODEL_ROOT"] else {
            throw XCTSkip(
                "Set MERERUN_GEMMA4_VLM_MODEL_ROOT to the installed Gemma 4 VLM model root."
            )
        }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: tempURL,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let imageURL = tempURL.appendingPathComponent("training.png")
        try MediaImageIO.writePNG(
            try MediaImage(
                width: 64,
                height: 48,
                rgba8: Array(repeating: 128, count: 64 * 48 * 4)
            ),
            to: imageURL
        )
        let imageDigest = try TextSFTDataset.fileDigest(imageURL)
        let outputURL = tempURL.appendingPathComponent("adapter.safetensors")
        let example = TextSFTExample(
            id: "official-vlm-lora-e2e",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Describe visible evidence precisely."),
                ChatMessage(
                    role: .user,
                    content: "What is visible in this image?",
                    imageUrl: imageURL.path
                ),
                ChatMessage(role: .assistant, content: "A uniform gray test image is visible."),
            ]
        )

        let report = try await Gemma4VLMLoRATrainingPipeline.train(
            Gemma4VLMLoRATrainingPipelineRequest(
                modelId: Gemma4Resources.visionTwelveBModelId,
                modelPath: path,
                examples: [example],
                trainingImageDigestsByPath: [imageURL.path: imageDigest],
                outputURL: outputURL,
                trainingConfig: TextLoRATrainingConfig(
                    trainingSteps: 1,
                    batchSize: 1,
                    learningRate: 1e-5
                ),
                maxSequenceLength: 512,
                rank: 2,
                alpha: 2,
                targetSuffixes: ["q_proj"]
            )
        )

        XCTAssertEqual(report.steps, 1)
        XCTAssertGreaterThan(report.layerCount, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
        XCTAssertGreaterThan(
            try outputURL.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0,
            0
        )
    }
}
