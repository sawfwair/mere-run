#if os(macOS)
import Foundation
import MLX
import XCTest
@testable import MereRunCore

/// Runs exact OCR checks on an external manifest without changing its labels.
final class Q38ProjectionOCRCheckpointTests: MereRunCoreTestCase {
    func testInstalledOCRManifest() async throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_FLASH_NEXT_CHECKPOINTS"] == "1",
                          "Installed OCR qualification is opt-in.")
        let root = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"])
        let path = try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_OCR_MANIFEST"])
        let manifest = try JSONDecoder().decode(
            Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: path))
        )
        XCTAssertFalse(manifest.cases.isEmpty)
        XCTAssertEqual(Set(manifest.cases.map(\.id)).count, manifest.cases.count)
        let generator = Q35Generator(
            modelId: Q35Resources.q38FlashNextMixedModelId,
            prefixKVCacheEnabled: false, continuousBatchingEnabled: false
        )
        do {
            for item in manifest.cases {
                let response = try await generator.chat(
                    ChatRequest(
                        messages: [.init(role: .user, content: item.prompt, imageUrl: item.image)],
                        maxTokens: 64, temperature: 0, topP: 1,
                        showThinking: false, maxContextTokens: 8_192
                    ),
                    modelPath: root, progressHandler: nil
                )
                let actual = response.response.trimmingCharacters(in: .whitespacesAndNewlines)
                let record = Record(
                    id: item.id, expected: item.expected, actual: actual,
                    exactMatch: actual == item.expected,
                    promptTokens: response.promptTokens ?? 0,
                    outputTokens: response.tokensGenerated,
                    prefillSeconds: response.timing?.prefillSeconds,
                    decodeSeconds: response.timing?.decodeSeconds,
                    activeBytes: Memory.activeMemory, cacheBytes: Memory.cacheMemory
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.sortedKeys]
                let encoded = try encoder.encode(record)
                FileHandle.standardError.write(Data("[q38-projection-ocr] ".utf8) + encoded + Data("\n".utf8))
                XCTAssertEqual(actual, item.expected, item.id)
                XCTAssertEqual(response.acceleration?.route, "final-target-pipelined")
            }
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private struct Manifest: Decodable {
        let cases: [OCRCase]
    }

    private struct OCRCase: Decodable {
        let id: String
        let image: String
        let prompt: String
        let expected: String
    }

    private struct Record: Encodable {
        let id: String
        let expected: String
        let actual: String
        let exactMatch: Bool
        let promptTokens: Int
        let outputTokens: Int
        let prefillSeconds: Double?
        let decodeSeconds: Double?
        let activeBytes: Int
        let cacheBytes: Int
    }
}
#endif
