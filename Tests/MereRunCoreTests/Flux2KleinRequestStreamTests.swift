import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Flux2KleinRequestStreamTests: MereRunCoreTestCase {
    func testImageLoadingOwnsStreamsAndReleasesThemAfterFailure() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-klein-stream-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for component in ["transformer", "text_encoder", "tokenizer", "vae"] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(component), withIntermediateDirectories: true
            )
        }
        _ = try XCTUnwrap(MereRunModelManifest.writeTemplateIfKnown(modelId: "image-klein-nano", to: root))
        let generator = Flux2KleinGenerator()
        let parent = StreamOrDevice.default.stream
        let owned = await generator.withRequestStream { StreamOrDevice.default.stream }
        let loading = expectation(description: "Transformer loading runs inside the request stream")
        let request = GenerationRequest(
            prompt: "A red leaf", width: 512, height: 512, steps: 4,
            outputURL: root.appendingPathComponent("output.png"), model: root.path
        )

        do {
            _ = try await generator.generate(request) { progress in
                if progress.stage == .loadingTransformer {
                    XCTAssertEqual(StreamOrDevice.default.stream, owned)
                    XCTAssertNotEqual(StreamOrDevice.default.stream, parent)
                    loading.fulfill()
                }
            }
            XCTFail("The fixture has no transformer configuration")
        } catch {
            XCTAssertTrue(String(describing: error).contains("config.json"), "Unexpected loading error: \(error)")
            XCTAssertEqual(StreamOrDevice.default.stream, parent)
        }
        await fulfillment(of: [loading], timeout: 1)
        let next = await generator.withRequestStream { StreamOrDevice.default.stream }
        XCTAssertEqual(next, owned)
    }

    func testChatEntryPointsOwnStreamsBeforeModelLoading() async {
        let generator = Flux2KleinGenerator()
        let parent = StreamOrDevice.default.stream
        let owned = await generator.withRequestStream { StreamOrDevice.default.stream }
        let loading = expectation(description: "Chat loading runs inside the request stream")
        // Standalone loading emits its own stage before reading the root config.
        loading.expectedFulfillmentCount = 4
        let progress: @Sendable (ChatProgress) -> Void = { event in
            if event.stage == .loadingModel {
                XCTAssertEqual(StreamOrDevice.default.stream, owned)
                XCTAssertNotEqual(StreamOrDevice.default.stream, parent)
                loading.fulfill()
            }
        }
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-missing-klein-\(UUID().uuidString)").path
        let request = ChatRequest(messages: [ChatMessage(role: .user, content: "Hello")], maxTokens: 1)

        do {
            try await generator.prepareChat(modelPath: missing, standalone: false, progressHandler: progress)
            XCTFail("Expected missing model")
        } catch {
            XCTAssertEqual(StreamOrDevice.default.stream, parent)
        }
        do {
            _ = try await generator.chat(request, modelPath: missing, progressHandler: progress)
            XCTFail("Expected missing model")
        } catch {
            XCTAssertEqual(StreamOrDevice.default.stream, parent)
        }
        do {
            _ = try await generator.chatStandalone(request, modelPath: missing, progressHandler: progress)
            XCTFail("Expected missing model")
        } catch {
            XCTAssertEqual(StreamOrDevice.default.stream, parent)
        }
        await fulfillment(of: [loading], timeout: 1)
    }
}
