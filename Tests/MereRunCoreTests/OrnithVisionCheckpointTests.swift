#if os(macOS)
import CoreGraphics
import Foundation
import ImageIO
import MLX
import XCTest
import UniformTypeIdentifiers
@testable import MereRunCore

final class OrnithVisionCheckpointTests: MereRunCoreTestCase {
    func testInstalledFourBitCompositeLoadsEveryVisionTensorAndAnswersImage() async throws {
        guard let root = ProcessInfo.processInfo.environment[
            "MERERUN_TEST_ORNITH_VISION_4BIT_MODEL_ROOT"
        ] else {
            throw XCTSkip(
                "Set MERERUN_TEST_ORNITH_VISION_4BIT_MODEL_ROOT for installed Ornith 4-bit vision qualification."
            )
        }
        let rootURL = URL(fileURLWithPath: root, isDirectory: true)
        Self.report("validating bundled 4-bit target, MTP head, and vision tensors")
        try validatePublishedConfigurationAndVisionWeights(primaryRootURL: rootURL)
        Self.report("vision tensor validation complete")
        Memory.clearCache()

        let fixtureDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-ornith-vision-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
        let fixture = try Self.makeFixture(in: fixtureDirectory)
        let generator = Q35Generator(
            modelId: Q35Resources.ornith35BMLX4BitModelId,
            prefixKVCacheEnabled: true,
            continuousBatchingEnabled: false
        )
        let started = Date()
        do {
            Self.report("starting full model load and image answer")
            let response = try await generator.chat(
                ChatRequest(
                    messages: [ChatMessage(
                        role: .user,
                        content: "Describe the two shapes from left to right, giving each shape's color and shape.",
                        imageUrl: fixture.path
                    )],
                    maxTokens: 128,
                    temperature: 0,
                    topP: 1,
                    showThinking: false,
                    maxContextTokens: 8_192
                ),
                modelPath: root,
                progressHandler: { progress in
                    Self.report("\(progress.stage.rawValue): \(progress.message ?? "")")
                }
            )
            Self.report("image answer complete")
            let output = response.response.lowercased()
            XCTAssertTrue(output.contains("red") && output.contains("circle"), output)
            XCTAssertTrue(output.contains("blue") && output.contains("square"), output)
            XCTAssertEqual(response.acceleration?.route, "final-target-pipelined")
            XCTAssertNil(response.acceleration?.acceptedDraftTokens)
            print("[ornith-vision] elapsed_seconds=\(Date().timeIntervalSince(started)) "
                + "prompt_tokens=\(response.promptTokens ?? 0) output_tokens=\(response.tokensGenerated) "
                + "text=\(String(reflecting: response.response))")
        } catch {
            await generator.unload()
            throw error
        }
        await generator.unload()
    }

    private static func report(_ message: String) {
        FileHandle.standardError.write(Data("[ornith-vision] \(message)\n".utf8))
    }

    private func validatePublishedConfigurationAndVisionWeights(primaryRootURL: URL) throws {
        let primaryConfig = try JSONDecoder().decode(
            Q35Config.self,
            from: Data(contentsOf: primaryRootURL.appendingPathComponent("config.json"))
        )
        XCTAssertEqual(primaryConfig.modelType, "qwen3_5_moe")
        XCTAssertEqual(primaryConfig.quantization?.bits, 4)
        XCTAssertNil(primaryConfig.visionConfig)

        let mtpRoot = primaryRootURL.appendingPathComponent(
            Q35Resources.ornith35BMTPComponentPath,
            isDirectory: true
        )
        XCTAssertTrue(Q35Resources(rootURL: mtpRoot).validateOrnith35BMTPCompanion().isEmpty)

        let resources = Q35Resources(rootURL: primaryRootURL).ornithVisionComponentResources
        let config = try JSONDecoder().decode(
            Q35Config.self,
            from: Data(contentsOf: resources.configURL)
        )
        let vision = try XCTUnwrap(config.visionConfig)
        XCTAssertEqual(config.modelType, "qwen3_5_moe")
        XCTAssertEqual(config.textConfig.numHiddenLayers, 40)
        XCTAssertEqual(config.textConfig.numExperts, 256)
        XCTAssertEqual(vision.depth, 27)
        XCTAssertEqual(vision.hiddenSize, 1_152)
        XCTAssertEqual(vision.outHiddenSize, 2_048)
        XCTAssertEqual(vision.numPositionEmbeddings, 2_304)
        XCTAssertEqual(vision.patchSize, 16)
        XCTAssertEqual(vision.temporalPatchSize, 2)

        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: resources.modelIndexURL)
        )
        let expected = Set(index.weightMap.keys.flatMap {
            Q35VisionTower.mapVisionWeight($0, MLXArray.zeros([1])).map(\.0)
        })
        let tower = Q35VisionTower(config: config)
        try tower.loadWeights(from: resources)
        let parameters = tower.parameters().flattened()
        XCTAssertEqual(expected, Set(parameters.map(\.0)))
        print("[ornith-vision-weights] checkpoint_tensors=\(expected.count) runtime_tensors=\(parameters.count)")

        Self.report("starting isolated 256x256 vision forward")
        let embeddings = try tower.encodeImage(
            pixelValues: MLXArray.zeros([1, 3, 256, 256], dtype: .float32),
            gridTHW: (1, 16, 16)
        )
        XCTAssertEqual(embeddings.shape, [64, config.textConfig.hiddenSize])
        Self.report("isolated vision forward complete")
    }

    private static func makeFixture(in directory: URL) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: 512,
            height: 384,
            bitsPerComponent: 8,
            bytesPerRow: 512 * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ))
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 512, height: 384))
        context.setFillColor(CGColor(red: 0.9, green: 0.05, blue: 0.05, alpha: 1))
        context.fillEllipse(in: CGRect(x: 55, y: 117, width: 150, height: 150))
        context.setFillColor(CGColor(red: 0.05, green: 0.15, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 307, y: 117, width: 150, height: 150))

        let url = directory.appendingPathComponent("red-circle-blue-square.png")
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        )
        CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }
}
#endif
