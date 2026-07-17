import Foundation
import XCTest
@testable import MereRunCore

final class ModelResolverTests: MereRunCoreTestCase {
    private func writeMinimalImageModel(at root: URL, id: ModelResolver.ModelID) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoder = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformer = root.appendingPathComponent("transformer", isDirectory: true)
        let vae = root.appendingPathComponent("vae", isDirectory: true)
        let scheduler = root.appendingPathComponent("scheduler", isDirectory: true)

        try TestFileSystem.createDirectory(tokenizer)
        try TestFileSystem.createDirectory(textEncoder)
        try TestFileSystem.createDirectory(transformer)
        try TestFileSystem.createDirectory(vae)
        try TestFileSystem.createDirectory(scheduler)

        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("merges.txt"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("vocab.json"), contents: Data("{}".utf8))

        try TestFileSystem.writeFile(textEncoder.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(textEncoder.appendingPathComponent("model.safetensors"), contents: Data())

        try TestFileSystem.writeFile(transformer.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(transformer.appendingPathComponent("diffusion_pytorch_model.safetensors"), contents: Data())

        try TestFileSystem.writeFile(vae.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(vae.appendingPathComponent("diffusion_pytorch_model.safetensors"), contents: Data())

        try TestFileSystem.writeFile(scheduler.appendingPathComponent("scheduler_config.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalTextRoot(
        at root: URL,
        id: ModelResolver.ModelID
    ) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalHiDreamRoot(
        at root: URL,
        id: ModelResolver.ModelID
    ) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("preprocessor_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalSAM31Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionSegmentSAM31, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizer)
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalFalconModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizer)
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizer.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalLTX23DevModel(
        at root: URL,
        id: ModelResolver.ModelID,
        includeVocoder: Bool
    ) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        for file in [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-dev.safetensors",
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
        ] {
            try TestFileSystem.writeFile(root.appendingPathComponent(file), contents: Data())
        }
        if includeVocoder {
            try TestFileSystem.writeFile(root.appendingPathComponent("vocoder.safetensors"), contents: Data())
        }
    }

    func testResolvesFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot
            .appendingPathComponent("image-zimage-nano", isDirectory: true)
        try writeMinimalImageModel(at: modelRoot, id: .zetaNano)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.zetaNano)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testMebotDoesNotFallBackToKleinImageModel() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let nanoRoot = modelsRoot
            .appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalImageModel(at: nanoRoot, id: .kleinNano)

        let resolver = ModelResolver()
        XCTAssertThrowsError(try resolver.resolve(.mebot))
    }

    func testLTX23FullAndLegacyA2VidIDsResolveCompatibleInstalledRoots() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)
        let legacyRoot = modelsRoot.appendingPathComponent(
            ModelResolver.ModelID.ltxVideo23A2VMLX.rawValue,
            isDirectory: true
        )
        try writeMinimalLTX23DevModel(
            at: legacyRoot,
            id: .ltxVideo23A2VMLX,
            includeVocoder: false
        )

        XCTAssertEqual(
            try ModelResolver().resolve(.ltxVideo23FullMLX).rootURL.standardizedFileURL,
            legacyRoot.standardizedFileURL
        )

        try FileManager.default.removeItem(at: legacyRoot)
        let fullRoot = modelsRoot.appendingPathComponent(
            ModelResolver.ModelID.ltxVideo23FullMLX.rawValue,
            isDirectory: true
        )
        try writeMinimalLTX23DevModel(
            at: fullRoot,
            id: .ltxVideo23FullMLX,
            includeVocoder: true
        )
        XCTAssertEqual(
            try ModelResolver().resolve(.ltxVideo23A2VMLX).rootURL.standardizedFileURL,
            fullRoot.standardizedFileURL
        )
    }

    func testResolvesStandaloneMebotModel() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let mebotRoot = modelsRoot
            .appendingPathComponent("text-chat-mebot", isDirectory: true)
        try writeMinimalTextRoot(at: mebotRoot, id: .mebot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.mebot)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, mebotRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testRejectsMismatchedManifestID() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let mismatchedRoot = modelsRoot
            .appendingPathComponent("image-zimage-base", isDirectory: true)
        try TestFileSystem.createDirectory(mismatchedRoot)

        // Wrong manifest ID for the requested directory.
        let wrongManifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))
        try wrongManifest.write(to: mismatchedRoot)

        let resolver = ModelResolver()
        XCTAssertThrowsError(try resolver.resolve(.zetaBase))
    }

    func testResolvesHiDreamFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("image-hidream-o1-dev", isDirectory: true)
        try writeMinimalHiDreamRoot(at: modelRoot, id: .hidreamO1Dev)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.hidreamO1Dev)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesHiDreamFullFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("image-hidream-o1", isDirectory: true)
        try writeMinimalHiDreamRoot(at: modelRoot, id: .hidreamO1)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.hidreamO1)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesGemma4FromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("text-chat-gemma4", isDirectory: true)
        try writeMinimalTextRoot(at: modelRoot, id: .gemma4)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.gemma4)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testGemma4AliasFallsBackToMaxInstall() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("text-chat-gemma4-max", isDirectory: true)
        try writeMinimalTextRoot(at: modelRoot, id: .gemma4Max)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.gemma4)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesGemma4NanoFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("text-chat-gemma4-nano", isDirectory: true)
        try writeMinimalTextRoot(at: modelRoot, id: .gemma4Nano)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.gemma4Nano)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesGemma4TurboFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("text-chat-gemma4-turbo", isDirectory: true)
        try writeMinimalTextRoot(at: modelRoot, id: .gemma4Turbo)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.gemma4Turbo)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesVisionSegmentSAM31FromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("vision-segment-sam31", isDirectory: true)
        try writeMinimalSAM31Model(at: modelRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.visionSegmentSAM31)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }

    func testResolvesVisionGroundFalconPerceptionFromProcessModelStoreOverride() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalFalconModel(at: modelRoot)

        let resolver = ModelResolver()
        let resolved = try resolver.resolve(.visionGroundFalconPerception)

        XCTAssertEqual(resolved.rootURL.standardizedFileURL, modelRoot.standardizedFileURL)
        XCTAssertEqual(resolved.source, .localModelStore)
    }
}
