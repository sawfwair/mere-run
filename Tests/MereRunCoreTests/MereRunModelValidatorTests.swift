import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelValidatorTests: MereRunCoreTestCase {

    private func writeMinimalValidModel(at root: URL, id: ModelResolver.ModelID) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        // Root marker
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        // Required components
        for component in ["transformer", "text_encoder", "vae"] {
            let dir = root.appendingPathComponent(component, isDirectory: true)
            try TestFileSystem.createDirectory(dir)
            try TestFileSystem.writeFile(dir.appendingPathComponent("config.json"), contents: Data("{}".utf8))
            try TestFileSystem.writeFile(dir.appendingPathComponent("model.safetensors"))
        }

        // Optional-but-expected components (warnings if missing)
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))

        let schedulerDir = root.appendingPathComponent("scheduler", isDirectory: true)
        try TestFileSystem.createDirectory(schedulerDir)
        try TestFileSystem.writeFile(schedulerDir.appendingPathComponent("scheduler_config.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidQ35FamilyModel(at root: URL, id: ModelResolver.ModelID = .q36Nano) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidLFM2Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .lfm25A1B8Bit, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data(#"{"model_type":"lfm2_moe"}"#.utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidGemma4Model(at root: URL, id: ModelResolver.ModelID = .gemma4) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidGemma4MTPAssistant(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .gemma4TwelveBMTP, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidHiDreamO1Model(at root: URL, id: ModelResolver.ModelID = .hidreamO1Dev) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("preprocessor_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidKrea2Model(
        at root: URL,
        id: ModelResolver.ModelID = .krea2Turbo
    ) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: id, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        let textEncoderDir = root.appendingPathComponent("text_encoder", isDirectory: true)
        let transformerDir = root.appendingPathComponent("transformer", isDirectory: true)
        let vaeDir = root.appendingPathComponent("vae", isDirectory: true)
        let schedulerDir = root.appendingPathComponent("scheduler", isDirectory: true)

        for directory in [tokenizerDir, textEncoderDir, transformerDir, vaeDir, schedulerDir] {
            try TestFileSystem.createDirectory(directory)
        }
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(textEncoderDir.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(textEncoderDir.appendingPathComponent("model.safetensors"), contents: Data())
        try TestFileSystem.writeFile(transformerDir.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(
            transformerDir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"),
            contents: Data("{}".utf8)
        )
        try TestFileSystem.writeFile(
            transformerDir.appendingPathComponent("diffusion_pytorch_model-00001-of-00003.safetensors"),
            contents: Data()
        )
        try TestFileSystem.writeFile(vaeDir.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(vaeDir.appendingPathComponent("diffusion_pytorch_model.safetensors"), contents: Data())
        try TestFileSystem.writeFile(schedulerDir.appendingPathComponent("scheduler_config.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidSAM31Model(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionSegmentSAM31, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidFalconModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidPrivacyFilterModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .privacyFilter, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidQwen3ASRModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .qwen3ASR, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        try TestFileSystem.writeFile(
            root.appendingPathComponent("config.json"),
            contents: Data(#"{"quantization":{"bits":8,"group_size":64}}"#.utf8)
        )
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("tokenizer_config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors.index.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidQwen35AgentGGUFModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .qwen35Agent9B, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("Qwen3.5-9B-Q4_K_M.gguf"), contents: Data())
    }

    private func writeMinimalValidMagentaRT2Model(at root: URL, modelID: ModelResolver.ModelID = .magentaRT2Small) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        let modelName = MagentaRT2Resources.modelName(for: modelID.rawValue)
        let modelDir = root.appendingPathComponent("models/\(modelName)", isDirectory: true)
        let musicCoCa = root.appendingPathComponent("resources/musiccoca", isDirectory: true)
        let spectrostream = root.appendingPathComponent("resources/spectrostream", isDirectory: true)
        for directory in [modelDir, musicCoCa, spectrostream] {
            try TestFileSystem.createDirectory(directory)
        }
        try TestFileSystem.writeFile(modelDir.appendingPathComponent("\(modelName).mlxfn"), contents: Data())
        try TestFileSystem.writeFile(modelDir.appendingPathComponent("\(modelName)_state.safetensors"), contents: Data())
        for file in [
            "audio_preprocessor.tflite",
            "mapper.tflite",
            "music_encoder.tflite",
            "pretrained_vector_quantizer.tflite",
            "spm.model",
            "text_encoder.tflite",
        ] {
            try TestFileSystem.writeFile(musicCoCa.appendingPathComponent(file), contents: Data())
        }
        for file in [
            "decoder.safetensors",
            "encoder.safetensors",
            "quantizer.safetensors",
            "spectrostream_encoder.mlxfn",
        ] {
            try TestFileSystem.writeFile(spectrostream.appendingPathComponent(file), contents: Data())
        }
    }

    private func writeMinimalValidMuScriptorModel(
        at root: URL,
        modelID: ModelResolver.ModelID = .muScriptorLarge
    ) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("config.json"), contents: Data("{}".utf8))
        try TestFileSystem.writeFile(root.appendingPathComponent("model.safetensors"), contents: Data())
    }

    private func writeMinimalValidWooshDFlowModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .wooshDFlow, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)

        let checkpoints = root.appendingPathComponent("checkpoints", isDirectory: true)
        for component in ["Woosh-DFlow", "Woosh-AE", "TextConditionerA"] {
            let componentDir = checkpoints.appendingPathComponent(component, isDirectory: true)
            try TestFileSystem.createDirectory(componentDir)
            try TestFileSystem.writeFile(componentDir.appendingPathComponent("config.yaml"), contents: Data("{}".utf8))
            try TestFileSystem.writeFile(componentDir.appendingPathComponent("weights.safetensors"), contents: Data())
        }
        let tokenizerDir = checkpoints.appendingPathComponent("TextConditionerA/tokenizer", isDirectory: true)
        try TestFileSystem.createDirectory(tokenizerDir)
        try TestFileSystem.writeFile(tokenizerDir.appendingPathComponent("tokenizer.json"), contents: Data("{}".utf8))
    }

    private func writeMinimalValidLTX23MLXModel(at root: URL) throws {
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(
            for: .ltxVideo23AVMLX,
            createdAt: Date(timeIntervalSince1970: 0)
        ).write(to: root)

        for file in [
            "config.json",
            "embedded_config.json",
            "split_model.json",
            "connector.safetensors",
            "transformer-distilled.safetensors",
            "vae_decoder.safetensors",
            "vae_encoder.safetensors",
            "audio_vae.safetensors",
            "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors",
            "spatial_upscaler_x2_v1_1_config.json",
            "spatial_upscaler_x1_5_v1_0.safetensors",
            "spatial_upscaler_x1_5_v1_0_config.json",
            "temporal_upscaler_x2_v1_0.safetensors",
            "temporal_upscaler_x2_v1_0_config.json",
        ] {
            try TestFileSystem.writeFile(
                root.appendingPathComponent(file),
                contents: file.hasSuffix(".json") ? Data(#"{"model_version":"2.3.0"}"#.utf8) : Data()
            )
        }
    }

    func testValidModelPasses() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-nano")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testGeometryAndDepthManagedModelsUseDirectArtifactValidation() throws {
        let cases: [(ModelResolver.ModelID, String)] = [
            (.visionGeometryMoGe2Small, "model.onnx"),
            (.visionDepthVDASmall, "video_depth_anything_vits.pth"),
        ]
        for (modelID, artifact) in cases {
            let root = try TestFileSystem.makeTempDir()
            defer { try? FileManager.default.removeItem(at: root) }
            try MereRunModelManifest.template(
                for: modelID,
                createdAt: Date(timeIntervalSince1970: 0)
            ).write(to: root)
            try TestFileSystem.writeFile(root.appendingPathComponent(artifact))

            let report = MereRunModelValidator.validate(
                modelRoot: root,
                expectedModelID: modelID.rawValue
            )
            XCTAssertTrue(report.isValid, "\(modelID.rawValue): \(report.errors)")
            XCTAssertFalse(report.warnings.contains { $0.contains("model root marker") })
            XCTAssertFalse(report.errors.contains { $0.contains("text_encoder") || $0.contains("tokenizer") })
        }
    }

    func testMagentaRT2RootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("music-magenta-rt2-small", isDirectory: true)
        try writeMinimalValidMagentaRT2Model(at: root)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.magentaRT2Small.rawValue
        )
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertFalse(report.warnings.contains { $0.contains("model root marker") })
        XCTAssertEqual(report.manifest?.engine, .magentaRT2)
        XCTAssertEqual(report.manifest?.family, .music)
        XCTAssertEqual(report.manifest?.tier, .small)
    }

    func testMuScriptorRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(ModelResolver.ModelID.muScriptorLarge.rawValue, isDirectory: true)
        try writeMinimalValidMuScriptorModel(at: root)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.muScriptorLarge.rawValue
        )
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertFalse(report.warnings.contains { $0.contains("engine mismatch") })
        XCTAssertEqual(report.manifest?.engine, .muScriptor)
        XCTAssertEqual(report.manifest?.family, .music)
        XCTAssertEqual(report.manifest?.tier, .max)
    }

    func testMuScriptorMissingWeightsFailsValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(ModelResolver.ModelID.muScriptorLarge.rawValue, isDirectory: true)
        try writeMinimalValidMuScriptorModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("model.safetensors"))

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.muScriptorLarge.rawValue
        )
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("model.safetensors") })
    }

    func testWooshDFlowRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(ModelResolver.ModelID.wooshDFlow.rawValue, isDirectory: true)
        try writeMinimalValidWooshDFlowModel(at: root)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.wooshDFlow.rawValue
        )
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertFalse(report.warnings.contains { $0.contains("model root marker") })
        XCTAssertFalse(report.errors.contains { $0.contains("text_encoder/config.json") })
        XCTAssertEqual(report.manifest?.engine, .woosh)
        XCTAssertEqual(report.manifest?.family, .sfx)
    }

    func testMagentaRT2MissingRuntimeAssetFailsValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("music-magenta-rt2-small", isDirectory: true)
        try writeMinimalValidMagentaRT2Model(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("models/mrt2_small/mrt2_small.mlxfn"))

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.magentaRT2Small.rawValue
        )
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("mrt2_small.mlxfn") })
    }

    func testLTX23MLXRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(ModelResolver.ModelID.ltxVideo23AVMLX.rawValue, isDirectory: true)
        try writeMinimalValidLTX23MLXModel(at: root)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: ModelResolver.ModelID.ltxVideo23AVMLX.rawValue
        )
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.engine, .ltxVideo)
        XCTAssertEqual(report.manifest?.family, .video)
        XCTAssertEqual(report.manifest?.upstreamRepoId, "dgrauet/ltx-2.3-mlx@main")
    }

    func testMissingWeightsFails() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        // Remove transformer weights to trigger a validation error.
        try FileManager.default.removeItem(at: root.appendingPathComponent("transformer/model.safetensors"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-nano")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains("No *.safetensors weights found in transformer/"))
    }

    func testManifestIdMismatchFails() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-klein-nano", isDirectory: true)
        try writeMinimalValidModel(at: root, id: .kleinNano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-klein-max")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains("Manifest id mismatch: expected=image-klein-max found=image-klein-nano"))
    }

    func testQ36NanoChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-q36-nano", isDirectory: true)
        try writeMinimalValidQ35FamilyModel(at: root, id: .q36Nano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-q36-nano")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.tier, .nano)
        XCTAssertEqual(report.manifest?.family, .qwen)
        XCTAssertEqual(report.manifest?.upstreamRepoId, "\(Q35Resources.q36NanoUpstreamRepoId)@\(Q35Resources.q36NanoUpstreamRevision)")
    }

    func testOrnithQ35CodeRootLayoutPassesValidationWithoutEngineWarning() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(Q35Resources.ornith9BModelId, isDirectory: true)
        try writeMinimalValidQ35FamilyModel(at: root, id: .ornith9B)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Q35Resources.ornith9BModelId)
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .code)
        XCTAssertFalse(report.warnings.contains { $0.contains("family=code expects") })
    }

    func testOrnith35BMLXQ35RootLayoutPassesValidationWithoutEngineWarning() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(Q35Resources.ornith35BMLXModelId, isDirectory: true)
        try writeMinimalValidQ35FamilyModel(at: root, id: .ornith35BMLX)

        let report = MereRunModelValidator.validate(
            modelRoot: root,
            expectedModelID: Q35Resources.ornith35BMLXModelId
        )
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .code)
        XCTAssertEqual(report.manifest?.tier, .small)
        XCTAssertFalse(report.warnings.contains { $0.contains("family=code expects") })
    }

    func testLFM2ChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-lfm25-a1b-8bit", isDirectory: true)
        try writeMinimalValidLFM2Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: LFM2Resources.defaultModelId)
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.engine, .lfm2)
        XCTAssertEqual(report.manifest?.family, .liquid)
        XCTAssertEqual(report.manifest?.precision, .int8)
        XCTAssertEqual(report.manifest?.upstreamRepoId, "\(LFM2Resources.upstreamRepoId)@\(LFM2Resources.upstreamRevision)")
    }

    func testGemma4ChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .gemma)
    }

    func testHiDreamO1UnifiedRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-hidream-o1-dev", isDirectory: true)
        try writeMinimalValidHiDreamO1Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-hidream-o1-dev")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .hidream)
        XCTAssertEqual(report.manifest?.engine, .hidreamO1)
        XCTAssertEqual(report.manifest?.defaults?.steps, 28)
    }

    func testHiDreamO1FullUnifiedRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("image-hidream-o1", isDirectory: true)
        try writeMinimalValidHiDreamO1Model(at: root, id: .hidreamO1)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "image-hidream-o1")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .hidream)
        XCTAssertEqual(report.manifest?.engine, .hidreamO1)
        XCTAssertEqual(report.manifest?.defaults?.steps, 50)
        XCTAssertEqual(report.manifest?.defaults?.cfg, 5.0)
    }

    func testKrea2TurboRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(Krea2Resources.modelId, isDirectory: true)
        try writeMinimalValidKrea2Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Krea2Resources.modelId)
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .krea)
        XCTAssertEqual(report.manifest?.engine, .krea2)
        XCTAssertEqual(report.manifest?.defaults?.steps, 8)
        XCTAssertEqual(report.manifest?.defaults?.cfg, 0.0)
        XCTAssertEqual(report.manifest?.defaults?.sigmaShift, Double(Krea2SampleBuilder.defaultMu))
    }

    func testKrea2RawRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent(Krea2RawResources.modelId, isDirectory: true)
        try writeMinimalValidKrea2Model(at: root, id: .krea2Raw)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Krea2RawResources.modelId)
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .krea)
        XCTAssertEqual(report.manifest?.engine, .krea2)
        XCTAssertEqual(report.manifest?.variant, .base)
        XCTAssertEqual(report.manifest?.defaults?.steps, 52)
        XCTAssertEqual(report.manifest?.defaults?.cfg, 3.5)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.txt2img, .loraTraining]))
    }

    func testKrea2TurboSymlinkedComponentLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let snapshot = temp.appendingPathComponent("snapshot", isDirectory: true)
        try writeMinimalValidKrea2Model(at: snapshot)

        let root = temp.appendingPathComponent(Krea2Resources.modelId, isDirectory: true)
        try TestFileSystem.createDirectory(root)
        try MereRunModelManifest.template(for: .krea2Turbo, createdAt: Date(timeIntervalSince1970: 0)).write(to: root)
        try TestFileSystem.writeFile(root.appendingPathComponent("model_index.json"), contents: Data("{}".utf8))

        for component in ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"] {
            try FileManager.default.createSymbolicLink(
                at: root.appendingPathComponent(component, isDirectory: true),
                withDestinationURL: snapshot.appendingPathComponent(component, isDirectory: true)
            )
        }

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Krea2Resources.modelId)
        XCTAssertTrue(report.isValid, report.errors.joined(separator: "\n"))
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testGemma4MaxChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4-max", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root, id: .gemma4Max)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4-max")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.tier, .max)
        XCTAssertEqual(report.manifest?.family, .gemma)
    }

    func testGemma4NanoChatOnlyRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4-nano", isDirectory: true)
        try writeMinimalValidGemma4Model(at: root, id: .gemma4Nano)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-chat-gemma4-nano")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.tier, .nano)
        XCTAssertEqual(report.manifest?.family, .gemma)
    }

    func testGemma4MTPAssistantRootLayoutPassesValidationWithoutManifestComponents() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-chat-gemma4-12b-mtp", isDirectory: true)
        try writeMinimalValidGemma4MTPAssistant(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: Gemma4MTPResources.modelId)
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertNil(report.manifest?.components)
    }

    func testSAM31VisionSegmentationRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-segment-sam31", isDirectory: true)
        try writeMinimalValidSAM31Model(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-segment-sam31")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.visionSegmentation, .visionTracking]))
    }

    func testFalconVisionGroundRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .falcon)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.visionGrounding, .visionDetection, .visionSegmentation]))
    }

    func testFalconValidationFailsWithoutTokenizer() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer/tokenizer.json"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("tokenizer.json") })
    }

    func testFalconValidationFailsWithoutConfig() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("config.json"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("config.json") })
    }

    func testFalconValidationFailsWithoutWeights() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("vision-ground-falcon-perception", isDirectory: true)
        try writeMinimalValidFalconModel(at: root)
        try FileManager.default.removeItem(at: root.appendingPathComponent("model.safetensors"))

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "vision-ground-falcon-perception")
        XCTAssertFalse(report.isValid)
        XCTAssertTrue(report.errors.contains { $0.contains("model.safetensors.index.json") })
    }

    func testPrivacyFilterRootLayoutPassesValidation() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-anonymize-privacy-filter", isDirectory: true)
        try writeMinimalValidPrivacyFilterModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-anonymize-privacy-filter")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertEqual(report.manifest?.family, .privacy)
        XCTAssertEqual(Set(report.manifest?.supports ?? []), Set([.textAnonymization]))
    }

    func testQwen3ASRUsesHFQuantizationConfigWithoutManifestQuantization() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("speech-asr-qwen3", isDirectory: true)
        try writeMinimalValidQwen3ASRModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "speech-asr-qwen3")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
    }

    func testGGUFCodeModelDoesNotRequireManifestQuantization() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let root = temp.appendingPathComponent("text-agent-qwen35-9b", isDirectory: true)
        try writeMinimalValidQwen35AgentGGUFModel(at: root)

        let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: "text-agent-qwen35-9b")
        XCTAssertTrue(report.isValid)
        XCTAssertTrue(report.errors.isEmpty)
        XCTAssertFalse(report.warnings.contains("Missing model root marker (expected model_index.json)."))
    }
}
