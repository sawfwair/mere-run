import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class SFXGenerateCommandParsingTests: XCTestCase {
    func testSFXGenerateParsesDefaults() throws {
        let cmd = try SFXGenerate.parse([
            "door slam in a concrete stairwell",
        ])

        XCTAssertEqual(cmd.prompt, "door slam in a concrete stairwell")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDFlow.rawValue)
        XCTAssertEqual(cmd.durationSeconds, 5.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 4.5, accuracy: 0.0001)
        XCTAssertNil(cmd.seed)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(), [])
    }

    func testSFXGenerateParsesWooshOverrides() throws {
        let cmd = try SFXGenerate.parse([
            "metal wrench dropping onto concrete",
            "--model", "/tmp/woosh",
            "--duration", "7.5",
            "--steps", "4",
            "--cfg", "3.25",
            "--seed", "42",
            "--renoise", "0,0.5,0.5,0.3",
            "-o", "/tmp/wrench-clang.wav",
        ])

        XCTAssertEqual(cmd.model, "/tmp/woosh")
        XCTAssertEqual(cmd.durationSeconds, 7.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 3.25, accuracy: 0.0001)
        XCTAssertEqual(cmd.seed, 42)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(), [0, 0.5, 0.5, 0.3])
        XCTAssertEqual(cmd.output, "/tmp/wrench-clang.wav")
    }

    func testSFXGenerateParsesMMAudioConditioningOptions() throws {
        let cmd = try SFXGenerate.parse([
            "waves breaking against a stone pier",
            "--negative-prompt", "speech, music",
            "--model", MMAudioResources.modelID,
            "--duration", "8",
            "--steps", "25",
            "--cfg", "4.5",
        ])

        XCTAssertEqual(cmd.negativePrompt, "speech, music")
        XCTAssertEqual(cmd.model, MMAudioResources.modelID)
        XCTAssertEqual(cmd.durationSeconds, 8)
        XCTAssertEqual(cmd.steps, 25)
        XCTAssertEqual(cmd.guidanceScale, 4.5)
    }

    func testSFXGenerateUsesModelSpecificMMAudioDefaults() throws {
        let cmd = try SFXGenerate.parse([
            "waves breaking against a stone pier",
            "--model", MMAudioResources.modelID,
        ])

        XCTAssertEqual(cmd.durationSeconds, MMAudioResources.defaultDurationSeconds)
        XCTAssertEqual(cmd.steps, MMAudioResources.defaultSteps)
        XCTAssertEqual(cmd.guidanceScale, MMAudioResources.defaultGuidanceScale)
    }

    func testSFXGeneratePreservesExplicitWooshDefaultsForMMAudio() throws {
        let cmd = try SFXGenerate.parse([
            "short impact",
            "--model", MMAudioResources.modelID,
            "--duration", "5",
            "--steps", "4",
        ])

        XCTAssertEqual(cmd.durationSeconds, 5)
        XCTAssertEqual(cmd.steps, 4)
    }

    func testSFXGenerateRejectsWrongRenoiseCount() throws {
        let cmd = try SFXGenerate.parse([
            "glass break",
            "--steps", "4",
            "--renoise", "0.1,0.2",
        ])

        XCTAssertThrowsError(try cmd.parseRenoiseSchedule())
    }

    func testSFXAEEncodeParsesOptions() throws {
        let cmd = try SFXAEEncode.parse([
            "/tmp/input.wav",
            "--model", ModelResolver.ModelID.wooshFlow.rawValue,
            "-o", "/tmp/latents.npy",
            "--quiet",
        ])

        XCTAssertEqual(cmd.input, "/tmp/input.wav")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/latents.npy")
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXAEDecodeParsesOptions() throws {
        let cmd = try SFXAEDecode.parse([
            "/tmp/latents.npy",
            "--model", ModelResolver.ModelID.wooshDFlow.rawValue,
            "-o", "/tmp/output.wav",
        ])

        XCTAssertEqual(cmd.input, "/tmp/latents.npy")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/output.wav")
        XCTAssertFalse(cmd.quiet)
    }

    func testSFXConditionTextParsesOptions() throws {
        let cmd = try SFXConditionText.parse([
            "glass breaking",
            "--model", ModelResolver.ModelID.wooshFlow.rawValue,
            "-o", "/tmp/condition.safetensors",
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "glass breaking")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshFlow.rawValue)
        XCTAssertEqual(cmd.output, "/tmp/condition.safetensors")
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXCLAPScoreParsesOptions() throws {
        let cmd = try SFXCLAPScoreCommand.parse([
            "glass breaking",
            "/tmp/glass.wav",
            "--model", ModelResolver.ModelID.wooshClap.rawValue,
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "glass breaking")
        XCTAssertEqual(cmd.audio, "/tmp/glass.wav")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshClap.rawValue)
        XCTAssertTrue(cmd.quiet)
    }

    func testSFXVideoGenerateParsesOptions() throws {
        let cmd = try SFXVideoGenerate.parse([
            "footsteps echoing in a hallway",
            "/tmp/synch.npy",
            "--model", ModelResolver.ModelID.wooshDVFlow8s.rawValue,
            "--synchformer-model", ModelResolver.ModelID.wooshSynchformer.rawValue,
            "--steps", "4",
            "--cfg", "3",
            "--renoise", "0,0.5,0.5,0.3",
            "--sync-batch-size", "2",
            "-o", "/tmp/video-sfx.wav",
            "--quiet",
        ])

        XCTAssertEqual(cmd.prompt, "footsteps echoing in a hallway")
        XCTAssertEqual(cmd.input, "/tmp/synch.npy")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.wooshDVFlow8s.rawValue)
        XCTAssertEqual(cmd.synchformerModel, ModelResolver.ModelID.wooshSynchformer.rawValue)
        XCTAssertEqual(cmd.steps, 4)
        XCTAssertEqual(cmd.guidanceScale, 3)
        XCTAssertEqual(cmd.syncBatchSize, 2)
        XCTAssertEqual(cmd.clipBatchSize, 4)
        XCTAssertEqual(try cmd.parseRenoiseSchedule(steps: 4), [0, 0.5, 0.5, 0.3])
        XCTAssertEqual(cmd.output, "/tmp/video-sfx.wav")
        XCTAssertTrue(cmd.quiet)
        XCTAssertFalse(cmd.preflight)
        XCTAssertFalse(cmd.json)
    }

    func testSFXVideoGenerateParsesMMAudioOptions() throws {
        let cmd = try SFXVideoGenerate.parse([
            "a skateboard rolling over rough pavement",
            "/tmp/skateboard.mp4",
            "--negative-prompt", "music",
            "--model", MMAudioResources.modelID,
            "--clip-batch-size", "3",
            "--sync-batch-size", "2",
        ])

        XCTAssertEqual(cmd.negativePrompt, "music")
        XCTAssertEqual(cmd.model, MMAudioResources.modelID)
        XCTAssertEqual(cmd.clipBatchSize, 3)
        XCTAssertEqual(cmd.syncBatchSize, 2)
    }

    func testSFXVideoPreflightReportsNativeMMAudioPlan() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("mmaudio", isDirectory: true)
        try writeMinimalMMAudioModel(at: modelRoot)
        let videoURL = try makeTempFile(name: "skateboard.mp4", in: temp)
        let outputURL = temp.appendingPathComponent("skateboard.wav")
        let cmd = try SFXVideoGenerate.parse([
            "a skateboard rolling over rough pavement",
            videoURL.path,
            "--model", modelRoot.path,
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: videoURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.model.variant, "large_44k_v2")
        XCTAssertTrue(envelope.result.model.supportedForVideo == true)
        XCTAssertNil(envelope.result.synchformer)
        XCTAssertFalse(envelope.result.input.requiresSynchformer)
        XCTAssertEqual(envelope.result.plan.effectiveSteps, MMAudioResources.defaultSteps)
        XCTAssertEqual(
            envelope.result.plan.effectiveGuidanceScale,
            MMAudioResources.defaultGuidanceScale,
            accuracy: 0.0001
        )
        XCTAssertEqual(envelope.result.plan.sampleRate, MMAudioResources.sampleRate)
    }

    func testSFXVideoGenerateParsesPreflightJSONFlags() throws {
        let cmd = try SFXVideoGenerate.parse([
            "footsteps echoing in a hallway",
            "/tmp/synch.npy",
            "--preflight",
            "--json",
        ])

        XCTAssertTrue(cmd.preflight)
        XCTAssertTrue(cmd.json)
    }

    func testSFXVideoPreflightReportsRawVideoPlan() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("woosh", isDirectory: true)
        try writeMinimalWooshModel(at: modelRoot, variant: .dvflow8s)
        let synchformerRoot = temp.appendingPathComponent("synchformer", isDirectory: true)
        try writeMinimalSynchformer(at: synchformerRoot)
        let videoURL = try makeTempFile(name: "silent-hallway.mp4", in: temp)
        let outputURL = temp.appendingPathComponent("hallway-footsteps.wav")
        let cmd = try SFXVideoGenerate.parse([
            "footsteps echoing in a hallway",
            videoURL.path,
            "--model", modelRoot.path,
            "--synchformer-model", synchformerRoot.path,
            "--duration", "8",
            "--renoise", "0,0.5,0.5,0.3",
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: videoURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.command, ["sfx", "video", "generate"])
        XCTAssertEqual(envelope.mode, .preflight)
        XCTAssertEqual(envelope.result.input.kind, "video")
        XCTAssertTrue(envelope.result.input.requiresSynchformer)
        XCTAssertEqual(envelope.result.model.kind, "local_path")
        XCTAssertEqual(envelope.result.model.variant, WooshVariant.dvflow8s.rawValue)
        XCTAssertTrue(envelope.result.model.installed)
        XCTAssertEqual(envelope.result.synchformer?.kind, "local_path")
        XCTAssertTrue(envelope.result.synchformer?.installed == true)
        XCTAssertEqual(envelope.result.plan.effectiveSteps, 4)
        XCTAssertEqual(envelope.result.plan.effectiveGuidanceScale, 3.0, accuracy: 0.0001)
        XCTAssertEqual(envelope.result.plan.renoiseSchedule, [0, 0.5, 0.5, 0.3])
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-sfx-video-generation" && $0.enabled })

        let encoded = try StructuredRunOutput.encode(envelope)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(SFXVideoPreflightEnvelope.self, from: Data(encoded.utf8))
        XCTAssertEqual(decoded, envelope)
    }

    func testSFXVideoPreflightFeatureInputDoesNotRequireSynchformer() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("woosh", isDirectory: true)
        try writeMinimalWooshModel(at: modelRoot, variant: .vflow8s)
        let featuresURL = try makeTempFile(name: "features.npy", in: temp)
        let outputURL = temp.appendingPathComponent("features.wav")
        let cmd = try SFXVideoGenerate.parse([
            "mechanical whir",
            featuresURL.path,
            "--model", modelRoot.path,
            "--synchformer-model", temp.appendingPathComponent("missing-synchformer").path,
            "--steps", "2",
            "--cfg", "4.25",
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: featuresURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .ok)
        XCTAssertEqual(envelope.result.input.kind, "features")
        XCTAssertFalse(envelope.result.input.requiresSynchformer)
        XCTAssertNil(envelope.result.synchformer)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "feature_shape_unverified" })
        XCTAssertEqual(envelope.result.plan.effectiveSteps, 2)
        XCTAssertEqual(envelope.result.plan.effectiveGuidanceScale, 4.25, accuracy: 0.0001)
    }

    func testSFXVideoPreflightBlocksMissingInput() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("woosh", isDirectory: true)
        try writeMinimalWooshModel(at: modelRoot, variant: .dvflow8s)
        let synchformerRoot = temp.appendingPathComponent("synchformer", isDirectory: true)
        try writeMinimalSynchformer(at: synchformerRoot)
        let inputURL = temp.appendingPathComponent("missing.mp4")
        let outputURL = temp.appendingPathComponent("out.wav")
        let cmd = try SFXVideoGenerate.parse([
            "footsteps echoing in a hallway",
            inputURL.path,
            "--model", modelRoot.path,
            "--synchformer-model", synchformerRoot.path,
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: inputURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "input_missing" })
        XCTAssertTrue(envelope.actions.contains { $0.id == "start-sfx-video-generation" && !$0.enabled })
    }

    func testSFXVideoPreflightBlocksWrongWooshVariant() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("woosh", isDirectory: true)
        try writeMinimalWooshModel(at: modelRoot, variant: .dflow)
        let featuresURL = try makeTempFile(name: "features.npy", in: temp)
        let outputURL = temp.appendingPathComponent("out.wav")
        let cmd = try SFXVideoGenerate.parse([
            "mechanical whir",
            featuresURL.path,
            "--model", modelRoot.path,
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: featuresURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "model_not_video_to_audio" })
    }

    func testSFXVideoPreflightBlocksInvalidRenoiseSchedule() throws {
        let temp = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let modelRoot = temp.appendingPathComponent("woosh", isDirectory: true)
        try writeMinimalWooshModel(at: modelRoot, variant: .dvflow8s)
        let featuresURL = try makeTempFile(name: "features.npy", in: temp)
        let outputURL = temp.appendingPathComponent("out.wav")
        let cmd = try SFXVideoGenerate.parse([
            "mechanical whir",
            featuresURL.path,
            "--model", modelRoot.path,
            "--steps", "4",
            "--renoise", "0.1,0.2",
            "--output", outputURL.path,
            "--preflight",
            "--json",
        ])

        let envelope = cmd.makePreflightEnvelope(
            inputURL: featuresURL,
            outputURL: outputURL,
            fileManager: .default,
            now: { Date(timeIntervalSince1970: 0) }
        )

        XCTAssertEqual(envelope.status, .blocked)
        XCTAssertTrue(envelope.diagnostics.contains { $0.id == "renoise_invalid" })
    }

    private func makeTempDir() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-sfx-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeTempFile(name: String, in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data("fixture".utf8).write(to: url)
        return url
    }

    private func writeMinimalWooshModel(at root: URL, variant: WooshVariant) throws {
        for component in variant.requiredComponents {
            let componentRoot = root.appendingPathComponent(component, isDirectory: true)
            try FileManager.default.createDirectory(at: componentRoot, withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: componentRoot.appendingPathComponent("config.yaml"))
            try Data("fixture".utf8).write(to: componentRoot.appendingPathComponent("weights.safetensors"))
        }
        let conditioner = variant == .vflow8s || variant == .dvflow8s ? "TextConditionerV" : "TextConditionerA"
        let tokenizer = root
            .appendingPathComponent(conditioner, isDirectory: true)
            .appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))
    }

    private func writeMinimalSynchformer(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(
            to: root.appendingPathComponent(WooshResources.synchformerFilename)
        )
    }

    private func writeMinimalMMAudioModel(at root: URL) throws {
        let resources = MMAudioModelResources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.clipTokenizerURL,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: resources.bigVGANURL,
            withIntermediateDirectories: true
        )
        for url in [
            resources.networkWeightsURL,
            resources.clipWeightsURL,
            resources.synchformerWeightsURL,
            resources.vaeWeightsURL,
            resources.clipTokenizerURL.appendingPathComponent("tokenizer.json"),
            resources.bigVGANConfigURL,
            resources.bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename),
        ] {
            XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data()))
        }
    }
}
