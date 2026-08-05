import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class MusicAnalyzeCommandParsingTests: XCTestCase {
    func testMusicAnalyzeParsesDefaults() throws {
        let cmd = try MusicAnalyze.parse([
            "/tmp/song.mp3",
        ])

        XCTAssertEqual(cmd.audio, "/tmp/song.mp3")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertNil(cmd.checkpointsRoot)
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae")
        XCTAssertNil(cmd.lmSubdirectory)
        XCTAssertNil(cmd.lmModel)
        XCTAssertNil(cmd.durationSeconds)
        XCTAssertEqual(cmd.maxNewTokens, 2048)
        XCTAssertEqual(cmd.lmTemperature, 0.3, accuracy: 0.0001)
        XCTAssertEqual(cmd.lmTopK, 0)
        XCTAssertEqual(cmd.lmTopP, 0.9, accuracy: 0.0001)
        XCTAssertFalse(cmd.includeRawLM)
        XCTAssertFalse(cmd.includeAudioCodes)
        XCTAssertFalse(cmd.quiet)
    }

    func testMusicAnalyzeParsesModelAndLMOverrides() throws {
        let cmd = try MusicAnalyze.parse([
            "/tmp/song.mp3",
            "--model", ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue,
            "--checkpoints-root", "/tmp/acestep",
            "--turbo-subdirectory", "acestep-v15-xl-turbo",
            "--vae-subdirectory", "vae-custom",
            "--lm-subdirectory", "acestep-5Hz-lm-4B",
            "--duration", "30",
            "--max-new-tokens", "512",
            "--lm-temperature", "0.2",
            "--lm-top-k", "32",
            "--lm-top-p", "0.75",
            "--include-raw-lm",
            "--include-audio-codes",
            "--quiet",
        ])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStepXLTurboLM4B.rawValue)
        XCTAssertEqual(cmd.checkpointsRoot, "/tmp/acestep")
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-xl-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae-custom")
        XCTAssertEqual(cmd.lmSubdirectory, "acestep-5Hz-lm-4B")
        XCTAssertEqual(cmd.durationSeconds, 30)
        XCTAssertEqual(cmd.maxNewTokens, 512)
        XCTAssertEqual(cmd.lmTemperature, 0.2, accuracy: 0.0001)
        XCTAssertEqual(cmd.lmTopK, 32)
        XCTAssertEqual(cmd.lmTopP, 0.75, accuracy: 0.0001)
        XCTAssertTrue(cmd.includeRawLM)
        XCTAssertTrue(cmd.includeAudioCodes)
        XCTAssertTrue(cmd.quiet)
    }

    func testMusicAnalyzeParsesIndependentPlannerModel() throws {
        let cmd = try MusicAnalyze.parse([
            "/tmp/song.mp3",
            "--model", ModelResolver.ModelID.aceStepXLSFT.rawValue,
            "--lm-model", ModelResolver.ModelID.aceStepLM4B.rawValue,
        ])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStepXLSFT.rawValue)
        XCTAssertEqual(cmd.lmModel, ModelResolver.ModelID.aceStepLM4B.rawValue)
        XCTAssertNil(cmd.lmSubdirectory)
    }

    func testACEPlannerDiscoveryPrefersUpstreamDefault17B() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let planner17 = root.appendingPathComponent("acestep-5Hz-lm-1.7B", isDirectory: true)
        let planner4 = root.appendingPathComponent("acestep-5Hz-lm-4B", isDirectory: true)
        try writeMinimalLM(at: planner17)
        try writeMinimalLM(at: planner4)

        XCTAssertEqual(
            try ACEStepCLIHelper.resolveLMSubdirectory(at: root, explicit: nil),
            "acestep-5Hz-lm-1.7B"
        )
        XCTAssertEqual(ACEStepCLIHelper.resolveLMRoot(at: planner17), planner17)
    }

    func testACEPlannerResolutionReusesInstalled17BWithoutCheckpointSymlink() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: root)
        }
        let models = root.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(models)

        let selectedCheckpoint = root.appendingPathComponent("xl-sft", isDirectory: true)
        try FileManager.default.createDirectory(
            at: selectedCheckpoint,
            withIntermediateDirectories: true
        )
        let base = models.appendingPathComponent(
            ModelResolver.ModelID.aceStep.rawValue,
            isDirectory: true
        )
        for subdirectory in [
            "acestep-v15-turbo",
            "vae",
            "Qwen3-Embedding-0.6B",
        ] {
            try FileManager.default.createDirectory(
                at: base.appendingPathComponent(subdirectory, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        let planner = base.appendingPathComponent("acestep-5Hz-lm-1.7B", isDirectory: true)
        try writeMinimalLM(at: planner)
        try MereRunModelManifest.template(for: .aceStep).write(to: base)

        let resolved = try await ACEStepCLIHelper.resolveLMResources(
            checkpointsRoot: selectedCheckpoint,
            lmModel: nil,
            lmSubdirectory: nil
        )

        XCTAssertEqual(resolved.source, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertEqual(resolved.rootURL, planner)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: selectedCheckpoint
                    .appendingPathComponent("acestep-5Hz-lm-1.7B")
                    .path
            )
        )
    }

    func testACEPlannerResolutionLabelsExplicitLocalRootTruthfully() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpoint = directory.appendingPathComponent("xl-sft", isDirectory: true)
        let planner = directory.appendingPathComponent("custom-planner", isDirectory: true)
        try FileManager.default.createDirectory(
            at: checkpoint,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: planner,
            withIntermediateDirectories: true
        )
        try writeMinimalLM(at: planner)

        let resolved = try await ACEStepCLIHelper.resolveLMResources(
            checkpointsRoot: checkpoint,
            lmModel: planner.path,
            lmSubdirectory: nil
        )

        XCTAssertEqual(resolved.source, "local")
        XCTAssertEqual(resolved.rootURL, planner)
    }

    func testMusicAnalyzeOutputRoundTrips() throws {
        let output = MusicAnalyzeOutput(
            audio: "/tmp/song.mp3",
            model: ModelResolver.ModelID.aceStep.rawValue,
            checkpointsRoot: "/tmp/acestep",
            turboSubdirectory: "acestep-v15-turbo",
            lmSubdirectory: "acestep-5Hz-lm-1.7B",
            languageModelSource: ModelResolver.ModelID.aceStepLM17B.rawValue,
            languageModelRoot: "/tmp/acestep/acestep-5Hz-lm-1.7B",
            inputDurationSeconds: 180,
            analyzedDurationSeconds: 30,
            metadata: ACEStepMusicUnderstandingMetadata(
                caption: "bright source song",
                lyrics: "[Verse]\nhello",
                bpm: 125,
                durationSeconds: 30,
                keyscale: "D major",
                language: "en",
                timesignature: "4"
            ),
            rawLMOutput: nil,
            audioCodes: nil
        )

        let data = try JSONEncoder().encode(output)
        let decoded = try JSONDecoder().decode(MusicAnalyzeOutput.self, from: data)

        XCTAssertEqual(decoded, output)
    }

    func testMusicCommandExposesAnalyze() {
        let commandNames = Set(Music.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(
            commandNames,
            Set([
                "analyze",
                "generate",
                "realtime",
                "separate",
                "serve",
                "train-adapter",
                "transcribe",
            ])
        )
    }

    private func writeMinimalLM(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for filename in [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
            "added_tokens.json",
        ] {
            XCTAssertTrue(
                FileManager.default.createFile(
                    atPath: root.appendingPathComponent(filename).path,
                    contents: Data()
                )
            )
        }
    }
}
