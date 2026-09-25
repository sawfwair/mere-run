import AudioCore
import Foundation
import MediaIO
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

/// The capability gate's warnings reach a command's JSON as a top-level `warnings` array, one
/// family of output at a time: a run's result, a preflight report, and a dry run. Each test takes
/// the warnings from the gate for the same command line, so the JSON says what stderr says; and
/// the same command without warnings prints what it always printed.
final class GateWarningsJSONTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-gate-warnings-json", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        SpeechTranscribe.transcriptionExecutorOverride = nil
        try? FileManager.default.removeItem(at: directory)
    }

    private struct FixtureASR: CLIASRTranscriptionExecutor {
        func transcribeQwen(
            request: ASRRequest, modelID: String, modelPath: String?,
            progressHandler: (@Sendable (ASRProgress) -> Void)?
        ) async throws -> ASRResult {
            ASRResult(text: "hello", language: "en", duration: 1)
        }

        func transcribeParakeet(
            request: ASRRequest, modelID: String, modelPath: String?,
            progressHandler: (@Sendable (ASRProgress) -> Void)?
        ) async throws -> ASRResult {
            ASRResult(text: "hello", language: "en", duration: 1)
        }
    }

    /// Everything the body prints to file descriptor 1.
    private func capturingStandardOutput(_ body: () async throws -> Void) async throws -> String {
        fflush(stdout)
        let pipe = Pipe()
        let saved = dup(STDOUT_FILENO)
        XCTAssertNotEqual(dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO), -1)
        var failure: Error?
        do {
            try await body()
        } catch {
            failure = error
        }
        fflush(stdout)
        dup2(saved, STDOUT_FILENO)
        close(saved)
        try pipe.fileHandleForWriting.close()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        if let failure { throw failure }
        return String(decoding: data, as: UTF8.self)
    }

    private func object(_ json: String) throws -> NSDictionary {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? NSDictionary)
    }

    /// The warned JSON is the plain JSON plus `warnings`, and the plain JSON has no such key.
    private func assertAddsOnlyWarnings(_ warned: String, to plain: String, _ warnings: [String]) throws {
        XCTAssertFalse(plain.contains("\"warnings\""), plain)
        let warnedObject = try XCTUnwrap(try object(warned).mutableCopy() as? NSMutableDictionary)
        XCTAssertEqual(warnedObject["warnings"] as? [String], warnings)
        warnedObject.removeObject(forKey: "warnings")
        XCTAssertEqual(warnedObject, try object(plain))
    }

    func testWithoutWarningsTheJSONIsByteIdentical() throws {
        struct Payload: Encodable {
            let path = "/tmp/out.wav"
            let sampleRate = 48_000
            let role: String? = nil
            let stems = ["vocals", "drums"]
        }
        let pretty = StructuredRunOutput.encoder()
        let snakeCase = JSONEncoder()
        snakeCase.keyEncodingStrategy = .convertToSnakeCase
        snakeCase.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        // Byte identity needs sorted keys: an unsorted JSONEncoder may order keys differently on
        // each call, so it is compared as a decoded object instead.
        XCTAssertEqual(
            try object(String(decoding: JSONEncoder().encode(GateWarned(Payload(), warnings: [])), as: UTF8.self)),
            try object(String(decoding: JSONEncoder().encode(Payload()), as: UTF8.self))
        )
        for encoder in [pretty, snakeCase] {
            XCTAssertEqual(try encoder.encode(GateWarned(Payload(), warnings: [])), try encoder.encode(Payload()))
        }
        for encoder in [JSONEncoder(), pretty, snakeCase] {
            let warned = try encoder.encode(GateWarned(Payload(), warnings: ["--cfg has no effect."]))
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: warned) as? [String: Any])
            XCTAssertEqual(object["warnings"] as? [String], ["--cfg has no effect."])
            XCTAssertEqual(object.count, 4, "the other keys stay as they were")
        }
        // Outside `MereRunCLI.main` nothing is bound.
        XCTAssertEqual(try pretty.encode(GateWarned(Payload())), try pretty.encode(Payload()))
    }

    // MARK: - Run result: the receipt

    func testTranscribeReceiptCarriesTheGateWarnings() async throws {
        let audio = directory.appendingPathComponent("talk.wav")
        try Data("RIFF".utf8).write(to: audio)
        let arguments = [audio.path, "--backend", "parakeet", "--max-tokens", "64", "--receipt", "--quiet"]
        let gate = try CLICapabilityGate.pass(arguments: ["mere.run", "speech", "transcribe"] + arguments)
        XCTAssertEqual(gate.warnings, ["--max-tokens has no effect with Parakeet. It applies to Qwen3-ASR."])
        XCTAssertEqual(gate.stderrLines, [], "--quiet keeps stderr clean; the JSON still carries them")

        SpeechTranscribe.transcriptionExecutorOverride = FixtureASR()
        let command = try SpeechTranscribe.parse(arguments)
        let warned = try await capturingStandardOutput {
            try await CLIGateWarnings.$current.withValue(gate.warnings) { try await command.run() }
        }
        let plain = try await capturingStandardOutput { try await command.run() }

        XCTAssertEqual(plain, "hello\n{\"event\":\"result\",\"exit\":0,\"outputs\":[]}\n")
        XCTAssertEqual(
            warned,
            "hello\n{\"event\":\"result\",\"exit\":0,\"outputs\":[],"
                + "\"warnings\":[\"--max-tokens has no effect with Parakeet. It applies to Qwen3-ASR.\"]}\n"
        )
        let receipt = try JSONDecoder().decode(RunReceipt.self, from: Data(warned.split(separator: "\n")[1].utf8))
        XCTAssertEqual(receipt.warnings, gate.warnings)
    }

    // MARK: - Preflight report

    func testVideoPreflightJSONCarriesTheGateWarnings() throws {
        let arguments = ["a cat", "--model", "video-ltx25-distilled-bf16", "--steps", "8", "--preflight", "--json"]
        let gate = try CLICapabilityGate.pass(arguments: ["mere.run", "video", "generate"] + arguments)
        XCTAssertEqual(gate.warnings.count, 1)
        XCTAssertTrue(gate.warnings[0].hasPrefix("--steps has no effect with LTX-2.5 Distilled."), "\(gate.warnings)")

        let command = try VideoGenerate.parse(arguments)
        let envelope = command.makePreflightEnvelope(
            outputURL: directory.appendingPathComponent("out.mp4"),
            adaptersRoot: directory.appendingPathComponent("adapters"),
            now: { Date(timeIntervalSince1970: 1_700_000_000) }
        )
        let plain = try VideoGenerate.encodePreflight(envelope)
        let warned = try CLIGateWarnings.$current.withValue(gate.warnings) { try VideoGenerate.encodePreflight(envelope) }
        try assertAddsOnlyWarnings(warned, to: plain, gate.warnings)
    }

    // MARK: - Dry run

    func testTrainLoRADryRunJSONCarriesTheGateWarnings() async throws {
        let images = directory.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
        try MediaImageIO.writePNG(
            try MediaImage(width: 2, height: 2, rgba8: Array(repeating: 96, count: 16)),
            to: images.appendingPathComponent("frame.png")
        )
        let example = TextSFTExample(
            id: "gate-warnings",
            sources: ["test"],
            messages: [
                ChatMessage(role: .system, content: "Describe visible evidence."),
                ChatMessage(role: .user, content: "What is visible?", imageUrl: "images/frame.png"),
                ChatMessage(role: .assistant, content: "A test frame."),
            ]
        )
        let dataset = directory.appendingPathComponent("pairs.jsonl")
        try (JSONEncoder().encode(example) + Data("\n".utf8)).write(to: dataset)
        let arguments = [
            "--model", Gemma4Resources.visionTwelveBModelId,
            "--data", dataset.path,
            "--output", directory.appendingPathComponent("adapter.safetensors").path,
            "--reasoning-effort", "0.5",
            "--dry-run",
            "--json",
        ]
        let gate = try CLICapabilityGate.pass(arguments: ["mere.run", "text", "train-lora"] + arguments)
        XCTAssertEqual(gate.warnings, ["--reasoning-effort has no effect with Gemma 4 12B vision. It applies to Inkling-Small."])

        let command = try TextTrainLoRA.parse(arguments)
        let warned = try await capturingStandardOutput {
            try await CLIGateWarnings.$current.withValue(gate.warnings) { try await command.run() }
        }
        let plain = try await capturingStandardOutput { try await command.run() }
        try assertAddsOnlyWarnings(warned, to: plain, gate.warnings)
    }
}
