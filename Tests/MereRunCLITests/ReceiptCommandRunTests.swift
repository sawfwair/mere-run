import AudioCore
import Foundation
import XCTest
@testable import MereRunCLI

/// Runs real command paths against fixture backends and checks the stdout
/// contract: without `--receipt` the output is exactly what the command has
/// always printed, and with it the receipt is the final line.
final class ReceiptCommandRunTests: XCTestCase {
    private struct FixtureASR: CLIASRTranscriptionExecutor {
        let text: String

        func transcribeQwen(
            request: ASRRequest, modelID: String, modelPath: String?,
            progressHandler: (@Sendable (ASRProgress) -> Void)?
        ) async throws -> ASRResult {
            ASRResult(text: text, language: "en", duration: 1.5)
        }

        func transcribeParakeet(
            request: ASRRequest, modelID: String, modelPath: String?,
            progressHandler: (@Sendable (ASRProgress) -> Void)?
        ) async throws -> ASRResult {
            ASRResult(text: text, language: "en", duration: 1.5)
        }
    }

    private struct FixtureTTS: SpeechSynthesisExecutor {
        func generate(
            _ request: TTSRequest,
            progressHandler: (@Sendable (TTSProgress) -> Void)?
        ) async throws -> AudioWaveform {
            progressHandler?(TTSProgress(stage: .generating, tokensGenerated: 25))
            return try AudioWaveform(interleaved: [0, 0.25, -0.25], channels: 1, sampleRate: 24_000)
        }

        func generateStream(_ request: TTSRequest, options: TTSStreamingOptions) -> AsyncThrowingStream<TTSStreamingEvent, Error> {
            AsyncThrowingStream { continuation in
                if options.emitTokenEvents {
                    for token in 0..<25 { continuation.yield(.token(id: token)) }
                }
                continuation.yield(.audioChunk(samples: [0, 0.25, -0.25], sampleRate: 24_000))
                continuation.yield(.completed(result: TTSResult(audioURL: request.outputURL, duration: 3.0 / 24_000)))
                continuation.finish()
            }
        }
    }

    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-receipt-tests", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        SpeechTranscribe.transcriptionExecutorOverride = nil
        SpeechSynthesize.synthesizerOverride = nil
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    /// Captures everything the body prints to file descriptor 1.
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

    private func receipt(fromLastLineOf stdout: String) throws -> [String: Any] {
        XCTAssertTrue(stdout.hasSuffix("\n"))
        let lastLine = try XCTUnwrap(stdout.dropLast().split(separator: "\n", omittingEmptySubsequences: false).last)
        let object = try JSONSerialization.jsonObject(with: Data(lastLine.utf8))
        let receipt = try XCTUnwrap(object as? [String: Any])
        XCTAssertEqual(receipt["event"] as? String, "result")
        XCTAssertEqual(receipt["exit"] as? Int, 0)
        return receipt
    }

    // MARK: - speech transcribe

    private func transcribeCommand(_ extra: [String]) throws -> SpeechTranscribe {
        let audioURL = temporaryDirectory.appendingPathComponent("talk.wav")
        try Data("RIFF".utf8).write(to: audioURL)
        SpeechTranscribe.transcriptionExecutorOverride = FixtureASR(text: "hello from the fixture")
        return try SpeechTranscribe.parse([audioURL.path, "--backend", "parakeet", "--quiet"] + extra)
    }

    func testTranscribeStdoutWithoutReceiptIsJustTheTranscript() async throws {
        let command = try transcribeCommand([])
        let stdout = try await capturingStandardOutput { try await command.run() }
        XCTAssertEqual(stdout, "hello from the fixture\n")
    }

    func testTranscribeReceiptIsTheLastLineAndListsTheOutputFile() async throws {
        let transcriptURL = temporaryDirectory.appendingPathComponent("talk.txt")
        let command = try transcribeCommand(["--receipt", "--output", transcriptURL.path])
        let stdout = try await capturingStandardOutput { try await command.run() }

        XCTAssertTrue(stdout.hasPrefix("hello from the fixture\n"))
        XCTAssertEqual(stdout.split(separator: "\n").count, 2)
        let receipt = try receipt(fromLastLineOf: stdout)
        let outputs = try XCTUnwrap(receipt["outputs"] as? [[String: Any]])
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(outputs[0]["kind"] as? String, "text")
        XCTAssertEqual(outputs[0]["path"] as? String, transcriptURL.standardizedFileURL.path)
        XCTAssertEqual(try String(contentsOf: transcriptURL, encoding: .utf8), "hello from the fixture")
    }

    func testTranscribeReceiptHasNoOutputsWhenTheTranscriptOnlyWentToStdout() async throws {
        let command = try transcribeCommand(["--receipt"])
        let stdout = try await capturingStandardOutput { try await command.run() }

        XCTAssertEqual(stdout, "hello from the fixture\n{\"event\":\"result\",\"exit\":0,\"outputs\":[]}\n")
    }

    func testRecordedTranscriptionPreservesStdoutAndReceiptContract() async throws {
        let directory = temporaryDirectory.appendingPathComponent("recorded-run")
        let transcript = temporaryDirectory.appendingPathComponent("copy.txt")
        let command = try transcribeCommand(["--run-dir", directory.path, "--receipt", "--output", transcript.path])
        let stdout = try await capturingStandardOutput { try await command.run() }
        XCTAssertTrue(stdout.hasPrefix("hello from the fixture\n"))
        XCTAssertEqual(stdout.split(separator: "\n").count, 2)
        let record = try SpeechTranscriptionRunRecord.inspect(at: directory)
        XCTAssertEqual(record.state, .succeeded)
        XCTAssertEqual(record.artifacts.count, 2)
        XCTAssertEqual(try String(contentsOf: transcript, encoding: .utf8), "hello from the fixture")
        let receipt = try receipt(fromLastLineOf: stdout)
        let outputs = try XCTUnwrap(receipt["outputs"] as? [[String: Any]])
        XCTAssertEqual(outputs.count, 1)
        XCTAssertEqual(outputs[0]["path"] as? String, transcript.path)
    }

    // MARK: - speech synthesize

    private func synthesizeCommand(_ extra: [String]) throws -> (SpeechSynthesize, URL) {
        let outputURL = temporaryDirectory.appendingPathComponent("hello.wav")
        SpeechSynthesize.synthesizerOverride = FixtureTTS()
        let command = try SpeechSynthesize.parse(["Hello there", "--output", outputURL.path, "--quiet"] + extra)
        return (command, outputURL)
    }

    func testSynthesizeStdoutWithoutReceiptIsJustTheOutputPath() async throws {
        let (command, outputURL) = try synthesizeCommand([])
        let stdout = try await capturingStandardOutput { try await command.run() }
        XCTAssertEqual(stdout, outputURL.standardizedFileURL.path + "\n")
        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
    }

    func testSynthesizeReceiptFollowsThePathLineAndListsTheWAV() async throws {
        let (command, outputURL) = try synthesizeCommand(["--receipt", "--progress-json"])
        let stdout = try await capturingStandardOutput { try await command.run() }

        let lines = stdout.split(separator: "\n")
        XCTAssertEqual(lines.count, 2)
        XCTAssertEqual(String(lines[0]), outputURL.standardizedFileURL.path)
        let receipt = try receipt(fromLastLineOf: stdout)
        let outputs = try XCTUnwrap(receipt["outputs"] as? [[String: Any]])
        XCTAssertEqual(outputs.map { $0["kind"] as? String }, ["audio"])
        XCTAssertEqual(outputs[0]["path"] as? String, outputURL.standardizedFileURL.path)
        XCTAssertNil(outputs[0]["role"])
    }

    func testStreamingSynthesizeReceiptNamesACompletedFloatWAV() async throws {
        let (command, outputURL) = try synthesizeCommand(["--stream", "--receipt", "--progress-json"])
        let stdout = try await capturingStandardOutput { try await command.run() }
        XCTAssertEqual(stdout.split(separator: "\n").count, 2)
        let receipt = try receipt(fromLastLineOf: stdout)
        let outputs = try XCTUnwrap(receipt["outputs"] as? [[String: Any]])
        XCTAssertEqual(outputs[0]["path"] as? String, outputURL.standardizedFileURL.path)
        let data = try Data(contentsOf: outputURL)
        XCTAssertEqual(data.count, 56)
        XCTAssertEqual(data[20], 3)
        XCTAssertEqual(data[34], 32)
        XCTAssertEqual(data[40], 12)
    }
}
