import Foundation
import XCTest
@testable import MereRunCLI

final class RunReceiptTests: XCTestCase {
    private func decode(_ line: String) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(line.utf8))
        return try XCTUnwrap(object as? [String: Any])
    }

    private func outputs(in line: String) throws -> [[String: Any]] {
        try XCTUnwrap(try decode(line)["outputs"] as? [[String: Any]])
    }

    func testReceiptLineIsOneSortedJSONObject() throws {
        let receipt = RunReceipt(outputs: [
            .init(path: "/tmp/out.png", kind: .image),
            .init(path: "/tmp/prompt.json", kind: .json, role: "structured-prompt"),
        ])

        let line = try receipt.line()
        XCTAssertEqual(
            line,
            #"{"event":"result","exit":0,"outputs":[{"kind":"image","path":"/tmp/out.png"},"# +
                #"{"kind":"json","path":"/tmp/prompt.json","role":"structured-prompt"}]}"#
        )
        XCTAssertFalse(line.contains("\n"))

        let decoded = try decode(line)
        XCTAssertEqual(decoded["event"] as? String, "result")
        XCTAssertEqual(decoded["exit"] as? Int, 0)
        XCTAssertEqual(try outputs(in: line).count, 2)
    }

    func testEmitIsANoOpUnlessEnabled() throws {
        var written: [String] = []
        let audioOutputs = [RunReceipt.Output(path: "/tmp/out.wav", kind: .audio)]

        try RunReceipt.emit(audioOutputs, enabled: false, write: { written.append($0) })
        XCTAssertTrue(written.isEmpty)

        try RunReceipt.emit(audioOutputs, enabled: true, write: { written.append($0) })
        XCTAssertEqual(written.count, 1)
        XCTAssertEqual(try outputs(in: written[0]).first?["kind"] as? String, "audio")
    }

    func testOutputURLsAreStandardizedAbsolutePaths() {
        let output = RunReceipt.Output(url: URL(fileURLWithPath: "/tmp/../tmp/./clip.mp4"), kind: .video)
        XCTAssertEqual(output.path, "/tmp/clip.mp4")
        XCTAssertNil(output.role)
    }

    func testGeneratedImageOutputsListStructuredPromptAsSidecar() throws {
        let plain = RunReceipt.generatedImageOutputs(image: URL(fileURLWithPath: "/tmp/a.png"), structuredPrompt: nil)
        XCTAssertEqual(plain.map(\.kind), [.image])

        let line = try RunReceipt(outputs: RunReceipt.generatedImageOutputs(
            image: URL(fileURLWithPath: "/tmp/a.png"),
            structuredPrompt: URL(fileURLWithPath: "/tmp/a-prompt.json")
        )).line()
        let entries = try outputs(in: line)
        XCTAssertEqual(entries.map { $0["kind"] as? String }, ["image", "json"])
        XCTAssertEqual(entries.map { $0["path"] as? String }, ["/tmp/a.png", "/tmp/a-prompt.json"])
        XCTAssertEqual(entries[1]["role"] as? String, "structured-prompt")
    }

    func testGeneratedVideoOutputsCoverMP4EXRDirectoryAndTimings() throws {
        let mp4 = RunReceipt.generatedVideoOutputs(primary: URL(fileURLWithPath: "/tmp/v.mp4"), kind: .video, timings: nil)
        XCTAssertEqual(mp4.map(\.kind), [.video])

        let exr = RunReceipt.generatedVideoOutputs(
            primary: URL(fileURLWithPath: "/tmp/v_exr"),
            kind: .directory,
            timings: URL(fileURLWithPath: "/tmp/v-timings.json")
        )
        let entries = try outputs(in: try RunReceipt(outputs: exr).line())
        XCTAssertEqual(entries.map { $0["kind"] as? String }, ["directory", "json"])
        XCTAssertEqual(entries[1]["role"] as? String, "timings")
    }

    func testGeneratedAudioOutputsKeepThePrimaryWAVFirst() throws {
        let entries = try outputs(in: try RunReceipt(outputs: RunReceipt.generatedAudioOutputs(
            audio: URL(fileURLWithPath: "/tmp/song.wav"),
            sidecars: [
                .init(url: URL(fileURLWithPath: "/tmp/song.candidate-2.seed-7.wav"), kind: .audio, role: "candidate"),
                .init(url: URL(fileURLWithPath: "/tmp/song.lrc"), kind: .text, role: "lyrics"),
                .init(url: URL(fileURLWithPath: "/tmp/song.recipe.json"), kind: .json, role: "recipe"),
                .init(url: URL(fileURLWithPath: "/tmp/song-bundle"), kind: .directory, role: "daw-bundle"),
            ]
        )).line())

        XCTAssertEqual(entries.first?["path"] as? String, "/tmp/song.wav")
        XCTAssertNil(entries.first?["role"])
        XCTAssertEqual(
            entries.dropFirst().map { $0["role"] as? String },
            ["candidate", "lyrics", "recipe", "daw-bundle"]
        )
        XCTAssertEqual(entries.map { $0["kind"] as? String }, ["audio", "audio", "text", "json", "directory"])
    }

    func testTranscriptOutputsAreEmptyWhenTranscriptOnlyWentToStdout() throws {
        XCTAssertTrue(RunReceipt.transcriptOutputs(nil).isEmpty)
        let line = try RunReceipt(outputs: RunReceipt.transcriptOutputs(nil)).line()
        XCTAssertEqual(line, #"{"event":"result","exit":0,"outputs":[]}"#)

        let file = RunReceipt.transcriptOutputs(URL(fileURLWithPath: "/tmp/talk.txt"))
        XCTAssertEqual(file.map(\.kind), [.text])
        XCTAssertEqual(file.first?.path, "/tmp/talk.txt")
    }

    func testAnnotatedImageOutputsListDetectionsAndOptionalMasks() throws {
        let withoutMasks = RunReceipt.annotatedImageOutputs(
            image: URL(fileURLWithPath: "/tmp/photo_grounded.jpg"),
            detections: URL(fileURLWithPath: "/tmp/photo_grounded.json"),
            masks: nil
        )
        XCTAssertEqual(withoutMasks.map(\.kind), [.image, .json])
        XCTAssertEqual(withoutMasks.map(\.role), [nil, "detections"])

        let entries = try outputs(in: try RunReceipt(outputs: RunReceipt.annotatedImageOutputs(
            image: URL(fileURLWithPath: "/tmp/photo_segmented.png"),
            detections: URL(fileURLWithPath: "/tmp/photo_segmented.json"),
            masks: URL(fileURLWithPath: "/tmp/masks")
        )).line())
        XCTAssertEqual(entries.map { $0["kind"] as? String }, ["image", "json", "directory"])
        XCTAssertEqual(entries[2]["role"] as? String, "masks")
        XCTAssertEqual(entries[2]["path"] as? String, "/tmp/masks")
    }

    func testAnnotatedVideoOutputsListTrackingJSONAndMasks() throws {
        let entries = try outputs(in: try RunReceipt(outputs: RunReceipt.annotatedVideoOutputs(
            videoPath: "/tmp/clip_tracked.mp4",
            trackingPath: "/tmp/clip_tracked.json",
            masks: URL(fileURLWithPath: "/tmp/clip-masks")
        )).line())
        XCTAssertEqual(entries.map { $0["kind"] as? String }, ["video", "json", "directory"])
        XCTAssertEqual(entries.map { $0["role"] as? String }, [nil, "tracking", "masks"])

        let minimal = RunReceipt.annotatedVideoOutputs(videoPath: "/tmp/clip_tracked.mp4", trackingPath: nil, masks: nil)
        XCTAssertEqual(minimal.map(\.kind), [.video])
    }

    // MARK: - Command flag parsing

    func testEveryReceiptCommandParsesTheFlag() throws {
        XCTAssertTrue(try ImageGenerate.parse(["--prompt", "a cat", "--receipt", "--progress-json"]).receipt)
        XCTAssertTrue(try VideoGenerate.parse(["a cat", "--receipt", "--progress-json"]).receipt)
        XCTAssertTrue(try VideoGenerate.parse(["a cat", "--progress-json"]).progressJson)
        XCTAssertTrue(try MusicGenerate.parse(["lofi beat", "--receipt", "--progress-json"]).receipt)
        XCTAssertTrue(try MusicGenerate.parse(["lofi beat", "--progress-json"]).progressJson)
        XCTAssertTrue(try SFXGenerate.parse(["door slam", "--receipt", "--progress-json"]).receipt)
        XCTAssertTrue(try SFXGenerate.parse(["door slam", "--progress-json"]).progressJson)
        XCTAssertTrue(try SpeechSynthesize.parse(["hello", "--output", "/tmp/h.wav", "--receipt", "--progress-json"]).receipt)
        XCTAssertTrue(try SpeechSynthesize.parse(["hello", "--output", "/tmp/h.wav", "--progress-json"]).progressJson)
        XCTAssertTrue(try SpeechTranscribe.parse(["/tmp/talk.wav", "--receipt"]).receipt)
        XCTAssertTrue(try VisionGround.parse(["/tmp/a.png", "--query", "cat", "--receipt"]).receipt)
        XCTAssertTrue(try VisionSegment.parse(["/tmp/a.png", "--prompt", "cat", "--receipt"]).receipt)
        XCTAssertTrue(try VisionTrack.parse(["/tmp/a.mp4", "--prompt", "cat", "--receipt"]).receipt)
    }

    func testReceiptDefaultsOffSoHumanOutputIsUnchanged() throws {
        XCTAssertFalse(try ImageGenerate.parse(["--prompt", "a cat"]).receipt)
        XCTAssertFalse(try VideoGenerate.parse(["a cat"]).receipt)
        XCTAssertFalse(try VideoGenerate.parse(["a cat"]).progressJson)
        XCTAssertFalse(try MusicGenerate.parse(["lofi beat"]).receipt)
        XCTAssertFalse(try SFXGenerate.parse(["door slam"]).receipt)
        XCTAssertFalse(try SpeechSynthesize.parse(["hello", "--output", "/tmp/h.wav"]).receipt)
        XCTAssertFalse(try SpeechTranscribe.parse(["/tmp/talk.wav"]).receipt)
        XCTAssertFalse(try VisionGround.parse(["/tmp/a.png", "--query", "cat"]).receipt)
        XCTAssertFalse(try VisionSegment.parse(["/tmp/a.png", "--prompt", "cat"]).receipt)
        XCTAssertFalse(try VisionTrack.parse(["/tmp/a.mp4", "--prompt", "cat"]).receipt)
    }

    func testReceiptDoesNotRequirePreflightUnlikeJSON() throws {
        XCTAssertNoThrow(try VisionGround.parse(["/tmp/a.png", "--query", "cat", "--receipt"]).validate())
        XCTAssertNoThrow(try VisionSegment.parse(["/tmp/a.png", "--prompt", "cat", "--receipt"]).validate())
        XCTAssertNoThrow(try VisionTrack.parse(["/tmp/a.mp4", "--prompt", "cat", "--receipt"]).validate())
        XCTAssertThrowsError(try VisionGround.parse(["/tmp/a.png", "--query", "cat", "--json"]))
    }

    func testSpeechTranscribeRejectsReceiptForRawStreamingStdin() {
        XCTAssertThrowsError(try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000", "--receipt",
        ]))
        XCTAssertNoThrow(try SpeechTranscribe.parse([
            "-", "--stream", "--input-format", "pcm-s16le", "--sample-rate", "16000", "--jsonl",
        ]))
    }

    func testReceiptAndProgressFlagsShareOneHelpString() {
        for help in [
            ImageGenerate.helpMessage(), VideoGenerate.helpMessage(), MusicGenerate.helpMessage(),
            SFXGenerate.helpMessage(), SpeechSynthesize.helpMessage(), SpeechTranscribe.helpMessage(),
            VisionGround.helpMessage(), VisionSegment.helpMessage(), VisionTrack.helpMessage(),
        ] {
            XCTAssertTrue(help.contains("--receipt"))
            XCTAssertTrue(help.contains("JSON result line"))
        }
        for help in [
            ImageGenerate.helpMessage(), VideoGenerate.helpMessage(), MusicGenerate.helpMessage(),
            SFXGenerate.helpMessage(), SpeechSynthesize.helpMessage(),
        ] {
            XCTAssertTrue(help.contains("--progress-json"))
        }
    }
}
