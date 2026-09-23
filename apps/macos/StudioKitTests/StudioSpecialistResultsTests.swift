@testable import StudioKit
import XCTest

/// The typed readings of the CLI output the specialist pages render: `speech diarize --format
/// json`, `music analyze`, and the three shapes of `run inspect --json`. Each sample is written
/// the way the CLI's own Codable types encode it.
final class StudioSpecialistResultsTests: XCTestCase {
    // MARK: - vision face detect

    /// `vision face detect --json-output` on a 512×512 portrait, as `FaceAnalysisResult` encodes it.
    func testFaceDetectionResultDecodesForTheOverlay() throws {
        let json = """
        {"elapsedMilliseconds":558.9,"height":512,"image":"/tmp/portrait-512.png","modelID":"vision-face-buffalo-l","width":512,
         "faces":[{"index":0,"detection":{"score":0.7824,
           "boundingBox":{"height":369.15,"width":259.37,"x":126.61,"y":46.76},
           "landmarks":[{"x":194.73,"y":187.0},{"x":315.77,"y":186.16},{"x":257.02,"y":270.04},{"x":189.64,"y":295.28},{"x":326.75,"y":294.3}]}}]}
        """
        let result = try JSONDecoder().decode(StudioFaceOverlayResult.self, from: Data(json.utf8))
        XCTAssertEqual(result.width, 512)
        XCTAssertEqual(result.height, 512)
        XCTAssertEqual(result.faces.map(\.index), [0])
        let face = try XCTUnwrap(result.faces.first)
        XCTAssertEqual(face.detection.score, 0.7824)
        XCTAssertEqual(face.detection.boundingBox, .init(x: 126.61, y: 46.76, width: 259.37, height: 369.15))
        XCTAssertEqual(face.detection.landmarks.count, 5, "Buffalo-L reports five landmarks")
        XCTAssertEqual(face.detection.landmarks.first, .init(x: 194.73, y: 187.0))

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("face-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        XCTAssertEqual(StudioFaceOverlayResult.load(from: url), result)
        XCTAssertNil(StudioFaceOverlayResult.load(from: url.appendingPathExtension("missing")))
        try Data(#"{"width":1}"#.utf8).write(to: url)
        XCTAssertNil(StudioFaceOverlayResult.load(from: url), "a document without faces is not a face result")
    }

    // MARK: - speech diarize

    func testDiarizationLoadsThePayloadAndSummarizesSpeakers() throws {
        let url = try temporaryFile("speakers.json", contents: """
        {
          "schema_version" : 1,
          "model" : "speech-diarization-sortformer",
          "source" : "/Users/example/Music/mere.run/Audio/standup.wav",
          "runtime" : "mlx",
          "device" : "gpu",
          "duration_seconds" : 192.4,
          "speaker_count" : 2,
          "processing_seconds" : 6.1,
          "segments" : [
            { "speaker" : "speaker_0", "speaker_index" : 0, "start_seconds" : 0.0, "end_seconds" : 4.0, "duration_seconds" : 4.0 },
            { "speaker" : "speaker_1", "speaker_index" : 1, "start_seconds" : 4.5, "end_seconds" : 8.0, "duration_seconds" : 3.5 },
            { "speaker" : "speaker_0", "speaker_index" : 0, "start_seconds" : 8.2, "end_seconds" : 70.2, "duration_seconds" : 62.0 }
          ]
        }
        """)

        let document = try XCTUnwrap(StudioDiarizationDocument.load(from: url))

        XCTAssertEqual(document.summary, "2 speakers · 3 turns · 3:12")
        XCTAssertEqual(document.speakers.map(\.name), ["Speaker 1", "Speaker 2"])
        XCTAssertEqual(document.speakers.map(\.turnCount), [2, 1])
        XCTAssertEqual(document.speakers.map(\.talkTimeDescription), ["1:06", "0:04"])
        // The Analyze panel's rows name the turn's span, since a diarized turn has no words.
        XCTAssertEqual(StudioAnalyzeDocument.diarization(document).speechSegments.map(\.text), ["Spoke for 0:04", "Spoke for 0:04", "Spoke for 1:02"])
    }

    func testAnRTTMTimelineIsNotADocument() throws {
        let url = try temporaryFile("speakers.rttm", contents: "SPEAKER standup 1 0.000 4.000 <NA> <NA> speaker_0 <NA> <NA>\n")
        XCTAssertNil(StudioDiarizationDocument.load(from: url))
    }

    // MARK: - music analyze

    func testMusicAnalysisDecodesTheCLIOutput() throws {
        let document = try XCTUnwrap(StudioMusicAnalysisDocument.decode("""
        {
          "analyzedDurationSeconds" : 30,
          "audio" : "/Users/example/Music/demo.wav",
          "audioCodes" : "<|audio_code_1|><|audio_code_2|>",
          "checkpointsRoot" : "/Users/example/Library/Application Support/MereRun/models/music-acestep",
          "inputDurationSeconds" : 192.4,
          "languageModelRoot" : "/Users/example/Library/Application Support/MereRun/models/music-acestep/lm",
          "languageModelSource" : "bundled",
          "lmSubdirectory" : "acestep-5Hz-lm-1.7B",
          "metadata" : {
            "bpm" : 120,
            "caption" : "Warm analog synth pop with a steady four-on-the-floor pulse.",
            "durationSeconds" : 192.4,
            "keyscale" : "C# minor",
            "language" : "en",
            "lyrics" : "[verse]\\nCity lights are calling",
            "timesignature" : "4/4"
          },
          "model" : "music-acestep",
          "rawLMOutput" : "<bpm>120</bpm>",
          "turboSubdirectory" : "acestep-v15-turbo"
        }
        """))

        XCTAssertEqual(document.tempoDescription, "120 BPM")
        XCTAssertEqual(document.metadata.keyscale, "C# minor")
        XCTAssertEqual(document.metadata.timesignature, "4/4")
        XCTAssertEqual(document.languageDescription, Locale.current.localizedString(forLanguageCode: "en") ?? "en")
        XCTAssertEqual(document.analyzedDescription, "0:30 of 3:12")
        XCTAssertEqual(document.caption, "Warm analog synth pop with a steady four-on-the-floor pulse.")
        XCTAssertEqual(document.lyrics, "[verse]\nCity lights are calling")
        XCTAssertEqual(document.rawLMOutput, "<bpm>120</bpm>")
    }

    func testMusicAnalysisWithNothingDetectedStillReads() throws {
        let document = try XCTUnwrap(StudioMusicAnalysisDocument.decode("""
        {
          "analyzedDurationSeconds" : 12.2,
          "audio" : "/tmp/noise.wav",
          "checkpointsRoot" : "/tmp/models",
          "inputDurationSeconds" : 12.2,
          "languageModelRoot" : "/tmp/models/lm",
          "languageModelSource" : "bundled",
          "lmSubdirectory" : "lm",
          "metadata" : {},
          "model" : "music-acestep",
          "turboSubdirectory" : "turbo"
        }
        """))

        XCTAssertNil(document.tempoDescription)
        XCTAssertNil(document.languageDescription)
        XCTAssertNil(document.caption)
        XCTAssertEqual(document.analyzedDescription, "0:12")
        XCTAssertNil(StudioMusicAnalysisDocument.decode("error: model music-acestep is not installed"))
    }

    /// A Library row keeps stdout and stderr together; the object is read out of the middle.
    func testMusicAnalysisReadsTheObjectOutOfSurroundingText() throws {
        let document = try XCTUnwrap(StudioMusicAnalysisDocument.decode("""
        ACE-Step source analysis: bpm=120, keyscale=C# minor
        {
          "analyzedDurationSeconds" : 12.2, "audio" : "/tmp/noise.wav", "checkpointsRoot" : "/tmp/models",
          "inputDurationSeconds" : 12.2, "languageModelRoot" : "/tmp/models/lm", "languageModelSource" : "bundled",
          "lmSubdirectory" : "lm", "metadata" : { "bpm" : 120 }, "model" : "music-acestep", "turboSubdirectory" : "turbo"
        }
        STDERR: Completed with exit code 0.
        """))
        XCTAssertEqual(document.tempoDescription, "120 BPM")
    }

    // MARK: - run inspect

    func testRunInspectionReadsARelayJob() throws {
        let inspection = try XCTUnwrap(StudioRunInspection.decode("""
        {
          "artifacts" : [
            { "content_type" : "image/png", "kind" : "image", "name" : "preview", "path" : "preview.png", "sha256" : "ab", "size_bytes" : 42 }
          ],
          "created_at" : "2026-07-28T12:00:00Z",
          "error" : null,
          "executor" : "relay:fleet",
          "job_id" : "job-1",
          "job_reference" : "relay://fleet/job-1",
          "run_directory" : null,
          "state" : "finished",
          "updated_at" : "2026-07-28T12:01:12Z"
        }
        """))

        guard case .remoteJob(let job) = inspection else { return XCTFail("Expected a relay job, got \(inspection)") }
        XCTAssertEqual(job.jobReference, "relay://fleet/job-1")
        let presentation = inspection.presentation
        XCTAssertEqual(presentation.state, "finished")
        XCTAssertEqual(presentation.title, "job-1")
        XCTAssertEqual(presentation.facts.map(\.label), ["Executor", "Started", "Updated", "Duration"])
        XCTAssertEqual(presentation.facts.last?.value, "1:12")
        XCTAssertEqual(presentation.outputs.map(\.name), ["preview.png"])
        XCTAssertEqual(presentation.outputs.first?.detail, "image · 42 bytes")
        XCTAssertTrue(presentation.problems.isEmpty)
    }

    func testRunInspectionReadsAGraphRunWithItsNodes() throws {
        let inspection = try XCTUnwrap(StudioRunInspection.decode("""
        {
          "attempt" : 1,
          "contract_version" : "mere.run/graph-run.v1",
          "created_at" : "2026-07-28T12:00:00Z",
          "error" : "render: exit status 1",
          "executor" : { "kind" : "local", "profile" : null, "job_reference" : null },
          "graph_fingerprint" : "f1",
          "graph_name" : "poster",
          "job_id" : "poster-2026-07-28",
          "nodes" : [
            { "artifacts" : [], "attempt" : 1, "completed_at" : "2026-07-28T12:00:20Z", "fingerprint" : "n1", "id" : "prompt", "kind" : "text.chat", "max_attempts" : 1, "models" : [], "outputs" : [], "started_at" : "2026-07-28T12:00:02Z", "state" : "finished" },
            { "artifacts" : [ { "content_type" : "image/png", "kind" : "image", "name" : "poster", "path" : "/tmp/runs/poster/poster.png", "sha256" : "cd", "size_bytes" : 2048 } ], "attempt" : 2, "error" : "exit status 1", "fingerprint" : "n2", "id" : "render", "kind" : "image.generate", "max_attempts" : 3, "models" : [], "outputs" : [], "started_at" : "2026-07-28T12:00:20Z", "state" : "failed" }
          ],
          "outputs" : [
            { "content_type" : "image/png", "kind" : "image", "name" : "poster", "path" : "/tmp/runs/poster/poster.png", "sha256" : "cd", "size_bytes" : 2048 }
          ],
          "state" : "failed",
          "updated_at" : "2026-07-28T12:00:41Z"
        }
        """))

        guard case .graphRun = inspection else { return XCTFail("Expected a graph run, got \(inspection)") }
        let presentation = inspection.presentation
        XCTAssertEqual(presentation.state, "failed")
        XCTAssertEqual(presentation.facts.prefix(2).map(\.value), ["poster", "1"])
        XCTAssertEqual(presentation.steps.map(\.title), ["prompt", "render"])
        XCTAssertEqual(presentation.steps.map(\.state), ["finished", "failed"])
        XCTAssertEqual(presentation.steps[0].detail, "text.chat · 0:18")
        XCTAssertEqual(presentation.steps[1].detail, "image.generate · attempt 2 of 3 · 1 file")
        XCTAssertEqual(presentation.outputs.map(\.path), ["/tmp/runs/poster/poster.png"])
        // The node's own error is already quoted by the run's, so it is not repeated.
        XCTAssertEqual(presentation.problems, ["render: exit status 1"])
    }

    func testRunInspectionReadsALocalImageRunEnvelope() throws {
        let inspection = try XCTUnwrap(StudioRunInspection.decode("""
        {
          "actions" : [],
          "command" : [ "run", "inspect" ],
          "created_at" : "2026-07-28T12:05:00Z",
          "cwd" : "/Users/example",
          "diagnostics" : [
            { "id" : "image.retry", "locations" : [], "message" : "The run can be retried.", "severity" : "note", "suggestedActionIDs" : [], "title" : "Retryable" }
          ],
          "mere_run_version" : "0.55.0",
          "mode" : "inspection",
          "request" : { "path" : "/tmp/runs/mug" },
          "result" : {
            "image_run" : {
              "artifacts" : [ { "byteCount" : 812000, "sha256" : "ef", "url" : "file:///tmp/runs/mug/mug.png" } ],
              "createdAt" : "2026-07-28T12:00:00Z",
              "effective" : { "seed" : 8812 },
              "id" : "6B29FC40-CA47-1067-B31D-00DD010662DA",
              "inputs" : [],
              "issue" : { "code" : "process_interrupted", "message" : "The image process stopped before recording a terminal result." },
              "modelSelector" : "image-zimage-nano",
              "requested" : {},
              "schemaVersion" : 1,
              "state" : "interrupted",
              "updatedAt" : "2026-07-28T12:00:09Z"
            },
            "kind" : "image_run",
            "path" : "/tmp/runs/mug"
          },
          "schema_version" : 1,
          "status" : "ok",
          "summary" : "Image run mug: interrupted"
        }
        """))

        guard case .local = inspection else { return XCTFail("Expected an envelope, got \(inspection)") }
        let presentation = inspection.presentation
        XCTAssertEqual(presentation.state, "interrupted")
        XCTAssertEqual(presentation.title, "Image run mug: interrupted")
        XCTAssertEqual(presentation.facts.map(\.label), ["Model", "Seed", "Started", "Updated", "Duration"])
        XCTAssertEqual(presentation.facts[1].value, "8812")
        XCTAssertEqual(presentation.outputs.map(\.path), ["/tmp/runs/mug/mug.png"])
        XCTAssertEqual(presentation.outputs.first?.detail, "812 KB")
        // Notes are not problems; the record's own issue is.
        XCTAssertEqual(presentation.problems, ["The image process stopped before recording a terminal result."])
    }

    func testRunInspectionReadsATrainingRunDirectory() throws {
        let inspection = try XCTUnwrap(StudioRunInspection.decode("""
        {
          "actions" : [],
          "command" : [ "run", "inspect" ],
          "created_at" : "2026-07-28T12:05:00Z",
          "cwd" : "/Users/example",
          "diagnostics" : [
            { "id" : "run.stale", "locations" : [], "message" : "No event in 20 minutes.", "severity" : "warning", "suggestedActionIDs" : [], "title" : "Stale run" }
          ],
          "mere_run_version" : "0.55.0",
          "mode" : "inspection",
          "request" : { "path" : "/tmp/runs/lighthouse-lora" },
          "result" : {
            "kind" : "run_directory",
            "path" : "/tmp/runs/lighthouse-lora",
            "run_directory" : {
              "actions" : [],
              "artifacts" : [
                { "exists" : true, "is_image" : false, "kind" : "adapter", "name" : "adapter.safetensors", "path" : "/tmp/runs/lighthouse-lora/adapter.safetensors", "size_bytes" : 41943040 },
                { "exists" : false, "is_image" : true, "kind" : "sample", "name" : "sample-0250.png", "path" : "/tmp/runs/lighthouse-lora/samples/sample-0250.png", "size_bytes" : null }
              ],
              "event_paths" : [ "/tmp/runs/lighthouse-lora/events.ndjson" ],
              "events" : { "count" : 412, "latest" : null, "types" : [ "step", "checkpoint" ] },
              "manifest" : { "checkpoint_files" : {}, "created_at" : "2026-07-28T11:00:00Z", "data_root" : null, "format" : "lora", "is_edit" : false, "model" : "image-zimage-nano", "progress" : 0.5, "seed" : 42, "step" : 500, "total_steps" : 1000 },
              "metrics" : { "adapter_count" : 1, "checkpoint_count" : 2, "first_loss" : 0.91, "first_step" : 1, "latest_loss" : 0.3125, "latest_step" : 500, "loss_point_count" : 500, "max_loss" : 0.95, "min_loss" : 0.2981, "sample_image_count" : 2 },
              "path" : "/tmp/runs/lighthouse-lora",
              "status" : "running"
            }
          },
          "schema_version" : 1,
          "status" : "warning",
          "summary" : "Training run lighthouse-lora: running"
        }
        """))

        let presentation = inspection.presentation
        XCTAssertEqual(presentation.state, "running")
        XCTAssertEqual(presentation.facts.first { $0.label == "Progress" }?.value, "500 of 1000 steps")
        XCTAssertEqual(presentation.facts.first { $0.label == "Latest loss" }?.value, "0.3125 at step 500")
        XCTAssertEqual(presentation.facts.first { $0.label == "Checkpoints" }?.value, "2")
        XCTAssertEqual(presentation.outputs.map(\.exists), [true, false])
        XCTAssertEqual(presentation.outputs[0].detail, "adapter · 41.9 MB")
        XCTAssertEqual(presentation.problems, ["Stale run: No event in 20 minutes."])
    }

    func testRunInspectionReadsARunPlan() throws {
        let inspection = try XCTUnwrap(StudioRunInspection.decode("""
        {
          "actions" : [], "command" : [ "run", "inspect" ], "created_at" : "2026-07-28T12:05:00Z",
          "cwd" : "/Users/example", "diagnostics" : [], "mere_run_version" : "0.55.0", "mode" : "inspection",
          "request" : { "path" : "/Users/example/plans/lighthouse.json" },
          "result" : {
            "kind" : "run_plan",
            "path" : "/Users/example/plans/lighthouse.json",
            "plan" : { "schema_version" : 1, "kind" : "image.train", "command" : [ "image", "train" ], "created_at" : "2026-07-27T09:30:00Z", "cwd" : "/Users/example" }
          },
          "schema_version" : 1, "status" : "ok", "summary" : "Run plan lighthouse.json: image.train"
        }
        """))

        let presentation = inspection.presentation
        XCTAssertEqual(presentation.state, "ok")
        XCTAssertEqual(presentation.facts.map(\.label), ["Plan", "Command", "Written"])
        XCTAssertEqual(presentation.facts[1].value, "image train")
    }

    func testRunInspectionLeavesUnreadableOutputToTheRawView() {
        XCTAssertNil(StudioRunInspection.decode("error: /tmp/nowhere is not a run directory"))
        XCTAssertNil(StudioRunInspection.decode("{\"unexpected\": true}"))
    }

    // MARK: - Helpers

    private func temporaryFile(_ name: String, contents: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioSpecialistResultsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent(name, isDirectory: false)
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
