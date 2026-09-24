@testable import StudioKit
import StudioTestSupport
import MereRunContract
import XCTest

/// Music ▸ Analyze and Music ▸ Transcribe on the shared task workspace: their task drafts build
/// the argv the Music Tools page built for the same settings, the well takes the recording, the
/// instruments editor owns both instrument flags, the destinations routing fills are the page's
/// two files, and a draft the page kept seeds the task draft once.
final class StudioMusicTasksTests: XCTestCase {
    // MARK: Argv parity with the page

    func testAnalyzeTaskDraftBuildsThePagesArgv() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicAnalyze))
        var page = template.defaultDraft()
        page.inputPath = "/tmp/harbor-lights.wav"
        page.model = "music-acestep-xl-turbo-lm4b"
        page.useDuration = true
        page.durationSeconds = 30
        page.musicAnalysisMaxTokens = 4_096
        page.musicAnalysisTemperature = 0.5
        page.musicLMTopK = 40
        page.musicLMTopP = 0.85
        page.musicIncludeRawLM = true
        page.musicIncludeAudioCodes = true
        page.musicCheckpointsRoot = "/Volumes/Models/acestep"
        page.musicDecoderSubdirectory = "acestep-v15-turbo"
        page.musicVAESubdirectory = "vae"
        page.musicLMModel = "music-acestep-lm-1.7b"
        page.musicLMSubdirectory = "lm"
        page.musicTextSubdirectory = "text"

        let draft = StudioTaskDraft(templateID: .musicAnalyze, form: StudioConsoleCommand.seed(template: template, draft: page))
        XCTAssertEqual(draft.arguments, template.arguments(from: page))
        XCTAssertEqual(draft.primaryInputPath, "/tmp/harbor-lights.wav")
        XCTAssertEqual(draft.model, "music-acestep-xl-turbo-lm4b")
        XCTAssertEqual(draft.text("--duration"), "30")
        XCTAssertEqual(Array(draft.arguments.prefix(3)), ["music", "analyze", "/tmp/harbor-lights.wav"])
        XCTAssertTrue(draft.arguments.contains("--include-raw-lm"))
        XCTAssertEqual(draft.request()?.mode, .music, "attribution is the template's own")
    }

    func testTranscribeTaskDraftBuildsThePagesArgv() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicTranscribe))
        var page = template.defaultDraft()
        page.inputPath = "/tmp/harbor-lights.wav"
        page.model = "music-muscriptor-large"
        page.musicTranscribeVariant = "large"
        page.musicTranscribeFormat = "json"
        page.musicInstruments = "voice,drums,electric_bass"
        page.musicBeamSize = 4
        page.musicChunkBatchSize = 2
        page.musicMaxTokensPerChunk = 1_024
        page.musicDType = "float16"
        page.musicSampling = true
        page.temperature = 0.7
        page.musicStrictEOS = true
        page.outputPath = "/tmp/out/transcription.json"
        page.musicContextOutput = "/tmp/out/musical-context.json"

        // The page's builder and the contract order `--strict-eos` differently; the command is
        // the same, so compare the commands as the console reads them and as token sets.
        let capability = try XCTUnwrap(template.id.capability)
        func sameCommand(_ lhs: [String], _ rhs: [String], line: UInt = #line) {
            XCTAssertEqual(Array(lhs.prefix(3)), Array(rhs.prefix(3)), line: line)
            XCTAssertEqual(lhs.sorted(), rhs.sorted(), line: line)
            XCTAssertEqual(
                StudioConsoleCommand.seed(capability: capability, arguments: lhs),
                StudioConsoleCommand.seed(capability: capability, arguments: rhs),
                line: line
            )
        }
        let draft = StudioTaskDraft(templateID: .musicTranscribe, form: StudioConsoleCommand.seed(template: template, draft: page))
        sameCommand(draft.arguments, template.arguments(from: page))
        XCTAssertEqual(draft.text("--instruments"), "voice,drums,electric_bass")
        XCTAssertEqual(StudioInstrumentList.decode(draft.text("--instruments")), ["voice", "drums", "electric_bass"])
        XCTAssertEqual(draft.arguments.firstIndex(of: "--instruments").map { draft.arguments[$0 + 1] }, "voice,drums,electric_bass")
        XCTAssertEqual(draft.arguments.firstIndex(of: "--format").map { draft.arguments[$0 + 1] }, "json")
        XCTAssertTrue(draft.arguments.contains("--sampling"))
        XCTAssertTrue(draft.arguments.contains("--strict-eos"))
        XCTAssertFalse(draft.arguments.contains("--no-musical-context"), "the page's context toggle stays on by default")

        var quiet = page
        quiet.musicNoMusicalContext = true
        quiet.musicContextOutput = ""
        let noContext = StudioTaskDraft(templateID: .musicTranscribe, form: StudioConsoleCommand.seed(template: template, draft: quiet))
        sameCommand(noContext.arguments, template.arguments(from: quiet))
        XCTAssertTrue(noContext.arguments.contains("--no-musical-context"))
        XCTAssertFalse(noContext.arguments.contains("--context-output"))
    }

    // MARK: The surface

    func testTheWellTakesTheRecordingAndNothingElse() {
        for templateID in [CommandTemplateID.musicAnalyze, .musicTranscribe] {
            let slots = StudioTaskSchema.slots(for: templateID)
            XCTAssertEqual(slots.map(\.id), ["audio"], "\(templateID)")
            XCTAssertEqual(slots.first?.acceptedTypes, [.audio], "\(templateID)")
            XCTAssertEqual(slots.first?.storage, .argument(0), "\(templateID)")
        }
        XCTAssertEqual(StudioTaskSchema.slots(for: .musicAnalyze).first?.isRequired, true)
        XCTAssertEqual(StudioTaskSchema.slots(for: .musicTranscribe).first?.isRequired, false, "the CLI's audio positional is optional")
        for task in [StudioTask.musicAnalyze, .musicTranscribe] {
            let presentation = task.presentation.attaching(StudioTaskSchema.primarySlot(for: task.variantTemplates[0].id))
            XCTAssertEqual(presentation.attachLabel, "Choose audio…", "\(task)")
            XCTAssertTrue(presentation.requiresAttachment, "\(task)")
            XCTAssertNil(StudioTaskSchema.variantField(for: task), "\(task) runs one template")
            XCTAssertEqual(task.analyzeArchetype?.inputKind(for: nil), .audio)
        }
        XCTAssertEqual(StudioTask.musicAnalyze.analyzeArchetype?.views, [.analysis, .json])
        XCTAssertEqual(StudioTask.musicTranscribe.analyzeArchetype?.views, [.notes, .json])
    }

    func testTheInstrumentsEditorOwnsBothInstrumentFlags() throws {
        let draft = StudioTaskDraft(templateID: .musicTranscribe)
        let fields = StudioTaskSchema.fields(for: .musicTranscribe, draft: draft)
        let instruments = try XCTUnwrap(fields.first { $0.overrideID == .instruments })
        XCTAssertEqual(Set(instruments.bindings.map(\.fieldID)), ["--instruments", "--list-instruments"])
        XCTAssertEqual(fields.filter { $0.overrideID == .instruments }.count, 1, "one editor, drawn where --instruments is declared")
        XCTAssertFalse(fields.contains { $0.overrideID == nil && ["--instruments", "--list-instruments"].contains($0.flag) },
                       "neither flag is also a plain control")
        let capability = try XCTUnwrap(CommandTemplateID.musicTranscribe.capability)
        let hidden = StudioTaskSchema.hiddenFlags(for: capability)
        XCTAssertTrue(hidden.isSuperset(of: ["--output", "--context-output"]), "routing owns both destinations")
        XCTAssertFalse(fields.contains { ["--output", "--context-output"].contains($0.flag) })
        XCTAssertTrue(fields.contains { $0.flag == "--format" })
        XCTAssertTrue(fields.contains { $0.flag == "--no-musical-context" })
        XCTAssertTrue(fields.contains { $0.overrideID == .model })

        let analyze = StudioTaskSchema.fields(for: .musicAnalyze, draft: StudioTaskDraft(templateID: .musicAnalyze))
        XCTAssertTrue(analyze.contains { $0.flag == "--duration" })
        XCTAssertTrue(analyze.contains { $0.flag == "--checkpoints-root" }, "the checkpoint layout stays reachable")
        XCTAssertEqual(analyze.first { $0.flag == "--checkpoints-root" }?.group, .model, "a model location files with the model")
    }

    func testModelScopeIsTheMusicCategory() {
        XCTAssertEqual(StudioTaskSchema.modelScope(for: StudioTaskDraft(templateID: .musicAnalyze)).categories, ["music"])
        // The page forced these models only when the template's default was blank; it never is,
        // so a fresh task draft carries the same `--model` the page sent.
        XCTAssertEqual(StudioTaskDraft(templateID: .musicAnalyze).text("--model"), "music-acestep")
        XCTAssertEqual(StudioTaskDraft(templateID: .musicTranscribe).text("--model"), "music-muscriptor-medium")
        XCTAssertEqual(StudioTaskSchema.modelID(for: StudioTaskDraft(templateID: .musicAnalyze)), "music-acestep")
        XCTAssertEqual(StudioTaskSchema.modelID(for: StudioTaskDraft(templateID: .musicTranscribe)), "music-muscriptor-medium")
    }

    // MARK: Destinations

    func testTranscribeRoutesTheMIDIAndItsContextTogether() throws {
        try withConfiguredRoot { root in
            var draft = StudioTaskDraft(templateID: .musicTranscribe)
            draft.setArgument(0, "/tmp/harbor-lights.wav")
            let named = StudioOutputLocation.destination(for: draft)
            let output = URL(fileURLWithPath: named.text("--output"))
            XCTAssertEqual(output.deletingLastPathComponent().path, root.appendingPathComponent("Music").path)
            XCTAssertTrue(output.lastPathComponent.hasPrefix("harbor-lights-"), output.path)
            XCTAssertEqual(output.pathExtension, "mid")
            XCTAssertEqual(named.text("--context-output"), output.deletingPathExtension().path + "-context.json")
            let request = try XCTUnwrap(named.request())
            XCTAssertEqual(request.draft.outputPath, output.path, "the job lifecycle reads the same file")
            XCTAssertEqual(request.execution?.arguments.firstIndex(of: "--context-output").map { request.execution!.arguments[$0 + 1] },
                           named.text("--context-output"))
        }
    }

    /// Without a configured root a transcription files by what it is: MIDI is audio and lands
    /// under Music, a JSON or JSON Lines event list is text and lands under Documents.
    func testTranscriptionsFileByTheirFormat() {
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/song.mid")), .audio)
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/song.midi")), .audio)
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/song.json")), .text)
        XCTAssertEqual(StudioOutputFileKind.classify(URL(fileURLWithPath: "/tmp/song.jsonl")), .text)
        let home = URL(fileURLWithPath: "/Users/example", isDirectory: true)
        XCTAssertEqual(StudioOutputLocation.directory(domain: .music, kind: .audio, configuredRoot: "", home: home).path,
                       "/Users/example/Music/mere.run/Music")
        XCTAssertEqual(StudioOutputLocation.directory(domain: .music, kind: .text, configuredRoot: "", home: home).path,
                       "/Users/example/Documents/mere.run/Music")
    }

    func testAnalyzeHasNoDestinationToRoute() {
        let draft = StudioTaskDraft(templateID: .musicAnalyze)
        XCTAssertNil(CommandTemplateID.musicAnalyze.capability?.output.flag, "music analyze prints its result")
        XCTAssertEqual(StudioOutputLocation.destination(for: draft), draft)
    }

    // MARK: The page's drafts

    @MainActor
    func testThePagesDraftsSeedTheTaskDraftsOnce() throws {
        let sessions = StudioTaskSessions()
        var analyze = try XCTUnwrap(CommandCatalog.template(id: .musicAnalyze)).defaultDraft()
        analyze.inputPath = "/tmp/harbor-lights.wav"
        analyze.useDuration = true
        analyze.durationSeconds = 45
        sessions.set(analyze, for: StudioTask.musicAnalyze.rawValue + ".MusicTools.analyzeDraft")
        let importedAnalyze = try XCTUnwrap(sessions.taskDraft(for: .musicAnalyze))
        XCTAssertEqual(importedAnalyze.primaryInputPath, "/tmp/harbor-lights.wav")
        XCTAssertEqual(importedAnalyze.text("--duration"), "45")

        // The page stamped a transcription and a context path into every draft it kept; those
        // were the last run's files, not settings, so the import drops them and routing names
        // fresh ones beside the recording rather than writing over the old run.
        try withConfiguredRoot { root in
            let pageMusic = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Music/mere.run/Music", isDirectory: true)
            var transcribe = try XCTUnwrap(CommandCatalog.template(id: .musicTranscribe)).defaultDraft()
            transcribe.inputPath = "/tmp/harbor-lights.wav"
            transcribe.musicInstruments = "voice,drums"
            transcribe.musicTranscribeFormat = "jsonl"
            transcribe.outputPath = pageMusic.appendingPathComponent("transcription-20260903-101500.jsonl").path
            transcribe.musicContextOutput = pageMusic.appendingPathComponent("musical-context-20260903-101500.json").path
            sessions.set(transcribe, for: StudioTask.musicTranscribe.rawValue + ".MusicTools.transcribeDraft")
            let importedTranscribe = try XCTUnwrap(sessions.taskDraft(for: .musicTranscribe))
            XCTAssertEqual(importedTranscribe.text("--instruments"), "voice,drums")
            XCTAssertEqual(importedTranscribe.text("--format"), "jsonl")
            XCTAssertEqual(importedTranscribe.text("--output"), "", "the page's stamped transcription path is not a setting")
            XCTAssertEqual(importedTranscribe.text("--context-output"), "", "nor is its context path")
            let named = StudioOutputLocation.destination(for: importedTranscribe)
            let output = URL(fileURLWithPath: named.text("--output"))
            XCTAssertEqual(output.deletingLastPathComponent().path, root.appendingPathComponent("Music").path)
            XCTAssertTrue(output.lastPathComponent.hasPrefix("harbor-lights-"), output.path)
            XCTAssertEqual(output.pathExtension, "jsonl")
            XCTAssertEqual(named.text("--context-output"), output.deletingPathExtension().path + "-context.json")
        }
    }

    private func withConfiguredRoot(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioMusicTasksTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        StudioTestDefaults.redirectOutputs(under: root, outputs: root)
        defer {
            StudioTestDefaults.restore()
            try? FileManager.default.removeItem(at: root)
        }
        try body(root)
    }
}
