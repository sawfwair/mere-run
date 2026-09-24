import Foundation
import MereRunContract
import StudioTestSupport
@testable import StudioKit
import XCTest

/// Audio ▸ Who Spoke, Enhance, Separate, Live, and Voice ▸ Voices on their task drafts: each
/// builds the argv its page built from a `CommandDraft`, the documents the pages read decode,
/// the live launch turns on what the session needs, and the Voices page's create and delete
/// run through the task runner.
@MainActor
final class StudioAudioVoiceTests: XCTestCase {
    // MARK: - Argv parity with the retired pages

    private func fixedOutput(_ path: String, in draft: inout StudioTaskDraft, flag: String) {
        draft.form[flag] = .text(path)
    }

    func testWhoSpokeTaskDraftBuildsTheVoicePagesArgv() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechDiarize))
        var legacy = template.defaultDraft()
        legacy.inputPath = "/tmp/standup.wav"
        legacy.outputPath = "/tmp/standup-speakers.json"
        var draft = StudioTaskDraft(templateID: .speechDiarize)
        draft.setArgument(0, "/tmp/standup.wav")
        fixedOutput("/tmp/standup-speakers.json", in: &draft, flag: "--output")
        XCTAssertEqual(draft.arguments, template.arguments(from: legacy))
        XCTAssertEqual(draft.primaryInputPath, "/tmp/standup.wav")

        // The Nemotron 3 switch and its input buffer, as the page's buttons set them.
        legacy.model = "speech-diarization-nemotron3"
        legacy.speechDiarizationLatency = "0.64"
        legacy.speechDiarizationFormat = "rttm"
        legacy.outputPath = "/tmp/standup-speakers.rttm"
        draft.model = "speech-diarization-nemotron3"
        draft.form["--latency"] = .text("0.64")
        draft.form["--format"] = .text("rttm")
        fixedOutput("/tmp/standup-speakers.rttm", in: &draft, flag: "--output")
        XCTAssertEqual(draft.arguments, template.arguments(from: legacy))
    }

    func testWhoSpokeOffersTheModelFormatAndInputBufferAsChips() throws {
        let draft = StudioTaskDraft(templateID: .speechDiarize)
        let chips = StudioTaskSchema.essentials(for: .audioWhoSpoke, draft: draft).map(\.flag)
        XCTAssertEqual(chips, ["--format", "--latency"], "the model chip is drawn separately")
        let sections = StudioTaskSchema.sections(for: .audioWhoSpoke, draft: draft)
        XCTAssertEqual(Set(sections.flatMap(\.fields).map(\.flag)),
                       ["--model", "--format", "--threshold", "--min-duration", "--merge-gap", "--latency"])
        XCTAssertEqual(StudioTaskSchema.advanced(for: .audioWhoSpoke, draft: draft).map(\.flag), ["--quiet"])
        let slot = try XCTUnwrap(StudioTaskSchema.primarySlot(for: .speechDiarize))
        XCTAssertEqual(slot.storage, .argument(0))
        XCTAssertTrue(slot.acceptedTypes.contains(.audio))
    }

    func testDiarizeInputBufferIsValidatedAgainstTheModel() throws {
        let capability = try XCTUnwrap(CommandTemplateID.speechDiarize.capability)
        var draft = StudioTaskDraft(templateID: .speechDiarize)
        draft.setArgument(0, "/tmp/standup.wav")
        XCTAssertNil(StudioConsoleCommand.validationMessage(for: capability, draft: draft.form))
        draft.form["--latency"] = .text("1.04")
        XCTAssertEqual(
            StudioConsoleCommand.validationMessage(for: capability, draft: draft.form),
            "Input buffer latency applies to Nemotron 3 only; choose Offline for speech-diarization-sortformer."
        )
        draft.model = "speech-diarization-nemotron3"
        XCTAssertNil(StudioConsoleCommand.validationMessage(for: capability, draft: draft.form))
        draft.model = ""
        draft.form["--latency"] = .text("offline")
        XCTAssertNil(StudioConsoleCommand.validationMessage(for: capability, draft: draft.form))
    }

    func testEnhanceTaskDraftBuildsTheAudioToolsPagesArgv() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .audioEnhance))
        var legacy = template.defaultDraft()
        legacy.inputPath = "/tmp/memo.wav"
        legacy.outputPath = "/tmp/memo-48k.wav"
        legacy.model = "audio-enhance-universr-audio"
        legacy.audioInputRate = 16_000
        legacy.audioODEMethod = "rk4"
        legacy.audioODESteps = 8
        legacy.audioGuidanceScale = 2
        legacy.audioChunkSeconds = 12
        legacy.audioDType = "float16"
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.setArgument(0, "/tmp/memo.wav")
        fixedOutput("/tmp/memo-48k.wav", in: &draft, flag: "--output")
        draft.model = "audio-enhance-universr-audio"
        draft.form["--input-rate"] = .integer(16_000)
        draft.form["--ode-method"] = .text("rk4")
        draft.form["--ode-steps"] = .integer(8)
        draft.form["--guidance-scale"] = .number(2)
        draft.form["--chunk-seconds"] = .integer(12)
        draft.form["--dtype"] = .text("float16")
        XCTAssertEqual(draft.arguments, template.arguments(from: legacy))
        XCTAssertEqual(StudioTaskSchema.essentials(for: .audioEnhance, draft: draft).map(\.flag), ["--dtype"])
    }

    func testAudioEditIsNotAnEnhanceVariant() {
        XCTAssertEqual(StudioTask.audioEnhance.variantTemplates.map(\.id), [.audioEnhance])
        XCTAssertNil(StudioTaskSchema.variantField(for: .audioEnhance))
    }

    func testSeparateTaskDraftBuildsTheAudioToolsPagesArgvUnderBothTasks() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicSeparate))
        var legacy = template.defaultDraft()
        legacy.inputPath = "/tmp/late-set.wav"
        legacy.outputPath = "/tmp/late-set-stems"
        legacy.model = "music-separate-bs-roformer-4stem"
        legacy.audioOverlap = 4
        legacy.audioDType = "float32"
        for task in [StudioTask.audioSeparate, .musicSeparate] {
            var draft = try XCTUnwrap(StudioTaskDraft(task: task))
            XCTAssertEqual(draft.templateID, .musicSeparate)
            draft.setArgument(0, "/tmp/late-set.wav")
            fixedOutput("/tmp/late-set-stems", in: &draft, flag: "--output-dir")
            draft.model = "music-separate-bs-roformer-4stem"
            draft.form["--overlap"] = .integer(4)
            draft.form["--dtype"] = .text("float32")
            XCTAssertEqual(draft.arguments, template.arguments(from: legacy), "\(task)")
            XCTAssertTrue(task.runs(.musicSeparate))
        }
    }

    func testSeparateRowsShowOnTheAudioAndTheMusicPage() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicSeparate))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/late-set.wav"
        let row = StudioLibraryItem(
            id: UUID(), mode: .music, prompt: "", inputURL: URL(fileURLWithPath: "/tmp/late-set.wav"),
            outputURL: URL(fileURLWithPath: "/tmp/late-set-stems"), createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "mere.run music separate late-set.wav",
            outputText: nil, templateID: .musicSeparate, commandDraft: draft
        )
        for task in [StudioTask.audioSeparate, .musicSeparate] {
            XCTAssertEqual(StudioFeedCardBuilder.cards(items: [row], task: task) { _ in nil }.map(\.id), [row.id], "\(task)")
        }
        XCTAssertTrue(StudioFeedCardBuilder.cards(items: [row], task: .audioEnhance) { _ in nil }.isEmpty)
    }

    func testAudioToolsPageDraftsImportOnce() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicSeparate))
        var parked = template.defaultDraft()
        parked.inputPath = "/tmp/late-set.wav"
        parked.model = "music-separate-mel-roformer-denoise"
        let sessions = StudioTaskSessions()
        sessions.set(parked, for: StudioTask.audioSeparate.rawValue + ".AudioTools.separationDraft")
        let imported = try XCTUnwrap(sessions.taskDraft(for: .audioSeparate))
        XCTAssertEqual(imported.primaryInputPath, "/tmp/late-set.wav")
        XCTAssertEqual(imported.model, "music-separate-mel-roformer-denoise")
    }

    // MARK: - Documents

    func testRTTMTimelineReadsIntoTheDiarizationDocument() throws {
        let rttm = """
        SPEAKER standup 1 0.400 9.400 <NA> <NA> speaker_0 <NA> <NA>
        SPEAKER standup 1 10.300 13.800 <NA> <NA> speaker_1 <NA> <NA>
        SPEAKER standup 1 24.600 3.300 <NA> <NA> speaker_0 <NA> <NA>

        """
        let document = try XCTUnwrap(StudioDiarizationDocument.rttm(rttm))
        XCTAssertEqual(document.speakerCount, 2)
        XCTAssertEqual(document.segments.map(\.speakerIndex), [0, 1, 0])
        XCTAssertEqual(document.segments.map(\.endSeconds), [9.8, 24.1, 27.9])
        XCTAssertEqual(document.durationSeconds, 27.9)
        XCTAssertEqual(document.speakers.map(\.name), ["Speaker 1", "Speaker 2"])
        guard case .diarization(let decoded) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(rttm.utf8))) else {
            return XCTFail("An RTTM timeline decodes as the diarization document")
        }
        XCTAssertEqual(decoded, document)
        XCTAssertNil(StudioDiarizationDocument.rttm("[00:04.120 --> 00:07.400] and that is the whole idea."))
        XCTAssertEqual(StudioAnalyzeDocumentSource.preferredExtensions(for: .speechDiarize), ["rttm"])
    }

    func testSeparationManifestDecodesAndTheAnalyzeDocumentPicksIt() throws {
        let json = """
        {
          "chunk_size" : 352800,
          "chunks" : 3,
          "created_at" : "2026-09-24T14:02:11Z",
          "elapsed_seconds" : 21.5,
          "manifest_path" : "/tmp/late-set-stems/separation.json",
          "model" : {
            "compute_type" : "float16",
            "id" : "music-separate-bs-roformer-viperx-1297",
            "license" : "MIT",
            "repository" : "mere-run/roformer",
            "revision" : "abc",
            "weights_sha256" : "00"
          },
          "overlap" : 2,
          "schema_version" : 1,
          "source" : {
            "channels" : 2,
            "frames" : 529200,
            "path" : "/tmp/late-set.wav",
            "sample_rate" : 44100,
            "sha256" : "11"
          },
          "stems" : [
            { "name" : "vocals", "path" : "/tmp/late-set-stems/vocals.wav", "sha256" : "22" },
            { "name" : "instrumental", "path" : "/tmp/late-set-stems/instrumental.wav", "sha256" : "33" }
          ]
        }
        """
        let manifest = try XCTUnwrap(StudioSeparationManifest.decode(Data(json.utf8)))
        XCTAssertEqual(manifest.stems.map(\.title), ["Vocals", "Instrumental"])
        XCTAssertEqual(manifest.summary, "2 stems · 3 chunks")
        XCTAssertEqual(manifest.model.id, "music-separate-bs-roformer-viperx-1297")
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        XCTAssertEqual(document, .separation(manifest))
        XCTAssertEqual(document.modelID, manifest.model.id)
        XCTAssertEqual(document.summary(detectionCount: 0), "2 stems · 3 chunks")
        XCTAssertTrue(document.speechSegments.isEmpty)
    }

    // MARK: - Live listen

    func testLiveListenLaunchTurnsOnTheStreamingSwitchesWithoutParkingThem() throws {
        XCTAssertEqual(StudioTask.audioLive.variantTemplates.map(\.id), [.speechListen, .speechDiarizeLive])
        var listen = StudioTaskDraft(templateID: .speechListen)
        listen.form["--device"] = .text("BuiltInMicrophoneDevice")
        listen.form["--language"] = .text("en")
        XCTAssertEqual(listen.arguments, ["speech", "listen", "--device", "BuiltInMicrophoneDevice", "--language", "en"])
        XCTAssertEqual(
            listen.liveListenLaunch().arguments,
            ["speech", "listen", "--device", "BuiltInMicrophoneDevice", "--language", "en", "--quiet", "--jsonl"]
        )
        XCTAssertFalse(listen.arguments.contains("--jsonl"), "the parked draft never carries the launch switches")

        var speakers = listen
        speakers.switchTemplate(to: .speechDiarizeLive)
        XCTAssertEqual(speakers.text("--device"), "BuiltInMicrophoneDevice", "the microphone carries across the switch")
        XCTAssertEqual(speakers.model, "", "the switch clears the model; the CLI's default is Nemotron 3")
        speakers.form["--latency"] = .text("0.64")
        XCTAssertEqual(
            speakers.liveListenLaunch().arguments,
            ["speech", "diarize-live", "--device", "BuiltInMicrophoneDevice", "--latency", "0.64", "--quiet"]
        )
        XCTAssertEqual(
            StudioTaskDraft(templateID: .speechDiarizeLive).liveListenLaunch().arguments,
            ["speech", "diarize-live", "--model", "speech-diarization-nemotron3", "--quiet"]
        )
        for field in StudioTaskSchema.fields(for: .audioLive, draft: listen) + StudioTaskSchema.fields(for: .audioLive, draft: speakers)
        where StudioTaskDraft.liveListenOwnedFlags.contains(field.flag) {
            XCTAssertTrue(["--device", "--list-devices", "--stdin", "--jsonl", "--quiet"].contains(field.flag))
        }
    }

    func testListenDeviceListKeepsUIDSeparateFromDisplayName() {
        let devices = StudioListenDevice.parseList(
            "* BuiltInDeviceUID\tMacBook Pro Microphone\n  USBDeviceUID\tStudio USB Mic\n"
        )
        XCTAssertEqual(devices, [
            .init(uid: "BuiltInDeviceUID", name: "MacBook Pro Microphone", isDefault: true),
            .init(uid: "USBDeviceUID", name: "Studio USB Mic", isDefault: false),
        ])
    }

    // MARK: - Voices

    func testVoiceProfileDraftsBuildTheVoicePagesArgv() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechProfileCreate))
        var legacy = template.defaultDraft()
        legacy.prompt = "Narrator"
        legacy.inputPath = "/tmp/narrator.wav"
        legacy.secondaryText = "A calm reading of the opening paragraph."
        legacy.language = "en"
        var draft = StudioTaskDraft(templateID: .speechProfileCreate)
        draft.form["--name"] = .text("Narrator")
        let slot = try XCTUnwrap(StudioTaskSchema.primarySlot(for: .speechProfileCreate), "the reference audio is a well slot")
        XCTAssertEqual(slot.storage, .flag("--audio"))
        XCTAssertTrue(slot.isRequired)
        slot.attach([URL(fileURLWithPath: "/tmp/narrator.wav")], to: &draft)
        draft.form["--text"] = .text("A calm reading of the opening paragraph.")
        draft.form["--language"] = .text("en")
        XCTAssertEqual(draft.arguments, template.arguments(from: legacy))
        XCTAssertEqual(draft.primaryInputPath, "/tmp/narrator.wav")

        let id = UUID()
        XCTAssertEqual(StudioTaskDraft.deletingVoiceProfile(id).arguments, ["speech", "profile", "delete", "--id", id.uuidString])
        XCTAssertEqual(StudioTask.voiceVoices.variantTemplates.map(\.id), [.speechProfileCreate])
    }

    func testVoiceProfileRecordResolvesItsReference() throws {
        let json = """
        [{"id":"6F9B2C1E-0D44-4C1B-9A7E-3B2C4D5E6F70","name":"Narrator","createdAt":800000000,"updatedAt":800000000,
          "transcript":"A calm reading.","language":"en","referenceAudioRelativePath":"6F9B2C1E/reference.wav","modelFingerprint":null}]
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let record = try XCTUnwrap(decoder.decode([StudioVoiceProfileRecord].self, from: Data(json.utf8)).first)
        XCTAssertEqual(record.referenceAudioURL, StudioVoiceProfileStore.voicesDirectory.appendingPathComponent("6F9B2C1E/reference.wav"))
        let absolute = StudioVoiceProfileRecord(
            id: record.id, name: record.name, createdAt: record.createdAt, updatedAt: record.updatedAt, transcript: record.transcript,
            language: nil, referenceAudioRelativePath: "/tmp/reference.wav", modelFingerprint: nil
        )
        XCTAssertEqual(absolute.referenceAudioURL, URL(fileURLWithPath: "/tmp/reference.wav"))
    }

    // MARK: - Runner

    func testVoiceCreateAndDeleteRunThroughTheTaskRunner() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("audio-voice-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let processRunner = RecordingProcessRunner()
        let controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: processRunner, resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json"))
        )
        defer { controller.terminateAllProcesses() }
        let library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        library.observe(controller: controller)
        let runner = StudioTaskRunner(controller: controller, library: library)
        controller.checkReadiness(for: .voiceVoices, modelID: StudioTaskSchema.modelID(for: StudioTaskDraft(templateID: .speechProfileCreate)))
        XCTAssertEqual(controller.readiness(for: .voiceVoices), .ready, "profile create runs no managed model")

        let reference = root.appendingPathComponent("narrator.wav")
        try Data([0x52, 0x49, 0x46, 0x46]).write(to: reference)
        var create = StudioTaskDraft(templateID: .speechProfileCreate)
        XCTAssertThrowsError(try runner.run(create, task: .voiceVoices), "the reference audio is required") { error in
            XCTAssertEqual((error as? StudioValidationError)?.message, "Reference audio (--audio) is required.")
        }
        create.form["--name"] = .text("Narrator")
        create.form["--audio"] = .text(reference.path)
        let created = try runner.run(create, task: .voiceVoices)
        XCTAssertEqual(created.mode, .speak, "profiles file under Voice like Speak's runs")
        XCTAssertEqual(
            created.execution?.arguments,
            ["speech", "profile", "create", "--name", "Narrator", "--audio", reference.path, "--language", "auto"],
            "the template's default language rides along, as the page's draft always sent it"
        )
        XCTAssertEqual(runner.currentJob(for: .voiceVoices)?.request.requestID, created.id)
        XCTAssertEqual(library.items.first?.templateID, .speechProfileCreate)
        runner.stop(task: .voiceVoices)

        let id = UUID()
        let deleted = try runner.run(.deletingVoiceProfile(id), task: .voiceVoices)
        XCTAssertEqual(deleted.execution?.arguments, ["speech", "profile", "delete", "--id", id.uuidString])
        XCTAssertEqual(deleted.mode, .speak)
        XCTAssertEqual(runner.currentJob(for: .voiceVoices)?.request.requestID, deleted.id)
    }
}
