import XCTest
import MLX
@testable import MereRunCLI
@testable import MereRunCore

final class MusicGenerateCommandParsingTests: XCTestCase {
    func testMusicGenerateParsesManagedDefaultModel() throws {
        let cmd = try MusicGenerate.parse([
            "warm synthwave groove",
        ])

        XCTAssertEqual(cmd.caption, "warm synthwave groove")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.aceStep.rawValue)
        XCTAssertNil(cmd.checkpointsRoot)
        XCTAssertEqual(cmd.turboSubdirectory, "acestep-v15-turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "vae")
        XCTAssertFalse(cmd.useLM)
        XCTAssertEqual(cmd.durationSeconds, 10.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.shift, 3.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.coverNoiseStrength, 0.0, accuracy: 0.0001)
        XCTAssertFalse(cmd.resolvedACEStepIsCover)
    }

    func testMusicGenerateParsesModelAndAdvancedOverrides() throws {
        let cmd = try MusicGenerate.parse([
            "club track",
            "--model", "/tmp/acestep",
            "--checkpoints-root", "/tmp/checkpoints",
            "--turbo-subdirectory", "turbo",
            "--vae-subdirectory", "custom-vae",
            "--lm-subdirectory", "custom-lm",
            "--text-subdirectory", "text-encoder",
            "--use-lm",
            "--duration", "18",
            "--steps", "12",
        ])

        XCTAssertEqual(cmd.model, "/tmp/acestep")
        XCTAssertEqual(cmd.checkpointsRoot, "/tmp/checkpoints")
        XCTAssertEqual(cmd.turboSubdirectory, "turbo")
        XCTAssertEqual(cmd.vaeSubdirectory, "custom-vae")
        XCTAssertEqual(cmd.lmSubdirectory, "custom-lm")
        XCTAssertEqual(cmd.textSubdirectory, "text-encoder")
        XCTAssertTrue(cmd.useLM)
        XCTAssertEqual(cmd.durationSeconds, 18.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.steps, 12)
    }

    func testMusicGenerateCheckpointCandidatesHonorRequestedManagedModel() throws {
        let cmd = try MusicGenerate.parse([
            "xl track",
            "--model", ModelResolver.ModelID.aceStepXLTurbo.rawValue,
        ])

        let candidates = cmd.buildAcestepCheckpointCandidates().map(\.standardizedFileURL.path)
        let requestedRoot = MereRunModelPaths.modelsDir
            .appendingPathComponent(ModelResolver.ModelID.aceStepXLTurbo.rawValue, isDirectory: true)
            .standardizedFileURL
            .path
        let defaultRoot = MereRunModelPaths.modelsDir
            .appendingPathComponent(ModelResolver.ModelID.aceStep.rawValue, isDirectory: true)
            .standardizedFileURL
            .path

        XCTAssertTrue(candidates.contains(requestedRoot))
        XCTAssertFalse(candidates.contains(defaultRoot))
    }

    func testMusicGenerateResolvesCoverOnlyForCoverTask() throws {
        let cover = try MusicGenerate.parse([
            "cover this",
            "--task-type", "cover",
        ])
        XCTAssertTrue(cover.resolvedACEStepIsCover)

        let forcedNonCover = try MusicGenerate.parse([
            "cover this",
            "--task-type", "cover",
            "--non-cover",
        ])
        XCTAssertFalse(forcedNonCover.resolvedACEStepIsCover)

        let textToMusic = try MusicGenerate.parse([
            "new song",
            "--task-type", "text2music",
        ])
        XCTAssertFalse(textToMusic.resolvedACEStepIsCover)
    }

    func testMusicGenerateSourceAudioImpliesCoverMode() throws {
        let cmd = try MusicGenerate.parse([
            "dream-pop cover",
            "--source-audio", "~/Downloads/song.mp3",
            "--reference-audio", "/tmp/ref-a.wav", "/tmp/ref-b.wav",
            "--analyze-source-audio",
            "--audio-cover-strength", "0.75",
            "--cover-noise-strength", "0.45",
        ])

        XCTAssertEqual(cmd.sourceAudio, "~/Downloads/song.mp3")
        XCTAssertEqual(cmd.referenceAudio, ["/tmp/ref-a.wav", "/tmp/ref-b.wav"])
        XCTAssertTrue(cmd.analyzeSourceAudio)
        XCTAssertEqual(cmd.audioCoverStrength, 0.75, accuracy: 0.0001)
        XCTAssertEqual(cmd.coverNoiseStrength, 0.45, accuracy: 0.0001)
        XCTAssertTrue(cmd.resolvedACEStepIsCover)
        XCTAssertEqual(cmd.resolvedACEStepTaskType(isCover: cmd.resolvedACEStepIsCover), "cover")
    }

    func testMusicGenerateSkipsLMForCoverParity() throws {
        let cover = try MusicGenerate.parse([
            "faithful cover",
            "--source-audio", "~/Downloads/song.mp3",
            "--use-lm",
            "--lm-subdirectory", "acestep-5Hz-lm-4B",
        ])
        let taskType = cover.resolvedACEStepTaskType(isCover: cover.resolvedACEStepIsCover)

        XCTAssertEqual(taskType, "cover")
        XCTAssertFalse(cover.resolvedACEStepUsesLM(taskType: taskType))

        let textToMusic = try MusicGenerate.parse([
            "fresh song",
            "--use-lm",
            "--lm-subdirectory", "acestep-5Hz-lm-4B",
        ])

        XCTAssertTrue(textToMusic.resolvedACEStepUsesLM(taskType: "text2music"))
    }

    func testMusicGenerateCoverDurationUsesSourceAudioLength() throws {
        let cmd = try MusicGenerate.parse([
            "dream-pop cover",
            "--source-audio", "~/Downloads/song.mp3",
            "--duration", "30",
        ])
        let sourceAudio = MLXArray(Array(repeating: Float(0), count: 48_000 * 2 * 2), [1, 48_000 * 2, 2])

        let duration = cmd.resolvedACEStepDurationSeconds(
            isCover: cmd.resolvedACEStepIsCover,
            sourceAudio48kHz: sourceAudio
        )

        XCTAssertEqual(duration, 2.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.resolvedMetadataDuration(effectiveDurationSeconds: duration), "2 seconds")
    }

    func testMusicGenerateSourceAnalysisFillsOnlyMissingMetadata() throws {
        let cmd = try MusicGenerate.parse([
            "reggaeton cover",
            "--source-audio", "~/Downloads/song.mp3",
        ])
        let metadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: nil,
            caption: "reggaeton cover",
            duration: "180 seconds",
            keyscale: "A minor",
            language: nil,
            timesignature: nil
        )
        let analysis = ACEStepMusicUnderstandingMetadata(
            bpm: 95,
            durationSeconds: 180,
            keyscale: "G major",
            language: "en",
            timesignature: "4"
        )

        let merged = cmd.mergedMetadataWithSourceAnalysis(metadata, analysis)

        XCTAssertEqual(merged.metadata.bpm, "95")
        XCTAssertEqual(merged.metadata.caption, "reggaeton cover")
        XCTAssertEqual(merged.metadata.duration, "180 seconds")
        XCTAssertEqual(merged.metadata.keyscale, "A minor")
        XCTAssertEqual(merged.metadata.language, "en")
        XCTAssertEqual(merged.metadata.timesignature, "4")
        XCTAssertEqual(merged.filledFields, ["bpm", "language", "timesignature"])
    }

    func testMusicGenerateParsesMagentaControls() throws {
        let cmd = try MusicGenerate.parse([
            "ambient modular synths",
            "--model", ModelResolver.ModelID.magentaRT2Small.rawValue,
            "--duration", "4",
            "--style-conditioning", "full",
            "--temperature", "0.8",
            "--top-k", "64",
            "--cfg-musiccoca", "2.5",
            "--cfg-notes", "4.5",
            "--cfg-drums", "0.75",
            "--drumless",
            "--unmask-width", "8",
            "--seed-rotation", "3",
            "--prefill-silence",
            "--prefill-duration", "1.25",
        ])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.magentaRT2Small.rawValue)
        XCTAssertEqual(cmd.durationSeconds, 4.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.magentaStyleConditioning, .full)
        XCTAssertEqual(cmd.magentaTemperature, 0.8, accuracy: 0.0001)
        XCTAssertEqual(cmd.magentaTopK, 64)
        XCTAssertEqual(cmd.magentaCFGMusicCoCa, 2.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.magentaCFGNotes, 4.5, accuracy: 0.0001)
        XCTAssertEqual(cmd.magentaCFGDrums, 0.75, accuracy: 0.0001)
        XCTAssertTrue(cmd.magentaDrumless)
        XCTAssertEqual(cmd.magentaUnmaskWidth, 8)
        XCTAssertEqual(cmd.magentaSeedRotation, 3)
        XCTAssertTrue(cmd.magentaPrefillSilence)
        XCTAssertEqual(cmd.magentaPrefillDuration, 1.25, accuracy: 0.0001)
    }

    func testMusicRealtimeParsesMagentaDefaults() throws {
        let cmd = try MusicRealtime.parse([
            "ambient pads",
        ])

        XCTAssertEqual(cmd.prompt, "ambient pads")
        XCTAssertEqual(cmd.model, ModelResolver.ModelID.magentaRT2Small.rawValue)
        XCTAssertEqual(cmd.durationSeconds, 30.0, accuracy: 0.0001)
        XCTAssertTrue(cmd.play)
        XCTAssertNil(cmd.output)
    }

    func testMusicRealtimeParsesHeadlessCapture() throws {
        let cmd = try MusicRealtime.parse([
            "ambient pads",
            "--model", ModelResolver.ModelID.magentaRT2Base.rawValue,
            "--duration", "2",
            "--output", "/tmp/live.wav",
            "--no-play",
            "--interactive",
            "--style-conditioning", "full",
            "--temperature", "0.7",
            "--top-k", "32",
            "--cfg-musiccoca", "2",
            "--cfg-notes", "3",
            "--cfg-drums", "0.5",
            "--drumless",
            "--unmask-width", "4",
            "--seed-rotation", "9",
            "--prefill-silence",
            "--prefill-duration", "1.5",
        ])

        XCTAssertEqual(cmd.model, ModelResolver.ModelID.magentaRT2Base.rawValue)
        XCTAssertEqual(cmd.durationSeconds, 2.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.output, "/tmp/live.wav")
        XCTAssertFalse(cmd.play)
        XCTAssertTrue(cmd.interactive)
        XCTAssertEqual(cmd.styleConditioning, .full)
        XCTAssertEqual(cmd.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(cmd.topK, 32)
        XCTAssertEqual(cmd.cfgMusicCoCa, 2.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.cfgNotes, 3.0, accuracy: 0.0001)
        XCTAssertEqual(cmd.cfgDrums, 0.5, accuracy: 0.0001)
        XCTAssertTrue(cmd.drumless)
        XCTAssertEqual(cmd.unmaskWidth, 4)
        XCTAssertEqual(cmd.seedRotation, 9)
        XCTAssertTrue(cmd.prefillSilence)
        XCTAssertEqual(cmd.prefillDuration, 1.5, accuracy: 0.0001)
    }

    func testMusicRealtimeParsesMIDIControls() throws {
        let cmd = try MusicRealtime.parse([
            "minimal synth pop",
            "--midi-input", "OP-1",
            "--midi-log-events",
            "--midi-log-raw",
            "--midi-channel", "2",
            "--midi-note-offset", "12",
            "--midi-cc", "1=temp:0.2:1.4",
            "--midi-cc", "2=drums:0:2",
        ])

        XCTAssertEqual(cmd.prompt, "minimal synth pop")
        XCTAssertEqual(cmd.midiInput, "OP-1")
        XCTAssertTrue(cmd.midiLogEvents)
        XCTAssertTrue(cmd.midiLogRaw)
        XCTAssertEqual(cmd.midiChannel, "2")
        XCTAssertEqual(cmd.midiNoteOffset, 12)
        XCTAssertEqual(cmd.midiCCMappings, ["1=temp:0.2:1.4", "2=drums:0:2"])
    }

    func testMusicRealtimeParsesMIDIInputListingWithoutPrompt() throws {
        let cmd = try MusicRealtime.parse([
            "--list-midi-inputs",
        ])

        XCTAssertNil(cmd.prompt)
        XCTAssertTrue(cmd.listMIDIInputs)
    }

    func testMusicRealtimeParsesMIDIMonitorWithoutPrompt() throws {
        let cmd = try MusicRealtime.parse([
            "--midi-monitor",
            "--midi-input", "OP-1",
            "--midi-log-raw",
            "--duration", "3",
        ])

        XCTAssertNil(cmd.prompt)
        XCTAssertTrue(cmd.midiMonitor)
        XCTAssertEqual(cmd.midiInput, "OP-1")
        XCTAssertTrue(cmd.midiLogRaw)
        XCTAssertEqual(cmd.durationSeconds, 3.0, accuracy: 0.0001)
    }

    func testMusicRealtimeRequiresMIDIInputForLogging() async throws {
        let cmd = try MusicRealtime.parse([
            "minimal synth pop",
            "--midi-log-events",
        ])

        do {
            try await cmd.run()
            XCTFail("Expected --midi-log-events without --midi-input to fail")
        } catch {
            XCTAssertTrue(String(describing: error).contains("Use --midi-input"))
        }
    }

    func testMagentaRT2MIDICCMappingScalesValues() throws {
        let mapping = try MagentaRT2MIDICCMapping.parse("1=temp:0.2:1.4")

        XCTAssertEqual(mapping.controller, 1)
        XCTAssertEqual(mapping.scaledValue(0), 0.2, accuracy: 0.0001)
        XCTAssertEqual(mapping.scaledValue(127), 1.4, accuracy: 0.0001)
    }

    func testMagentaRT2LiveControlQueueCollectsMIDINotesAndControls() throws {
        let queue = MagentaRT2LiveControlQueue(initialControls: MagentaRT2Controls())
        queue.enqueueNoteOn(60)
        queue.enqueueNoteOff(64)
        try MagentaRT2MIDICCMapping.parse("1=temp:0.2:1.4").apply(rawValue: 127, to: queue)

        let snapshot = try XCTUnwrap(queue.snapshot())
        XCTAssertEqual(snapshot.noteEvents, [.on(60), .off(64)])
        XCTAssertEqual(snapshot.noteOn, [60])
        XCTAssertEqual(snapshot.noteOff, [64])
        let controls = try XCTUnwrap(snapshot.controls)
        XCTAssertEqual(controls.temperature, Float(1.4), accuracy: 0.0001)
    }

    func testMagentaRT2LiveControlQueuePreservesShortMIDITapOrder() throws {
        let queue = MagentaRT2LiveControlQueue(initialControls: MagentaRT2Controls())
        queue.enqueueNoteOn(43)
        queue.enqueueNoteOff(43)

        let snapshot = try XCTUnwrap(queue.snapshot())

        XCTAssertEqual(snapshot.noteEvents, [.on(43), .off(43)])
        XCTAssertEqual(snapshot.noteOn, [43])
        XCTAssertEqual(snapshot.noteOff, [43])
    }

    private func parsedEvents(
        _ parser: inout MagentaRT2MIDIStreamParser,
        _ bytes: [UInt8]
    ) -> [MagentaRT2MIDIEvent] {
        var events: [MagentaRT2MIDIEvent] = []
        parser.consume(bytes) { events.append($0) }
        return events
    }

    func testMIDIStreamParserParsesMultipleMessagesPerPacket() {
        var parser = MagentaRT2MIDIStreamParser()
        let events = parsedEvents(&parser, [0x90, 60, 100, 0x80, 60, 0, 0xB1, 1, 64])

        XCTAssertEqual(events, [
            .noteOn(channel: 1, note: 60, velocity: 100),
            .noteOff(channel: 1, note: 60),
            .controlChange(channel: 2, controller: 1, value: 64),
        ])
    }

    func testMIDIStreamParserHandlesRunningStatus() {
        var parser = MagentaRT2MIDIStreamParser()
        // One 0x90 status followed by three note pairs: a chord under
        // running status, with a velocity-0 pair meaning note-off.
        let events = parsedEvents(&parser, [0x90, 60, 100, 64, 100, 60, 0])

        XCTAssertEqual(events, [
            .noteOn(channel: 1, note: 60, velocity: 100),
            .noteOn(channel: 1, note: 64, velocity: 100),
            .noteOff(channel: 1, note: 60),
        ])
    }

    func testMIDIStreamParserSurvivesInterleavedRealtimeAndPacketBoundaries() {
        var parser = MagentaRT2MIDIStreamParser()
        // Clock (0xF8) interrupts a note-on mid-message, and the message's
        // final byte arrives in the next packet.
        var events = parsedEvents(&parser, [0x90, 0xF8, 60])
        XCTAssertTrue(events.isEmpty)
        events = parsedEvents(&parser, [100])
        XCTAssertEqual(events, [.noteOn(channel: 1, note: 60, velocity: 100)])
    }

    func testMIDIStreamParserSkipsUnrelatedMessagesAndSysEx() {
        var parser = MagentaRT2MIDIStreamParser()
        let events = parsedEvents(&parser, [
            0xC0, 5, // program change: 1 data byte, ignored
            0xE0, 0, 64, // pitch bend: 2 data bytes, ignored
            0xF0, 1, 2, 3, 0xF7, // sysex: dropped, cancels running status
            33, 44, // stray data bytes with no status: dropped
            0x91, 62, 90, // then a normal note-on still parses
        ])

        XCTAssertEqual(events, [.noteOn(channel: 2, note: 62, velocity: 90)])
    }
}
