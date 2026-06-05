import XCTest
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
        XCTAssertEqual(cmd.shift, 1.0, accuracy: 0.0001)
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
}
