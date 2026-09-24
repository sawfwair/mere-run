@testable import StudioKit
@testable import StudioUI
import XCTest

/// The Music renderers register by `(view, document)`: the analysis view draws `music analyze`'s
/// document, the notes view draws a MIDI file, and neither claims the other's pair or the JSON
/// view, which stays the raw document.
final class StudioMusicRenderersTests: XCTestCase {
    private let analysis = StudioMusicAnalysisDocument.decode("""
    {"audio":"/tmp/harbor-lights.wav","model":"music-acestep","inputDurationSeconds":214.6,
     "analyzedDurationSeconds":30,"metadata":{"bpm":96,"keyscale":"D major","timesignature":"4/4",
     "language":"en","caption":"Warm indie folk.","lyrics":"Harbour lights are blinking slow"}}
    """)

    private var midi: StudioMIDISummary {
        StudioMIDISummary(
            format: 0, trackCount: 1, ticksPerQuarter: 480,
            notes: [StudioMIDINote(id: 0, startTick: 0, durationTicks: 480, pitch: 60, velocity: 100, channel: 0)],
            tempoMicrosecondsPerQuarter: 625_000
        )
    }

    func testAnalysisViewDrawsTheAnalysisDocument() throws {
        let document = try XCTUnwrap(analysis)
        XCTAssertEqual(StudioResultRenderers.rendering(for: .analysis, document: .musicAnalysis(document)), .musicAnalysis(document))
        XCTAssertNil(StudioResultRenderers.rendering(for: .json, document: .musicAnalysis(document)), "JSON stays the raw document")
        XCTAssertNil(StudioResultRenderers.rendering(for: .analysis, document: .midi(midi)))
        XCTAssertNil(StudioResultRenderers.rendering(for: .analysis, document: nil))
        XCTAssertEqual(StudioAnalyzeDocument.musicAnalysis(document).summary(detectionCount: 0), "96 BPM · D major · 0:30 of 3:35")
    }

    func testNotesViewDrawsTheMIDIFile() throws {
        XCTAssertEqual(StudioResultRenderers.rendering(for: .notes, document: .midi(midi)), .pianoRoll(midi))
        XCTAssertNil(StudioResultRenderers.rendering(for: .notes, document: .musicAnalysis(try XCTUnwrap(analysis))))
        XCTAssertNil(StudioResultRenderers.rendering(for: .json, document: .midi(midi)))
        XCTAssertEqual(StudioAnalyzeDocument.midi(midi).summary(detectionCount: 0), "1 note · 1 track")
        XCTAssertEqual(StudioAnalyzeDocumentSource.preferredExtensions(for: .musicTranscribe), ["mid", "midi"],
                       "the MIDI is the document even with the context JSON beside it")
    }
}
