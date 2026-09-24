@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import XCTest

/// The Sound renderers in the registry: the CLAP gauge answers the Score view, and a finished
/// card in the Sound feed draws Video Foley's sync review or a tensor's header in place of the
/// output grid, keyed by what the run wrote rather than by task.
final class StudioSoundRendererTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("sound-renderers-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    func testTheScoreViewRendersTheCLAPGauge() {
        let output = StudioCLAPScore.Output(score: 0.63, prompt: "a glass bottle breaking", audio: "/tmp/bottle.wav", model: "sfx-woosh-clap")
        let item = row(templateID: .sfxClapScore, input: root.appendingPathComponent("bottle.wav"), output: nil)
        XCTAssertEqual(StudioResultRenderers.rendering(for: .score, document: .clap(output), item: item), .clap(output))
        XCTAssertNil(StudioResultRenderers.rendering(for: .json, document: .clap(output), item: item), "the JSON view keeps the raw text")
        XCTAssertNil(StudioResultRenderers.rendering(for: .score, document: nil, item: item), "no result yet")
    }

    /// Decode's input is a latents file, so the panel plays the decoded audio itself; Enhance's
    /// input is audio the canvas already plays in the input column, so its panel lists the file.
    func testTheAudioViewPlaysTheOutputOnlyWhenTheInputIsNotAudio() throws {
        let latents = root.appendingPathComponent("hit.npy")
        let decoded = root.appendingPathComponent("hit-decoded.wav")
        try Data([0]).write(to: latents)
        try Data([0]).write(to: decoded)
        let decode = row(templateID: .sfxAEDecode, input: latents, output: decoded)
        XCTAssertEqual(StudioResultRenderers.rendering(for: .audio, document: nil, item: decode), .audioOutput(decoded))
        XCTAssertEqual(StudioResultRenderers.renderedFiles(of: .audioOutput(decoded), item: decode), [decoded])
        XCTAssertNil(StudioResultRenderers.rendering(for: .json, document: nil, item: decode))
        let enhance = row(templateID: .audioEnhance, input: root.appendingPathComponent("memo.wav"), output: decoded)
        XCTAssertNil(StudioResultRenderers.rendering(for: .audio, document: nil, item: enhance))
        let missing = row(templateID: .sfxAEDecode, input: latents, output: root.appendingPathComponent("gone.wav"))
        XCTAssertNil(StudioResultRenderers.rendering(for: .audio, document: nil, item: missing), "nothing on disk to play")
    }

    func testAFoleyCardReviewsTheClipAgainstItsWaveform() throws {
        let clip = root.appendingPathComponent("walk.mp4")
        let foley = root.appendingPathComponent("walk-foley.wav")
        try Data([0]).write(to: clip)
        try Data([0]).write(to: foley)
        let item = row(templateID: .sfxVideo, input: clip, output: foley)

        XCTAssertEqual(StudioResultRenderers.cardRendering(for: item, files: [foley]), .syncReview(video: clip, audio: foley))
        XCTAssertEqual(StudioResultRenderers.renderedFiles(of: .syncReview(video: clip, audio: foley), item: item), [foley])
        XCTAssertNil(StudioResultRenderers.cardRendering(for: item, files: []), "the WAV is not on disk yet")
        let gone = row(templateID: .sfxVideo, input: root.appendingPathComponent("missing.mp4"), output: foley)
        XCTAssertNil(StudioResultRenderers.cardRendering(for: gone, files: [foley]), "no picture to review against")
        let speak = row(templateID: .speechSynthesize, input: clip, output: foley)
        XCTAssertNil(StudioResultRenderers.cardRendering(for: speak, files: [foley]), "only Foley's clip is its picture")
    }

    func testATensorOutputCardShowsItsHeader() throws {
        let safetensors = root.appendingPathComponent("door.safetensors")
        try TensorFixtures.safetensors([("pooled", [1, 1_024])]).write(to: safetensors)
        let item = row(templateID: .sfxConditionText, input: nil, output: safetensors)

        guard case .tensor(.safetensors(let header))? = StudioResultRenderers.cardRendering(for: item, files: [safetensors]) else {
            return XCTFail("the conditioning tensors were not read")
        }
        XCTAssertEqual(header.tensors.map(\.name), ["pooled"])
        XCTAssertEqual(StudioResultRenderers.renderedFiles(of: .tensor(.safetensors(header)), item: item), [safetensors])

        let picture = root.appendingPathComponent("fox.png")
        try Data([0]).write(to: picture)
        XCTAssertNil(StudioResultRenderers.cardRendering(for: row(templateID: .imageGenerate, input: nil, output: picture), files: [picture]),
                     "an image keeps the grid")
    }

    private func row(templateID: CommandTemplateID, input: URL?, output: URL?) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(), mode: templateID.studioTask.mode ?? .sfx, prompt: "a sound", inputURL: input, outputURL: output,
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0, commandPreview: "mere.run",
            outputText: nil, templateID: templateID, commandDraft: nil, artifactURLs: output.map { [$0] } ?? []
        )
    }

}
