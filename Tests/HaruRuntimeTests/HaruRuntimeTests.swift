import Foundation
import MereRunMLXTestSupport
import MLX
import XCTest
@testable import AudioHaruModel

final class HaruRuntimeTests: MLXTestCase {
    func testCheckpointBackboneAndMeanFlowMatchReference() throws {
        let (model, reference) = try loadFixture()
        let cache = model.makeCache()
        let tokens = try XCTUnwrap(reference["text"])
        let speaker = try XCTUnwrap(reference["speaker"])
        let user = try XCTUnwrap(reference["user"])

        let spans = [0..<4, 4..<5, 5..<6]
        let hidden = concatenated(spans.map { span in
            let ids = tokens[0..., span]
            let embeddings = model.embeddings(ids, model: speaker[0..., span], user: user[0..., span])
            let states = model.lm.decisionHiddenStates(ids, cache: cache, inputEmbeddings: embeddings)
            eval(states)
            return states
        }, axis: 1)

        try assertClose(hidden, to: reference["hidden"], tolerance: 2e-4)
        try assertClose(model.lm.logits(from: hidden[0..., -1]), to: reference["logits"], tolerance: 2e-4)
        let noise = try XCTUnwrap(reference["noise"])
        let conditions = model.head.conditioning(dtype: model.dtype)
        let latent = model.head.sample(hidden: hidden[0..., -1], noise: noise, conditioning: conditions)
        try assertClose(latent, to: reference["head"], tolerance: 2e-4)
    }

    func testCodecMatchesReferenceAcrossChunkBoundaryAndPartialHop() throws {
        let (model, reference) = try loadFixture()
        let clip = try XCTUnwrap(reference["clip"])
        let latents = try XCTUnwrap(reference["latents"])[0]
        let boundary = HaruConfiguration.frameSamples

        var encoder = model.codec.encoderState
        let means = concatenated([
            model.codec.encode(clip[..<boundary], state: &encoder),
            model.codec.encode(clip[boundary...], state: &encoder),
        ])
        try assertClose(means, to: reference["mean"]?[0], tolerance: 2e-4)

        var decoder = model.codec.decoderState
        let audio = concatenated([
            model.codec.decode(latents[..<1], state: &decoder),
            model.codec.decode(latents[1...], state: &decoder),
        ])
        XCTAssertEqual(audio.shape, [2 * boundary])
        try assertClose(audio, to: reference["audio"]?[0], tolerance: 2e-4)
    }

    func testSessionsKeepIndependentStateAndAnInvalidFrameDoesNotAdvance() throws {
        let (model, reference) = try loadFixture()
        let session = HaruSession(model: model, seed: 42, temperature: 0)
        let control = HaruSession(model: model, seed: 42, temperature: 0)
        let blank = Array(repeating: Float(0), count: HaruConfiguration.frameSamples)
        XCTAssertThrowsError(try session.step(blank, decode: describe))
        let audio = try XCTUnwrap(reference["clip"]).asArray(Float.self)
        let a = try session.start(card: [15, 4, 91], reference: audio, speakFirst: true, decode: describe)
        XCTAssertThrowsError(try session.start(card: [], reference: [], speakFirst: false, decode: describe))
        let b = try control.start(card: [15, 4, 91], reference: audio, speakFirst: true, decode: describe)
        XCTAssertEqual(a.tokenID, b.tokenID)
        XCTAssertEqual(a.audio, b.audio)
        XCTAssertFalse([125, 126, 127].contains(a.tokenID))
        XCTAssertThrowsError(try session.step([0], decode: describe))
        let next = try session.step(Array(audio.prefix(blank.count)), decode: describe)
        let expected = try control.step(Array(audio.prefix(blank.count)), decode: describe)
        XCTAssertEqual(next.tokenID, expected.tokenID)
        XCTAssertEqual(next.audio, expected.audio)
        XCTAssertEqual(next.audio.count, blank.count)
        XCTAssertTrue(next.audio.allSatisfy { $0.isFinite && abs($0) <= 1 })
        let silent = HaruSession(model: model, seed: 42, temperature: 0)
        _ = try silent.start(card: [15, 4, 91], reference: audio, speakFirst: true, decode: describe)
        let silence = try silent.step(blank, decode: describe)
        XCTAssertNotEqual(next.audio, silence.audio)
    }

    private func describe(_ tokens: [Int]) -> String {
        tokens.map(String.init).joined(separator: " ")
    }

    private func loadFixture() throws -> (HaruModel, [String: MLXArray]) {
        let root = try XCTUnwrap(Bundle.module.resourceURL).appending(path: "Fixtures")
        let model = try HaruModel.load(from: root)
        return (model, try MLX.loadArrays(url: root.appending(path: "reference.safetensors")))
    }

    private func assertClose(
        _ value: MLXArray, to expected: MLXArray?, tolerance: Float, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let expected = try XCTUnwrap(expected, file: file, line: line)
        XCTAssertEqual(value.shape, expected.shape, file: file, line: line)
        guard value.shape == expected.shape else { return }
        let maximumError = abs(value - expected).max().item(Float.self)
        XCTAssertLessThan(maximumError, tolerance, file: file, line: line)
    }
}
