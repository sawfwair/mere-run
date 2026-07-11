import AudioCodecs
import Foundation
import MLX
@testable import MereRunCore
import XCTest

final class MuScriptorTests: XCTestCase {
    func testPublishedConfigurations() throws {
        XCTAssertEqual(MuScriptorConfiguration.small.dim, 768)
        XCTAssertEqual(MuScriptorConfiguration.medium.numLayers, 24)
        XCTAssertEqual(MuScriptorConfiguration.large.numHeads, 24)
        try MuScriptorConfiguration.small.validate()
        try MuScriptorConfiguration.medium.validate()
        try MuScriptorConfiguration.large.validate()
    }

    func testModelParameterNamesMatchPublishedCheckpointAndRoundTrip() throws {
        let config = MuScriptorConfiguration(dim: 8, numHeads: 2, numLayers: 1, card: 1_393)
        let model = MuScriptorModel(configuration: config)
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let keys = Set(arrays.keys)
        XCTAssertTrue(keys.contains("condition_provider.conditioners.self_wav.output_proj.weight"))
        XCTAssertTrue(keys.contains("condition_provider.conditioners.instrument_group.embed.weight"))
        XCTAssertTrue(keys.contains("condition_provider.conditioners.dataset_name.embed.weight"))
        XCTAssertTrue(keys.contains("emb.weight"))
        XCTAssertTrue(keys.contains("transformer.layers.0.self_attn.in_proj_weight"))
        XCTAssertTrue(keys.contains("transformer.layers.0.self_attn.out_proj.weight"))
        XCTAssertTrue(keys.contains("transformer.layers.0.linear1.weight"))
        XCTAssertTrue(keys.contains("out_norm.weight"))
        XCTAssertTrue(keys.contains("linear.weight"))

        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try JSONEncoder().encode(config).write(to: root.appendingPathComponent("config.json"))
        try MLX.save(arrays: arrays, url: root.appendingPathComponent("model.safetensors"))

        let loaded = try MuScriptorModel.load(
            resources: MuScriptorResources(rootURL: root),
            variant: .small,
            dtype: .float32
        )
        let prefix = try loaded.conditioningPrefix(
            mel: MLX.zeros([1, 501, 512]),
            instruments: ["acoustic_piano", "drums"]
        )
        let logits = loaded.logits(
            tokenID: loaded.configuration.card,
            prefix: prefix,
            caches: loaded.makeCaches()
        )
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [1_393])
    }

    func testBatchedLogitsMatchIndependentChunkLanes() {
        let config = MuScriptorConfiguration(dim: 8, numHeads: 2, numLayers: 1, card: 1_393)
        let model = MuScriptorModel(configuration: config)
        let firstPrefix = MLXArray((0..<24).map { Float($0) / 24 }, [1, 3, 8])
        let secondPrefix = MLXArray((0..<24).map { Float(24 - $0) / 24 }, [1, 3, 8])
        let batchPrefix = MLX.concatenated([firstPrefix, secondPrefix], axis: 0)
        let batchCaches = model.makeCaches()
        let firstCaches = model.makeCaches()
        let secondCaches = model.makeCaches()

        let batchPrefill = model.batchedLogits(
            tokenIDs: MLXArray([Int32(config.card), Int32(config.card)]).reshaped(2, 1),
            prefix: batchPrefix,
            caches: batchCaches
        )
        let firstPrefill = model.logits(
            tokenID: config.card,
            prefix: firstPrefix,
            caches: firstCaches
        )
        let secondPrefill = model.logits(
            tokenID: config.card,
            prefix: secondPrefix,
            caches: secondCaches
        )
        MLX.eval(batchPrefill, firstPrefill, secondPrefill)

        XCTAssertEqual(batchPrefill.shape, [2, 1_393])
        XCTAssertLessThan(
            MLX.max(MLX.abs(batchPrefill[0] - firstPrefill)).item(Float.self),
            1e-4
        )
        XCTAssertLessThan(
            MLX.max(MLX.abs(batchPrefill[1] - secondPrefill)).item(Float.self),
            1e-4
        )

        let batchDecode = model.batchedLogits(
            tokenIDs: MLXArray([Int32(12), Int32(27)]).reshaped(2, 1),
            prefix: nil,
            caches: batchCaches
        )
        let firstDecode = model.logits(tokenID: 12, prefix: nil, caches: firstCaches)
        let secondDecode = model.logits(tokenID: 27, prefix: nil, caches: secondCaches)
        MLX.eval(batchDecode, firstDecode, secondDecode)

        XCTAssertLessThan(
            MLX.max(MLX.abs(batchDecode[0] - firstDecode)).item(Float.self),
            1e-4
        )
        XCTAssertLessThan(
            MLX.max(MLX.abs(batchDecode[1] - secondDecode)).item(Float.self),
            1e-4
        )
    }

    func testTranscriptionOptionsValidateChunkBatchSize() throws {
        try MuScriptorTranscriptionOptions(chunkBatchSize: 4).validate()
        XCTAssertThrowsError(
            try MuScriptorTranscriptionOptions(chunkBatchSize: 0).validate()
        )
    }

    func testMelFrontendMatchesPublishedShapeAndSilenceFloor() {
        let mel = MuScriptorMelSpectrogram().extract(
            from: [Float](repeating: 0, count: MuScriptorMelSpectrogram.chunkSampleCount)
        )
        XCTAssertEqual(mel.shape, [1, 501, 512])
        let values = mel.asArray(Float.self)
        XCTAssertEqual(values[0], log(1e-6), accuracy: 1e-5)
        XCTAssertEqual(values[values.count - 1], log(1e-6), accuracy: 1e-5)

        var impulse = [Float](repeating: 0, count: MuScriptorMelSpectrogram.chunkSampleCount)
        impulse[40_000] = 1
        let impulseValues = MuScriptorMelSpectrogram().extract(from: impulse).asArray(Float.self)
        XCTAssertEqual(impulseValues[250 * 512 + 100], -0.48411807, accuracy: 2e-4)
        XCTAssertEqual(impulseValues[250 * 512 + 300], 0.65676868, accuracy: 2e-4)
        XCTAssertEqual(impulseValues[250 * 512 + 511], 1.68770206, accuracy: 2e-4)
    }

    func testInstrumentResolutionSupportsUniqueAbbreviations() throws {
        XCTAssertEqual(
            try MuScriptorInstruments.resolve(["timp", "dist", "voice"]),
            ["timpani", "distorted_electric_guitar", "voice"]
        )
        XCTAssertThrowsError(try MuScriptorInstruments.resolve(["piano"]))
    }

    func testTokenDecoderBuildsNoteEvents() throws {
        let transcription = try MuScriptorTokenDecoder().decode(chunks: [[
            1_134, // tie prologue terminator
            13, // shift 10 = 0.10 seconds
            1_135, // program 0
            1_133, // velocity on
            1_064, // pitch 60
            23, // shift 20 = 0.20 seconds
            1_132, // velocity off
            1_064,
        ]])

        XCTAssertEqual(transcription.notes.count, 1)
        XCTAssertEqual(transcription.notes[0].instrument, "acoustic_piano")
        XCTAssertEqual(transcription.notes[0].pitch, 60)
        XCTAssertEqual(transcription.notes[0].onset, 0.1, accuracy: 1e-8)
        XCTAssertEqual(transcription.notes[0].offset, 0.2, accuracy: 1e-8)
        XCTAssertEqual(transcription.events.map(\.type), [.start, .end])
    }

    func testTokenDecoderCarriesTiedNotesAcrossChunks() throws {
        let transcription = try MuScriptorTokenDecoder().decode(chunks: [
            [1_134, 493, 1_135, 1_133, 1_064], // note on at 4.90
            [1_135, 1_064, 1_134, 23, 1_135, 1_132, 1_064], // tied, off at 5.20
        ])

        XCTAssertEqual(transcription.notes.count, 1)
        XCTAssertEqual(transcription.notes[0].onset, 4.9, accuracy: 1e-8)
        XCTAssertEqual(transcription.notes[0].offset, 5.2, accuracy: 1e-8)
    }

    func testMIDIEncoderWritesFormatOneAndInstrumentTracks() throws {
        let data = try MuScriptorMIDI.encode(notes: [
            MuScriptorNote(
                instrument: "acoustic_piano",
                program: 0,
                pitch: 60,
                onset: 0,
                offset: 1,
                isDrum: false
            ),
            MuScriptorNote(
                instrument: "drums",
                program: 128,
                pitch: 36,
                onset: 0,
                offset: 0.01,
                isDrum: true
            ),
        ])
        XCTAssertEqual(String(data: data.prefix(4), encoding: .ascii), "MThd")
        XCTAssertEqual(Array(data[8..<10]), [0, 1])
        XCTAssertEqual(Array(data[10..<12]), [0, 3])
        XCTAssertEqual(data.windows(ofCount: 4).filter { Array($0) == Array("MTrk".utf8) }.count, 3)
    }
}

private extension Data {
    func windows(ofCount count: Int) -> [Data.SubSequence] {
        guard self.count >= count else { return [] }
        return (startIndex...index(endIndex, offsetBy: -count)).map { start in
            self[start..<index(start, offsetBy: count)]
        }
    }
}
