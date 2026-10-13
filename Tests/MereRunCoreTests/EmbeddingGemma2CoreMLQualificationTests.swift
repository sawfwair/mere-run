#if canImport(CoreML) && os(macOS) && arch(arm64)
import Foundation
import MLX
import MereRunCoreMLModel
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

/// Opt-in released-checkpoint qualification; the normal gate needs no model assets.
final class EmbeddingGemma2CoreMLQualificationTests: MLXTestCase {
    private struct Observation: Encodable {
        let name: String
        let tokens: Int
        let cosine: Double
        let maxAbsoluteError: Float
        let coreML: [Float]
        let mlxFloat32: [Float]
    }

    func testExpandedReleasedCheckpointParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundle = env["MERERUN_EMBEDDINGGEMMA2_COREML_EXPANDED"],
              let source = env["MERERUN_EMBEDDINGGEMMA2_SOURCE"],
              let output = env["MERERUN_EMBEDDINGGEMMA2_COREML_RECEIPT"] else {
            throw XCTSkip("Set expanded bundle, source checkpoint, and receipt paths to qualify Core ML against MLX.")
        }
        try await Device.withDefaultDevice(Device(.gpu)) {
            let coreML = try EmbeddingGemma2(bundle: URL(fileURLWithPath: bundle))
            XCTAssertEqual(coreML.config.maximumTokens, 8192)
            let mlx = try await EmbeddingGemma2Model(resources: EmbeddingGemma2Resources(rootURL: URL(fileURLWithPath: source)), dtype: .float32)
            var observations: [Observation] = []
            func observe(_ name: String, sequence: MultimodalSequence, reference: (embeddings: [[Float]], tokenCounts: [Int])) async throws {
                let actual = try await coreML.embed(sequence: sequence)
                let expected = try XCTUnwrap(reference.embeddings.first)
                XCTAssertEqual(sequence.ids.count, reference.tokenCounts.first)
                XCTAssertTrue(actual.allSatisfy(\.isFinite))
                let dot = zip(actual, expected).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
                let normA = sqrt(actual.reduce(0.0) { $0 + Double($1) * Double($1) })
                let normB = sqrt(expected.reduce(0.0) { $0 + Double($1) * Double($1) })
                let cosine = dot / (normA * normB)
                let error = zip(actual, expected).map { abs($0 - $1) }.max()!
                observations.append(Observation(name: name, tokens: sequence.ids.count, cosine: cosine,
                                                maxAbsoluteError: error, coreML: actual, mlxFloat32: expected))
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(observations).write(to: URL(fileURLWithPath: output), options: .atomic)
                FileHandle.standardError.write(Data("Core ML parity \(name): \(sequence.ids.count) tokens, cosine \(cosine)\n".utf8))
                XCTAssertGreaterThan(cosine, 0.999, name)
                // The retained short graph normalizes in FP16. Core's public
                // wrapper re-normalizes its output; longer graphs return FP32.
                let tolerance = sequence.ids.count <= 512 ? 0.002 : 0.00001
                XCTAssertEqual(normA, 1, accuracy: tolerance, name)
                MLX.Memory.clearCache()
            }
            let paragraphs = [
                "Auroras form when charged solar particles excite gases in Earth's atmosphere. ",
                "A Swift function sorts transaction records by date and preserves their identifiers. ",
                "海洋の温度と塩分を測定して、生態系の変化を調べます。 ",
                "Sourdough fermentation depends on time, temperature, flour, and a healthy starter. "
            ]
            let text = (0..<1000).map { paragraphs[$0 % paragraphs.count] }.joined()
            let body = coreML.tokenizer.encode(text)
            XCTAssertGreaterThan(body.count, 8192)
            for length in [513, 1024, 1025, 2048, 2049, 4096, 4097, 8192] {
                let sequence = MultimodalSequence(ids: [2] + body.prefix(length - 2) + [1])
                let reference = try mlx.embed(texts: [text], maxTokens: length)
                try await observe("text-\(length)", sequence: sequence, reference: reference)
            }
            // Exact 30-second input exercises all 2999 valid frames / 750 soft tokens.
            let root = URL(fileURLWithPath: output).deletingLastPathComponent()
            for seconds in [11, 20, 30] {
                let url = root.appending(path: "tone-\(seconds)s.wav")
                try writeTone(seconds: seconds, to: url)
                let sequence = try await coreML.audioSequence(url: url)
                let reference = try mlx.embed(inputs: [EmbeddingGemma2Input(content: [.audio(url)])])
                try await observe("audio-\(seconds)s", sequence: sequence, reference: reference)
            }
            let audioURL = root.appending(path: "tone-30s.wav")
            var mixed = MultimodalSequence(ids: [2] + coreML.tokenizer.encode("Describe this recording. "))
            mixed.appendMedia(try await coreML.audioSequence(url: audioURL))
            mixed.ids += coreML.tokenizer.encode(" It contains a synthetic tone.")
            mixed.ids.append(1)
            let reference = try mlx.embed(inputs: [EmbeddingGemma2Input(content: [
                .text("Describe this recording. "), .audio(audioURL), .text(" It contains a synthetic tone.")
            ])])
            try await observe("mixed-audio-30s", sequence: mixed, reference: reference)
            var pair = MultimodalSequence(ids: [2])
            pair.appendMedia(try await coreML.audioSequence(url: audioURL))
            pair.ids += coreML.tokenizer.encode(" followed by ")
            pair.appendMedia(try await coreML.audioSequence(url: audioURL))
            pair.ids.append(1)
            let pairReference = try mlx.embed(inputs: [EmbeddingGemma2Input(content: [
                .audio(audioURL), .text(" followed by "), .audio(audioURL)
            ])])
            try await observe("two-ordered-30s-segments", sequence: pair, reference: pairReference)
            if let path = env["MERERUN_EMBEDDINGGEMMA2_AUDIO_FIXTURE"] {
                let url = URL(fileURLWithPath: path)
                let sequence = try await coreML.audioSequence(url: url)
                let reference = try mlx.embed(inputs: [EmbeddingGemma2Input(content: [.audio(url)])])
                try await observe("speech-30s", sequence: sequence, reference: reference)
            }
            do {
                _ = try await coreML.audioSequence(samples: [Float](repeating: 0, count: 480001))
                XCTFail("Audio longer than 30 seconds must be rejected.")
            } catch let error as EmbedderError {
                XCTAssertTrue(error.description.contains("at most 30 seconds"))
            }
        }
    }

    func testPackedReleasedCheckpointMatchesIndependentPredictionsAndSessionReuse() async throws {
        guard let bundle = ProcessInfo.processInfo.environment["MERERUN_EMBEDDINGGEMMA2_COREML_EXPANDED"] else {
            throw XCTSkip("Set the expanded Core ML bundle to qualify packed predictions.")
        }
        let root = URL(fileURLWithPath: bundle)
        let model = try EmbeddingGemma2(bundle: root)
        let texts = ["Auroras in the northern sky", "A stable Swift sorting function", "海洋の温度と塩分", "", "Café déjà vu"]
        var sequences = (0..<73).map { index in
            MultimodalSequence(ids: [2] + model.tokenizer.encode(texts[index % texts.count]) + [1])
        }
        // Both sides of the pack boundary, ordinary 512-token graphs, and the long
        // GPU path must preserve output order when interspersed with packed texts.
        for (index, length) in [(3, 256), (11, 257), (20, 512), (31, 513)] {
            sequences[index] = MultimodalSequence(ids: [2] + Array(repeating: 100, count: length - 2) + [1])
        }
        let actual = try await model.embed(sequences: sequences)
        XCTAssertEqual(actual.count, sequences.count)
        for (sequence, vector) in zip(sequences, actual) {
            let expected = try await model.embed(sequence: sequence)
            let dot = zip(vector, expected).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
            let norm = sqrt(vector.reduce(0.0) { $0 + Double($1) * Double($1) }
                            * expected.reduce(0.0) { $0 + Double($1) * Double($1) })
            XCTAssertGreaterThan(dot / norm, 0.9999)
        }
        let session = try EmbeddingGemma2CoreMLSession(root: root)
        let inputs = texts.map { EmbeddingGemma2Input(content: [.text($0)]) }
        for dimension in [128, 256, 512, 768] {
            let batch = try await session.embed(inputs: inputs, task: .document, title: "Auroras", dimensions: dimension)
            for (index, input) in inputs.enumerated() {
                let single = try await session.embed(inputs: [input], task: .document, title: "Auroras", dimensions: dimension)
                XCTAssertEqual(batch.tokenCounts[index], single.tokenCounts[0])
                let dot = zip(batch.embeddings[index], single.embeddings[0]).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
                XCTAssertGreaterThan(dot, 0.9999)
            }
        }
        async let first = session.embed(inputs: inputs, task: .query)
        async let second = session.embed(inputs: inputs, task: .similarity)
        let concurrent = try await [first, second]
        XCTAssertEqual(concurrent[0].embeddings.count, texts.count)
        XCTAssertEqual(concurrent[1].embeddings.count, texts.count)
    }

    private func writeTone(seconds: Int, to url: URL) throws {
        let samples = (0..<(seconds * 16000)).map { index -> Int16 in
            let t = Double(index) / 16000
            let envelope = 0.5 + 0.5 * sin(2 * Double.pi * 0.7 * t)
            return Int16(10000 * envelope * (sin(2 * Double.pi * 440 * t) + 0.3 * sin(2 * Double.pi * 677 * t)))
        }
        var data = Data()
        func ascii(_ string: String) { data.append(contentsOf: string.utf8) }
        func u32(_ value: UInt32) { var word = value.littleEndian; withUnsafeBytes(of: &word) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { var word = value.littleEndian; withUnsafeBytes(of: &word) { data.append(contentsOf: $0) } }
        ascii("RIFF"); u32(UInt32(36 + samples.count * 2)); ascii("WAVEfmt "); u32(16)
        u16(1); u16(1); u32(16000); u32(32000); u16(2); u16(16); ascii("data"); u32(UInt32(samples.count * 2))
        for sample in samples { u16(UInt16(bitPattern: sample)) }
        try data.write(to: url)
    }
}
#endif
