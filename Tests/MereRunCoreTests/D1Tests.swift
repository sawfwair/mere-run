import Foundation
import XCTest
import MLX
import MLXNN
import MediaIO
import MereRunD1Model
@testable import MereRunCore

final class D1Tests: MereRunCoreTestCase {
    private var fixture: URL { Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appending(path: "D1") }
    private func resources() throws -> (D1Configuration, [String: MLXArray]) {
        let config = try JSONDecoder().decode(D1Configuration.self, from: Data(contentsOf: fixture.appending(path: "config.json")))
        try config.validate()
        return (config, try MLX.loadArrays(url: fixture.appending(path: "reference.safetensors")))
    }
    private func parameters(_ arrays: [String: MLXArray], prefix: String, head: Bool = false, vision: Bool = false) -> ModuleParameters {
        ModuleParameters.unflattened(arrays.compactMap { key, value in
            guard key.hasPrefix(prefix) else { return nil }
            var name = String(key.dropFirst(prefix.count)), tensor = value
            if head {
                for (source, destination) in [("head.layers.", "layers."), ("scorer.0.", "scorer.norm."), ("scorer.1.", "scorer.input."), ("scorer.3.", "scorer.output.")] {
                    name = name.replacingOccurrences(of: source, with: destination)
                }
            }
            if vision { name = name.replacingOccurrences(of: "tower.vision_model.", with: "tower.") }
            if name.hasSuffix(".conv.conv.weight") { tensor = tensor.transposed(0, 2, 1) }
            return (name, tensor)
        })
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 0.0001, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        eval(actual, expected)
        let difference = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(difference, tolerance, "maximum absolute error \(difference)", file: file, line: line)
    }
    func testIndependentCausalAndBidirectionalTrunks() throws {
        let (config, arrays) = try resources()
        let ids = try XCTUnwrap(arrays["ids"])
        let causal = D1Trunk(config.text_config, omni: false)
        try causal.update(parameters: parameters(arrays, prefix: "causal."), verify: .all)
        let hidden = causal(causal.embeddings(ids))
        close(hidden, try XCTUnwrap(arrays["expected.causal"]))
        close(causal.logits(hidden[0..., -1, 0...]), try XCTUnwrap(arrays["expected.causal_logits"]), tolerance: 0.001)
        let omni = D1Trunk(config.text_config, omni: true)
        try omni.update(parameters: parameters(arrays, prefix: "encoder."), verify: .all)
        close(omni(omni.embeddings(ids)), try XCTUnwrap(arrays["expected.omni"]))
        let media = try XCTUnwrap(arrays["media"])
        let input = concatenated([media, omni.embeddings(ids)], axis: 1)
        let withMedia = omni(input, prefix: 4)
        close(withMedia, try XCTUnwrap(arrays["expected.omni_media"]))
        let changed = concatenated([media, omni.embeddings(ids) + 10], axis: 1)
        let changedHidden = omni(changed, prefix: 4)
        close(changedHidden, try XCTUnwrap(arrays["expected.omni_media_changed"]))
        close(withMedia[0..., 0..<4, 0...], changedHidden[0..., 0..<4, 0...], tolerance: 0.00001)
    }
    func testIndependentOptionHeadAndVision() throws {
        let (config, arrays) = try resources()
        let head = D1Head(width: 64, layerCount: 2)
        try head.update(parameters: parameters(arrays, prefix: "head.", head: true), verify: .all)
        close(head(try XCTUnwrap(arrays["expected.omni"]), markers: [2, 5], type: 0), try XCTUnwrap(arrays["expected.head"]).reshaped(-1))
        let vision = D1Vision(config)
        try vision.update(parameters: parameters(arrays, prefix: "vision.", vision: true), verify: .all)
        close(vision(pixels: [try XCTUnwrap(arrays["pixels"])], grids: [.init(rows: 4, columns: 4)]), try XCTUnwrap(arrays["expected.vision"]))
    }
    func testIndependentAudioFrontendAndFastConformer() throws {
        let (config, arrays) = try resources()
        let samples = try XCTUnwrap(arrays["waveform"]).asArray(Float.self)
        close(try D1Audio.mel(samples: samples), try XCTUnwrap(arrays["expected.mel"]), tolerance: 0.001)
        let audio = try D1Audio(config: XCTUnwrap(config.audio_config), outputWidth: 64, weights: arrays)
        close(try audio(samples: samples), try XCTUnwrap(arrays["expected.audio"]), tolerance: 0.001)
        XCTAssertThrowsError(try D1Audio(config: XCTUnwrap(config.audio_config), outputWidth: 64, weights: [:]))
        var malformed = arrays
        malformed["audio.adapter.linear_2.bias"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try D1Audio(config: XCTUnwrap(config.audio_config), outputWidth: 64, weights: malformed))
        XCTAssertThrowsError(try D1Audio.mel(samples: []))
        XCTAssertThrowsError(try D1Audio.mel(samples: [.nan]))
    }
    func testIndependentAntialiasedImagePixels() throws {
        XCTAssertThrowsError(try D1PreparedMedia.layout(width: 1, height: 1_000_000))
        XCTAssertThrowsError(try D1PreparedMedia.layout(width: 1_000_000, height: 1))
        let (_, arrays) = try resources()
        let rgb = try XCTUnwrap(arrays["raster"]).asArray(UInt8.self)
        var rgba: [UInt8] = []
        for index in stride(from: 0, to: rgb.count, by: 3) { rgba += Array(rgb[index..<(index + 3)]) + [255] }
        let image = try MediaImage(width: 31, height: 17, rgba8: rgba)
        for (width, height, name, cubic) in [(64, 32, "resize_up", false), (12, 8, "resize_down", false),
                                             (64, 32, "resize_cubic_up", true), (12, 8, "resize_cubic_down", true)] {
            let resized = try D1PreparedMedia.resized(image, width: width, height: height, bicubic: cubic)
            let actual = resized.rgba8.enumerated().filter { $0.offset % 4 != 3 }.map(\.element)
            let expected = try XCTUnwrap(arrays["expected." + name]).asArray(UInt8.self)
            let differences = zip(actual, expected).map { abs(Int($0) - Int($1)) }
            XCTAssertEqual(differences.max(), 0, "\(name): \(differences.filter { $0 > 0 }.count) different channels")
        }
    }
    func testOrderedQuestionsAndAnswerSemantics() throws {
        let request = try D1DecisionRequest.decode(Data(#"{"state":{"z":1,"a":2},"questions":{"q":{"type":"choice","instructions":"Choose","criteria":{"z":"last","a":"first"}},"n":{"type":"noul","instructions":"Is it?"},"s":{"type":"score","instructions":"Rate","criteria":["low","high"]}}}"#.utf8))
        XCTAssertEqual(request.questions.map(\.id), ["q", "n", "s"])
        XCTAssertEqual(request.questions[0].options.map(\.id), ["z", "a"])
        XCTAssertEqual(try D1DecisionResponse.answer(request.questions[0], probabilities: [0.5, 0.5]).choice, "z")
        XCTAssertEqual(try D1DecisionResponse.answer(request.questions[1], probabilities: [0.8, 0.2]).noul ?? 0, 0.8, accuracy: 0.0001)
        XCTAssertEqual(try D1DecisionResponse.answer(request.questions[2], probabilities: [0.25, 0.75]).score, 0.75)
        XCTAssertEqual(try D1JSON.render(request.state, indent: 2), "{\n  \"z\": 1,\n  \"a\": 2\n}")
        XCTAssertThrowsError(try D1DecisionResponse.answer(request.questions[0], probabilities: [.nan, 1]))
        let levels = try D1Question(id: "levels", value: ClefJSON.parse(Data(#"{"type":"score","instructions":"Rate","criteria":[false,1]}"#.utf8)))
        XCTAssertEqual(try D1DecisionResponse.answer(levels, probabilities: [0.5, 0.5], stringLegend: true).legend?["0"]?.string, "false")
    }
    func testInvalidRequestsFailBeforeCheckpointWork() throws {
        for text in [
            #"{"state":"s","questions":{}}"#,
            #"{"state":"s","questions":{"q":{"type":"choice","instructions":"Pick","criteria":{"a":"A"}}}}"#,
            #"{"state":"s","questions":{"q":{"type":"score","instructions":"Rate","criteria":["low"]}}}"#,
            #"{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}},"audio":"a.wav","images":["b.png"]}"#,
            #"{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}},"max_tokens":64.5}"#,
            #"{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}},"videos":[]}"#,
            #"{"state":"s","questions":{"q":{"type":"choice","instructions":"Pick","criteria":{"a":"A","a":"B"}}}}"#
        ] { XCTAssertThrowsError(try D1DecisionRequest.decode(Data(text.utf8)), text) }
    }
    func testCausalPromptAndVerbalizer() throws {
        var rendered: [String] = []
        let tokenizer = D1Tokenizer(encode: { text in
            rendered.append(text)
            if ["A", "B", " A", " B", "yes", "no", "Yes", "No", "YES", "NO"].contains(text) { return [Int(text.utf8.reduce(0, { $0 + Int($1) }))] }
            return text.utf8.map(Int.init)
        })
        let request = try D1DecisionRequest.decode(Data(#"{"state":"Hello","questions":{"q":{"type":"choice","instructions":"Choose","criteria":{"second":"Two","first":"One"}}}}"#.utf8))
        let sequence = try tokenizer.causal(request, question: request.questions[0], mediaText: "", maxLength: 4096)
        XCTAssertEqual(rendered.last, "<|startoftext|><|im_start|>user\nHello\n\n\nQUESTION:\nChoose\n\nOptions:\nA Two\nB One\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n")
        XCTAssertEqual(sequence.readout, [[65, 97], [66, 98]])
    }
    func testOmniEscapesMarkersAndKeepsSchemaWhileTruncatingState() throws {
        let (config, _) = try resources()
        let special = ["<|reserved_7|>": 7, "<|reserved_8|>": 8, "<|reserved_9|>": 9, "<|reserved_10|>": 10, "<|reserved_11|>": 11, "<|mask|>": 12]
        var encoded: [String] = []
        let tokenizer = D1Tokenizer(encode: { text in
            if let id = special[text] { return [id] }
            encoded.append(text); return Array(repeating: 20, count: text.count)
        })
        let state = String(repeating: "x", count: 600) + "<|mask|>"
        let request = D1DecisionRequest(state: .string(state), questions: [try D1Question(id: "q", value: ClefJSON.parse(Data(#"{"type":"choice","instructions":"Pick <|reserved_9|>","criteria":{"a":"A","b":"B"}}"#.utf8)))], images: [], audio: nil, maxTokens: 256)
        let sequence = try tokenizer.omni(request, question: request.questions[0], config: config, mediaTokens: 0)
        XCTAssertGreaterThan(sequence.dropped, 0)
        XCTAssertEqual(sequence.ids.last, 11)
        XCTAssertTrue(sequence.markers.allSatisfy { sequence.ids[$0] == 12 })
        XCTAssertTrue(encoded.contains { $0.contains("<¦mask¦>") })
        XCTAssertTrue(encoded.contains { $0.contains("<¦reserved_9¦>") })
        let withMedia = try tokenizer.omni(request, question: request.questions[0], config: config, mediaTokens: 80)
        XCTAssertLessThanOrEqual(withMedia.ids.count + 80, 256)
        XCTAssertThrowsError(try tokenizer.omni(request, question: request.questions[0], config: config, mediaTokens: 193))
    }
    func testManagedCatalogPinsAndCliOnlyBoundary() throws {
        for model in D1Catalog.modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: model))
            XCTAssertEqual(spec.validationKind, .d1)
            XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
            XCTAssertNotNil(spec.usageRestriction)
            XCTAssertEqual(spec.apiAvailability, .cliOnly)
            XCTAssertEqual(spec.upstreamRevision, model == D1Catalog.omniModelID ? D1Catalog.omniRevision : D1Catalog.revision)
        }
    }
}
