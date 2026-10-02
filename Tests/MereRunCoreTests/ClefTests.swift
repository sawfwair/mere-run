import Foundation
import XCTest
import MLX
import MLXNN
@testable import MereRunCore
@testable import MereRunQwenModel

final class ClefTests: MereRunCoreTestCase {
    private var fixture: URL { Bundle.module.resourceURL!.appending(path: "Fixtures/Clef") }
    private var byteTokenizer: ClefTokenizer { ClefTokenizer(encode: { $0.utf8.map(Int.init) }) }

    private func request() throws -> ClefDecisionRequest {
        try ClefDecisionRequest.decode(Data(contentsOf: fixture.appending(path: "request.json")))
    }

    func testSchemaTokensAndSpansMatchPinnedReference() throws {
        struct Reference: Decodable {
            let ids: [Int]
            let fields: [Field]
            struct Field: Decodable {
                let id: String
                let type: Int
                let question_span: [Int]
                let option_spans: [[Int]]
                let option_ids: [String]
            }
        }
        let expected = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixture.appending(path: "sequence.json")))
        let sequence = try byteTokenizer.sequence(request(), modelID: ClefCatalog.modelID)
        XCTAssertEqual(sequence.ids, expected.ids)
        XCTAssertEqual(sequence.plan.questions.map(\.id), ["z_route", "urgency", "a_outage"])
        for (index, field) in sequence.plan.questions.enumerated() {
            XCTAssertEqual(field.id, expected.fields[index].id)
            XCTAssertEqual(field.questionSpan, expected.fields[index].question_span)
            XCTAssertEqual(field.optionSpans, expected.fields[index].option_spans)
            XCTAssertEqual(field.optionIDs, expected.fields[index].option_ids)
            XCTAssertEqual(sequence.fields[index].type, expected.fields[index].type)
        }
    }

    func testRealQwenTokenizerMatchesPinnedReferenceWhenProvided() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_CLEF_TOKENIZER"] else {
            throw XCTSkip("Set MERERUN_CLEF_TOKENIZER to the pinned checkpoint's tokenizer directory.")
        }
        struct Reference: Decodable { let ids: [Int] }
        let expected = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixture.appending(path: "sequence_qwen.json")))
        let tokenizer = try ClefTokenizer.load(root: URL(fileURLWithPath: root))
        let sequence = try tokenizer.sequence(request(), modelID: ClefCatalog.modelID)
        XCTAssertEqual(sequence.ids, expected.ids)
    }

    func testEntireJointHeadMatchesIndependentUpstreamFixture() throws {
        let config = try JSONDecoder().decode(ClefHeadConfiguration.self, from: Data(contentsOf: fixture.appending(path: "joint_head_config.json")))
        let head = try ClefJointHead(configuration: config)
        try head.load(MLX.loadArrays(url: fixture.appending(path: "head.safetensors")), dtype: .float32)
        let reference = try MLX.loadArrays(url: fixture.appending(path: "reference.safetensors"))
        let hidden = try XCTUnwrap(reference["hidden"])
        let lexical = try XCTUnwrap(reference["lexical"])
        let sequence = try byteTokenizer.sequence(request(), modelID: ClefCatalog.modelID)
        let ids = MLXArray(sequence.ids.map(Int32.init))
        let embeddings = sequence.fields.map { field in field.optionSpans.map { lexical[ids[$0]] } }
        let logits = try head(hidden: hidden, fields: sequence.fields, lexical: embeddings)
        for (index, output) in logits.enumerated() {
            let expected = try XCTUnwrap(reference["logits_\(index)"])
            XCTAssertEqual(output.shape, expected.shape)
            XCTAssertLessThan(abs(output - expected).max().item(Float.self), 2e-5)
        }
        // Joint fields must depend on evidence outside their own spans.
        let changed = try head(hidden: hidden + MLXArray(Float(0.5)) * lexical[0], fields: sequence.fields, lexical: embeddings)
        XCTAssertGreaterThan(abs(changed[0] - logits[0]).max().item(Float.self), 1e-5)
    }

    func testHeadRejectsIncompleteExtraAndWrongShapeWeights() throws {
        let config = try JSONDecoder().decode(ClefHeadConfiguration.self, from: Data(contentsOf: fixture.appending(path: "joint_head_config.json")))
        let original = try MLX.loadArrays(url: fixture.appending(path: "head.safetensors"))
        var missing = original
        missing.removeValue(forKey: "residual_gate")
        var extra = original
        extra["unknown.weight"] = MLX.zeros([1])
        var wrong = original
        wrong["memory_projection.weight"] = MLX.zeros([2, 2])
        for weights in [missing, extra, wrong] {
            let head = try ClefJointHead(configuration: config)
            XCTAssertThrowsError(try head.load(weights, dtype: .float32))
        }
        XCTAssertThrowsError(try config.validate(backboneHiddenSize: config.hiddenSize + 1))
    }

    func testBackboneRetainsAllHiddenStatesAndGathersQuantizedOutputRows() throws {
        let json = #"{"model_type":"qwen3_5","tie_word_embeddings":false,"text_config":{"model_type":"qwen3_5_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,"layer_types":["full_attention"],"linear_num_value_heads":1,"linear_num_key_heads":1,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,"max_position_embeddings":2048,"rms_norm_eps":0.000001,"attention_bias":false,"attention_dropout":0,"vocab_size":128,"rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}"#
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(json.utf8))
        let model = Q35Model(config: config)
        let input = MLXArray([Int32(1), 3, 2], [1, 3])
        let hidden = model.decisionHiddenStates(input, cache: nil)
        XCTAssertEqual(hidden.shape, [1, 3, 64])
        XCTAssertLessThan(abs(hidden - model.forward(input, cache: nil).hidden).max().item(Float.self), 1e-6)
        let weight = MLXArray(Array(0..<8192).map { sin(Float($0)) }, [128, 64])
        let quantized = QuantizedLinear(Linear(weight: weight), groupSize: 64, bits: 4)
        model.update(modules: ModuleChildren.unflattened([("lm_head", quantized)]))
        let ids = MLXArray([Int32(7), 2, 7])
        let rows = model.decisionLexicalEmbeddings(ids)
        let dense = dequantized(quantized.weight, scales: quantized.scales, biases: quantized.biases, groupSize: 64, bits: 4)
        XCTAssertEqual(rows.shape, [3, 64])
        XCTAssertLessThan(abs(rows - dense[ids]).max().item(Float.self), 1e-6)
        XCTAssertGreaterThan(abs(rows - model.embeddings(for: ids)).max().item(Float.self), 0.1)
    }

    func testStateTruncationRetainsCompleteSchemaAndSuffix() throws {
        let json = try ClefJSON.parse(Data(contentsOf: fixture.appending(path: "request.json")))
        let original = try byteTokenizer.sequence(request(), modelID: ClefCatalog.modelID)
        let fixed = original.ids.count - original.plan.stateTokens
        let fields = try XCTUnwrap(json.fields).filter { $0.key != "max_tokens" }
        let short = try ClefDecisionRequest(json: .object(fields + [.init(key: "max_tokens", value: .number(String(fixed + 3)))]))
        let truncated = try byteTokenizer.sequence(short, modelID: ClefCatalog.modelID)
        XCTAssertEqual(truncated.ids.count, fixed + 3)
        XCTAssertEqual(truncated.plan.stateTokens, 3)
        XCTAssertEqual(truncated.plan.stateTokensDropped, original.plan.stateTokens - 3)
        XCTAssertEqual(Array(truncated.ids.suffix(100)), Array(original.ids.suffix(100)))
        let impossible = try ClefDecisionRequest(json: .object(fields + [.init(key: "max_tokens", value: .number(String(fixed - 1)))]))
        XCTAssertThrowsError(try byteTokenizer.sequence(impossible, modelID: ClefCatalog.modelID))
    }

    func testMalformedAndAmbiguousRequestsAreRejected() throws {
        let payloads = [
            #"{"state":"x","questions":{}}"#,
            #"{"state":"x","questions":[]}"#,
            #"{"state":"x","questions":{"a":{"type":"chat"}}}"#,
            #"{"state":"x","questions":{"a":{"type":"choice","criteria":{}}}}"#,
            #"{"state":"x","questions":{"a":{"type":"score","criteria":{}}}}"#,
            #"{"state":"x","questions":{"a":{"type":"noul"},"a":{"type":"noul"}}}"#,
            #"{"state":"x","state":"y","questions":{"a":{"type":"noul"}}}"#,
            #"{"state":01,"questions":{"a":{"type":"noul"}}}"#,
            #"{"state":"x","questions":{"a":{"type":"noul"}},"max_tokens":0}"#,
            #"{"state":"x","questions":{"a":{"type":"noul"}},"images":["a"],"videos":[["b"]]}"#,
            #"{"state":"x","questions":{"a":{"type":"noul"}},"media_kwargs":{"fps":4}}"#,
            #"{"state":"x","questions":{"a":{"type":"noul"}}} trailing"#,
        ]
        for payload in payloads { XCTAssertThrowsError(try ClefDecisionRequest.decode(Data(payload.utf8)), payload) }
    }

    func testTypedAnswersAndChoiceTieMatchSystemOne() throws {
        let request = try request()
        let choice = try ClefDecisionResponse.answer(question: request.questions[0], probabilities: [0.1, 0.2, 0.7])
        XCTAssertEqual(choice.choice, "technical")
        XCTAssertEqual(choice.confidence, 0.7)
        let tied = try ClefDecisionResponse.answer(question: request.questions[0], probabilities: [0.5, 0, 0.5])
        XCTAssertEqual(tied.choice, "technical") // Original criterion order, rather than sorted encoding order.
        let score = try ClefDecisionResponse.answer(question: request.questions[1], probabilities: [0.1, 0.2, 0.3, 0.4])
        XCTAssertEqual(score.score, 2)
        XCTAssertEqual(score.legend?["3"]?.string, "Critical")
        let noul = try ClefDecisionResponse.answer(question: request.questions[2], probabilities: [0.75, 0.25])
        XCTAssertEqual(noul.noul, 0.75)
        XCTAssertNil(noul.probabilities)
        XCTAssertThrowsError(try ClefDecisionResponse.answer(question: request.questions[2], probabilities: [.nan, 0]))
    }

    func testVideoSamplingAndGeometryMatchReference() throws {
        XCTAssertEqual(ClefPreparedMedia.sampleFrameIndices(count: 1), [0])
        XCTAssertEqual(ClefPreparedMedia.sampleFrameIndices(count: 5), [0, 1, 3, 4])
        XCTAssertEqual(ClefPreparedMedia.sampleFrameIndices(count: 48), [0, 16, 31, 47])
        let size = try ClefPreparedMedia.videoSize(width: 1920, height: 1080, frames: 4, minPixels: 4096, maxPixels: 25_165_824)
        XCTAssertEqual(size.width, 1920)
        XCTAssertEqual(size.height, 1088)
    }

    func testPinnedCatalogManifestAndDiscovery() throws {
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ClefCatalog.repository))
        XCTAssertEqual(spec.id, ClefCatalog.modelID)
        XCTAssertEqual(spec.validationKind, .clef)
        XCTAssertEqual(spec.upstreamRevision, ClefCatalog.revision)
        XCTAssertEqual(spec.defaultCLICommands, ["text decide"])
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        XCTAssertEqual(spec.apiAvailability, .cliOnly)
        XCTAssertNil(spec.apiProfile)
        XCTAssertFalse(try XCTUnwrap(spec.hubFallback).patterns.contains("clef_mlx.py"))
        let manifest = MereRunModelManifest.template(for: .clef4Bit)
        XCTAssertEqual(manifest.engine, .clef)
        XCTAssertEqual(manifest.supports, [.textDecision])
    }
}
