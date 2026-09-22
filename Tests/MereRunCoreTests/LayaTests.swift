import Foundation
import XCTest
import MLX
import MereRunLayaModel
@testable import MereRunCore

final class LayaTests: MereRunCoreTestCase {
    private var fixture: URL {
        Bundle.module.resourceURL!.appending(path: "Fixtures/Laya")
    }

    func testNativeFullGraphMatchesPinnedPyTorchReference() throws {
        let configuration = try LayaResources(root: fixture).configuration()
        let model = try LayaNetwork(configuration: configuration.encoder, agent: configuration.agent,
                                    arrays: MLX.loadArrays(url: fixture.appending(path: "model.safetensors")))
        let reference = try MLX.loadArrays(url: fixture.appending(path: "reference.safetensors"))
        func array(_ name: String) throws -> MLXArray { try XCTUnwrap(reference[name]) }
        let result = try model(inputIDs: array("input_ids"), attentionMask: array("attention_mask").asType(.bool),
                                markerPositions: array("marker_positions"), markerMask: array("marker_mask"),
                                questionTypes: array("question_types"))
        try assertClose(result.logits, array("logits"))
        try assertClose(result.actionLogits, array("action_logits"))
        let single = try model(inputIDs: array("input_ids")[2..<3], attentionMask: array("attention_mask")[2..<3].asType(.bool),
                                markerPositions: array("marker_positions")[2..<3, 0..<1], markerMask: array("marker_mask")[2..<3, 0..<1],
                                questionTypes: array("question_types")[2..<3])
        try assertClose(single.logits, array("single_logits"))
        try assertClose(single.actionLogits, array("single_action_logits"))
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        let error = abs(actual - expected).max().item(Float.self)
        XCTAssertLessThan(error, 2e-5, file: file, line: line)
    }

    func testWeightLoaderRejectsMissingExtraAndWrongShape() throws {
        let configuration = try LayaResources(root: fixture).configuration()
        let original = try MLX.loadArrays(url: fixture.appending(path: "model.safetensors"))
        var missing = original
        missing.removeValue(forKey: "scorer.1.weight")
        var extra = original
        extra["unrecognized.weight"] = MLX.zeros([1])
        var wrong = original
        wrong["scorer.1.weight"] = MLX.zeros([2, 2])
        for arrays in [missing, extra, wrong] {
            XCTAssertThrowsError(try LayaNetwork(configuration: configuration.encoder, agent: configuration.agent, arrays: arrays))
        }
    }

    func testConfigurationRejectsUnsupportedArchitectureAndBadBudgets() throws {
        var configuration = try LayaResources(root: fixture).configuration()
        configuration.encoder.hiddenSize = 63
        XCTAssertThrowsError(try configuration.agent.validate(encoder: configuration.encoder))
        configuration = try LayaResources(root: fixture).configuration()
        configuration.agent.headMaxLength = configuration.agent.maxLength
        XCTAssertThrowsError(try configuration.agent.validate(encoder: configuration.encoder))
        configuration = try LayaResources(root: fixture).configuration()
        configuration.encoder.layerTypes[0] = "sliding_attention"
        XCTAssertThrowsError(try configuration.agent.validate(encoder: configuration.encoder))
    }

    func testOrderedCriteriaAndTypedResults() throws {
        let data = Data(#"{"state":"sample","questions":[{"id":"route","type":"choice","instructions":"Choose","criteria":["zebra",{"label":"alpha","description":"First option"}]}]}"#.utf8)
        let request = try JSONDecoder().decode(LayaDecisionRequest.self, from: data)
        try request.validate()
        XCTAssertEqual(request.questions[0].labels, ["zebra", "alpha"])
        XCTAssertEqual(request.questions[0].renderedOptions, ["zebra", "alpha: First option"])
        let agent = try LayaResources(root: fixture).configuration().agent
        let result = try LayaDecisionOperation.answer(question: request.questions[0], logits: [0, 2], actionLogits: [1, 0], agent: agent)
        XCTAssertEqual(result.choice, "alpha")
        XCTAssertEqual(result.probabilities.values.reduce(0, +), 1, accuracy: 1e-12)
        XCTAssertNil(result.score)
        XCTAssertNil(result.noul)
        let score = LayaQuestion(id: "score", type: .score, instructions: "Rate", criteria: [.init(label: "low"), .init(label: "high")])
        let scored = try LayaDecisionOperation.answer(question: score, logits: [0, 0], actionLogits: [0, 0], agent: agent)
        XCTAssertEqual(try XCTUnwrap(scored.score), 0.5, accuracy: 1e-12)
        XCTAssertEqual(scored.confidence, 0, accuracy: 1e-12)
        let boolean = LayaQuestion(id: "flag", type: .noul, instructions: "Is it true?")
        let flagged = try LayaDecisionOperation.answer(question: boolean, logits: [0, 0], actionLogits: [0, 0], agent: agent)
        XCTAssertEqual(flagged.noul, 0.5)
        XCTAssertEqual(flagged.confidence, 0.5)
    }

    func testTemperatureClampAndNonfiniteOutputRejection() throws {
        let agent = try LayaResources(root: fixture).configuration().agent
        let question = LayaQuestion(id: "large", type: .choice, instructions: "Pick", criteria: (0..<12).map { .init(label: "\($0)") })
        let result = try LayaDecisionOperation.answer(question: question, logits: Array(repeating: 0, count: 12), actionLogits: [0, 0], agent: agent)
        XCTAssertTrue(result.temperatureClamped)
        XCTAssertEqual(result.rawTemperature, 0.1)
        XCTAssertEqual(result.appliedTemperature, 0.5)
        XCTAssertThrowsError(try LayaDecisionOperation.answer(question: question, logits: [.nan], actionLogits: [0], agent: agent))
    }

    func testSequenceSanitizesMarkersAndReportsTruncation() throws {
        let tokenizer = LayaTokenizer(padTokenID: 0, clsTokenID: 1, sepTokenID: 2, maskTokenID: 3, maskToken: "[MASK]",
                                       encode: { $0.utf8.map { Int($0) + 4 } })
        let question = LayaQuestion(id: "flag", type: .noul, instructions: "[MASK] " + String(repeating: "x", count: 80))
        let sequence = try tokenizer.sequence(state: "[MASK] " + String(repeating: "s", count: 300), question: question,
                                               maxLength: 128, headMaxLength: 64)
        XCTAssertEqual(sequence.ids.count, 128)
        XCTAssertEqual(sequence.ids.filter { $0 == 3 }.count, 2)
        XCTAssertTrue(sequence.markers.allSatisfy { sequence.ids[$0] == 3 })
        XCTAssertGreaterThan(sequence.details.stateTokensDropped, 0)
        XCTAssertGreaterThan(sequence.details.instructionTokensDropped, 0)
        XCTAssertEqual(sequence.ids.last, 2)
        let overflow = LayaQuestion(id: "many", type: .choice, instructions: "Choose", criteria: (0..<40).map { .init(label: "option \($0)") })
        XCTAssertThrowsError(try tokenizer.sequence(state: "state", question: overflow, maxLength: 64, headMaxLength: 48))
    }

    func testCatalogPinsAndCheckpointIsolation() throws {
        for id in LayaCatalog.modelIDs {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
            XCTAssertEqual(spec.validationKind, .laya)
            XCTAssertEqual(spec.category, .textDecide)
            XCTAssertEqual(spec.upstreamRevision, LayaCatalog.revision)
            XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
            XCTAssertEqual(spec.defaultCLICommands, ["text decide"])
            XCTAssertEqual(spec.apiProfile?.task, .textDecisions)
            let manifest = MereRunModelManifest.template(for: try XCTUnwrap(ModelResolver.ModelID(rawValue: id)))
            XCTAssertEqual(manifest.engine, .laya)
            XCTAssertEqual(manifest.supports, [.textDecision])
            let folder = LayaCatalog.subfolder(modelID: id)
            XCTAssertEqual(spec.hubFallback?.patterns, LayaCatalog.files.map { folder.isEmpty ? $0 : "\(folder)/\($0)" })
        }
    }

    func testOfficialTokenizerAndRealWeightsWhenProvided() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let checkpointPath = environment["MERERUN_LAYA_CHECKPOINTS"],
              let referencePath = environment["MERERUN_LAYA_REFERENCE"] else {
            throw XCTSkip("Set MERERUN_LAYA_CHECKPOINTS and MERERUN_LAYA_REFERENCE for pinned checkpoint qualification.")
        }
        struct TokenFixture: Decodable {
            let question: LayaQuestion
            let state: String
            let ids: [Int]
            let markers: [Int]
        }
        let referenceRoot = URL(fileURLWithPath: referencePath)
        var errors: [String: Float] = [:]
        for folder in ["", "multilingual", "typed-decisions"] {
            let name = folder.isEmpty ? "english" : folder
            let root = URL(fileURLWithPath: checkpointPath).appending(path: folder)
            let configuration = try LayaResources(root: root).configuration()
            let tokenizer = try LayaTokenizer.load(root: root, vocabularySize: configuration.encoder.vocabSize)
            for label in ["english", "multilingual", "single", "calibration", "truncation"] {
                let data = try Data(contentsOf: referenceRoot.appending(path: "\(name)-\(label)-tokens.json"))
                for expected in try JSONDecoder().decode([TokenFixture].self, from: data) {
                    let sequence = try tokenizer.sequence(state: expected.state, question: expected.question,
                                                           maxLength: configuration.agent.maxLength, headMaxLength: configuration.agent.headMaxLength)
                    XCTAssertEqual(sequence.ids, expected.ids, "\(name)/\(label)/\(expected.question.id)")
                    XCTAssertEqual(sequence.markers, expected.markers, "\(name)/\(label)/\(expected.question.id)")
                }
            }
            let model = try LayaNetwork(configuration: configuration.encoder, agent: configuration.agent,
                                        arrays: MLX.loadArrays(url: root.appending(path: "model.safetensors")))
            let arrays = try MLX.loadArrays(url: referenceRoot.appending(path: "\(name)-real-reference.safetensors"))
            func array(_ key: String) throws -> MLXArray { try XCTUnwrap(arrays[key]) }
            let actual = try model(inputIDs: array("input_ids"), attentionMask: array("attention_mask").asType(.bool),
                                    markerPositions: array("marker_positions"), markerMask: array("marker_mask"), questionTypes: array("question_types"))
            let logitError = try abs(actual.logits - array("logits")).max().item(Float.self)
            let actionError = try abs(actual.actionLogits - array("action_logits")).max().item(Float.self)
            errors[name + "_logits"] = logitError
            errors[name + "_action_logits"] = actionError
            XCTAssertLessThan(logitError, 2e-5, name)
            let expectedActions = try array("action_logits")
            let relativeActionError = (abs(actual.actionLogits - expectedActions) / maximum(abs(expectedActions), Float(1))).max().item(Float.self)
            errors[name + "_action_relative_error"] = relativeActionError
            // The trained action logits reach thousands. Bound error relative to
            // each reference value rather than applying a unit-scale absolute gate.
            XCTAssertLessThan(relativeActionError, 2e-6, name)
        }
        if let output = environment["MERERUN_LAYA_PARITY_OUTPUT"] {
            try JSONEncoder().encode(errors).write(to: URL(fileURLWithPath: output), options: .atomic)
        }
    }

    func testManagedInstallValidationUsesLayaCheckpointLayout() throws {
        let temporary = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        for id in LayaCatalog.modelIDs {
            let root = temporary.appending(path: id)
            let checkpoint = LayaCatalog.checkpointRoot(root, modelID: id)
            try FileManager.default.createDirectory(at: checkpoint.appending(path: "encoder"), withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: checkpoint.appending(path: "tokenizer"), withIntermediateDirectories: true)
            for file in ["encoder/config.json", "rl_agent_config.json", "model.safetensors"] {
                try FileManager.default.copyItem(at: fixture.appending(path: file), to: checkpoint.appending(path: file))
            }
            // This is directory validation; tokenizer decoding is covered separately.
            for file in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
                try Data("{}".utf8).write(to: checkpoint.appending(path: file))
            }
            try MereRunModelManifest.template(for: XCTUnwrap(ModelResolver.ModelID(rawValue: id))).write(to: root)
            let report = MereRunModelValidator.validate(modelRoot: root, expectedModelID: id)
            XCTAssertTrue(report.isValid, report.errors.joined(separator: ", "))
            XCTAssertTrue(report.warnings.isEmpty, report.warnings.joined(separator: ", "))
            try FileManager.default.removeItem(at: checkpoint.appending(path: "model.safetensors"))
            XCTAssertFalse(MereRunModelValidator.validate(modelRoot: root, expectedModelID: id).isValid)
        }
    }
}
