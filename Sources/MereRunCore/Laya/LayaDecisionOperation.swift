import Foundation
import MLX
import MereRunLayaModel

public final class LayaDecisionOperation {
    public let modelID: String
    private let resources: LayaResources
    private let encoderConfiguration: LayaEncoderConfiguration
    private let agentConfiguration: LayaAgentConfiguration
    private let tokenizer: LayaTokenizer
    private var network: LayaNetwork?

    public init(root: URL, modelID: String) throws {
        self.modelID = modelID
        resources = LayaResources(root: root)
        let configuration = try resources.configuration()
        encoderConfiguration = configuration.encoder
        agentConfiguration = configuration.agent
        tokenizer = try LayaTokenizer.load(root: root, vocabularySize: configuration.encoder.vocabSize)
        guard tokenizer.padTokenID == configuration.encoder.padTokenID else {
            throw LayaModelError.invalidConfiguration("Laya tokenizer and encoder padding ids differ.")
        }
    }

    public func prepare(_ request: LayaDecisionRequest) throws -> (plan: LayaDecisionPlan, sequences: [LayaTokenSequence]) {
        try request.validate()
        let maxTokens = request.maxTokens ?? agentConfiguration.maxLength
        let headMaxTokens = request.headMaxTokens ?? agentConfiguration.headMaxLength
        guard maxTokens <= encoderConfiguration.maxPositionEmbeddings else {
            throw LayaModelError.invalidInput("max_tokens exceeds this encoder's position limit.")
        }
        let stateIDs = tokenizer.encodeState(request.state)
        let sequences = try request.questions.map {
            try tokenizer.sequence(stateIDs: stateIDs, question: $0, maxLength: maxTokens, headMaxLength: headMaxTokens)
        }
        guard sequences.allSatisfy({ $0.ids.allSatisfy { (0..<encoderConfiguration.vocabSize).contains($0) } }) else {
            throw LayaModelError.invalidInput("Tokenizer produced a token outside the checkpoint vocabulary.")
        }
        return (LayaDecisionPlan(model: modelID, maxTokens: maxTokens, headMaxTokens: headMaxTokens,
                                 questions: sequences.map(\.details)), sequences)
    }

    public func predict(_ request: LayaDecisionRequest) throws -> LayaDecisionResponse {
        let prepared = try prepare(request)
        try Task.checkCancellation()
        if network == nil {
            network = try LayaNetwork(configuration: encoderConfiguration, agent: agentConfiguration,
                                      arrays: MLX.loadArrays(url: resources.root.appending(path: "model.safetensors")))
        }
        guard let network else { throw LayaModelError.invalidWeights("Laya network was not loaded.") }
        var answers: [String: LayaDecisionAnswer] = [:]
        var offset = 0
        while offset < prepared.sequences.count {
            try Task.checkCancellation()
            let candidates = Array(prepared.sequences[offset..<min(offset + 8, prepared.sequences.count)])
            let longest = candidates.map { $0.ids.count }.max() ?? 1
            // Bound both padded tokens and quadratic attention storage.
            let batchCount = max(1, min(candidates.count, 8_192 / longest, 8_388_608 / (longest * longest)))
            let batch = Array(candidates.prefix(batchCount))
            let questions = Array(request.questions[offset..<(offset + batchCount)])
            let result = infer(network: network, sequences: batch, questions: questions)
            for index in batch.indices {
                let question = questions[index]
                answers[question.id] = try Self.answer(
                    question: question, logits: result.logits[index], actionLogits: result.actions[index], agent: agentConfiguration)
            }
            offset += batchCount
        }
        return LayaDecisionResponse(model: modelID, runtime: "native-swift-mlx-fp32", answers: answers,
                                    plan: prepared.plan, inputTokens: prepared.sequences.reduce(0) { $0 + $1.ids.count }, outputTokens: 0)
    }

    private func infer(network: LayaNetwork, sequences: [LayaTokenSequence], questions: [LayaQuestion]) -> (logits: [[Float]], actions: [[Float]]) {
        let length = sequences.map { $0.ids.count }.max() ?? 1
        let options = sequences.map { $0.markers.count }.max() ?? 1
        let ids = sequences.flatMap { $0.ids + Array(repeating: tokenizer.padTokenID, count: length - $0.ids.count) }
        let mask = sequences.flatMap { Array(repeating: true, count: $0.ids.count) + Array(repeating: false, count: length - $0.ids.count) }
        let markers = sequences.flatMap { $0.markers + Array(repeating: 0, count: options - $0.markers.count) }
        let markerMask = sequences.flatMap { Array(repeating: true, count: $0.markers.count) + Array(repeating: false, count: options - $0.markers.count) }
        let result = network(inputIDs: MLXArray(ids).reshaped([sequences.count, length]),
                             attentionMask: MLXArray(mask).reshaped([sequences.count, length]),
                             markerPositions: MLXArray(markers).reshaped([sequences.count, options]),
                             markerMask: MLXArray(markerMask).reshaped([sequences.count, options]),
                             questionTypes: MLXArray(questions.map { $0.type.index }))
        eval(result.logits, result.actionLogits)
        return (sequences.indices.map { result.logits[$0].asArray(Float.self) },
                sequences.indices.map { result.actionLogits[$0].asArray(Float.self) })
    }

    static func answer(question: LayaQuestion, logits: [Float], actionLogits: [Float], agent: LayaAgentConfiguration) throws -> LayaDecisionAnswer {
        let labels = question.labels
        guard logits.count >= labels.count, !labels.isEmpty, !actionLogits.isEmpty,
              logits.allSatisfy(\.isFinite), actionLogits.allSatisfy(\.isFinite) else {
            throw LayaModelError.invalidWeights("Laya produced invalid decision logits.")
        }
        let temperature = agent.temperature(type: question.type.index, optionCount: labels.count)
        func distribution(_ values: [Double]) -> [Double] {
            let peak = values.max() ?? 0
            let values = values.map { exp($0 - peak) }
            let total = values.reduce(0, +)
            return values.map { $0 / total }
        }
        let probabilities = distribution(logits.prefix(labels.count).map { Double($0) / temperature.applied })
        let best = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
        let confidence = labels.count < 2 ? 1 : min(1, max(0, 1 - entropy / log(Double(labels.count))))
        return LayaDecisionAnswer(
            type: question.type, choice: question.type == .choice ? labels[best] : nil,
            score: question.type == .score ? probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element } : nil,
            noul: question.type == .noul ? probabilities[1] : nil,
            probabilities: Dictionary(uniqueKeysWithValues: zip(labels, probabilities)),
            confidence: question.type == .noul ? max(probabilities[0], probabilities[1]) : confidence,
            actProbability: distribution(actionLogits.map(Double.init))[0],
            rawTemperature: temperature.raw, appliedTemperature: temperature.applied,
            temperatureClamped: temperature.raw != temperature.applied)
    }
}
