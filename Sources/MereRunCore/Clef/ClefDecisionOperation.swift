#if !os(iOS)
import Foundation
import MLX
import MereRunQwenModel

/// Native single-pass structured decisions. The operation owns one checkpoint and is used serially.
final class ClefDenseDecisionOperation {
    public let modelID: String
    private let root: URL
    private let config: Q35Config
    private let headConfig: ClefHeadConfiguration
    private let processor: ClefProcessorConfiguration
    private let tokenizer: ClefTokenizer
    private var model: Q35Model?
    private var head: ClefJointHead?
    private var tower: Q35VisionTower?

    public init(root: URL, modelID: String = ClefCatalog.modelID) throws {
        self.root = root
        self.modelID = modelID
        let configuration = try ClefResources(root: root).configuration()
        config = configuration.backbone
        headConfig = configuration.head
        processor = configuration.processor
        tokenizer = try ClefTokenizer.load(root: root)
    }

    private func prepared(_ request: ClefDecisionRequest) throws -> (ClefTokenSequence, ClefPreparedMedia) {
        guard request.audio.isEmpty, request.videoFiles.isEmpty, request.images.isEmpty || request.videos.isEmpty, request.maxTokens <= 16_384 else {
            throw ClefError.invalidInput("Audio, mixed media, video files, and contexts above 16384 require Clef Omni.")
        }
        let media = try ClefPreparedMedia.prepare(request, processor: processor)
        let sequence = try tokenizer.sequence(request, modelID: modelID, mediaText: media.text)
        guard sequence.ids.allSatisfy({ (0..<config.textConfig.vocabSize).contains($0) }) else {
            throw ClefError.invalidInput("Clef tokenizer produced a token outside the backbone vocabulary.")
        }
        return (sequence, media)
    }

    public func prepare(_ request: ClefDecisionRequest) throws -> ClefDecisionPlan { try prepared(request).0.plan }

    public func predict(_ request: ClefDecisionRequest) throws -> ClefDecisionResponse {
        let (sequence, media) = try prepared(request)
        try Task.checkCancellation()
        let helper = Q35Generator(modelId: modelID)
        if model == nil {
            let loaded = Q35Model(config: config)
            try helper.loadTextWeights(into: loaded, from: Q35Resources(rootURL: root), groupSize: 64, bits: 4)
            let loadedHead = try ClefJointHead(configuration: headConfig)
            try loadedHead.load(MLX.loadArrays(url: root.appending(path: "joint_head.safetensors")))
            model = loaded
            head = loadedHead
        }
        guard let model, let head else { throw ClefError.invalidWeights("Clef model was not loaded.") }
        let ids = MLXArray(sequence.ids.map(Int32.init), [1, sequence.ids.count])
        var embeddings: MLXArray?
        var positions: MLXArray?
        if !media.items.isEmpty {
            if tower == nil {
                let loaded = Q35VisionTower(config: config)
                loaded.useCheckpointPositionInterpolation()
                try loaded.loadWeights(from: Q35Resources(rootURL: root))
                tower = loaded
            }
            guard let tower else { throw ClefError.invalidWeights("Clef vision tower was not loaded.") }
            let replacements = try media.replacements(tower: tower)
            guard let tokenID = request.videos.isEmpty ? config.imageTokenId : config.videoTokenId else {
                throw ClefError.invalidConfiguration("Clef media token id is missing.")
            }
            positions = try helper.buildMRoPEPositionData(inputIds: ids, imageTokenId: tokenID,
                                                          replacements: replacements, spatialMergeSize: tower.spatialMergeSize)?.positionIds
            embeddings = helper.insertVisionEmbeddings(hiddenStates: model.embeddings(for: ids), inputIds: ids,
                                                         imageTokenId: tokenID, replacements: replacements)
        }
        // No generation, sampling, chat template, or vocabulary projection is involved.
        let hidden = model.decisionHiddenStates(ids, cache: nil, inputEmbeddings: embeddings, positionIds: positions)[0]
        let lexical = sequence.fields.map { field in
            field.optionSpans.map { model.decisionLexicalEmbeddings(ids[0, $0]) }
        }
        try Task.checkCancellation()
        let logits = try head(hidden: hidden, fields: sequence.fields, lexical: lexical)
        let probabilities = logits.map { softmax($0.asType(.float32), axis: 0) }
        MLX.eval(probabilities)
        try Task.checkCancellation()
        var answers: [String: ClefDecisionResponse.Answer] = [:]
        for (question, values) in zip(request.questions, probabilities) {
            answers[question.id] = try ClefDecisionResponse.answer(question: question, probabilities: values.asArray(Float.self))
        }
        return ClefDecisionResponse(model: modelID, answers: answers,
                                     usage: .init(input_tokens: sequence.ids.count, output_tokens: 0))
    }

    public func unload() {
        model = nil
        head = nil
        tower = nil
    }
}
#endif
