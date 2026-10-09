import Foundation
import MLX
import AudioCodecs
import AudioQwen3ASRModel
import MereRunQwenModel

final class ClefOmniDecisionOperation {
    let modelID: String
    let resources: ClefOmniResources
    let config: Qwen3OmniConfiguration
    let headConfig: ClefHeadConfiguration
    let tokenizer: ClefTokenizer
    private var model: Qwen3OmniThinker?
    private var head: ClefJointHead?
    private var vision: Qwen3OmniVision?
    private var audio: Qwen3ASRAudioTower?

    init(root: URL, modelID: String) throws {
        self.modelID = modelID
        resources = ClefOmniResources(root: root)
        (config, headConfig) = try resources.configuration()
        tokenizer = try ClefTokenizer.load(root: root)
    }

    private func prepared(_ request: ClefDecisionRequest) throws -> (ClefTokenSequence, ClefOmniMedia, ClefOmniLayout) {
        let media = try ClefOmniMedia.prepare(request)
        let sequence = try tokenizer.sequence(request, modelID: modelID, mediaIDs: media.blocks.flatMap { tokenizer.encode($0.text) })
        guard sequence.ids.allSatisfy({ (0..<config.thinkerConfig.textConfig.vocabSize).contains($0) }) else {
            throw ClefError.invalidInput("Clef Omni input contains a token outside the thinker vocabulary.")
        }
        return (sequence, media, try ClefOmniLayout(media: media, tokenizer: tokenizer, sequence: sequence, config: config))
    }

    func prepare(_ request: ClefDecisionRequest) throws -> ClefDecisionPlan { try prepared(request).0.plan }

    func predict(_ request: ClefDecisionRequest) throws -> ClefDecisionResponse {
        let (sequence, media, layout) = try prepared(request)
        try Task.checkCancellation()
        if model == nil {
            let model = Qwen3OmniThinker(config: config.thinkerConfig.textConfig, quantization: config.quantization)
            try resources.loadText(model)
            self.model = model
            let head = try ClefJointHead(configuration: headConfig)
            try head.load(MLX.loadArrays(url: resources.root.appending(path: "joint_head.safetensors")))
            self.head = head
        }
        guard let model, let head else { throw ClefError.invalidWeights("Omni thinker was not loaded.") }
        let ids = MLXArray(sequence.ids.map(Int32.init), [1, sequence.ids.count])
        let embeddings = model.embedding(ids)
        var visualIndices: [Int] = []
        var deepstack: [[MLXArray]] = Array(repeating: [], count: config.thinkerConfig.visionConfig.deepstackVisualIndexes.count)
        for (index, block) in media.blocks.enumerated() {
            try Task.checkCancellation()
            let indices = layout.indices[index]
            if let visual = block.visual {
                if vision == nil {
                    let tower = Qwen3OmniVision(config: config.thinkerConfig.visionConfig)
                    try resources.loadVision(tower)
                    vision = tower
                }
                guard let vision else { throw ClefError.invalidWeights("Omni vision encoder was not loaded.") }
                let features = try vision(patches: visual.patches(), temporal: visual.temporal, height: visual.height, width: visual.width)
                let selected = zip(indices, block.kinds).filter { $0.1 != 3 }.map(\.0)
                guard features.embeddings.dim(0) == selected.count, features.deepstack.count == deepstack.count else {
                    throw ClefError.invalidInput("Omni vision placeholders differ from encoder output.")
                }
                embeddings[0, MLXArray(selected.map(Int32.init))] = features.embeddings
                visualIndices += selected
                for layer in deepstack.indices { deepstack[layer].append(features.deepstack[layer]) }
            }
            if let clip = block.audio {
                if audio == nil {
                    let config = config.thinkerConfig.audioConfig
                    let tower = Qwen3ASRAudioTower(config: Qwen3ASRAudioEncoderConfig(
                        dModel: config.dModel, numHiddenLayers: config.encoderLayers, numAttentionHeads: config.encoderAttentionHeads,
                        ffnDim: config.encoderFfnDim, maxSourcePositions: config.maxSourcePositions, numMelBins: config.numMelBins,
                        outputDim: config.outputDim, downsampleHiddenSize: config.downsampleHiddenSize,
                        nWindow: config.nWindow, nWindowInfer: config.nWindowInfer, convChunkSize: config.convChunksize))
                    try resources.loadAudio(tower)
                    audio = tower
                }
                guard let audio else { throw ClefError.invalidWeights("Omni audio encoder was not loaded.") }
                let mel = clip.mel().asType(.bfloat16)
                let features = audio(mel, checkpointPositionArithmetic: true)[0]
                let selected = zip(indices, block.kinds).filter { $0.1 == 3 }.map(\.0)
                guard features.dim(0) == selected.count else { throw ClefError.invalidInput("Omni audio placeholder count differs from encoder output.") }
                embeddings[0, MLXArray(selected.map(Int32.init))] = features
            }
        }
        let merged = visualIndices.isEmpty ? [] : deepstack.map { MLX.concatenated($0, axis: 0) }
        let positions = MLXArray(layout.positions.flatMap { $0 }, [3, 1, sequence.ids.count])
        let hidden = try model(ids: ids, positions: positions, embeddings: embeddings, visualIndices: visualIndices, deepstack: merged)[0]
        let lexical = sequence.fields.map { field in field.optionSpans.map { model.lexical(ids[0, $0]) } }
        let logits = try head(hidden: hidden, fields: sequence.fields, lexical: lexical)
        let probabilities = logits.map { softmax($0.asType(.float32), axis: 0) }
        MLX.eval(probabilities)
        var answers: [String: ClefDecisionResponse.Answer] = [:]
        for (question, probability) in zip(request.questions, probabilities) {
            answers[question.id] = try ClefDecisionResponse.answer(question: question, probabilities: probability.asArray(Float.self))
        }
        return ClefDecisionResponse(model: modelID, answers: answers, usage: .init(input_tokens: sequence.ids.count, output_tokens: 0))
    }

    func unload() { model = nil; head = nil; vision = nil; audio = nil }
}

struct ClefOmniLayout {
    let positions: [[Float]]
    let indices: [[Int]]

    init(media: ClefOmniMedia, tokenizer: ClefTokenizer, sequence: ClefTokenSequence, config: Qwen3OmniConfiguration) throws {
        let prefix = tokenizer.encode(ClefTokenizer.prefixText).count
        var positions = Array(repeating: [Float](), count: 3)
        var indices: [[Int]] = []
        var next: Float = 0
        func append(_ values: [Float]) { for axis in 0..<3 { positions[axis].append(values[axis]) } }
        func text(_ count: Int) { for _ in 0..<count { append(Array(repeating: next, count: 3)); next += 1 } }
        text(prefix)
        for block in media.blocks {
            let ids = tokenizer.encode(block.text)
            let newlineCount = tokenizer.encode("\n").count
            guard ids.count == block.bosCount + block.kinds.count + block.eosCount + newlineCount else {
                throw ClefError.invalidInput("Omni tokenizer does not preserve checkpoint media markers.")
            }
            text(block.bosCount)
            let start = positions[0].count
            let base = next
            var maximum = next - 1
            for (offset, position) in block.positions.enumerated() {
                let token = block.kinds[offset] == 1 ? config.thinkerConfig.imageTokenId
                    : block.kinds[offset] == 2 ? config.thinkerConfig.videoTokenId : config.thinkerConfig.audioTokenId
                guard ids[block.bosCount + offset] == token else { throw ClefError.invalidInput("Omni media token ID mismatch.") }
                let values = position.map { $0 + base }
                maximum = max(maximum, values.max()!)
                append(values)
            }
            indices.append(Array(start..<(start + block.kinds.count)))
            next = maximum + 1
            text(block.eosCount + newlineCount)
        }
        text(sequence.ids.count - positions[0].count)
        self.positions = positions
        self.indices = indices
    }
}
