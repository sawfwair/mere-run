// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation
import MLX
import MLXNN
import MereRunD1Model

/// One serial checkpoint owner; each question runs an independent forward with fresh state.
public final class D1DecisionOperation {
    public let modelID: String
    private let root: URL
    private let config: D1Configuration
    private let tokenizer: D1Tokenizer
    private var trunk: D1Trunk?
    private var head: D1Head?
    private var vision: D1Vision?
    private var audio: D1Audio?
    public init(root: URL, modelID: String) throws {
        self.root = root; self.modelID = modelID
        config = try D1Catalog.configuration(root: root)
        tokenizer = try D1Tokenizer.load(root: root)
        guard try tokenizer.token("<|startoftext|>") == config.bos_token_id else { throw D1Error.invalid("D1 tokenizer BOS differs from configuration.") }
    }
    public func unload() { trunk = nil; head = nil; vision = nil; audio = nil }
    private func prepared(_ request: D1DecisionRequest) throws -> (D1DecisionPlan, [D1TokenSequence], D1PreparedMedia) {
        let media = try D1PreparedMedia.prepare(request, config: config)
        let maximum = request.maxTokens ?? config.maxLength
        guard maximum <= config.maxLength else { throw D1Error.invalid("max_tokens exceeds the D1 checkpoint's context limit.") }
        let sequences = try request.questions.map {
            try config.isOmni ? tokenizer.omni(request, question: $0, config: config, mediaTokens: media.tokens)
                : tokenizer.causal(request, question: $0, mediaText: media.markup, maxLength: maximum)
        }
        guard sequences.allSatisfy({ sequence in
            (sequence.ids + sequence.readout.flatMap { $0 }).allSatisfy { (0..<config.text_config.vocab_size).contains($0) }
        }) else { throw D1Error.invalid("D1 tokenizer emitted an out-of-vocabulary token.") }
        if !config.isOmni, !media.grids.isEmpty {
            guard let imageID = config.image_token_id, sequences.allSatisfy({ $0.ids.filter { $0 == imageID }.count == media.tokens }) else {
                throw D1Error.invalid("D1 image placeholders and patch counts differ.")
            }
        }
        let fields = zip(request.questions, sequences).map { question, sequence in
            D1DecisionPlan.Field(id: question.id, inputTokens: sequence.ids.count + (config.isOmni ? media.tokens : 0),
                stateTokensDropped: sequence.dropped, optionIDs: question.options.map(\.id), markerPositions: sequence.markers)
        }
        let plan = D1DecisionPlan(model: modelID, runtime: config.isOmni ? "native-swift-mlx-fp32" : "native-swift-mlx-bf16",
            mediaTokens: media.tokens, inputTokens: fields.reduce(0) { $0 + $1.inputTokens }, questions: fields)
        return (plan, sequences, media)
    }
    public func prepare(_ request: D1DecisionRequest) throws -> D1DecisionPlan { try prepared(request).0 }
    public func predict(_ request: D1DecisionRequest) throws -> D1DecisionResponse {
        let (plan, sequences, media) = try prepared(request)
        try Task.checkCancellation()
        try load(requireVision: !media.grids.isEmpty, requireAudio: media.samples != nil)
        guard let trunk else { throw D1Error.invalid("D1 trunk was not loaded.") }
        let prefix: MLXArray?
        if let samples = media.samples {
            guard let audio else { throw D1Error.invalid("D1 audio encoder was not loaded.") }
            prefix = try audio(samples: samples)
        } else if !media.grids.isEmpty {
            guard let vision else { throw D1Error.invalid("D1 vision tower was not loaded.") }
            let pixels = zip(media.patches, media.grids).map { MLXArray($0).reshaped(1, $1.patchCount, 768) }
            prefix = vision(pixels: pixels, grids: media.grids)
        } else { prefix = nil }
        if let prefix { guard prefix.dim(1) == media.tokens else { throw D1Error.invalid("D1 media prefix differs from preflight.") }; eval(prefix) }
        var answers: [String: D1DecisionResponse.Answer] = [:]
        for (question, sequence) in zip(request.questions, sequences) {
            try Task.checkCancellation()
            let ids = MLXArray(sequence.ids).reshaped(1, -1)
            var embeddings = trunk.embeddings(ids)
            if let prefix {
                if config.isOmni { embeddings = concatenated([prefix, embeddings], axis: 1) }
                else {
                    var index = 0
                    let rows = sequence.ids.enumerated().map { position, token -> MLXArray in
                        if token == config.image_token_id { defer { index += 1 }; return prefix[0..., index..<(index + 1), 0...] }
                        return embeddings[0..., position..<(position + 1), 0...]
                    }
                    embeddings = concatenated(rows, axis: 1)
                }
            }
            let hidden = trunk(embeddings, prefix: config.isOmni ? media.tokens : 0)
            let probabilities: MLXArray
            if config.isOmni {
                guard let head else { throw D1Error.invalid("D1 decision head was not loaded.") }
                let type = question.type == .choice ? 0 : question.type == .score ? 1 : 2
                let text = hidden[0..., media.tokens..., 0...]
                let logits = head(text, markers: sequence.markers, type: type)
                let count = question.options.count
                let bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
                let temperature = media.tokens == 0 ? config.temperatures?[question.type.rawValue + ":" + bucket]
                    ?? config.temperatures?[question.type.rawValue] ?? 1 : 1
                probabilities = softmax(logits / temperature, axis: -1)
            } else {
                // Normalization over the full vocabulary cancels in the option softmax.
                let logits = trunk.logits(hidden[0..., (hidden.dim(1) - 1)..., 0...]).reshaped(-1).asType(.float32)
                let scores = sequence.readout.map { take(logits, MLXArray($0), axis: 0).max() }
                probabilities = softmax(stacked(scores), axis: -1)
            }
            eval(probabilities)
            var values = probabilities.asArray(Float.self)
            if config.isOmni, question.type == .noul { values.reverse() }
            answers[question.id] = try D1DecisionResponse.answer(question, probabilities: values, stringLegend: config.isOmni)
        }
        return D1DecisionResponse(model: modelID, answers: answers, usage: .init(input_tokens: plan.inputTokens, output_tokens: 0))
    }
    private func load(requireVision: Bool, requireAudio: Bool) throws {
        if trunk != nil, (!requireVision || vision != nil), (!requireAudio || audio != nil) { return }
        let index = root.appending(path: "model.safetensors.index.json")
        let arrays = try FileManager.default.fileExists(atPath: index.path)
            ? HFSafetensorsWeightsLoader.loadShardedArrays(indexURL: index) : MLX.loadArrays(url: root.appending(path: "model.safetensors"))
        func mapped(prefix: String, replacements: [(String, String)] = []) -> ModuleParameters {
            let values = arrays.compactMap { key, value -> (String, MLXArray)? in
                guard key.hasPrefix(prefix) else { return nil }
                var name = String(key.dropFirst(prefix.count)), tensor = value
                for (source, destination) in replacements { name = name.replacingOccurrences(of: source, with: destination) }
                if name.hasSuffix(".conv.conv.weight") { tensor = tensor.transposed(0, 2, 1) }
                return (name, tensor)
            }
            return ModuleParameters.unflattened(values)
        }
        let loadedTrunk = D1Trunk(config.text_config, omni: config.isOmni)
        try loadedTrunk.update(parameters: mapped(prefix: config.isOmni ? "encoder." : "model.language_model."), verify: .all)
        var loadedHead: D1Head?
        if config.isOmni {
            let head = D1Head(width: config.text_config.hidden_size, layerCount: config.head_layers ?? 2)
            try head.update(parameters: mapped(prefix: "head.", replacements: [("head.layers.", "layers."), ("scorer.0.", "scorer.norm."), ("scorer.1.", "scorer.input."), ("scorer.3.", "scorer.output.")]), verify: .all)
            loadedHead = head
        }
        var loadedVision: D1Vision?
        if requireVision {
            let vision = D1Vision(config)
            let parameters: ModuleParameters
            if config.isOmni { parameters = mapped(prefix: "vision.", replacements: [("tower.vision_model.", "tower.")]) }
            else { parameters = mapped(prefix: "model.", replacements: [("vision_tower.vision_model.", "tower."), ("multi_modal_projector.", "projector.")]) }
            // For the causal checkpoint, select only the tower and projector; the language model is loaded above.
            let selected = parameters.flattened().filter { $0.0.hasPrefix("tower.") || $0.0.hasPrefix("projector.") }
            try vision.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
            loadedVision = vision
        }
        let loadedAudio = try requireAudio ? config.audio_config.map { try D1Audio(config: $0, outputWidth: config.text_config.hidden_size, weights: arrays.filter { $0.key.hasPrefix("audio.") }) } : nil
        trunk = loadedTrunk; head = loadedHead; vision = loadedVision; audio = loadedAudio
    }
}
