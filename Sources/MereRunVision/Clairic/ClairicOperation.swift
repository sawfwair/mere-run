import MereRunCore
#if !os(iOS)
import Foundation
import MLX
import MereRunQwenModel

/// Local, serial inference. Prefill retains independent hybrid caches, then
/// active field slots advance together and stop on their own EOS tokens.
public final class ClairicOperation {
    private static let contextLimit = 65_273
    private static let outputLimit = 256
    private static let batchSize = 8

    private let root: URL
    private let modelID: String
    private let config: ClairicConfiguration
    private let tokenizer: ClairicTokenizer
    private let qwen: Q35Generator
    private var model: Q35Model?
    private var tower: Q35VisionTower?

    public init(root: URL, modelID: String? = nil) throws {
        self.root = root
        config = try ClairicConfiguration(root: root)
        self.modelID = modelID ?? (config.isFlash ? ClairicCatalog.flashModelID : ClairicCatalog.modelID)
        tokenizer = try ClairicTokenizer(root: root, vocabularySize: config.backbone.textConfig.vocabSize)
        guard config.backbone.eosTokenIds.contains(tokenizer.eos), config.backbone.textConfig.eosTokenId == tokenizer.eos else {
            throw ClairicError.invalid("Clairic checkpoint EOS does not match the tokenizer terminator.")
        }
        qwen = Q35Generator(modelId: self.modelID)
    }

    public func prepare(_ request: ClairicRequest) throws -> ClairicPlan {
        try prepared(request).plan
    }

    public func predict(_ request: ClairicRequest) throws -> ClairicResponse {
        let prepared = try prepared(request)
        var answers: [String: ClairicResponse.Answer] = [:]
        var outputTokens = 0
        if !prepared.slots.isEmpty {
            let model = try loadModel()
            let media = try loadVision(prepared.media)
            let seed = try prefillPrefix(prepared.slots, model: model, replacements: media)
            var grouped: [String: [ClairicResponse.Answer]] = [:]
            var grammars: [String: ClairicSlotGrammar] = [:]
            for start in stride(from: 0, to: prepared.slots.count, by: Self.batchSize) {
                let slots = prepared.slots[start..<min(start + Self.batchSize, prepared.slots.count)]
                let rows = try slots.map { slot in
                    let key = try slot.slot.spec.schemaJSON()
                    let grammar: ClairicSlotGrammar
                    if let shared = grammars[key] {
                        grammar = shared
                    } else {
                        grammar = try ClairicSlotGrammar(slot.slot.spec)
                        grammars[key] = grammar
                    }
                    return try prefill(slot, model: model, seed: seed, grammar: grammar)
                }
                try decode(rows, model: model)
                for row in rows {
                    grouped[row.slot.field, default: []].append(row.answer(source: prepared.state, hasMedia: !media.isEmpty))
                    outputTokens += row.tokens
                }
            }
            for (name, values) in grouped where request.fields[name]?.type != .list {
                answers[name] = values[0]
            }
            for (name, field) in request.fields where field.type == .list {
                answers[name] = Self.listAnswer(grouped[name] ?? [])
            }
        } else {
            for name in request.fields.keys { answers[name] = Self.listAnswer([]) }
        }
        return ClairicResponse(
            model: modelID, answers: answers, truncated: false,
            usage: .init(input_tokens: prepared.plan.input_tokens, output_tokens: outputTokens),
            state_text: request.options?.return_state_text == true ? prepared.state : nil)
    }

    public func unload() {
        model = nil
        tower = nil
    }

    private struct PreparedSlot {
        let slot: ClairicPrompt.Slot
        let ids: [Int]
    }

    private struct Prepared {
        let state: String
        let media: [ClairicMedia.Item]
        let slots: [PreparedSlot]
        let plan: ClairicPlan
    }

    private func prepared(_ request: ClairicRequest) throws -> Prepared {
        try request.validate()
        let state = switch request.state {
        case .none: ""
        case .string(let value): value
        case .some(let value): try ClairicField.json(value)
        }
        let media = try ClairicMedia.prepare(request, processor: config.processor)
        var prefix = tokenizer.encode(ClairicPrompt.header)
        if let start = config.backbone.visionStartTokenId,
           let image = config.backbone.imageTokenId,
           let end = config.backbone.visionEndTokenId {
            for frame in media {
                prefix += [start] + Array(repeating: image, count: frame.tokenCount) + [end]
            }
        }
        let stateIDs = try tokenizer.encodeLiteral(state)
        let footer = tokenizer.encode(ClairicPrompt.footer)
        let promptSlots = try ClairicPrompt.slots(for: request, compactLists: config.isFlash)
        guard promptSlots.count <= 256 else {
            throw ClairicError.invalid("Clairic accepts at most 256 generated slots per request. Reduce the list limits.")
        }
        let slots = try promptSlots.map { slot -> PreparedSlot in
            try Task.checkCancellation()
            var suffix = slot.instruction
            if media.isEmpty || !state.isEmpty { suffix += "\n\nDocument:\n" }
            let ids = prefix + (try tokenizer.encodeLiteral(suffix)) + stateIDs + footer
            guard ids.count <= min(Self.contextLimit, config.backbone.textConfig.maxPositionEmbeddings - Self.outputLimit) else {
                throw ClairicError.invalid("Clairic field '\(slot.field)' exceeds the model context.")
            }
            return PreparedSlot(slot: slot, ids: ids)
        }
        let counts = Dictionary(grouping: slots, by: { $0.slot.field }).mapValues(\.count)
        let fields = request.fields.keys.sorted().map { ClairicPlan.Field(name: $0, slots: counts[$0, default: 0]) }
        return Prepared(state: state, media: media, slots: slots,
                        plan: ClairicPlan(model: modelID, input_tokens: slots.reduce(0) { $0 + $1.ids.count },
                                          image_count: media.count, fields: fields))
    }

    private func loadModel() throws -> Q35Model {
        if let model { return model }
        let missing = ClairicCatalog.validate(root: root, fileManager: .default)
        guard missing.isEmpty else {
            throw ClairicError.invalid("Clairic checkpoint is incomplete: \(missing.map(\.lastPathComponent).joined(separator: ", ")).")
        }
        try Task.checkCancellation()
        let loaded = Q35Model(config: config.backbone)
        try qwen.loadTextWeights(into: loaded, from: Q35Resources(rootURL: root), groupSize: 64, bits: 4)
        model = loaded
        return loaded
    }

    private func loadVision(_ media: [ClairicMedia.Item]) throws -> [Q35VisionReplacement] {
        guard !media.isEmpty else { return [] }
        if tower == nil {
            let loaded = Q35VisionTower(config: config.backbone)
            loaded.useCheckpointPositionInterpolation()
            try loaded.loadWeights(from: Q35Resources(rootURL: root))
            tower = loaded
        }
        guard let tower else { throw ClairicError.inference("Clairic could not load its vision tower.") }
        return try media.map { try $0.encode(using: tower) }
    }

    private struct PrefillSeed {
        let count: Int
        let caches: [Q35LayerCache?]
        let ropeDelta: Int?
    }

    private func prefillPrefix(
        _ slots: [PreparedSlot], model: Q35Model, replacements: [Q35VisionReplacement]
    ) throws -> PrefillSeed? {
        guard let first = slots.first else { return nil }
        var count = first.ids.count - 1
        for slot in slots.dropFirst() {
            count = min(count, slot.ids.count - 1)
            count = (0..<count).first { first.ids[$0] != slot.ids[$0] } ?? count
        }
        guard (!replacements.isEmpty || slots.count > 1 && count >= 128) else { return nil }

        let ids = MLXArray(first.ids.prefix(count).map(Int32.init), [1, count])
        let caches = qwen.makeLayerCaches(config: config.backbone)
        var embeddings: MLXArray?
        var position: Q35MRoPEPositionData?
        if let imageID = config.backbone.imageTokenId, !replacements.isEmpty {
            position = try qwen.buildMRoPEPositionData(inputIds: ids, imageTokenId: imageID,
                                                       replacements: replacements, spatialMergeSize: 2)
            embeddings = qwen.insertVisionEmbeddings(hiddenStates: model.embeddings(for: ids), inputIds: ids,
                                                       imageTokenId: imageID, replacements: replacements)
        }
        _ = try prefillHidden(ids, caches: caches, model: model, embeddings: embeddings, positionIDs: position?.positionIds)
        return PrefillSeed(count: count, caches: caches, ropeDelta: position?.ropeDelta)
    }

    private func prefill(
        _ prepared: PreparedSlot, model: Q35Model, seed: PrefillSeed?, grammar: ClairicSlotGrammar
    ) throws -> ClairicDecodeRow {
        let start = seed?.count ?? 0
        let count = prepared.ids.count - start
        let ids = MLXArray(prepared.ids[start...].map(Int32.init), [1, count])
        let caches = seed?.caches.map { $0?.fork() } ?? qwen.makeLayerCaches(config: config.backbone)
        let positionIDs = seed?.ropeDelta.map { delta in
            let lane = (start..<(start + count)).map { Int32($0 + delta) }
            return MLXArray(lane + lane + lane, [3, 1, count])
        }
        let hidden = try prefillHidden(ids, caches: caches, model: model, embeddings: nil, positionIDs: positionIDs)
        let logits = model.logits(from: hidden)
        MLX.eval(logits)
        return ClairicDecodeRow(slot: prepared.slot, caches: caches, logits: logits, grammar: grammar,
                                nextPosition: start + count + (seed?.ropeDelta ?? 0), usesMRoPE: positionIDs != nil)
    }

    private func prefillHidden(
        _ ids: MLXArray, caches: [Q35LayerCache?], model: Q35Model, embeddings: MLXArray?, positionIDs: MLXArray?
    ) throws -> MLXArray {
        let count = ids.dim(1)
        var lastHidden: MLXArray?
        for start in stride(from: 0, to: count, by: 512) {
            try Task.checkCancellation()
            let range = start..<min(start + 512, count)
            let hidden = model.decisionHiddenStates(
                ids[0..., range], cache: caches, inputEmbeddings: embeddings?[0..., range, 0...],
                positionIds: positionIDs?[0..., 0..., range])
            MLX.eval(hidden)
            lastHidden = hidden[0, range.count - 1]
        }
        guard let lastHidden else { throw ClairicError.inference("Clairic slot prefill returned no tokens.") }
        return lastHidden
    }

    private func decode(_ rows: [ClairicDecodeRow], model: Q35Model) throws {
        var active = rows
        for step in 0..<Self.outputLimit {
            try Task.checkCancellation()
            var tokens: [Int32] = []
            active = try active.filter { row in
                guard let token = try row.step(tokenizer: tokenizer, isFirst: step == 0) else { return false }
                tokens.append(Int32(token))
                return true
            }
            guard !active.isEmpty, step + 1 < Self.outputLimit else { return }
            try advance(active, tokens: tokens, model: model)
        }
    }

    private func advance(_ rows: [ClairicDecodeRow], tokens: [Int32], model: Q35Model) throws {
        if rows.count > 1, let caches = qwen.makeBatchedLayerCaches(rows.map(\.caches)) {
            let ids = MLXArray(tokens, [rows.count, 1])
            let positions = rows.map { Int32($0.nextPosition) }
            let positionIDs = rows.contains(where: \.usesMRoPE)
                ? MLXArray(positions + positions + positions, [3, rows.count, 1]) : nil
            let logits = model(ids, cache: caches, positionIds: positionIDs)
            MLX.eval(logits)
            guard let split = qwen.splitBatchedLayerCaches(caches, rowCount: rows.count) else {
                throw ClairicError.inference("Clairic could not split the batched decoder state.")
            }
            for (index, row) in rows.enumerated() {
                row.logits = logits[index, 0]
                row.caches = split[index]
                row.nextPosition += 1
            }
            return
        }
        for (row, token) in zip(rows, tokens) {
            let ids = MLXArray([token], [1, 1])
            let position = Int32(row.nextPosition)
            let positionIDs = row.usesMRoPE ? MLXArray([position, position, position], [3, 1, 1]) : nil
            row.logits = model(ids, cache: row.caches, positionIds: positionIDs)[0, 0]
            MLX.eval(row.logits)
            row.nextPosition += 1
        }
    }

    private static func listAnswer(_ slots: [ClairicResponse.Answer]) -> ClairicResponse.Answer {
        let values = slots.filter { $0.valid && $0.value != .null }
        return .init(value: .array(values.map(\.value)), valid: slots.allSatisfy(\.valid), complete: slots.allSatisfy(\.complete),
                     confidence: values.map(\.confidence).min() ?? slots.first?.confidence ?? 1,
                     p_null: slots.first?.p_null ?? 0, span: .null)
    }
}
#endif
