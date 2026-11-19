import Foundation
import MereRunQwenModel
import MLX
import MLXRandom

package final class HaruSession {
    private let model: HaruModel
    private let temperature: Float
    private var encoder: [MLXArray]
    private var decoder: [MLXArray]
    private let caches: [Q35LayerCache?]
    private var key: MLXArray
    private var stream: HaruTokenStream
    private lazy var backboneOperations = Q35CompiledOperations()
    private var conditioning: [MLXArray] = []
    private var previous: (token: MLXArray, audio: MLXArray)?
    private var positions = 0
    private var started = false

    package init(model: HaruModel, seed: UInt64, temperature: Float) {
        self.model = model
        self.temperature = temperature
        encoder = model.codec.encoderState
        decoder = model.codec.decoderState
        caches = model.makeCache()
        key = MLXRandom.key(seed)
        stream = HaruTokenStream(pad: model.config.padTokenId, eop: model.config.eopTokenId, eos: model.config.lm.eosTokenIds)
    }

    package func start(
        card: [Int], reference: [Float], speakFirst: Bool, decode: ([Int]) -> String
    ) throws -> (audio: [Float], tokenID: Int, text: String) {
        guard !started else { throw HaruError.sessionAlreadyStarted }
        let frame = HaruConfiguration.frameSamples
        let reference = reference.suffix(180 * HaruConfiguration.sampleRate)
        let referenceFrames = (reference.count + frame - 1) / frame
        let opening = speakFirst || (card.isEmpty && reference.isEmpty)
        let count = card.count + referenceFrames + (opening ? (speakFirst ? 2 : 1) : 0)
        guard count <= model.config.lm.textConfig.maxPositionEmbeddings else { throw HaruError.contextLimit }

        // All prefill state is request-local. A cancelled start cannot reuse partially advanced caches.
        try Task.checkCancellation()
        started = true
        let referenceKey = nextKey()
        var hidden: MLXArray?
        for start in stride(from: 0, to: card.count, by: 64) {
            try Task.checkCancellation()
            let last = forward(Array(card[start..<min(card.count, start + 64)]))
            eval(last)
            hidden = last
        }
        var referenceLatents: MLXArray?
        if !reference.isEmpty {
            let latents = model.codec.sample(try encodeReference(reference), key: referenceKey)
            referenceLatents = latents
            var tokens = Array(repeating: model.config.padTokenId, count: referenceFrames)
            tokens[referenceFrames - 1] = model.config.eopTokenId
            for start in stride(from: 0, to: referenceFrames, by: 64) {
                try Task.checkCancellation()
                let span = start..<min(referenceFrames, start + 64)
                let last = forward(Array(tokens[span]), modelAudio: latents[span])
                eval(last)
                hidden = last
            }
        }
        if opening {
            let silenceSamples = Array(repeating: Float(0), count: frame * 16)
            let silence = try encodeReference(silenceSamples[...])[15]
            let tone = referenceLatents.flatMap { latents in
                quietestFrame(in: reference).map { latents[$0] }
            } ?? silence
            let tokens = speakFirst ? [model.config.padTokenId, model.config.eopTokenId] : [model.config.padTokenId]
            let shape = [tokens.count, model.config.latentDim]
            hidden = forward(tokens, modelAudio: broadcast(tone, to: shape), userAudio: broadcast(silence, to: shape))
        }
        guard let hidden else { throw HaruError.unsupportedConfiguration }
        conditioning = model.head.conditioning(dtype: model.dtype)
        eval(conditioning)
        return emit(hidden, forceText: speakFirst, decode: decode)
    }

    package func step(_ audio: [Float], decode: ([Int]) -> String) throws -> (audio: [Float], tokenID: Int, text: String) {
        guard let previous else { throw HaruError.sessionNotStarted }
        guard audio.count == HaruConfiguration.frameSamples else { throw HaruError.invalidFrame(audio.count) }
        guard positions < model.config.lm.textConfig.maxPositionEmbeddings else { throw HaruError.contextLimit }
        try Task.checkCancellation()
        let mean = model.codec.encode(MLXArray(audio), state: &encoder)
        let user = model.codec.sample(mean, key: nextKey())
        let input = previous.token.reshaped(1, 1)
        let hidden = forward(input, modelAudio: previous.audio, userAudio: user)
        return emit(hidden, forceText: false, decode: decode)
    }

    private func encodeReference(_ audio: ArraySlice<Float>) throws -> MLXArray {
        var state = model.codec.encoderState
        let chunk = 8 * HaruConfiguration.frameSamples
        var means: [MLXArray] = []
        for offset in stride(from: audio.startIndex, to: audio.endIndex, by: chunk) {
            try Task.checkCancellation()
            let samples = MLXArray(Array(audio[offset..<min(audio.endIndex, offset + chunk)]))
            let mean = model.codec.encode(samples, state: &state)
            eval(mean, state)
            means.append(mean)
        }
        return concatenated(means)
    }

    private func quietestFrame(in audio: ArraySlice<Float>) -> Int? {
        let frame = HaruConfiguration.frameSamples
        return (0..<(audio.count / frame)).min { first, second in
            func energy(_ index: Int) -> Float {
                let start = audio.startIndex + index * frame
                return audio[start..<(start + frame)].reduce(0) { $0 + $1 * $1 }
            }
            return energy(first) < energy(second)
        }
    }

    private func forward(_ tokens: [Int], modelAudio: MLXArray? = nil, userAudio: MLXArray? = nil) -> MLXArray {
        forward(MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count), modelAudio: modelAudio, userAudio: userAudio)
    }

    private func forward(_ tokens: MLXArray, modelAudio: MLXArray?, userAudio: MLXArray?) -> MLXArray {
        let embeddings = model.embeddings(
            tokens, model: modelAudio?.expandedDimensions(axis: 0), user: userAudio?.expandedDimensions(axis: 0)
        )
        positions += tokens.dim(1)
        return backboneOperations.withScope {
            model.lm.decisionHiddenStates(tokens, cache: caches, inputEmbeddings: embeddings)[0..., -1]
        }
    }

    private func nextKey() -> MLXArray {
        let (next, current) = MLXRandom.split(key: key)
        key = next
        return current
    }

    private func emit(
        _ hidden: MLXArray, forceText: Bool, decode: ([Int]) -> String
    ) -> (audio: [Float], tokenID: Int, text: String) {
        let logits = model.lm.logits(from: hidden).reshaped(-1).asType(.float32)
        stream.penalizeRepeats(in: logits)
        let controls = MLXArray([Int32(model.config.padTokenId), Int32(model.config.eopTokenId)])
        let rows = model.lm.embeddings(for: controls).asType(.float32)
        logits[controls] = matmul(hidden.asType(.float32), rows.T).reshaped(-1)
        if forceText {
            logits[MLXArray(stream.exempt.sorted().map(Int32.init))] = MLXArray(-Float.infinity)
        }
        let token = HaruTokenStream.sample(logits, temperature: temperature, key: nextKey())
        let noise = MLXRandom.normal([1, model.config.latentDim], key: nextKey()).asType(model.dtype)
        let latent = model.rawLatent(model.head.sample(hidden: hidden, noise: noise, conditioning: conditioning))
        let pcm = clip(model.codec.decode(latent, state: &decoder), min: -1, max: 1)
        eval(token, latent, pcm, encoder, decoder)
        previous = (token, latent)
        let tokenID = token.item(Int.self)
        return (pcm.asArray(Float.self), tokenID, stream.append(tokenID, decode: decode))
    }
}
