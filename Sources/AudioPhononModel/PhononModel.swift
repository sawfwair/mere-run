import Foundation
import MLX
import MLXRandom

/// Native autoregressive flow sampling and frame-by-frame Mimi synthesis.
/// Instances are confined to their owning generator actor.
public final class PhononModel {
    let config: PhononConfig
    let tokenizer: PhononTokenizer
    let transformer: PhononTransformer
    let flow: PhononFlow
    let mimi: PhononMimi
    let input: PhononLinear
    let outputNorm: PhononNorm
    let eos: PhononLinear
    let bos, mean, std, textEmbeddings, speakerProjection, conditions: MLXArray
    public var sampleRate: Int { config.sampleRate }

    public init(directory: URL) throws {
        config = try PhononConfig.load(from: directory.appendingPathComponent("config.json"))
        tokenizer = try PhononTokenizer(url: directory.appendingPathComponent("tokenizer.json"))
        guard tokenizer.vocabularySize == config.flowLm.nBins else {
            throw PhononError.invalidAsset("tokenizer vocabulary does not match the checkpoint")
        }
        let arrays = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        let weights = PhononWeights(arrays: arrays)
        let cfg = config.flowLm
        transformer = try PhononTransformer(weights, "flow_lm.transformer", width: cfg.dModel,
                                            heads: cfg.numHeads, count: cfg.numLayers,
                                            feedforward: cfg.dimFeedforward, period: cfg.maxPeriod)
        flow = try PhononFlow(weights, config: cfg)
        mimi = try PhononMimi(weights, config: config.mimi)
        input = try PhononLinear(weights, "flow_lm.input_linear", cfg.ldim, cfg.dModel)
        outputNorm = try PhononNorm(weights, "flow_lm.out_norm", cfg.dModel)
        eos = try PhononLinear(weights, "flow_lm.out_eos", cfg.dModel, 1, bias: true)
        bos = try weights.tensor("flow_lm.bos_emb", [cfg.ldim])
        mean = try weights.tensor("flow_lm.emb_mean", [cfg.ldim])
        std = try weights.tensor("flow_lm.emb_std", [cfg.ldim])
        textEmbeddings = try weights.tensor("flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight",
                                           [cfg.nBins + 1, cfg.dModel])
        speakerProjection = try weights.tensor("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight",
                                               [cfg.dModel, cfg.dModel])
        conditions = try Self.loadConditions(weights, config: config)
    }

    private static func loadConditions(_ weights: PhononWeights, config: PhononConfig) throws -> MLXArray {
        let prefix = "flow_lm.condition_provider.conditioners."
        guard let speakers = config.conditioners.first(where: { $0.name == "num_speakers" })?.lut,
              let duration = config.conditioners.first(where: { $0.name == "duration_delta" })?.continuous,
              let speakerID = speakers.possibleValues.firstIndex(of: "1") else {
            throw PhononError.invalidConfiguration("missing default conditioning")
        }
        let embedding = try weights.tensor(prefix + "num_speakers.embed.weight", [speakers.nBins + 1, speakers.dim])
        let projectSpeakers = try PhononLinear(weights, prefix + "num_speakers.output_proj", speakers.dim, config.flowLm.dModel)
        let projectDuration = try PhononLinear(weights, prefix + "duration_delta.output_proj", duration.dim, config.flowLm.dModel)
        // duration_delta=0 gives cos(0)=1 and sin(0)=0, independently of scale/period.
        let durationEmbedding = MLXArray([Float](repeating: 1, count: duration.dim / 2)
                                       + [Float](repeating: 0, count: duration.dim / 2))
        return (projectSpeakers(embedding[speakerID]) + projectDuration(durationEmbedding)).reshaped(1, 1, config.flowLm.dModel)
    }

    public func loadVoice(url: URL) throws -> MLXArray {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        if let model = metadata["model_ext"], !config.embeddingsExt.contains(model) {
            throw PhononError.invalidAsset("voice model_ext \(model) is incompatible with this checkpoint")
        }
        guard arrays.count == 1, let pair = arrays.first else {
            throw PhononError.invalidAsset("a voice must contain exactly one embedding or speaker_wavs tensor")
        }
        var voice = pair.value.asType(.float32)
        if voice.ndim == 2 { voice = voice.expandedDimensions(axis: 0) }
        guard voice.ndim == 3, voice.dim(0) == 1 else { throw PhononError.invalidAsset("voice must have batch size one and rank two or three") }
        if pair.key == "speaker_wavs" {
            guard voice.dim(1) == config.flowLm.dModel else { throw PhononError.invalidAsset("speaker latent width mismatch") }
            voice = matmul(voice.transposed(0, 2, 1), speakerProjection.T)
        } else {
            guard pair.key == "emb" || pair.key == "audio_prompt" else { throw PhononError.invalidAsset("unknown voice tensor \(pair.key)") }
        }
        guard voice.dim(2) == config.flowLm.dModel, (1...512).contains(voice.dim(1)) else {
            throw PhononError.invalidAsset("voice must contain between one and 512 projected frames")
        }
        eval(voice)
        guard all(isFinite(voice)).item(Bool.self) else { throw PhononError.invalidAsset("non-finite voice embedding") }
        return voice
    }

    /// Emits one 80 ms frame at a time. A fresh KV/codec state is used for each sentence chunk.
    /// A deterministic seed is local to this request and does not change MLX's global RNG.
    public func generate(text: String, voice: MLXArray, temperature: Float = 0.8, seed: UInt64? = nil,
                         onFrame: ([Float]) throws -> Void) throws {
        guard temperature.isFinite, (0...2).contains(temperature) else {
            throw PhononError.invalidRequest("Phonon requires a temperature between zero and two.")
        }
        var randomKey = seed.map { MLXRandom.key($0) } ?? MLXRandom.key(UInt64.random(in: .min ... .max))
        try generate(text: text, voice: voice, noiseSampler: {
            let keys = MLXRandom.split(key: randomKey)
            randomKey = keys.0
            return MLXRandom.normal([1, self.config.flowLm.ldim], key: keys.1) * sqrt(temperature)
        }, onFrame: onFrame)
    }

    /// Injects recorded noise for cross-runtime numerical qualification. The product sampler
    /// and this path share all model computation, chunking, and stopping behavior.
    func generate(text: String, voice: MLXArray, noiseSampler: () -> MLXArray,
                  onStep: ((MLXArray, MLXArray) -> Void)? = nil,
                  onFrame: ([Float]) throws -> Void) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw PhononError.invalidRequest("Phonon requires nonempty text.")
        }
        guard voice.ndim == 3, voice.dim(0) == 1, voice.dim(2) == config.flowLm.dModel else {
            throw PhononError.invalidAsset("invalid projected voice shape")
        }
        let chunks = try tokenizer.chunks(text)
        var voiceState = transformer.makeState()
        // Gradium primes voice first, then text; sum conditions apply only to generated latents.
        let voiceOutput = transformer(voice, state: &voiceState)
        eval([voiceOutput] + voiceState.flatMap(\.arrays))
        for chunk in chunks {
            try Task.checkCancellation()
            // MLX operations replace these arrays rather than mutating the shared prefix.
            var state = voiceState, decoderState = mimi.makeState()
            let tokens = chunk.tokens
            let embeddings = textEmbeddings[MLXArray(tokens.map(Int32.init))].reshaped(1, tokens.count, config.flowLm.dModel)
            eval(transformer(embeddings, state: &state))
            var previous = bos.reshaped(1, 1, config.flowLm.ldim)
            var remaining: Int?
            let budget = Int(ceil((Double(tokens.count) / 3 + 2) * config.mimi.frameRate))
            var finished = false
            for _ in 0..<budget {
                try Task.checkCancellation()
                let backbone = outputNorm(transformer(input(previous) + conditions, state: &state)).reshaped(1, config.flowLm.dModel)
                let eosLogit = eos(backbone)
                let noise = noiseSampler()
                let latent = flow.sample(noise, condition: backbone, steps: config.lsdDecodeSteps)
                previous = latent.reshaped(1, 1, config.flowLm.ldim)
                let audio = mimi.decode(previous * std + mean, state: &decoderState)
                // Materialize retained streaming state together with the outputs once per frame.
                // Evaluating individual layers/convolutions serializes every GPU operation.
                eval([previous, eosLogit, audio] + state.flatMap(\.arrays) + decoderState.arrays)
                onStep?(previous, eosLogit)
                let samples = audio.asArray(Float.self)
                guard samples.allSatisfy(\.isFinite) else { throw PhononError.invalidAsset("non-finite generated audio") }
                try onFrame(samples)
                if remaining == nil, eosLogit.item(Float.self) > config.eosThreshold { remaining = chunk.tailFrames }
                if let count = remaining {
                    if count == 0 { finished = true; break }
                    remaining = count - 1
                }
            }
            guard finished else { throw PhononError.frameBudgetExceeded }
        }
    }
}
