// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
#if canImport(CoreAI) && os(macOS) && arch(arm64)
// Copyright 2026 Everpaper. Licensed under the Apache License, Version 2.0 (see LICENSE in this directory and THIRD_PARTY_NOTICES.md at the repository root).
//
// EmbeddingGemma 2 (text + images + audio) on Apple Core AI, macOS 27. Loads the bundle folder (config.json,
// tokenizer/, the text asset and, on request, the vision and audio assets) and embeds texts, images and audio clips
// into the same 768-d space:
//   text   -> exact Gemma tokenizer -> <bos> prompt text <eos> -> `mm_<S>` with zero soft-token inputs -> embedding
//   image  -> decode (alpha dropped, no EXIF, no colour management) -> exact resize -> patches -> `vision_2520`
//             -> n soft tokens -> sequence <bos> <|image> [n slots] <image|> <eos> -> `mm_288` -> embedding
//   audio  -> decode -> mono -> 16 kHz -> log-mel frames -> `audio_<Tb>` -> n soft tokens
//             -> sequence <bos> <|audio> [n slots] <audio|> <eos> -> `mm_<S>` -> embedding
// All outputs are L2-normalized by the graph; Matryoshka prefixes (512/256/128) are re-normalized on the host.
//
// The v1.0 API (init(bundle:compute:), tokenize, embed(ids:), embed(_:kind:dim:), prewarm) is unchanged and still
// loads the text asset only; pass `loadVision: true` to embed images and `loadAudio: true` to embed audio. `compute`
// places the text asset, `imageCompute` and `audioCompute` the image and audio encoders (default: the same as
// `compute`): the Neural Engine uses the least CPU, GPU and power, the GPU is faster (README, "Images: Neural Engine
// or GPU" and "Audio: Neural Engine or GPU").

import CoreAI
import CoreGraphics
import Foundation

/// How audio longer than the model's 30 s window is embedded.
public enum LongAudioPolicy: String, Sendable {
    /// The first 30 s only (what the upstream model does).
    case truncate
    /// The mean of the normalized embeddings of consecutive 30 s windows, re-normalized.
    case chunkMean = "chunk_mean"
}

public enum ComputePreference: String, Sendable {
    case neuralEngine = "ane"
    case gpu
    case cpu
    case `default`
}

@available(macOS 27.0, *)
public final class EmbeddingGemma2 {
    public let bundleURL: URL
    public let config: BundleConfig
    public let tokenizer: GemmaTokenizer
    /// Where the text asset (texts, and the image sequence after the image encoder) runs.
    public let compute: ComputePreference
    /// Where the image encoder (`vision_2520`) runs; equals `compute` unless set with `imageCompute:`.
    public let imageCompute: ComputePreference
    /// Where the audio encoder (`audio_<Tb>`) runs; equals `compute` unless set with `audioCompute:`.
    public let audioCompute: ComputePreference
    /// Wall-clock seconds spent in `AIModel(contentsOf:)` + `loadFunction` (specialization) for the loaded assets.
    /// About 1 s warm; the first load by an executable that has not loaded these assets before takes minutes (README).
    public let loadSeconds: Double
    public let tokenizerLoadSeconds: Double

    private let textModel: AIModel
    private let functions: [Int: InferenceFunction]   // bucket -> mm_<S>
    private let visionModel: AIModel?                 // nil unless created with loadVision: true
    private let visionFunction: InferenceFunction?
    private let audioModel: AIModel?                  // nil unless created with loadAudio: true
    private let audioFunctions: [Int: InferenceFunction]   // frame bucket -> audio_<Tb>
    public var buckets: [Int] { functions.keys.sorted() }
    /// Frame buckets of the loaded audio asset (empty unless loaded).
    public var audioBuckets: [Int] { audioFunctions.keys.sorted() }
    /// True when the vision asset is loaded and `embedImage` / `softTokens` can be used.
    public var hasVision: Bool { visionFunction != nil }
    /// True when the audio asset is loaded and `embedAudio` / `audioSoftTokens` can be used.
    public var hasAudio: Bool { !audioFunctions.isEmpty }
    private var hidden: Int { config.hidden_size }

    /// `loadVision: false` (the default, as in v1.0) loads the text asset only. `loadVision: true` also loads the
    /// vision asset (~300 MB more memory, its own first-load specialization) so that images can be embedded.
    /// `imageCompute` places the image encoder independently of the text asset (nil = same as `compute`):
    /// `.neuralEngine` ~0.37 s per image with ~2 ms CPU and ~1 ms GPU time (lowest power); `.gpu` ~0.13 s per image
    /// with ~20-35 ms CPU and ~0.1 s GPU time (about 3x the power draw). Both match the reference within float16 noise.
    /// `loadAudio: true` also loads the audio asset (~560 MB) so that audio clips can be embedded; `audioCompute`
    /// places it independently (nil = same as `compute`), see README "Audio: Neural Engine or GPU".
    public init(bundle: URL, compute: ComputePreference = .neuralEngine, loadVision: Bool = false,
                imageCompute: ComputePreference? = nil, loadAudio: Bool = false,
                audioCompute: ComputePreference? = nil) async throws {
        bundleURL = bundle
        self.compute = compute
        self.imageCompute = imageCompute ?? compute
        self.audioCompute = audioCompute ?? compute
        config = try JSONDecoder().decode(BundleConfig.self, from: Data(contentsOf: bundle.appendingPathComponent("config.json")))
        try config.validate()
        var t0 = Date()
        tokenizer = try GemmaTokenizer(contentsOf: bundle.appendingPathComponent("tokenizer/tokenizer.json"))
        tokenizerLoadSeconds = Date().timeIntervalSince(t0)
        guard tokenizer.bosID == config.bos_token_id, tokenizer.eosID == config.eos_token_id, tokenizer.padID == config.pad_token_id else {
            throw EmbedderError(description: "tokenizer special ids do not match config.json")
        }
        // Refuse an unsupported audio request before loading anything (no specialization, no temp files).
        if loadAudio {
            guard config.audio != nil else { throw EmbedderError(description: "this bundle has no audio asset (config.json \"audio\")") }
            guard self.audioCompute != .cpu else {
                throw EmbedderError(description: "audioCompute: .cpu is not supported for audio; use .neuralEngine or .gpu")
            }
        }
        t0 = Date()
        textModel = try await AIModel(contentsOf: bundle.appendingPathComponent(config.asset), options: Self.options(compute))
        var fns: [Int: InferenceFunction] = [:]
        for name in config.functions {
            guard let s = Int(name.split(separator: "_").last ?? ""), let fn = try textModel.loadFunction(named: name) else {
                throw EmbedderError(description: "function \(name) missing from \(config.asset)")
            }
            fns[s] = fn
        }
        functions = fns
        if loadVision {
            let vm = try await AIModel(contentsOf: bundle.appendingPathComponent(config.vision.asset), options: Self.options(self.imageCompute))
            guard let vf = try vm.loadFunction(named: config.vision.function) else {
                throw EmbedderError(description: "function \(config.vision.function) missing from \(config.vision.asset)")
            }
            visionModel = vm
            visionFunction = vf
        } else {
            visionModel = nil
            visionFunction = nil
        }
        if loadAudio {
            guard let ac = config.audio else { throw EmbedderError(description: "this bundle has no audio asset (config.json \"audio\")") }
            let am = try await AIModel(contentsOf: bundle.appendingPathComponent(ac.asset), options: Self.options(self.audioCompute))
            var afns: [Int: InferenceFunction] = [:]
            for name in ac.functions {
                guard let tb = Int(name.split(separator: "_").last ?? ""), let fn = try am.loadFunction(named: name) else {
                    throw EmbedderError(description: "function \(name) missing from \(ac.asset)")
                }
                afns[tb] = fn
            }
            audioModel = am
            audioFunctions = afns
        } else {
            audioModel = nil
            audioFunctions = [:]
        }
        loadSeconds = Date().timeIntervalSince(t0)
    }

    private static func options(_ compute: ComputePreference) -> SpecializationOptions {
        switch compute {
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .cpu: return SpecializationOptions.cpuOnly
        case .neuralEngine: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .default: return SpecializationOptions.default
        }
    }

    // MARK: - Tokenization

    /// `<bos>` + prompt prefix + text + `<eos>`, truncated to the largest bucket (2048 tokens: longer input is cut).
    /// `kind` is a key of `config.prompts` ("document", "search_query", ...); nil = no prefix.
    public func tokenize(_ text: String, kind: String? = "document") -> [Int32] {
        let prefix = kind.flatMap { config.prompts[$0] } ?? ""
        var ids = tokenizer.encode(prefix + text)
        let maxBody = (buckets.last ?? 32) - 2
        if ids.count > maxBody { ids = Array(ids.prefix(maxBody)) }
        return [config.bos_token_id] + ids + [config.eos_token_id]
    }

    public func bucket(forTokenCount n: Int) throws -> Int {
        guard let b = buckets.first(where: { $0 >= n }) else {
            throw EmbedderError(description: "\(n) tokens exceeds the largest bucket \(buckets.last ?? 0)")
        }
        return b
    }

    // MARK: - Text

    /// Runs `mm_<bucket>` on already-tokenized ids (with `<bos>`/`<eos>`), right-padded with `<pad>`, no soft tokens.
    /// Returns the raw 768-d output (already L2-normalized by the graph), as Float.
    public func embed(ids: [Int32], bucket: Int? = nil) async throws -> [Float] {
        try await embed(sequence: MultimodalSequence(ids: ids), bucket: bucket)
    }

    /// Embeds one text. `dim` 768 (full) or a Matryoshka size 512/256/128: the leading dims, re-normalized.
    public func embed(_ text: String, kind: String? = "document", dim: Int = 768) async throws -> [Float] {
        try normalized(try await embed(ids: tokenize(text, kind: kind)), dim: dim)
    }

    // MARK: - Images

    /// Embeds an image file. `applyEXIFOrientation: false` reproduces the upstream reference exactly; `true`
    /// embeds camera photos upright (what an app usually wants; the two vectors differ by cosine ~0.97).
    public func embedImage(url: URL, applyEXIFOrientation: Bool = false, dim: Int = 768) async throws -> [Float] {
        try await embedImage(try ImagePreprocessor.prepare(url: url, applyEXIFOrientation: applyEXIFOrientation), dim: dim)
    }

    /// Embeds a CGImage (e.g. a screenshot); its raw samples are used, alpha dropped.
    public func embedImage(cgImage: CGImage, dim: Int = 768) async throws -> [Float] {
        try await embedImage(try ImagePreprocessor.prepare(try ImageDecoder.decodeRGB(cgImage: cgImage)), dim: dim)
    }

    /// Embeds an already prepared image (`ImagePreprocessor.prepare`).
    public func embedImage(_ inputs: VisionInputs, dim: Int = 768) async throws -> [Float] {
        let soft = try await softTokens(inputs)
        return try normalized(try await embed(sequence: imageSequence(softTokens: soft)), dim: dim)
    }

    /// Soft tokens of one prepared image (`vision_2520`): `[n, 512]` as a flat array (n = inputs.softTokenCount).
    public func softTokens(_ inputs: VisionInputs) async throws -> [Float16] {
        guard let visionFunction else {
            throw EmbedderError(description: "vision asset not loaded: create the model with loadVision: true")
        }
        let N = VisionInputs.maxPatches, D = VisionInputs.patchDim
        guard inputs.patches.count == N * D else { throw EmbedderError(description: "VisionInputs has the wrong size") }
        var patches = NDArray(shape: [1, N, D], scalarType: .float16)
        var posX = NDArray(shape: [1, N], scalarType: .int32)
        var posY = NDArray(shape: [1, N], scalarType: .int32)
        var cos = NDArray(shape: [1, 1, N, VisionInputs.headDim], scalarType: .float16)
        var sin = NDArray(shape: [1, 1, N, VisionInputs.headDim], scalarType: .float16)
        var mask = NDArray(shape: [1, N], scalarType: .float16)
        fill(&patches, inputs.patches)
        fill(&posX, inputs.posX)
        fill(&posY, inputs.posY)
        fill(&cos, inputs.cos)
        fill(&sin, inputs.sin)
        fill(&mask, inputs.patchMask)
        var out = try await visionFunction.run(inputs: ["patches": patches, "pos_x": posX, "pos_y": posY, "cos": cos, "sin": sin, "patch_mask": mask])
        guard let soft = out.remove("soft_tokens")?.ndArray else { throw EmbedderError(description: "output 'soft_tokens' missing") }
        let n = inputs.softTokenCount
        var result = [Float16](repeating: 0, count: n * hidden)
        let view = soft.view(as: Float16.self)
        view.withUnsafePointer { p, _, _ in
            for i in 0..<(n * hidden) { result[i] = p[i] }
        }
        return result
    }

    /// The upstream single-image sequence: `<bos> <|image> <|image|> x n <image|> <eos>` (no prompt).
    public func imageSequence(softTokens: [Float16]) -> MultimodalSequence {
        let n = softTokens.count / hidden
        var ids = [config.bos_token_id, config.boi_token_id]
        ids.append(contentsOf: [Int32](repeating: config.image_token_id, count: n))
        ids.append(contentsOf: [config.eoi_token_id, config.eos_token_id])
        return MultimodalSequence(ids: ids, softTokens: softTokens, slots: Array(2..<(2 + n)))
    }

    // MARK: - Audio

    /// Embeds an audio file (anything AVAudioFile reads). The clip is decoded, mixed to mono, resampled to 16 kHz
    /// and, with `.truncate` (the upstream behaviour), its first 30 s are embedded; `.chunkMean` averages the
    /// embeddings of consecutive 30 s windows instead.
    public func embedAudio(url: URL, dim: Int = 768, longAudio: LongAudioPolicy = .truncate) async throws -> [Float] {
        try await embedAudio(samples16k: try AudioFrontend.loadMono16k(url: url), dim: dim, longAudio: longAudio)
    }

    /// Embeds PCM samples (any rate, any channel count: `channels[c][i]`, channel-major, every channel the same
    /// length); they are mixed to mono and resampled. Throws for a non-positive rate, channels of different lengths
    /// and arrays that look frame-major (more "channels" than samples per channel: pass `channels[c][i]`, not `[i][c]`).
    public func embedAudio(samples: [[Float]], sampleRate: Int, dim: Int = 768, longAudio: LongAudioPolicy = .truncate) async throws -> [Float] {
        guard sampleRate > 0 else { throw EmbedderError(description: "sampleRate must be positive (got \(sampleRate))") }
        guard let n = samples.first?.count, samples.allSatisfy({ $0.count == n }) else {
            throw EmbedderError(description: samples.isEmpty ? "samples has no channels" : "every channel must have the same number of samples")
        }
        guard samples.count <= max(n, 1) else {
            throw EmbedderError(description: "samples looks frame-major (\(samples.count) channels of \(n) samples): pass channel-major samples[channel][frame]")
        }
        let mono = AudioFrontend.toMono(DecodedAudio(sampleRate: sampleRate, channelCount: samples.count, channels: samples))
        return try await embedAudio(samples16k: AudioResampler.resampleTo16k(mono, rate: sampleRate), dim: dim, longAudio: longAudio)
    }

    /// Embeds 16 kHz mono samples.
    public func embedAudio(samples16k: [Float], dim: Int = 768, longAudio: LongAudioPolicy = .truncate) async throws -> [Float] {
        switch longAudio {
        case .truncate:
            return try await embedAudio(try AudioFrontend.prepare(wav16k: samples16k), dim: dim)
        case .chunkMean:
            var acc = [Float](repeating: 0, count: config.embedding_dim)
            for chunk in AudioFrontend.chunks30s(samples16k) {
                let v = try await embedAudio(try AudioFrontend.prepare(wav16k: chunk), dim: config.embedding_dim)
                for i in acc.indices { acc[i] += v[i] }
            }
            return try normalized(acc, dim: dim)
        }
    }

    /// Embeds already prepared audio inputs (`AudioFrontend.prepare`).
    public func embedAudio(_ inputs: AudioInputs, dim: Int = 768) async throws -> [Float] {
        let soft = try await audioSoftTokens(inputs)
        return try normalized(try await embed(sequence: audioSequence(softTokens: soft)), dim: dim)
    }

    /// Soft tokens of one prepared clip (`audio_<bucket>`): `[n, 512]` as a flat array (n = inputs.softTokenCount).
    public func audioSoftTokens(_ inputs: AudioInputs) async throws -> [Float16] {
        guard let fn = audioFunctions[inputs.bucket] else {
            throw EmbedderError(description: hasAudio ? "no audio function for bucket \(inputs.bucket)" : "audio asset not loaded: create the model with loadAudio: true")
        }
        let Tb = inputs.bucket, M = AudioInputs.nMels
        guard inputs.features.count == Tb * M, inputs.frameMask.count == Tb, inputs.halfMask.count == Tb / 2, inputs.tokenMask.count == Tb / 4 else {
            throw EmbedderError(description: "AudioInputs has the wrong size")
        }
        var features = NDArray(shape: [1, Tb, M], scalarType: .float16)
        var frameMask = NDArray(shape: [1, Tb], scalarType: .float16)
        var halfMask = NDArray(shape: [1, Tb / 2], scalarType: .float16)
        var tokenMask = NDArray(shape: [1, Tb / 4], scalarType: .float16)
        fill(&features, inputs.features)
        fill(&frameMask, inputs.frameMask)
        fill(&halfMask, inputs.halfMask)
        fill(&tokenMask, inputs.tokenMask)
        var out = try await fn.run(inputs: ["features": features, "frame_mask": frameMask, "half_mask": halfMask, "token_mask": tokenMask])
        guard let soft = out.remove("soft_tokens")?.ndArray else { throw EmbedderError(description: "output 'soft_tokens' missing") }
        let n = inputs.softTokenCount
        var result = [Float16](repeating: 0, count: n * hidden)
        let view = soft.view(as: Float16.self)
        view.withUnsafePointer { p, _, _ in
            for i in 0..<(n * hidden) { result[i] = p[i] }
        }
        return result
    }

    /// The upstream single-clip sequence: `<bos> <|audio> <|audio|> x n <audio|> <eos>` (no prompt).
    public func audioSequence(softTokens: [Float16]) -> MultimodalSequence {
        let n = softTokens.count / hidden
        let boa = config.boa_token_id ?? 256000, audio = config.audio_token_id ?? 258881, eoa = config.eoa_token_id ?? 258883
        var ids = [config.bos_token_id, boa]
        ids.append(contentsOf: [Int32](repeating: audio, count: n))
        ids.append(contentsOf: [eoa, config.eos_token_id])
        return MultimodalSequence(ids: ids, softTokens: softTokens, slots: Array(2..<(2 + n)))
    }

    // MARK: - Any sequence

    /// Runs `mm_<bucket>` on a sequence (ids with `<bos>`/`<eos>`, right-padded with `<pad>`; soft tokens placed at
    /// `slots`). Any mix of text and images works as long as it fits a bucket (e.g. several images, or text + image).
    /// Returns the raw 768-d output (L2-normalized by the graph), as Float.
    public func embed(sequence seq: MultimodalSequence, bucket: Int? = nil) async throws -> [Float] {
        let S = try bucket ?? self.bucket(forTokenCount: seq.ids.count)
        guard let fn = functions[S] else { throw EmbedderError(description: "no function for bucket \(S)") }
        guard seq.ids.count <= S else { throw EmbedderError(description: "\(seq.ids.count) ids do not fit bucket \(S)") }
        guard seq.slots.count * hidden == seq.softTokens.count else { throw EmbedderError(description: "soft tokens do not match slots") }
        guard seq.slots.allSatisfy({ $0 >= 0 && $0 < seq.ids.count }) else { throw EmbedderError(description: "slot outside the sequence") }
        var inputIDs = NDArray(shape: [1, S], scalarType: .int32)
        var mask = NDArray(shape: [1, S], scalarType: .float16)
        var mm = NDArray(shape: [1, S, hidden], scalarType: .float16)
        var mmMask = NDArray(shape: [1, S], scalarType: .float16)
        do {
            let v = inputIDs.mutableView(as: Int32.self)
            let m = mask.mutableView(as: Float16.self)
            let e = mm.mutableView(as: Float16.self)
            let em = mmMask.mutableView(as: Float16.self)
            let pad = config.pad_token_id
            let ids = seq.ids
            v.withUnsafeMutablePointer { p, _, _ in for i in 0..<S { p[i] = i < ids.count ? ids[i] : pad } }
            m.withUnsafeMutablePointer { p, _, _ in for i in 0..<S { p[i] = i < ids.count ? 1 : 0 } }
            let h = hidden
            e.withUnsafeMutablePointer { p, _, _ in
                for i in 0..<(S * h) { p[i] = 0 }
                for (r, slot) in seq.slots.enumerated() {
                    for j in 0..<h { p[slot * h + j] = seq.softTokens[r * h + j] }
                }
            }
            em.withUnsafeMutablePointer { p, _, _ in
                for i in 0..<S { p[i] = 0 }
                for slot in seq.slots { p[slot] = 1 }
            }
        }
        var out = try await fn.run(inputs: ["input_ids": inputIDs, "attention_mask": mask, "mm_embeds": mm, "mm_mask": mmMask])
        guard let emb = out.remove("embedding")?.ndArray else { throw EmbedderError(description: "output 'embedding' missing") }
        let dim = config.embedding_dim
        var result = [Float](repeating: 0, count: dim)
        let view = emb.view(as: Float16.self)
        view.withUnsafePointer { p, _, _ in for i in 0..<dim { result[i] = Float(p[i]) } }
        return result
    }

    private func normalized(_ v: [Float], dim: Int) throws -> [Float] {
        guard config.matryoshka_dims.contains(dim) else { throw EmbedderError(description: "dim must be one of \(config.matryoshka_dims)") }
        var r = Array(v.prefix(dim))
        let norm = max(sqrt(r.reduce(0) { $0 + $1 * $1 }), 1e-12)
        for i in r.indices { r[i] /= norm }
        return r
    }

    /// Runs every text bucket (and the vision function, if loaded) once so the first real call pays no
    /// first-call cost. Call it right after init, off the main thread. Returns the seconds it took.
    @discardableResult
    public func prewarm() async throws -> Double {
        let t0 = Date()
        for S in buckets {
            _ = try await embed(ids: [config.bos_token_id, config.eos_token_id], bucket: S)
        }
        if visionFunction != nil {
            _ = try await softTokens(ImagePreprocessor.patchify(resizedRGB: [UInt8](repeating: 128, count: 48 * 48 * 3), width: 48, height: 48))
        }
        for tb in audioBuckets {
            let n = AudioInputs.nMels
            let inputs = AudioInputs(bucket: tb, frameCount: 4, softTokenCount: 1, features: [Float16](repeating: 0, count: tb * n),
                                     frameMask: [Float16](repeating: 0, count: tb), halfMask: [Float16](repeating: 0, count: tb / 2),
                                     tokenMask: [Float16](repeating: 0, count: tb / 4))
            _ = try await audioSoftTokens(inputs)
        }
        return Date().timeIntervalSince(t0)
    }

    private func fill<T: BitwiseCopyable>(_ arr: inout NDArray, _ values: [T]) {
        let v = arr.mutableView(as: T.self)
        v.withUnsafeMutablePointer { p, _, _ in
            values.withUnsafeBufferPointer { src in p.update(from: src.baseAddress!, count: values.count) }
        }
    }

}
#endif
