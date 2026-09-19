import AudioCodecs
import Foundation
import Hub
import MLX
import MLXRandom
import MereRunAudioModels
@preconcurrency import Tokenizers

public struct AuKGenerationOptions: Sendable {
    public var variant: AuKVariant = .base
    public var duration: Double?
    public var steps = 32
    public var guidance: Float = 2
    public var seed: UInt64 = 42
    public init() {}

    public func validate(hasReference: Bool) throws {
        _ = try AuKSampling.schedule(variant: variant, steps: steps)
        if let duration { _ = try AuKSampling.frameCount(seconds: duration) }
        guard hasReference || duration != nil else {
            throw AuKError.invalid("Text-only AuK generation requires an explicit duration")
        }
        guard guidance.isFinite, guidance >= 0 else {
            throw AuKError.invalid("AuK guidance must be finite and non-negative")
        }
    }
}

/// Sequential native pipeline. Each model is released after its evaluated output is retained.
public enum AuKGenerator {
    public static let sampleRate = 24000
    public static let sourceRevision = "6943a1e967409e8c73139a7a345f2a611cfb3dd6"

    public static func generate(instruction: String, modelRoot: URL, thinkerRoot: URL,
                                reference24k: [Float]? = nil, reference16k: [Float]? = nil,
                                options: AuKGenerationOptions = .init(),
                                progress: (String) -> Void = { _ in }) throws -> [Float] {
        try options.validate(hasReference: reference24k != nil)
        guard !instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AuKError.invalid("AuK instruction cannot be empty")
        }
        guard (reference24k == nil) == (reference16k == nil) else {
            throw AuKError.invalid("AuK reference audio requires both 24 kHz and 16 kHz mono samples")
        }
        for samples in [reference24k, reference16k].compactMap({ $0 }) {
            guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else {
                throw AuKError.invalid("AuK reference audio must contain finite samples")
            }
        }
        if let reference16k, !(400...4_800_000).contains(reference16k.count) {
            throw AuKError.invalid("AuK reference audio must be between 25 ms and 300 seconds")
        }
        let config = try AuKThinkerConfiguration.load(from: thinkerRoot.appendingPathComponent("config.json"))
        let checkpointURL = modelRoot.appendingPathComponent(options.variant.checkpoint)
        let vaeURL = modelRoot.appendingPathComponent("vae.safetensors")
        progress("Encoding reference audio")
        let reference = try reference24k.map { try encodeReference($0, vaeURL: vaeURL) }
        Memory.clearCache()
        let frameCount = try options.duration.map(AuKSampling.frameCount) ?? reference?.dim(1) ?? 0
        guard frameCount > 0 else { throw AuKError.invalid("AuK reference is too short for one latent frame") }
        progress("Encoding instruction with Qwen2.5-Omni-3B")
        let conditioning = try encodeInstruction(instruction, audio: reference16k, root: thinkerRoot,
                                                config: config, checkpoint: checkpointURL)
        Memory.clearCache()
        progress("Loading AuK \(options.variant.rawValue) diffusion transformer")
        let latent = try sample(conditioning, reference: reference, frameCount: frameCount,
                                checkpoint: checkpointURL, options: options, progress: progress)
        Memory.clearCache()
        progress("Decoding waveform")
        let vae = try AuKVAE(weights: AuKCheckpoint.vae(vaeURL))
        let waveform = try vae.decode(latent)
        eval(waveform)
        let samples = waveform.asArray(Float.self)
        guard samples.allSatisfy(\.isFinite) else { throw AuKError.invalid("AuK generated non-finite audio") }
        return samples
    }

    private static func encodeReference(_ samples: [Float], vaeURL: URL) throws -> MLXArray {
        let vae = try AuKVAE(weights: AuKCheckpoint.vae(vaeURL))
        let result = try vae.encode(MLXArray(samples, [1, samples.count, 1]))
        eval(result)
        return result
    }

    /// Fixed upstream user-only chat template; audio placeholders are expanded after BPE tokenization.
    public static func prompt(instruction: String, hasAudio: Bool) -> String {
        let content = instruction + (hasAudio ? "<|audio_bos|><|AUDIO|><|audio_eos|>" : "|<no_prompt_audio>|")
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
    }

    private static func encodeInstruction(_ instruction: String, audio: [Float]?, root: URL,
                                          config: AuKThinkerConfiguration, checkpoint: URL) throws -> MLXArray {
        let (tokens, mel) = try instructionInputs(instruction, audio: audio, root: root,
                                                  audioTokenIndex: config.audioTokenIndex)
        let fusion = try AuKCheckpoint.diffusion(checkpoint).1
        let thinker = try AuKThinker(weights: AuKCheckpoint.thinker(root), configuration: config)
        let output = try thinker.encode(tokens: tokens, audio: mel, layerWeights: fusion.tensor("layer_weights"),
                                        layerScale: fusion.tensor("layer_scale"))
        eval(output)
        return output
    }

    static func instructionInputs(_ instruction: String, audio: [Float]?, root: URL,
                                  audioTokenIndex: Int) throws -> ([Int], MLXArray?) {
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: HubApi.shared.configuration(fileURL: root.appendingPathComponent("tokenizer_config.json")),
            tokenizerData: HubApi.shared.configuration(fileURL: root.appendingPathComponent("tokenizer.json")))
        var tokens = tokenizer.encode(text: prompt(instruction: instruction, hasAudio: audio != nil), addSpecialTokens: false)
        let mel = audio.map { samples in
            // HF pads waveforms with zeros before STFT, then masks to ceil(samples / hop).
            // A full FFT of trailing zeros reproduces that boundary without processing 300 s of silence.
            let features = MelSpectrogram().extract(from: samples + Array(repeating: 0, count: 400))
            return features[0..., 0..., ..<((samples.count + 159) / 160)].transposed(0, 2, 1)
        }
        if let mel {
            let count = (mel.dim(1) + 1) / 2 / 2
            guard tokens.filter({ $0 == audioTokenIndex }).count == 1, count > 0 else {
                throw AuKError.invalid("AuK tokenizer must produce exactly one reference audio placeholder")
            }
            tokens = tokens.flatMap { $0 == audioTokenIndex ? Array(repeating: $0, count: count) : [$0] }
        }
        guard tokens.count <= 32768 else { throw AuKError.invalid("AuK instruction exceeds the encoder context") }
        return (tokens, mel)
    }

    private static func sample(_ text: MLXArray, reference: MLXArray?, frameCount: Int, checkpoint: URL,
                                options: AuKGenerationOptions, progress: (String) -> Void) throws -> MLXArray {
        let (weights, fusion) = try AuKCheckpoint.diffusion(checkpoint)
        let dit = try AuKDiT(weights: weights, frequencies: fusion.tensor("inv_freq").asArray(Float.self))
        let schedule = try AuKSampling.schedule(variant: options.variant, steps: options.steps)
        let initial = MLXRandom.normal([1, frameCount, 64], key: MLXRandom.key(options.seed))
        let output = try AuKSampling.integrate(initial: initial, schedule: schedule) { state, time in
            try dit.velocity(latent: state, text: text, time: time, reference: reference,
                             guidance: options.variant == .flash ? 0 : options.guidance)
        } progress: { step, total in progress("AuK diffusion: \(step)/\(total)") }
        guard all(isFinite(output)).item(Bool.self) else { throw AuKError.invalid("AuK generated non-finite latents") }
        return output
    }
}
