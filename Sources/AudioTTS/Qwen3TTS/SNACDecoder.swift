import Foundation
import MLX
import MLXNN

// NOTE: This SNAC decoder implementation is NOT USED by Qwen3-TTS models.
// The Qwen3-TTS-12Hz models use Qwen3TTSTokenizerV2 (SplitResidualVectorQuantizer)
// which has a different architecture (16 quantizers, transformer decoder, etc.).
// This file is kept for potential use with other models that use SNAC.

// MARK: - SNAC Decoder

/// SNAC (Neural Audio Codec) decoder for converting quantized codes to audio waveform
public final class SNACDecoder: Module {
    // Decoder architecture
    @ModuleInfo(key: "decoder") var decoder: SNACDecoderNetwork
    @ModuleInfo(key: "quantizer") var quantizer: ResidualVectorQuantize

    // Configuration
    let latentDim: Int
    let sampleRate: Int

    public init(
        latentDim: Int = 1024,
        codebookSize: Int = 4096,
        codebookDim: Int = 8,
        numQuantizers: Int = 3,
        sampleRate: Int = 24000
    ) {
        self.latentDim = latentDim
        self.sampleRate = sampleRate

        self._quantizer.wrappedValue = ResidualVectorQuantize(
            inputDim: latentDim,
            codebookSize: codebookSize,
            codebookDim: codebookDim,
            numQuantizers: numQuantizers
        )

        self._decoder.wrappedValue = SNACDecoderNetwork(latentDim: latentDim)
    }

    /// Decode audio tokens to waveform
    /// - Parameter codes: Hierarchical audio codes from TTS model
    /// - Returns: Audio waveform [1, samples]
    public func decode(codes: [[Int]]) -> MLXArray {
        // Reconstruct latent from codes
        let latent = quantizer.fromCodes(codes)

        // Decode latent to audio
        return decoder(latent)
    }

    /// Decode from MLXArray codes
    public func decode(codeArrays: [MLXArray]) -> MLXArray {
        let latent = quantizer.fromCodeArrays(codeArrays)
        return decoder(latent)
    }
}

// MARK: - SNAC Decoder Network

/// The actual decoder network that converts latents to audio
final class SNACDecoderNetwork: Module {
    @ModuleInfo(key: "model") var model: Sequential

    init(latentDim: Int = 1024) {
        // SNAC decoder architecture:
        // 1. Local attention
        // 2. Initial conv
        // 3. Decoder blocks with upsampling (strides: 8, 4, 2)
        // 4. Final conv to audio

        let channels = [512, 256, 128, 64]
        let strides = [8, 4, 2]

        var layers: [Module] = []

        // Local attention on latent
        layers.append(LocalMHA(dim: latentDim, heads: 8, windowSize: 32))

        // Initial projection from latent to first channel size
        layers.append(WNConv1d(inputChannels: latentDim, outputChannels: channels[0], kernelSize: 7, padding: 3))

        // Decoder blocks (upsample)
        for i in 0..<strides.count {
            layers.append(DecoderBlock(
                inputChannels: channels[i],
                outputChannels: channels[i + 1],
                stride: strides[i]
            ))
        }

        // Final activation and conv to audio
        layers.append(Snake1d(channels: channels.last!))
        layers.append(WNConv1d(inputChannels: channels.last!, outputChannels: 1, kernelSize: 7, padding: 3))
        layers.append(Tanh())

        self._model.wrappedValue = Sequential(layers)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, T, latentDim] -> [B, samples]
        let output = model(x)
        // Squeeze channel dimension
        return output.squeezed(axis: -1)
    }
}

// MARK: - Weight Loading

extension SNACDecoder {
    /// Load weights from safetensors file
    public func loadWeights(from url: URL) throws {
        let weights = try MLX.loadArrays(url: url)

        // Map weight keys from Python format to Swift format
        var mappedWeights: [(String, MLXArray)] = []

        for (key, value) in weights {
            let mappedKey = mapWeightKey(key)
            mappedWeights.append((mappedKey, value))
        }

        try update(parameters: ModuleParameters.unflattened(mappedWeights), verify: .none)
    }

    private func mapWeightKey(_ key: String) -> String {
        // Convert Python weight keys to MLX Swift format
        // The keys should map directly in most cases
        key
    }
}

// MARK: - Audio Token Parser

public enum SNACAudioTokenParser {
    /// Parse audio tokens from TTS model into hierarchical codes for SNAC decoder
    /// Each group of 7 tokens encodes hierarchical SNAC codes
    /// - Parameter tokens: Raw audio token IDs from TTS model (already offset by audioTokensStart)
    /// - Returns: 3 layers of codes for SNAC decoder
    public static func parseAudioTokens(_ tokens: [Int]) -> [[Int]] {
        var layer1: [Int] = []  // stride 8 (base layer)
        var layer2: [Int] = []  // stride 4
        var layer3: [Int] = []  // stride 2

        // Process in groups of 7
        var i = 0
        while i + 6 < tokens.count {
            // Token layout in each group of 7:
            // [0]: layer1 code (offset 0)
            // [1]: layer2 code (offset 4096)
            // [2]: layer3 code (offset 8192)
            // [3]: layer3 code (offset 12288)
            // [4]: layer2 code (offset 16384)
            // [5]: layer3 code (offset 20480)
            // [6]: layer3 code (offset 24576)

            layer1.append(tokens[i])
            layer2.append(tokens[i + 1] - 4096)
            layer3.append(tokens[i + 2] - 8192)
            layer3.append(tokens[i + 3] - 12288)
            layer2.append(tokens[i + 4] - 16384)
            layer3.append(tokens[i + 5] - 20480)
            layer3.append(tokens[i + 6] - 24576)

            i += 7
        }

        return [layer1, layer2, layer3]
    }

    /// Alternative parsing that handles codes directly without offsets
    /// Use this if tokens are already in their respective codebook ranges
    public static func parseRawCodes(layer1: [Int], layer2: [Int], layer3: [Int]) -> [[Int]] {
        [layer1, layer2, layer3]
    }
}

// MARK: - Audio Utilities

public enum SNACAudioWriter {
    /// Write audio samples to WAV file
    /// - Parameters:
    ///   - samples: Audio samples in range [-1, 1]
    ///   - url: Output URL for WAV file
    ///   - sampleRate: Sample rate (default 24000 for SNAC)
    public static func writeWAV(_ samples: MLXArray, to url: URL, sampleRate: Int = 24000) throws {
        // Evaluate and get samples
        MLX.eval(samples)
        let flatSamples = samples.reshaped(-1).asArray(Float.self)

        // Convert to 16-bit PCM
        let int16Samples = flatSamples.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * 32767.0)
        }

        // Build WAV file
        var data = Data()

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        let fileSize = UInt32(36 + int16Samples.count * 2)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // Format chunk
        data.append(contentsOf: "fmt ".utf8)
        let fmtSize = UInt32(16)
        data.append(contentsOf: withUnsafeBytes(of: fmtSize.littleEndian) { Array($0) })
        let audioFormat = UInt16(1)  // PCM
        data.append(contentsOf: withUnsafeBytes(of: audioFormat.littleEndian) { Array($0) })
        let numChannels = UInt16(1)  // Mono
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        let sampleRateU32 = UInt32(sampleRate)
        data.append(contentsOf: withUnsafeBytes(of: sampleRateU32.littleEndian) { Array($0) })
        let byteRate = UInt32(sampleRate * 2)  // sampleRate * numChannels * bytesPerSample
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        let blockAlign = UInt16(2)  // numChannels * bytesPerSample
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        let bitsPerSample = UInt16(16)
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

        // Data chunk
        data.append(contentsOf: "data".utf8)
        let dataSize = UInt32(int16Samples.count * 2)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        // Audio samples
        for sample in int16Samples {
            data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
        }

        try data.write(to: url)
    }
}
