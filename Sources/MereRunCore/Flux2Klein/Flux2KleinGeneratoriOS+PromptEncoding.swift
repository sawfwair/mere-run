import Foundation
import MLX
import MLXRandom
import MLXNN
import ImageIO

extension Flux2KleinGeneratoriOS {

    // MARK: - Prompt Encoding

    /// Load text encoder, encode prompts, save to disk, fully unload
    /// Returns true if negative embeddings were saved
    func encodePromptsAndSave(
        prompt: String,
        negativePrompt: String?,
        guidanceScale: Float,
        tokenizer: QwenTokenizer,
        textEncoderDirURL: URL,
        quantization: ModelWeightsLoader.QuantizationParams?,
        outputURL: URL,
        negOutputURL: URL,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> Bool {
        // Load text encoder
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 0, totalSteps: 1))
        let textEncoderConfig = try loadTextEncoderConfig(from: textEncoderDirURL)
        var textEncoder: QwenTextEncoder? = QwenTextEncoder(configuration: textEncoderConfig)

        try await loadTextEncoderWeights(from: textEncoderDirURL, to: textEncoder!, quantization: quantization)
        progressHandler?(GenerationProgress(stage: .loadingEncoder, stepIndex: 1, totalSteps: 1))

        // Encode prompt
        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
        let (promptEmbeds, _) = try encodePrompt(
            prompt: prompt,
            tokenizer: tokenizer,
            textEncoder: textEncoder!
        )
        MLX.eval(promptEmbeds)

        // Save to disk immediately
        try MLX.save(array: promptEmbeds, url: outputURL)

        // Encode negative prompt if CFG enabled
        let hasNegativePrompt = negativePrompt.map {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        } ?? false
        let useCFG = guidanceScale > 1.0 && hasNegativePrompt

        var hasNegative = false
        if useCFG, let negPrompt = negativePrompt {
            let (negEmbeds, _) = try encodePrompt(
                prompt: negPrompt,
                tokenizer: tokenizer,
                textEncoder: textEncoder!
            )
            MLX.eval(negEmbeds)
            try MLX.save(array: negEmbeds, url: negOutputURL)
            hasNegative = true
        }

        progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 1, totalSteps: 1))

        // Fully unload text encoder - it will be deallocated when function returns
        Stream.gpu.synchronize()
        textEncoder = nil

        return hasNegative
    }

    /// Encode a single prompt using tokenizer and text encoder
    private func encodePrompt(
        prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder
    ) throws -> (MLXArray, MLXArray) {
        let promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let encoded = tokenizer.encodePlain(prompts: [promptWithTemplate], maxLength: 512)

        let result = textEncoder.forwardWithHiddenStates(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        guard let hiddenStates = result.hiddenStates, hiddenStates.count >= 28 else {
            throw Flux2Error.insufficientHiddenStates
        }

        let h1 = hiddenStates[9]
        let h2 = hiddenStates[18]
        let h3 = hiddenStates[27]
        let promptEmbeds = concatenated([h1, h2, h3], axis: -1)

        let lastHidden = result.lastHiddenState
        let pooledEmbeds = lastHidden.mean(axis: 1)

        return (promptEmbeds, pooledEmbeds)
    }


}
