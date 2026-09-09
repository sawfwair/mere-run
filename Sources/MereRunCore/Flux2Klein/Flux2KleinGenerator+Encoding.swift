import Foundation
import MediaIO
import MLX
import MLXNN

extension Flux2KleinGenerator {
    // MARK: - Prompt Encoding

    func encodePrompt(
        prompt: String,
        tokenizer: QwenTokenizer,
        textEncoder: QwenTextEncoder,
        debugLog: ((String) -> Void)?
    ) throws -> (MLXArray, MLXArray) {
        // Klein uses Qwen3 conditioning; FLUX.2-dev uses Mistral Small 3.2.
        //
        // Important: we must NOT use `QwenTokenizer.encode(...)` here because it applies the
        // Z-Image system prefix/suffix (image-editing instructions), which breaks FLUX.2 prompts.
        let usesMistral = !textEncoder.configuration.useQKNorm
        let promptWithTemplate: String
        if usesMistral {
            let cleanedPrompt = prompt.replacingOccurrences(of: "[IMG]", with: "")
            let systemPrompt = """
            You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
            attribution and actions without speculation.
            """
            promptWithTemplate = "<s>[SYSTEM_PROMPT]\(systemPrompt)[/SYSTEM_PROMPT]"
                + "[INST]\(cleanedPrompt)[/INST]"
        } else {
            promptWithTemplate = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }
        let encoded = tokenizer.encodePlain(prompts: [promptWithTemplate], maxLength: 512)

        if let debugLog {
            let tokens = encoded.inputIds[0].asArray(Int32.self)
            debugLog("Tokenized \(tokens.count) tokens: \(tokens.prefix(20))...")
        }

        // Diffusers selects three architecture-specific intermediate hidden states.
        let result = textEncoder.forwardWithHiddenStates(
            inputIds: encoded.inputIds,
            attentionMask: encoded.attentionMask
        )

        let hiddenStateIndices = usesMistral ? [10, 20, 30] : [9, 18, 27]
        guard let hiddenStates = result.hiddenStates,
              let lastIndex = hiddenStateIndices.last,
              hiddenStates.count > lastIndex else {
            throw Flux2Error.insufficientHiddenStates
        }

        // Diffusers concatenates the raw states without applying RMSNorm.
        if let debugLog {
            debugLog("=== Hidden States Debug ===")
            debugLog("Total hidden states: \(hiddenStates.count)")
        }

        let h1 = hiddenStates[hiddenStateIndices[0]]
        let h2 = hiddenStates[hiddenStateIndices[1]]
        let h3 = hiddenStates[hiddenStateIndices[2]]

        if let debugLog {
            debugLog("h1 (idx \(hiddenStateIndices[0])): shape=\(h1.shape), mean=\(h1.mean().item(Float.self)), std=\(sqrt((h1 * h1).mean().item(Float.self)))")
            debugLog("h2 (idx \(hiddenStateIndices[1])): shape=\(h2.shape), mean=\(h2.mean().item(Float.self)), std=\(sqrt((h2 * h2).mean().item(Float.self)))")
            debugLog("h3 (idx \(hiddenStateIndices[2])): shape=\(h3.shape), mean=\(h3.mean().item(Float.self)), std=\(sqrt((h3 * h3).mean().item(Float.self)))")
        }

        // Concatenate the three states along the feature dimension.
        let promptEmbeds = concatenated([h1, h2, h3], axis: -1)

        if let debugLog {
            let mean = promptEmbeds.mean().item(Float.self)
            let std = sqrt((promptEmbeds * promptEmbeds).mean().item(Float.self))
            debugLog("promptEmbeds: shape=\(promptEmbeds.shape), mean=\(mean), std=\(std)")
            debugLog("=== Expected (diffusers): mean=0.0593, std=29.47 ===")
        }

        // Pooled embedding: mean pool the last hidden state
        let lastHidden = result.lastHiddenState
        let pooledEmbeds = lastHidden.mean(axis: 1)  // [batch, hidden_size]

        return (promptEmbeds, pooledEmbeds)
    }

    // MARK: - Reference Image Encoding

    /// Encode a reference image to patchified latent space for multi-reference editing
    /// Returns latent of shape [1, seqLen, 128] ready for concatenation
    func encodeReferenceImage(
        _ url: URL,
        vae: AutoencoderKL,
        width: Int,
        height: Int,
        patchedHeight: Int,
        patchedWidth: Int,
        bnMean: MLXArray,
        bnVar: MLXArray
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Flux2Error.referenceImageNotFound(url)
        }

        let image: MediaImage
        do {
            image = try MediaImageIO.decode(url)
        } catch {
            throw Flux2Error.referenceImageDecodeFailed(url)
        }

        // 1. Load and resize image to target dimensions
        let resizedArray = try QwenImageIO.resizedPixelArray(
            from: image,
            width: width,
            height: height,
            addBatchDimension: true,
            dtype: .float32
        )

        // 2. Normalize for VAE encoder
        let normalized = QwenImageIO.normalizeForEncoder(resizedArray)

        // 3. Encode through VAE
        let encoded = vae.encode(normalized)  // [1, latentChannels*2, H/8, W/8] (mean+var)

        // 4. Extract mean (clean latent) - first half of channels
        let latentChannels = vae.configuration.latentChannels  // 32 for FLUX
        let mean = encoded[0..., 0..<latentChannels, 0..., 0...]  // [1, 32, H/8, W/8]

        // 5. Apply VAE scale/shift (FLUX.2 Klein uses 1.0/0.0)
        let cleanLatent = (mean - MLXArray(vae.configuration.shiftFactor)) * MLXArray(vae.configuration.scalingFactor)

        // 6. Patchify FIRST: [1, 32, H/8, W/8] -> [1, 128, H/16, W/16]
        // This converts to 128-channel packed format before BatchNorm
        let patchified = Flux2LatentPacking.patchifyLatents(cleanLatent, height: patchedHeight * 2, width: patchedWidth * 2)

        // 7. Apply BatchNorm normalization (inverse of what we do for decode)
        // Decode does: packed * bnStd + bnMean
        // Encode should do: (packed - bnMean) / bnStd
        // mflux Flux2BatchNormStats uses eps = 0.0001 (1e-4)
        let normalizedPacked = Flux2KleinBatchNorm.normalizePackedLatents(
            patchified,
            mean: bnMean,
            variance: bnVar
        )

        // 8. Reshape to sequence format: [1, 128, H/16, W/16] -> [1, seqLen, 128]
        let seqLatent = normalizedPacked
            .transposed(0, 2, 3, 1)  // [1, H/16, W/16, 128]
            .reshaped([1, patchedHeight * patchedWidth, 128])
            .asType(.bfloat16)

        return seqLatent
    }

}
