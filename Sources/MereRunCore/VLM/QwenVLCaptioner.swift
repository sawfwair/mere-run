import Foundation
import MLX
import MLXNN

/// Minimal Swift/MLX autocaptioner using a Qwen-VL style text model + vision tower.
///
/// This is intentionally lightweight: it loads only the multimodal encoder pieces needed to
/// generate short captions for LoRA training datasets (one caption per image).
public final class QwenVLCaptioner: @unchecked Sendable {
    private static let debugLogits: Bool = {
        guard let raw = ProcessInfo.processInfo.environment["MERERUN_VLM_DEBUG_LOGITS"]?.lowercased() else {
            return false
        }
        return raw == "1" || raw == "true" || raw == "yes"
    }()

    public struct ModelConfig: Sendable, Hashable {
        public var maxNewTokens: Int
        public var temperature: Float
        public var topP: Float

        public init(
            maxNewTokens: Int = 128,
            temperature: Float = 0.2,
            topP: Float = 0.9
        ) {
            self.maxNewTokens = maxNewTokens
            self.temperature = temperature
            self.topP = topP
        }
    }

    public enum Error: Swift.Error {
        case missingTokenizer(URL)
        case missingTextEncoderConfig(URL)
        case missingTextEncoderWeights(URL)
        case missingVisionWeights(URL)
        case imageLoadFailed(URL)
        case tokenizerMissingVisionTokens
        case captionModelMissingVisionTokens
        case quantizedWeightsNotApplied
    }

    private let rootURL: URL
    private let tokenizer: QwenTokenizer
    private let encoder: QwenVLEncoder

    public init(modelRoot: URL) throws {
        self.rootURL = modelRoot.standardizedFileURL

        let tokenizerURL = rootURL.appendingPathComponent("tokenizer", isDirectory: true)
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw Error.missingTokenizer(tokenizerURL)
        }
        self.tokenizer = try QwenTokenizer.load(from: tokenizerURL, maxLengthOverride: 4096)

        let textEncoderConfigURL = rootURL
            .appendingPathComponent("text_encoder", isDirectory: true)
            .appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: textEncoderConfigURL.path) else {
            throw Error.missingTextEncoderConfig(textEncoderConfigURL)
        }
        let configData = try Data(contentsOf: textEncoderConfigURL)
        let cfg = try JSONDecoder().decode(QwenVLTextEncoderConfig.self, from: configData)

        let mropeSection = cfg.ropeScaling?.mropeSection ?? (cfg.isQwen3VL ? [24, 20, 20] : nil)
        let mropeInterleaved = cfg.ropeScaling?.mropeInterleaved ?? cfg.isQwen3VL

        let textCfg = QwenTextEncoderConfiguration(
            vocabSize: cfg.vocabSize,
            hiddenSize: cfg.hiddenSize,
            numHiddenLayers: cfg.numHiddenLayers,
            numAttentionHeads: cfg.numAttentionHeads,
            numKeyValueHeads: cfg.numKeyValueHeads ?? cfg.numAttentionHeads,
            intermediateSize: cfg.intermediateSize,
            ropeTheta: cfg.ropeTheta ?? 1_000_000,
            maxPositionEmbeddings: cfg.maxPositionEmbeddings ?? 4096,
            rmsNormEps: cfg.rmsNormEps ?? 1e-6,
            promptDropIndex: 0,
            headDim: cfg.headDim ?? 128,
            mropeSection: mropeSection,
            mropeInterleaved: mropeInterleaved
        )

        let visionEmbedDim = cfg.visionConfig?.embedDim ?? cfg.visionConfig?.hiddenSize ?? 1280
        // Qwen3-VL uses intermediateSize directly; Qwen2-VL uses mlpRatio
        let visionMlpHiddenDim: Int
        if let intermediateSize = cfg.visionConfig?.intermediateSize {
            visionMlpHiddenDim = intermediateSize
        } else {
            let visionMlpRatio = cfg.visionConfig?.mlpRatio ?? 4.0
            visionMlpHiddenDim = Int((Float(visionEmbedDim) * visionMlpRatio).rounded())
        }
        let visionOutHiddenDim = cfg.visionConfig?.outHiddenSize ?? cfg.visionConfig?.hiddenSize ?? cfg.hiddenSize
        let visionInChannels = cfg.visionConfig?.inChannels ?? cfg.visionConfig?.inChans ?? 3

        // Qwen3-VL uses deepstack visual features at layers 5, 11, 17
        let deepstackIndexes = cfg.visionConfig?.deepstackVisualIndexes ?? (cfg.isQwen3VL ? [5, 11, 17] : [])

        let visionCfg = QwenVisionConfiguration(
            depth: cfg.visionConfig?.depth ?? 32,
            embedDim: visionEmbedDim,
            mlpHiddenDim: visionMlpHiddenDim,
            hiddenAct: .geluApproximate,
            numHeads: cfg.visionConfig?.numHeads ?? 16,
            patchSize: cfg.visionConfig?.spatialPatchSize ?? cfg.visionConfig?.patchSize ?? 14,
            temporalPatchSize: cfg.visionConfig?.temporalPatchSize ?? 2,
            spatialMergeSize: cfg.visionConfig?.spatialMergeSize ?? 2,
            inChannels: visionInChannels,
            outHiddenDim: visionOutHiddenDim,
            windowSize: cfg.visionConfig?.windowSize ?? 112,
            fullAttentionBlockIndices: cfg.visionConfig?.fullattBlockIndexes ?? [7, 15, 23, 31],
            patchEmbedBias: cfg.isQwen3VL,
            numPositionEmbeddings: cfg.visionConfig?.numPositionEmbeddings,
            useLearnedPosEmbed: cfg.isQwen3VL,
            deepstackVisualIndexes: deepstackIndexes
        )

        self.encoder = QwenVLEncoder(
            textEncoderConfig: textCfg,
            visionConfig: visionCfg
        )

        try loadWeights(into: encoder, from: rootURL, config: cfg)
        MLX.eval(encoder)
        Memory.clearCache()
    }

    /// Test text-only generation (no vision)
    public func testTextOnly(prompt: String, maxTokens: Int = 50) throws -> String {
        let formattedPrompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"

        // Use unpadded tokens for generation to avoid RoPE issues with padding
        let promptTokens = tokenizer.encodeText(formattedPrompt)
        let inputLength = promptTokens.count
        let inputIds = MLXArray(promptTokens.map { Int32($0) }).reshaped(1, inputLength)

        let cache: [KVCache] = (0..<encoder.textEncoder.configuration.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }

        // Embed and forward
        let embeddings = encoder.textEncoder.encoder.embed(inputIds: inputIds)
        var logits = encoder.textEncoder.encoder.forwardCausal(
            embeddings: embeddings,
            cache: cache,
            lastPositionOnly: true
        )
        MLX.eval(logits)

        var tokens = promptTokens

        for _ in 0..<maxTokens {
            let lastLogits = logits[0, -1, 0...]
            let nextToken = Int(MLX.argMax(lastLogits).item(Int32.self))

            if nextToken == tokenizer.eosTokenId || nextToken == 151643 || nextToken == 151645 {
                break
            }

            tokens.append(nextToken)

            // Forward next token
            let nextInput = MLXArray([Int32(nextToken)], [1, 1])
            let nextEmbed = encoder.textEncoder.encoder.embed(inputIds: nextInput)
            logits = encoder.textEncoder.encoder.forwardCausal(embeddings: nextEmbed, cache: cache)
            MLX.eval(logits)
        }

        // Decode
        let generatedTokens = Array(tokens.dropFirst(inputLength))
        return tokenizer.decode(tokens: generatedTokens)
    }

    public func caption(
        imageURL: URL,
        prompt: String,
        config: ModelConfig = .init()
    ) throws -> String {
        guard let imageTokenId = tokenizer.imageTokenId,
              let visionStartTokenId = tokenizer.visionStartTokenId
        else {
            throw Error.tokenizerMissingVisionTokens
        }

        // Load and resize image using Qwen3-VL's max_pixels constraint
        let pixelValues: MLXArray
        do {
            pixelValues = try QwenVLImageLoader.pixelValues(
                imageURL: imageURL,
                patchSize: encoder.visionPatchSize,
                spatialMergeSize: encoder.visionSpatialMergeSize
            )
        } catch {
            throw Error.imageLoadFailed(imageURL)
        }
        let grid = [(1, pixelValues.dim(2) / encoder.visionPatchSize, pixelValues.dim(3) / encoder.visionPatchSize)]
        let imageTokenCount = max(
            1,
            grid.reduce(0) { partial, item in
                partial + item.0 * max(1, item.1 / encoder.visionSpatialMergeSize) * max(1, item.2 / encoder.visionSpatialMergeSize)
            }
        )
        let expandedImagePlaceholders = Array(
            repeating: "<|image_pad|>",
            count: imageTokenCount
        ).joined()

        // Match the HF processor by expanding the image placeholder count before tokenization.
        let formattedPrompt =
            "<|im_start|>user\n" +
            "<|vision_start|>\(expandedImagePlaceholders)<|vision_end|>" +
            prompt +
            "<|im_end|>\n" +
            "<|im_start|>assistant\n"

        let tokenIds = tokenizer.encodeText(formattedPrompt)
        let inputIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)

        // Prefill with vision embeddings replacing the expanded image-token span in place.
        let (logits, cache, finalSeqLen, ropeDelta) = try encoder.forwardPrefillForGeneration(
            inputIds: inputIds,
            imageTokenId: imageTokenId,
            visionStartTokenId: visionStartTokenId,
            pixelValues: pixelValues,
            gridThw: grid
        )

        var generatedTokens: [Int] = []
        var runningLogits = logits
        let cacheRef = cache

        if Self.debugLogits {
            Self.logTopLogits(label: "prefill", logits: runningLogits[0, -1, 0...], tokenizer: tokenizer)
        }

        let generationConfig = PromptEnhanceConfig(
            maxNewTokens: config.maxNewTokens,
            temperature: config.temperature,
            topP: config.topP,
            repetitionPenalty: 1.05,
            repetitionContextSize: 32,
            eosTokenId: tokenizer.eosTokenId ?? 151645,
            stopTokenIds: Set([tokenizer.eosTokenId ?? 151645, 151643])
        )

        for _ in 0..<config.maxNewTokens {
            let lastLogits = runningLogits[0, -1, 0...]
            let nextToken = sampleToken(
                logits: lastLogits,
                config: GenerationConfig(
                    maxTokens: generationConfig.maxNewTokens,
                    temperature: generationConfig.temperature,
                    topP: generationConfig.topP,
                    repetitionPenalty: generationConfig.repetitionPenalty,
                    repetitionContextSize: generationConfig.repetitionContextSize
                ),
                previousTokens: tokenIds + generatedTokens
            )

            if generationConfig.stopTokenIds.contains(nextToken) || nextToken == generationConfig.eosTokenId {
                break
            }

            generatedTokens.append(nextToken)

            // Position for new token = finalSeqLen + generatedCount - 1 + ropeDelta
            let nextInput = MLXArray([Int32(nextToken)]).reshaped(1, 1)
            let nextEmbed = encoder.textEncoder.encoder.embed(inputIds: nextInput)
            let posValue = Int32(finalSeqLen + generatedTokens.count - 1 + ropeDelta)
            let posIds = MLXArray([posValue]).reshaped(1, 1)
            runningLogits = encoder.textEncoder.encoder.forwardCausal(
                embeddings: nextEmbed,
                cache: cacheRef,
                positionIds: posIds
            )
            MLX.eval(runningLogits)

            if Self.debugLogits {
                Self.logTopLogits(label: "step\(generatedTokens.count)", logits: runningLogits[0, -1, 0...], tokenizer: tokenizer)
            }
        }

        let decoded = tokenizer.decode(tokens: generatedTokens)
        return decoded.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Weights

    private func loadWeights(into model: QwenVLEncoder, from root: URL, config: QwenVLTextEncoderConfig) throws {
        let textDir = root.appendingPathComponent("text_encoder", isDirectory: true)
        let visionDir = root.appendingPathComponent("vision_tower", isDirectory: true)

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4

        let textWeightsURL = textDir.appendingPathComponent("model.safetensors")
        let textIndexURL = textDir.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: textWeightsURL.path) || FileManager.default.fileExists(atPath: textIndexURL.path) else {
            throw Error.missingTextEncoderWeights(textDir)
        }

        let textIsQuantized = config.quantization != nil || Self.isQuantizedWeights(indexURL: textIndexURL)
        try Self.applyWeights(
            weightsURL: textWeightsURL,
            indexURL: textIndexURL,
            to: model,
            isQuantized: textIsQuantized,
            groupSize: groupSize,
            bits: bits
        )

        // Optional separate vision tower weights.
        // Skip if the vision_tower directory points to the same file as text_encoder
        // (unified models store both text and vision weights in a single safetensors).
        let visionWeightsURL = visionDir.appendingPathComponent("model.safetensors")
        let visionIndexURL = visionDir.appendingPathComponent("model.safetensors.index.json")
        let textWeightsResolved = textWeightsURL.resolvingSymlinksInPath()
        let visionWeightsResolved = visionWeightsURL.resolvingSymlinksInPath()
        let isSameFile = textWeightsResolved == visionWeightsResolved

        if !isSameFile,
           FileManager.default.fileExists(atPath: visionWeightsURL.path) || FileManager.default.fileExists(atPath: visionIndexURL.path) {
            let visionIsQuantized = Self.isQuantizedWeights(indexURL: visionIndexURL)
            try Self.applyWeights(
                weightsURL: visionWeightsURL,
                indexURL: visionIndexURL,
                to: model,
                isQuantized: visionIsQuantized,
                groupSize: groupSize,
                bits: bits
            )
        }

        if textIsQuantized, !(model.textEncoder.encoder.embedTokens is PreQuantizedEmbedding) {
            throw Error.quantizedWeightsNotApplied
        }
    }

    private static func isQuantizedWeights(indexURL: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data) else {
            return false
        }
        return index.weightMap.keys.contains { $0.hasSuffix(".scales") }
    }

    private static func applyWeights(
        weightsURL: URL,
        indexURL: URL,
        to model: Module,
        isQuantized: Bool,
        groupSize: Int,
        bits: Int
    ) throws {
        let mapKey: (String) -> String = { rawKey in
            Self.mapWeightKey(rawKey)
        }

        if isQuantized {
            if FileManager.default.fileExists(atPath: indexURL.path) {
                try HFSafetensorsWeightsLoader.applyQuantizedWeights(
                    indexURL: indexURL,
                    to: model,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: mapKey
                )
                return
            }

            if FileManager.default.fileExists(atPath: weightsURL.path) {
                let arrays = try MLX.loadArrays(url: weightsURL)
                try HFSafetensorsWeightsLoader.applyQuantizedWeightsFromArrays(
                    arrays,
                    to: model,
                    groupSize: groupSize,
                    bits: bits,
                    keyMapper: mapKey
                )
                return
            }
        }

        if FileManager.default.fileExists(atPath: indexURL.path) {
            try HFSafetensorsWeightsLoader.applyShardedWeights(
                indexURL: indexURL,
                to: model,
                dtype: .bfloat16,
                verify: [.shapeMismatch],
                mapper: { rawKey, value in
                    [(mapKey(rawKey), value)]
                }
            )
            return
        }

        guard FileManager.default.fileExists(atPath: weightsURL.path) else { return }
        try HFSafetensorsWeightsLoader.applyWeights(
            url: weightsURL,
            to: model,
            dtype: .bfloat16,
            verify: [.shapeMismatch],
            mapper: { rawKey, value in
                [(mapKey(rawKey), value)]
            }
        )
    }

    private static func mapWeightKey(_ rawKey: String) -> String {
        var key = rawKey

        // mlx-community/Qwen2-VL: language_model.model.* + vision_tower.*
        if key.hasPrefix("language_model.model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("language_model.model.".count))
        }
        if key.hasPrefix("vision_tower.") {
            key = "visionTower." + String(key.dropFirst("vision_tower.".count))
        }

        // Legacy Qwen-VL layout: model.* + visual.*
        if key.hasPrefix("model.") {
            key = "textEncoder.encoder." + String(key.dropFirst("model.".count))
        }
        if key.hasPrefix("visual.") {
            key = "visionTower." + String(key.dropFirst("visual.".count))
        }

        // Vision merger naming: merger.* → patch_merger.*
        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        // Qwen2-VL merger: mlp.0 → mlp_0, mlp.2 → mlp_2
        key = key.replacingOccurrences(of: ".patch_merger.mlp.0.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.mlp.2.", with: ".patch_merger.mlp_2.")
        // Qwen3-VL merger: norm → ln_q, linear_fc1 → mlp_0, linear_fc2 → mlp_2
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc1.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc2.", with: ".patch_merger.mlp_2.")

        // Qwen3-VL deepstack merger mapping (same structure as patch_merger)
        key = key.replacingOccurrences(of: ".deepstack_merger_list.", with: ".deepstack_merger_list.")
        // Map norm, linear_fc1, linear_fc2 to ln_q, mlp_0, mlp_2 for deepstack mergers
        if key.contains(".deepstack_merger_list.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")
            key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
            key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        }

        // Qwen3-VL vision MLP naming: linear_fc1 → fc1, linear_fc2 → fc2
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")

        return key
    }

    private static func logTopLogits(label: String, logits: MLXArray, tokenizer: QwenTokenizer, count: Int = 5) {
        let logitsF32 = logits.asType(.float32)
        MLX.eval(logitsF32)
        let sortedIndices = argSort(logitsF32, axis: -1)
        let topIndices = sortedIndices[(sortedIndices.dim(0) - count)...].asArray(Int32.self).map(Int.init)
        let sorted = topIndices.sorted { lhs, rhs in
            logitsF32[lhs].item(Float.self) > logitsF32[rhs].item(Float.self)
        }
        let formatted = sorted.map { token -> String in
            let logit = logitsF32[token].item(Float.self)
            let piece = tokenizer.decode(tokens: [token]).replacingOccurrences(of: "\n", with: "\\n")
            return "\(token):\(String(format: "%.3f", logit)):'\(piece)'"
        }
        print("[QwenVLCaptioner] \(label) top\(count)=\(formatted.joined(separator: ", "))")
    }
}
