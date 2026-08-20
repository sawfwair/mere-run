import Foundation
import MLX

public struct Qwen3VLEmbeddingInput: Sendable, Hashable {
    public var id: String?
    public var text: String?
    public var imageURLs: [URL]
    public var instruction: String?

    public init(
        id: String? = nil,
        text: String? = nil,
        imageURLs: [URL] = [],
        instruction: String? = nil
    ) {
        self.id = id
        self.text = text
        self.imageURLs = imageURLs
        self.instruction = instruction
    }
}

public struct Qwen3VLEmbeddingResult: Sendable, Hashable {
    public let id: String?
    public let embedding: [Float]
    public let tokenCount: Int
    public let imageCount: Int

    public init(id: String?, embedding: [Float], tokenCount: Int, imageCount: Int) {
        self.id = id
        self.embedding = embedding
        self.tokenCount = tokenCount
        self.imageCount = imageCount
    }
}

enum Qwen3VLEmbeddingPromptBuilder {
    static let defaultInstruction = "Represent the user's input."

    static func prompt(
        instruction: String?,
        text: String?,
        imageTokenCounts: [Int]
    ) -> String {
        let system = normalizedInstruction(instruction)
        let images = imageTokenCounts.map { count in
            let pads = Array(repeating: "<|image_pad|>", count: max(1, count)).joined()
            return "<|vision_start|>\(pads)<|vision_end|>"
        }.joined()
        let userText = text ?? ""
        return "<|im_start|>system\n\(system)<|im_end|>\n"
            + "<|im_start|>user\n\(images)\(userText)<|im_end|>\n"
            + "<|im_start|>assistant\n"
    }

    static func normalizedInstruction(_ instruction: String?) -> String {
        let trimmed = instruction?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return defaultInstruction }
        guard let scalar = trimmed.unicodeScalars.last else { return defaultInstruction }
        if CharacterSet.punctuationCharacters.contains(scalar) {
            return trimmed
        }
        return trimmed + "."
    }

    static func contiguousRanges(in tokenIDs: [Int], matching tokenID: Int) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start: Int?
        for (index, value) in tokenIDs.enumerated() {
            if value == tokenID {
                start = start ?? index
            } else if let lowerBound = start {
                ranges.append(lowerBound..<index)
                start = nil
            }
        }
        if let lowerBound = start {
            ranges.append(lowerBound..<tokenIDs.count)
        }
        return ranges
    }
}

public final class Qwen3VLEmbeddingModel: @unchecked Sendable {
    public enum EmbeddingError: LocalizedError {
        case missingFiles([URL])
        case noInputs
        case emptyInput(Int)
        case invalidDimensions(Int, maximum: Int)
        case invalidMaxTokens(Int)
        case invalidPixelBounds(minimum: Int, maximum: Int)
        case imageNotFound(URL)
        case imageLoadFailed(URL)
        case tokenizerMissingImageToken
        case imageTokenRangeMismatch(expected: Int, actual: Int)
        case inputExceedsMaxTokens(index: Int, actual: Int, maximum: Int)

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let urls):
                return "Missing Qwen3-VL embedding resources:\n" + urls.map(\.path).joined(separator: "\n")
            case .noInputs:
                return "At least one text or image input is required."
            case .emptyInput(let index):
                return "Embedding input \(index) has neither text nor images."
            case .invalidDimensions(let value, let maximum):
                return "dimensions must be between 1 and \(maximum) (received \(value))."
            case .invalidMaxTokens(let value):
                return "maxTokens must be positive (received \(value))."
            case .invalidPixelBounds(let minimum, let maximum):
                return "Image pixel bounds are invalid: minPixels=\(minimum), maxPixels=\(maximum)."
            case .imageNotFound(let url):
                return "Image not found: \(url.path)"
            case .imageLoadFailed(let url):
                return "Failed to decode image: \(url.path)"
            case .tokenizerMissingImageToken:
                return "Qwen3-VL tokenizer is missing the <|image_pad|> token."
            case .imageTokenRangeMismatch(let expected, let actual):
                return "Expected \(expected) image token ranges, found \(actual)."
            case .inputExceedsMaxTokens(let index, let actual, let maximum):
                return "Embedding input \(index) uses \(actual) tokens, exceeding maxTokens=\(maximum)."
            }
        }
    }

    public let resources: Qwen3VLEmbeddingResources
    public let dimensions: Int

    private let config: Qwen3VLEmbeddingRootConfig
    private let tokenizer: QwenTokenizer
    private let encoder: QwenVLEncoder

    public init(
        resources: Qwen3VLEmbeddingResources,
        fileManager: FileManager = .default
    ) throws {
        let missing = resources.validate(fileManager: fileManager)
        guard missing.isEmpty else { throw EmbeddingError.missingFiles(missing) }

        let config = try JSONDecoder().decode(
            Qwen3VLEmbeddingRootConfig.self,
            from: Data(contentsOf: resources.configURL)
        )
        let text = config.textConfig
        let vision = config.visionConfig
        let textConfiguration = QwenTextEncoderConfiguration(
            vocabSize: text.vocabSize,
            hiddenSize: text.hiddenSize,
            numHiddenLayers: text.numHiddenLayers,
            numAttentionHeads: text.numAttentionHeads,
            numKeyValueHeads: text.numKeyValueHeads ?? text.numAttentionHeads,
            intermediateSize: text.intermediateSize,
            ropeTheta: text.ropeTheta ?? 5_000_000,
            maxPositionEmbeddings: text.maxPositionEmbeddings ?? 262_144,
            rmsNormEps: text.rmsNormEps ?? 1e-6,
            promptDropIndex: 0,
            headDim: text.headDim ?? 128,
            mropeSection: text.ropeScaling?.mropeSection ?? [24, 20, 20],
            mropeInterleaved: text.ropeScaling?.mropeInterleaved ?? true
        )

        let visionEmbedDim = vision.embedDim ?? vision.hiddenSize ?? 1_024
        let visionIntermediate = vision.intermediateSize
            ?? Int((Float(visionEmbedDim) * (vision.mlpRatio ?? 4)).rounded())
        let visionConfiguration = QwenVisionConfiguration(
            depth: vision.depth ?? 24,
            embedDim: visionEmbedDim,
            mlpHiddenDim: visionIntermediate,
            hiddenAct: .geluApproximate,
            numHeads: vision.numHeads ?? 16,
            patchSize: vision.spatialPatchSize ?? vision.patchSize ?? 16,
            temporalPatchSize: vision.temporalPatchSize ?? 2,
            spatialMergeSize: vision.spatialMergeSize ?? 2,
            inChannels: vision.inChannels ?? vision.inChans ?? 3,
            outHiddenDim: vision.outHiddenSize ?? text.hiddenSize,
            windowSize: vision.windowSize ?? 112,
            fullAttentionBlockIndices: vision.fullattBlockIndexes ?? [7, 15, 23],
            patchEmbedBias: true,
            numPositionEmbeddings: vision.numPositionEmbeddings,
            useLearnedPosEmbed: true,
            deepstackVisualIndexes: vision.deepstackVisualIndexes ?? [5, 11, 17]
        )

        let encoder = QwenVLEncoder(
            textEncoderConfig: textConfiguration,
            visionConfig: visionConfiguration
        )
        try Qwen3VLEmbeddingWeights.load(
            resources: resources,
            config: config,
            into: encoder,
            fileManager: fileManager
        )

        self.resources = resources
        self.config = config
        self.tokenizer = try QwenTokenizer.load(
            from: resources.rootURL,
            maxLengthOverride: Qwen3VLEmbeddingCatalog.defaultMaxTokens
        )
        self.encoder = encoder
        self.dimensions = text.hiddenSize
        MLX.eval(encoder)
        Memory.clearCache()
    }

    public func embed(
        inputs: [Qwen3VLEmbeddingInput],
        dimensions requestedDimensions: Int? = nil,
        maxTokens: Int = Qwen3VLEmbeddingCatalog.defaultMaxTokens,
        minPixels: Int = Qwen3VLEmbeddingCatalog.defaultMinPixels,
        maxPixels: Int = Qwen3VLEmbeddingCatalog.defaultMaxPixels,
        fileManager: FileManager = .default
    ) throws -> [Qwen3VLEmbeddingResult] {
        guard !inputs.isEmpty else { throw EmbeddingError.noInputs }
        guard maxTokens > 0 else { throw EmbeddingError.invalidMaxTokens(maxTokens) }
        guard minPixels > 0, maxPixels >= minPixels else {
            throw EmbeddingError.invalidPixelBounds(minimum: minPixels, maximum: maxPixels)
        }

        let outputDimensions = requestedDimensions ?? dimensions
        guard (1...dimensions).contains(outputDimensions) else {
            throw EmbeddingError.invalidDimensions(outputDimensions, maximum: dimensions)
        }

        return try inputs.enumerated().map { index, input in
            try embedOne(
                input,
                index: index,
                dimensions: outputDimensions,
                maxTokens: min(maxTokens, config.textConfig.maxPositionEmbeddings ?? maxTokens),
                minPixels: minPixels,
                maxPixels: maxPixels,
                fileManager: fileManager
            )
        }
    }

    private func embedOne(
        _ input: Qwen3VLEmbeddingInput,
        index: Int,
        dimensions: Int,
        maxTokens: Int,
        minPixels: Int,
        maxPixels: Int,
        fileManager: FileManager
    ) throws -> Qwen3VLEmbeddingResult {
        let text = input.text?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text?.isEmpty == false || !input.imageURLs.isEmpty else {
            throw EmbeddingError.emptyInput(index)
        }

        var pixelValues: [MLXArray] = []
        var grids: [(height: Int, width: Int)] = []
        var imageTokenCounts: [Int] = []
        for imageURL in input.imageURLs {
            let url = imageURL.standardizedFileURL
            guard fileManager.fileExists(atPath: url.path) else {
                throw EmbeddingError.imageNotFound(url)
            }
            let pixels: MLXArray
            do {
                pixels = try QwenVLImageLoader.pixelValues(
                    imageURL: url,
                    minPixels: minPixels,
                    maxPixels: maxPixels,
                    patchSize: encoder.visionPatchSize,
                    spatialMergeSize: encoder.visionSpatialMergeSize
                )
            } catch {
                throw EmbeddingError.imageLoadFailed(url)
            }
            let height = pixels.dim(2) / encoder.visionPatchSize
            let width = pixels.dim(3) / encoder.visionPatchSize
            pixelValues.append(pixels)
            grids.append((height, width))
            imageTokenCounts.append(
                max(1, (height / encoder.visionSpatialMergeSize) * (width / encoder.visionSpatialMergeSize))
            )
        }

        let prompt = Qwen3VLEmbeddingPromptBuilder.prompt(
            instruction: input.instruction,
            text: text,
            imageTokenCounts: imageTokenCounts
        )
        let tokenIDs = tokenizer.encodeText(prompt)
        guard tokenIDs.count <= maxTokens else {
            throw EmbeddingError.inputExceedsMaxTokens(
                index: index,
                actual: tokenIDs.count,
                maximum: maxTokens
            )
        }

        let imageRanges: [Range<Int>]
        if input.imageURLs.isEmpty {
            imageRanges = []
        } else {
            guard let imageTokenID = tokenizer.imageTokenId else {
                throw EmbeddingError.tokenizerMissingImageToken
            }
            imageRanges = Qwen3VLEmbeddingPromptBuilder.contiguousRanges(
                in: tokenIDs,
                matching: imageTokenID
            )
            guard imageRanges.count == input.imageURLs.count else {
                throw EmbeddingError.imageTokenRangeMismatch(
                    expected: input.imageURLs.count,
                    actual: imageRanges.count
                )
            }
        }

        let conditioningImages = zip(zip(pixelValues, grids), imageRanges).map { pair, tokenRange in
            QwenVLEncoder.ConditioningImage(
                pixelValues: pair.0,
                tokenRange: tokenRange,
                heightPatchCount: pair.1.height,
                widthPatchCount: pair.1.width
            )
        }
        let inputIDs = MLXArray(tokenIDs.map(Int32.init)).reshaped(1, tokenIDs.count)
        let attentionMask = MLX.ones([1, tokenIDs.count], dtype: .int32)
        let hiddenStates = try encoder.forwardMultimodalHiddenState(
            inputIds: inputIDs,
            attentionMask: attentionMask,
            images: conditioningImages
        )
        let pooled = hiddenStates[0, tokenIDs.count - 1, 0..<dimensions].asType(.float32)
        let epsilon = MLXArray(Float32(1e-12))
        let norm = MLX.sqrt(MLX.sum(pooled * pooled) + epsilon)
        let normalized = pooled / norm
        MLX.eval(normalized)

        return Qwen3VLEmbeddingResult(
            id: input.id,
            embedding: normalized.asArray(Float.self),
            tokenCount: tokenIDs.count,
            imageCount: input.imageURLs.count
        )
    }
}
