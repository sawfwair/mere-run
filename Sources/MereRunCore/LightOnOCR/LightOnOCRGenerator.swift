import Foundation
import MLX
import MLXNN
import MLXFast
import ImageIO

/// Owns the public OCR entrypoint and runtime state for the LightOn stack.
/// Model loading and generation helpers live in companion files so readers can
/// follow the actor flow before dropping into vision/text details.
public actor LightOnOCRGenerator {
    public struct Config: Sendable {
        public let maxNewTokens: Int
        public let temperature: Float
        public let logProgress: Bool

        public init(
            maxNewTokens: Int = 4096,
            temperature: Float = 0.2,
            logProgress: Bool = false
        ) {
            self.maxNewTokens = maxNewTokens
            self.temperature = temperature
            self.logProgress = logProgress
        }
    }

    public struct Result: Sendable {
        public let text: String
        public let tokensGenerated: Int
    }

    var visionEncoder: PixtralVisionEncoder?
    var textDecoder: QwenTextEncoder?
    var visionProjection: VisionProjection?
    var tokenizer: QwenTokenizer?
    var loadedModelPath: String?
    var spatialMergeSize: Int = 2
    var logProgress: Bool = false

    let imageTokenId: Int = 151655
    let visionEndTokenId: Int = 151653
    let visionPadTokenId: Int = 151654
    let eosTokenId: Int = 151645
    let padTokenId: Int = 151643
    let imStartTokenId: Int = 151644

    public init() {}

    public func ocr(
        imageURL: URL,
        modelPath: String,
        config: Config = Config()
    ) async throws -> Result {
        logProgress = config.logProgress

        if loadedModelPath != modelPath {
            try await loadModels(from: modelPath)
        }

        guard let visionEncoder,
              let textDecoder,
              let visionProjection,
              let tokenizer else {
            throw LightOnOCRError.modelsNotLoaded
        }

        log("[OCR] Loading image...")
        let pixelValues = try loadAndPreprocessImage(
            imageURL,
            patchSize: visionEncoder.config.patchSize,
            spatialMergeSize: spatialMergeSize
        )
        MLX.eval(pixelValues)
        log("[OCR] Image shape: \(pixelValues.shape), memory: \(Memory.activeMemory / 1024 / 1024) MB")

        let imageH = pixelValues.dim(2)
        let imageW = pixelValues.dim(3)
        let gridH = imageH / visionEncoder.config.patchSize
        let gridW = imageW / visionEncoder.config.patchSize
        log("[OCR] Grid: \(gridH) x \(gridW) = \(gridH * gridW) patches")

        log("[OCR] Running vision encoder...")
        var visionFeatures = visionEncoder.encode(pixelValues: pixelValues)
        MLX.eval(visionFeatures)
        log("[OCR] Vision features: \(visionFeatures.shape), memory: \(Memory.activeMemory / 1024 / 1024) MB")

        log("[OCR] Running vision projection...")
        visionFeatures = visionProjection(
            visionFeatures,
            gridH: gridH,
            gridW: gridW,
            spatialMergeSize: spatialMergeSize
        )
        MLX.eval(visionFeatures)
        log("[OCR] Projected features: \(visionFeatures.shape), memory: \(Memory.activeMemory / 1024 / 1024) MB")

        let (preTokens, postTokens) = buildOCRPromptParts(tokenizer: tokenizer)
        let preIds = MLXArray(preTokens.map { Int32($0) }).reshaped(1, preTokens.count)
        let postIds = MLXArray(postTokens.map { Int32($0) }).reshaped(1, postTokens.count)
        let preEmbeds = textDecoder.encoder.embed(inputIds: preIds)
        let postEmbeds = textDecoder.encoder.embed(inputIds: postIds)

        MLX.eval(preEmbeds, visionFeatures, postEmbeds)
        log("[OCR] visionFeat: std=\(MLX.std(visionFeatures).item(Float.self)), max=\(MLX.max(MLX.abs(visionFeatures)).item(Float.self))")
        log("[OCR] preEmbeds: dtype=\(preEmbeds.dtype), std=\(MLX.std(preEmbeds).item(Float.self))")
        log("[OCR] postEmbeds: dtype=\(postEmbeds.dtype), std=\(MLX.std(postEmbeds).item(Float.self))")
        if visionFeatures.dtype != preEmbeds.dtype {
            visionFeatures = visionFeatures.asType(preEmbeds.dtype)
        }

        let mergedH = gridH / spatialMergeSize
        let mergedW = gridW / spatialMergeSize
        let hiddenDim = visionFeatures.dim(2)

        let visionPadEmbed = textDecoder.encoder.embed(
            inputIds: MLXArray([Int32(visionPadTokenId)]).reshaped(1, 1)
        )
        let visionEndEmbed = textDecoder.encoder.embed(
            inputIds: MLXArray([Int32(visionEndTokenId)]).reshaped(1, 1)
        )

        let visionGrid = visionFeatures.reshaped(1, mergedH, mergedW, hiddenDim)
        var visionParts: [MLXArray] = []
        visionParts.reserveCapacity(max(1, mergedH * 2))
        for row in 0..<mergedH {
            let rowEmbeds = visionGrid[0..., row, 0..<mergedW, 0...].reshaped(1, mergedW, hiddenDim)
            visionParts.append(rowEmbeds)
            if row < mergedH - 1 {
                visionParts.append(visionPadEmbed)
            }
        }
        let visionWithPads = MLX.concatenated(visionParts, axis: 1)
        let inputEmbeds = MLX.concatenated([preEmbeds, visionWithPads, visionEndEmbed, postEmbeds], axis: 1)

        let numVisionTokens = visionFeatures.dim(1)
        let promptSeqLen = inputEmbeds.dim(1)
        let numVisionPads = max(0, mergedH - 1)
        log("[OCR] Input embeddings: pre=\(preTokens.count), vision=\(numVisionTokens), pads=\(numVisionPads), post=\(postTokens.count), total=\(promptSeqLen)")

        let hasMRoPE = textDecoder.configuration.mropeSection?.isEmpty == false
        let positionIds: MLXArray?
        let ropeDelta: Int
        if hasMRoPE {
            let placeholderPos = preTokens.count
            let originalSeqLen = preTokens.count + 1 + postTokens.count
            let ids = computeExpandedRopePositions(
                originalSeqLen: originalSeqLen,
                placeholderPos: placeholderPos,
                numVisionTokens: numVisionTokens,
                gridThw: (1, gridH, gridW),
                spatialMergeSize: spatialMergeSize
            )
            MLX.eval(ids)
            let maxPos = ids.max().item(Int32.self)
            positionIds = ids
            ropeDelta = Int(maxPos) + 1 - promptSeqLen
            log("[OCR] MRoPE: placeholderPos=\(placeholderPos), maxPos=\(maxPos), ropeDelta=\(ropeDelta)")
        } else {
            positionIds = nil
            ropeDelta = 0
        }

        let generatedTokens = try generateFromEmbeddings(
            inputEmbeds: inputEmbeds,
            positionIds: positionIds,
            promptSeqLen: promptSeqLen,
            ropeDelta: ropeDelta,
            textDecoder: textDecoder,
            config: config
        )

        return Result(
            text: tokenizer.decode(tokens: generatedTokens),
            tokensGenerated: generatedTokens.count
        )
    }

    public func unload() {
        visionEncoder = nil
        textDecoder = nil
        visionProjection = nil
        tokenizer = nil
        loadedModelPath = nil
        Memory.clearCache()
    }

    func log(_ message: @autoclosure () -> String) {
        MereRunRuntimeDebug.write(message(), enabled: logProgress)
    }
}
