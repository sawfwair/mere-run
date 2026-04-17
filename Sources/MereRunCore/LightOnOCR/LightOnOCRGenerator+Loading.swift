import Foundation
import MLX
import MLXNN

/// Owns config parsing and weight application for the LightOn OCR stack.
/// This file intentionally stops before image preprocessing and decoding.
extension LightOnOCRGenerator {
    func loadModels(from path: String) async throws {
        let modelURL = URL(fileURLWithPath: path)
        let fm = FileManager.default

        log("[OCR] Starting model load from \(path)")
        log("[OCR] Memory before: \(Memory.activeMemory / 1024 / 1024) MB")

        let visionConfigURL = modelURL.appendingPathComponent("config.json")
        guard fm.fileExists(atPath: visionConfigURL.path) else {
            throw LightOnOCRError.configNotFound(visionConfigURL)
        }

        let configData = try Data(contentsOf: visionConfigURL)
        let fullConfig = try JSONDecoder().decode(LightOnOCRConfig.self, from: configData)

        let visionConfig = PixtralVisionConfiguration(
            hiddenSize: fullConfig.visionConfig.hiddenSize,
            numHiddenLayers: fullConfig.visionConfig.numHiddenLayers,
            numAttentionHeads: fullConfig.visionConfig.numAttentionHeads,
            headDim: fullConfig.visionConfig.headDim,
            intermediateSize: fullConfig.visionConfig.intermediateSize,
            patchSize: fullConfig.visionConfig.patchSize,
            imageSize: fullConfig.visionConfig.imageSize,
            numChannels: fullConfig.visionConfig.numChannels,
            ropeTheta: fullConfig.visionConfig.ropeTheta
        )
        visionEncoder = PixtralVisionEncoder(config: visionConfig)
        log("[OCR] After vision encoder init: \(Memory.activeMemory / 1024 / 1024) MB")

        let mropeSection = fullConfig.textConfig.ropeScaling?.mropeSection
        let mropeInterleaved = fullConfig.textConfig.ropeScaling?.mropeInterleaved ?? false
        let textConfig = QwenTextEncoderConfiguration(
            vocabSize: fullConfig.textConfig.vocabSize,
            hiddenSize: fullConfig.textConfig.hiddenSize,
            numHiddenLayers: fullConfig.textConfig.numHiddenLayers,
            numAttentionHeads: fullConfig.textConfig.numAttentionHeads,
            numKeyValueHeads: fullConfig.textConfig.numKeyValueHeads,
            intermediateSize: fullConfig.textConfig.intermediateSize,
            ropeTheta: fullConfig.textConfig.ropeTheta,
            maxPositionEmbeddings: fullConfig.textConfig.maxPositionEmbeddings,
            rmsNormEps: fullConfig.textConfig.rmsNormEps ?? 1e-6,
            promptDropIndex: 0,
            headDim: fullConfig.textConfig.headDim,
            mropeSection: mropeSection,
            mropeInterleaved: mropeInterleaved
        )
        textDecoder = QwenTextEncoder(configuration: textConfig)
        log("[OCR] Text config: heads=\(textConfig.numAttentionHeads), kv_heads=\(textConfig.numKeyValueHeads), headDim=\(textConfig.headDim), layers=\(textConfig.numHiddenLayers)")
        log("[OCR] After text decoder init: \(Memory.activeMemory / 1024 / 1024) MB")

        spatialMergeSize = fullConfig.spatialMergeSize
        visionProjection = VisionProjection(
            visionHiddenSize: visionConfig.hiddenSize,
            textHiddenSize: textConfig.hiddenSize,
            spatialMergeSize: spatialMergeSize
        )

        log("[OCR] Loading tokenizer...")
        let tokenizerURL = modelURL.appendingPathComponent("tokenizer")
        if fm.fileExists(atPath: tokenizerURL.path) {
            tokenizer = try QwenTokenizer.load(from: tokenizerURL, maxLengthOverride: 16384)
        } else {
            tokenizer = try QwenTokenizer.load(from: modelURL, maxLengthOverride: 16384)
        }
        log("[OCR] After tokenizer: \(Memory.activeMemory / 1024 / 1024) MB")

        log("[OCR] Loading weights...")
        try await loadWeights(from: modelURL)
        log("[OCR] After weights: \(Memory.activeMemory / 1024 / 1024) MB")

        loadedModelPath = path
    }

    func loadWeights(from modelURL: URL) async throws {
        let weightsURL = modelURL.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LightOnOCRError.weightsNotFound(modelURL)
        }
        guard let visionEncoder,
              let textDecoder,
              let visionProjection else {
            throw LightOnOCRError.modelsNotLoaded
        }

        log("[OCR] Loading safetensors file...")
        let allWeights = try MLX.loadArrays(url: weightsURL)
        log("[OCR] Loaded \(allWeights.count) tensors, memory: \(Memory.activeMemory / 1024 / 1024) MB")

        var visionWeights: [String: MLXArray] = [:]
        var textWeights: [String: MLXArray] = [:]
        var projWeights: [String: MLXArray] = [:]

        for (rawKey, value) in allWeights {
            var key = rawKey
            if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            }

            if key.hasPrefix("vision_encoder.") {
                visionWeights[String(key.dropFirst("vision_encoder.".count))] = value
            } else if key.hasPrefix("language_model.") {
                let mappedKey = "encoder." + key.dropFirst("language_model.".count)
                textWeights[String(mappedKey)] = value
            } else if key.hasPrefix("vision_projection.") {
                projWeights[String(key.dropFirst("vision_projection.".count))] = value
            }
        }

        log("[OCR] Mapped weights - vision: \(visionWeights.count), text: \(textWeights.count), proj: \(projWeights.count)")
        log("[OCR] Memory after mapping: \(Memory.activeMemory / 1024 / 1024) MB")
        log("[OCR] Sample vision keys: \(Array(visionWeights.keys.sorted().prefix(5)))")
        log("[OCR] Sample text keys: \(Array(textWeights.keys.sorted().prefix(5)))")
        let qkNormKeys = textWeights.keys.filter { $0.contains("q_norm") || $0.contains("k_norm") }.sorted()
        log("[OCR] QK norm keys (\(qkNormKeys.count)): \(Array(qkNormKeys.prefix(4)))")
        log("[OCR] Proj keys: \(Array(projWeights.keys.sorted()))")

        if let patchConv = visionWeights["patch_conv.weight"], patchConv.ndim == 4 {
            let inChannels = visionEncoder.config.numChannels
            let kernel = visionEncoder.config.patchSize
            if patchConv.dim(1) == inChannels && patchConv.dim(2) == kernel && patchConv.dim(3) == kernel {
                visionWeights["patch_conv.weight"] = patchConv.transposed(0, 2, 3, 1)
            }
        }

        log("[OCR] Applying vision weights...")
        do {
            try visionEncoder.update(parameters: ModuleParameters.unflattened(visionWeights), verify: .noUnusedKeys)
        } catch {
            log("[OCR] Vision weight error: \(error)")
            try visionEncoder.update(parameters: ModuleParameters.unflattened(visionWeights), verify: .none)
        }
        visionWeights.removeAll()
        log("[OCR] After vision update: \(Memory.activeMemory / 1024 / 1024) MB")

        log("[OCR] Applying text weights...")
        do {
            try textDecoder.update(parameters: ModuleParameters.unflattened(textWeights), verify: .noUnusedKeys)
        } catch {
            log("[OCR] Text weight error: \(error)")
            try textDecoder.update(parameters: ModuleParameters.unflattened(textWeights), verify: .none)
        }
        textWeights.removeAll()
        log("[OCR] After text update: \(Memory.activeMemory / 1024 / 1024) MB")

        log("[OCR] Applying projection weights...")
        try visionProjection.update(parameters: ModuleParameters.unflattened(projWeights), verify: .none)
        projWeights.removeAll()
        log("[OCR] After projection update: \(Memory.activeMemory / 1024 / 1024) MB")
    }
}
