import Foundation
import MLX
import MLXNN

final class QwenImage21Conditioner {
    static let systemPrompt = "Comprehend and analyze the provided prompt."
    let encoder: QwenVLEncoder
    let tokenizer: QwenTokenizer
    let layerCount: Int
    let dropCount: Int

    init(resources: QwenImage21Resources) throws {
        let config = try resources.decode("text_encoder/config.json", as: Qwen3VLEmbeddingRootConfig.self)
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

        let parameters = encoder.parameters().flattened()
        let expected = Dictionary(uniqueKeysWithValues: parameters.map { ($0.0, $0.1.shape) })
        let arrays = try resources.arrays("text_encoder", stem: "model")
        var mapped: [String: MLXArray] = [:]
        for (name, value) in arrays {
            if name == "lm_head.weight" { continue }
            for (key, tensor) in Qwen3VLEmbeddingWeights.mapWeight(name, value) {
                guard expected[key] == tensor.shape, mapped[key] == nil else {
                    throw QwenImage21Error.invalidWeights("Text encoder tensor mismatch: \(name).")
                }
                mapped[key] = tensor
            }
        }
        guard Set(mapped.keys) == Set(expected.keys) else {
            throw QwenImage21Error.invalidWeights("Missing text encoder tensors: \(Set(expected.keys).subtracting(mapped.keys).sorted())")
        }
        try encoder.update(parameters: ModuleParameters.unflattened(mapped), verify: [.all])
        self.encoder = encoder
        layerCount = text.numHiddenLayers
        tokenizer = try QwenTokenizer.load(from: resources.rootURL.appending(path: "processor"), maxLengthOverride: 262_144)
        // Qwen3-VL's system-only chat template emits this exact system message without an assistant prefix.
        dropCount = tokenizer.encodeText("<|im_start|>system\n" + Self.systemPrompt + "<|im_end|>\n").count
        eval(encoder)
    }

    func encode(prompt: String, images: [QwenImage21ImageIO.Reference], targetHeight: Int, targetWidth: Int) throws -> (MLXArray, QwenImage21Layout) {
        let counts = images.map { $0.height * $0.width / 1024 }
        let prefixes = counts.enumerated().map { index, count in
            "<image\(index + 1)><|vision_start|>" + String(repeating: "<|image_pad|>", count: count) + "<|vision_end|>"
        }.joined(separator: " ")
        let presentation = "<|im_start|>system\n" + Self.systemPrompt + "<|im_end|>\n<|im_start|>user\n"
            + prefixes + (prompt.isEmpty ? " " : prompt) + "<|im_end|>\n<|im_start|>assistant\n"
        let ids = tokenizer.encodeText(presentation)
        guard ids.count <= 262_144, let imageID = tokenizer.imageTokenId else {
            throw QwenImage21Error.invalidLayout("Invalid conditioning length or tokenizer image token.")
        }
        let ranges = Qwen3VLEmbeddingPromptBuilder.contiguousRanges(in: ids, matching: imageID)
        guard ranges.count == images.count, zip(ranges, counts).allSatisfy({ $0.0.count == $0.1 }) else {
            throw QwenImage21Error.invalidLayout("Reference placeholder count differs from the vision grid.")
        }
        let conditioning = zip(images, ranges).map { image, range in
            QwenVLEncoder.ConditioningImage(pixelValues: image.vision, tokenRange: range,
                heightPatchCount: image.height / 16, widthPatchCount: image.width / 16)
        }
        guard let hidden = try encoder.forwardMultimodalActivationHiddenState(
            inputIds: MLXArray(ids.map(Int32.init)).reshaped(1, ids.count),
            attentionMask: ones([1, ids.count], dtype: .int32), images: conditioning, activationLayer: layerCount - 1
        ) else { throw QwenImage21Error.invalidLayout("Text encoder did not return its final activation.") }
        let slots = ids.dropFirst(dropCount).map { $0 == imageID }
            + Array(repeating: true, count: targetHeight * targetWidth / 4)
        let shapes = images.map { (height: $0.height / 16, width: $0.width / 16) }
            + [(height: targetHeight, width: targetWidth)]
        let layout = try QwenImage21Layout(imageSlots: slots, imageShapes: shapes)
        let output = hidden[0..., dropCount..., 0...]
        eval(output)
        return (output, layout)
    }
}
