import Foundation
import MediaIO
import MLX
#if canImport(ImageIO)
import ImageIO
#endif

enum MiniMaxH3ImageConditioner {
    struct Reference {
        let vaeImage: MediaImage
        let vision: MLXArray
    }

    static func prepare(_ url: URL) throws -> Reference {
        var image = try MediaImageIO.decode(url)
        #if canImport(ImageIO)
        if let source = CGImageSourceCreateWithURL(url as CFURL, nil),
           let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let orientation = properties[kCGImagePropertyOrientation] as? Int {
            image = try Flux3ImageIO.oriented(image, orientation: orientation)
        }
        #endif
        guard min(image.width, image.height) >= 32 else {
            throw ImageGenerationIssue("reference_dimensions_invalid", "H3-Image references require at least 32 pixels on each side.")
        }
        let vaeSize = MiniMaxH3ImageGeometry.referenceSize(width: image.width, height: image.height, vision: false)
        let visionSize = MiniMaxH3ImageGeometry.referenceSize(width: image.width, height: image.height, vision: true)
        let vaeImage = try MuseGlimmerImageProcessor.resizedLanczos(image, width: vaeSize.width, height: vaeSize.height)
        let visionImage = try MuseGlimmerImageProcessor.resizedLanczos(image, width: visionSize.width, height: visionSize.height)
        let processorSize = MiniMaxH3ImageGeometry.visionProcessorSize(width: visionImage.width, height: visionImage.height)
        let processed = try MediaImageIO.bicubicResizedRGB(visionImage, width: processorSize.width, height: processorSize.height)
        let vision = MLXArray(MediaImageIO.rgbCHWFloat(processed, normalizedToMinusOneToOne: true))
            .reshaped(1, 3, processed.height, processed.width)
        return Reference(vaeImage: vaeImage, vision: vision)
    }

    static func encode(prompt: String, references: [Reference], resources: MiniMaxH3ImageResources,
                       didLoadEncoder: () -> Void = {}) throws -> (states: MLXArray, tags: [Int32]) {
        let tokenizer = try QwenTokenizer.load(from: resources.rootURL.appending(path: "tokenizer"), maxLengthOverride: 262_144)
        var ids: [Int] = [], tags: [Int32] = [], images: [QwenVLEncoder.ConditioningImage] = []
        for (index, reference) in references.enumerated() {
            guard let start = tokenizer.visionStartTokenId, let image = tokenizer.imageTokenId, let end = tokenizer.visionEndTokenId else {
                throw resources.invalid("Missing Qwen3-VL vision tokens.")
            }
            let labels = Array(tokenizer.encodeRawChatText("Picture \(index + 1): ").prefix(4_096))
            ids += labels
            tags += Array(repeating: 1, count: labels.count)
            ids.append(start)
            tags.append(0)
            let vision = reference.vision
            let count = vision.dim(2) / 32 * (vision.dim(3) / 32)
            let range = ids.count..<(ids.count + count)
            ids += Array(repeating: image, count: count)
            tags += Array(repeating: 0, count: count)
            ids.append(end)
            tags.append(0)
            images.append(.init(pixelValues: vision, tokenRange: range, heightPatchCount: vision.dim(2) / 16, widthPatchCount: vision.dim(3) / 16))
        }
        let promptIDs = Array(tokenizer.encodeRawChatText(prompt).prefix(4_096))
        ids += promptIDs
        tags += Array(repeating: 1, count: promptIDs.count)
        guard !promptIDs.isEmpty, ids.count <= 262_144 else { throw resources.invalid("Invalid conditioning sequence length.") }
        let encoder = try MiniMaxH3ImageLoader.conditioner(resources)
        didLoadEncoder()
        guard let states = try encoder.forwardMultimodalActivationHiddenState(
            inputIds: MLXArray(ids.map(Int32.init)).reshaped(1, -1),
            attentionMask: MLXArray.ones([1, ids.count], dtype: .int32), images: images, activationLayer: 49
        ) else { throw resources.invalid("Missing layer-50 conditioner activations.") }
        MLX.eval(states)
        return (states, tags)
    }
}
