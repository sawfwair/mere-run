import Foundation
import MediaIO
import MLX
import MLXRandom

public struct NemotronImageGenerator: ImageGenerator {
    public init() {}

    public func generate(_ request: GenerationRequest,
                         progressHandler: (@Sendable (GenerationProgress) -> Void)?) async throws -> GenerationResult {
        guard NemotronImageTokens.supports(width: request.width, height: request.height),
              request.outputURL.pathExtension.lowercased() == "png", request.inputImage == nil,
              request.referenceImages.isEmpty, request.negativePrompt?.isEmpty != false, request.loras.isEmpty,
              request.sigmas == nil, request.sigmaShift == nil, request.steps == 1, request.guidanceScale == 1,
              !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ImageGenerationIssue("nemotron_image_options_invalid", "Expected a text-to-image PNG request within the visual token budget.")
        }
        try Task.checkCancellation()
        let root = try ImageGenerationModelSelection(request.model ?? ManagedModelID.nemotronImageQ6.rawValue).resolveRoot()
        let format = try NemotronImageResources.read(NemotronImageTokens.self,
            at: root.appending(path: "unitron_extension_manifest.json"))
        try format.validate()
        let seed = ImageGenerationSeed.resolve(request.seed, prompt: request.prompt, backend: .nemotronImage)
        func progress(_ stage: GenerationStage, _ step: Int = 0, _ total: Int = 1) {
            progressHandler?(.init(stage: stage, stepIndex: step, totalSteps: total))
        }
        defer { Memory.clearCache() }
        progress(.encodingText)
        let config = try NemotronImageResources.languageConfig(at: root)
        let tokenizer = try QwenTokenizer.load(from: root)
        let system = "Generate an image matching the requested resolution by detailing the color, "
            + "quantity, text, shape, size, texture, spatial relationships of the objects and background."
        let prompt = try tokenizer.encodeChatTemplate(messages: [
            ["role": "system", "content": system],
            ["role": "user", "content": "Generate a \(request.width)x\(request.height) image.\n\n\(request.prompt)"]
        ], includeThinking: false, truncate: false)
        guard !prompt.isEmpty, prompt.count <= min(request.maxSequenceLength, config.maxPositionEmbeddings - NemotronImageTokens.budget) else {
            throw ImageGenerationIssue("prompt_length_invalid", "The tokenized image prompt exceeds the requested sequence length.")
        }
        progress(.loadingTransformer)
        let tokens = try Self.sample(root: root, config: config, prompt: prompt, end: format.ranges.framing.end,
                                     seed: seed) { count in progress(.decoding, count, NemotronImageTokens.budget) }
        let codes = try format.codes(from: tokens, width: request.width, height: request.height, decodeHeader: tokenizer.decode)
        Memory.clearCache()
        try Task.checkCancellation()
        progress(.loadingVAE)
        let directory = NemotronImageCommunityPack.isPresent(at: root) ? "vq" : NemotronImageResources.decoderDirectory
        let decoder = try NemotronImageLoader.decoder(at: root.appending(path: directory))
        progress(.decoding)
        let rgb = decoder.decode(MLXArray(codes), rows: request.height / 16, columns: request.width / 16)
        let image = try MediaImageIO.imageFromRGBHWC(rgb.asArray(UInt8.self), width: request.width, height: request.height)
        try Task.checkCancellation()
        progress(.saving)
        try MediaImageIO.writePNG(image, to: request.outputURL)
        progress(.saving, 1)
        return GenerationResult(outputURL: request.outputURL, seed: seed)
    }

    private static func sample(root: URL, config: NemotronHConfig, prompt: [Int], end: Int,
                               seed: UInt64, progress: (Int) -> Void) throws -> [Int] {
        let model = try NemotronImageLoader.generator(at: root, config: config)
        let cache = model.makeCache()
        var logits = model.prefill(MLXArray(prompt).reshaped(1, -1), cache: cache)
        var tokens: [Int] = []
        for step in 0..<NemotronImageTokens.budget {
            try Task.checkCancellation()
            let next = MLXRandom.categorical(logits.asType(.float32) / 0.9,
                                             key: MLXRandom.key(seed &+ UInt64(step)))
            let id = next.item(Int.self)
            tokens.append(id)
            progress(tokens.count)
            if id == end || config.eosTokenIDs.contains(id) { break }
            logits = model.lastPositionLogits(next.reshaped(1, 1), cache: cache)
        }
        return tokens
    }
}
