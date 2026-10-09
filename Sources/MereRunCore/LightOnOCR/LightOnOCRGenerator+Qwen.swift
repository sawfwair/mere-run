#if !os(iOS)
import Foundation

extension LightOnOCRGenerator {
    func ocrQwen(imageURL: URL, rootURL: URL, config: Config) async throws -> Result {
        if loadedModelPath != rootURL.path || qwenGenerator == nil {
            await unload()
            let modelConfig = try JSONDecoder().decode(
                Q35Config.self, from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
            )
            let modelID = modelConfig.textConfig.hiddenSize == 1024
                ? Q35Resources.lightOnOCR3SmallModelId : Q35Resources.lightOnOCR3FourBModelId
            qwenGenerator = Q35Generator(modelId: modelID)
            loadedModelPath = rootURL.path
        }
        guard let qwenGenerator else { throw LightOnOCRError.modelsNotLoaded }
        let response = try await qwenGenerator.chat(
            ChatRequest(
                messages: [ChatMessage(role: .user, content: config.mode.prompt, imageUrl: imageURL.path)],
                maxTokens: config.maxNewTokens,
                temperature: Double(config.temperature),
                topP: 0.9,
                showThinking: false,
                maxContextTokens: Q35Resources.defaultContextLength
            ),
            modelPath: rootURL.path,
            progressHandler: nil
        )
        return Result(text: response.response, tokensGenerated: response.tokensGenerated)
    }
}
#endif
