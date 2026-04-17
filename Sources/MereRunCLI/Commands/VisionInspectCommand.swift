import ArgumentParser
import Foundation
import MereRunCore

struct VisionInspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "Describe or answer questions about an image using Qwen3-VL.",
        discussion: """
        Prints the VLM response to stdout. Progress and diagnostics go to stderr.

        If no --model is provided, auto-downloads the default local vision-language model.
        """
    )

    @Argument(help: "Image file path.")
    var image: String

    @Argument(help: "Prompt / question about the image (default: \"Describe this image.\").")
    var prompt: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Local model root directory path (optional, auto-downloads if omitted).")
    var model: String?

    @Option(name: [.long], help: "Max new tokens to generate.")
    var maxTokens: Int = 2048

    @Option(name: [.long], help: "Sampling temperature.")
    var temperature: Float = 0.7

    @Option(name: [.long], help: "Top-p sampling.")
    var topP: Float = 0.9

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let imageURL = URL(fileURLWithPath: image).standardizedFileURL
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image not found: \(imageURL.path)")
        }

        let userPrompt = prompt.isEmpty ? "Describe this image." : prompt.joined(separator: " ")

        let modelURL: URL
        if let model {
            modelURL = URL(fileURLWithPath: model).standardizedFileURL
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw ValidationError("Model path not found: \(modelURL.path)")
            }
        } else {
            FileHandle.standardError.write(Data("Downloading \(Qwen3VLAutoCaptioner.modelId)...\n".utf8))
            let autoCaptioner = Qwen3VLAutoCaptioner()
            modelURL = try await autoCaptioner.ensureReady { progress in
                let msg = "\r\(progress.status) (\(Int(progress.fraction * 100))%)"
                FileHandle.standardError.write(Data(msg.utf8))
            }
            FileHandle.standardError.write(Data("\n".utf8))
        }

        let captioner = try QwenVLCaptioner(modelRoot: modelURL)
        let config = QwenVLCaptioner.ModelConfig(maxNewTokens: maxTokens, temperature: temperature, topP: topP)
        let result = try captioner.caption(imageURL: imageURL, prompt: userPrompt, config: config)
        print(result.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}
