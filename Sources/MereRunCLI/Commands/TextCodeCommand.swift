import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Text Code Command

struct TextCode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "code",
        abstract: "Run local code generation with GGUF models via llama.cpp.",
        discussion: """
        Uses llama.cpp for GGUF model inference.
        Supports Qwen3-Coder and other chat-format models.

        Example:
          mere.run text code -p "Write a Swift function to reverse a string"
          mere.run text code -m ./model.gguf -p "Explain this code"
        """
    )

    @Option(name: [.customShort("p"), .long], help: "User prompt.")
    var prompt: String

    @Option(name: [.customShort("s"), .customLong("system")], help: "System prompt.")
    var systemPrompt: String = "You are a helpful coding assistant."

    @Option(name: [.long], help: "Max new tokens.")
    var maxTokens: Int = 2048

    @Option(name: [.long], help: "Temperature (default 1.0 for Qwen3-Coder).")
    var temperature: Double = 1.0

    @Option(name: [.long], help: "Top-p (default 0.95).")
    var topP: Double = 0.95

    @Option(name: [.customLong("min-p")], help: "Min-p cutoff relative to the most likely token.")
    var minP: Double = 0

    @Option(name: [.customShort("m"), .long], help: "Path to GGUF model file.")
    var model: String?

    @Flag(name: [.customLong("stats")], help: "Print generation timing and tokens/sec.")
    var stats: Bool = false

    @Flag(name: [.short, .long], help: "Suppress progress output.")
    var quiet: Bool = false

    @Flag(name: [.customLong("stream")], help: "Stream tokens as they are generated.")
    var stream: Bool = false

    func run() async throws {
        // Use explicit model path if provided, otherwise use default model ID
        let modelId = model ?? CodeGenResources.defaultModelId
        let generator = CodeGenGenerator(modelId: modelId)

        var messages: [ChatMessage] = []
        if !systemPrompt.isEmpty {
            messages.append(ChatMessage(role: .system, content: systemPrompt))
        }
        messages.append(ChatMessage(role: .user, content: prompt))

        let request = ChatRequest(
            messages: messages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            minP: minP
        )

        let progressHandler: (@Sendable (ChatProgress) -> Void)?
        if quiet {
            progressHandler = nil
        } else if stream {
            progressHandler = { progress in
                if progress.stage == .generating, let token = progress.message {
                    CLIStdout.write(token)
                } else if progress.stage == .loadingModel {
                    CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
                }
            }
        } else {
            progressHandler = { progress in
                CLIStderr.write("[\(progress.stage.rawValue)] \(progress.message ?? "")\n")
            }
        }

        let startTime = Date()
        let result = try await generator.chat(request, modelPath: model, progressHandler: progressHandler)
        let elapsed = Date().timeIntervalSince(startTime)

        if stream {
            print() // Final newline after streaming
        }

        if stats {
            let tps = elapsed > 0 ? Double(result.tokensGenerated) / elapsed : 0
            let timing = String(format: "time=%.2fs tokens=%d tps=%.2f", elapsed, result.tokensGenerated, tps)
            CLIStderr.write("\(timing)\n")
        }

        if !stream {
            print(result.response)
        }
    }
}
