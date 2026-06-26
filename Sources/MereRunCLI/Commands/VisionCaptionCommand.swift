import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Vision Caption Command

struct VisionCaption: AsyncParsableCommand {
    static let defaultPrompt = "Write a short, concrete caption describing the image for LoRA training. Avoid fluff."

    static let configuration = CommandConfiguration(
        commandName: "caption",
        abstract: "Generate training-friendly captions for images.",
        discussion: """
        Writes one .txt caption per image next to the file (or into --output-dir).

        If no --model is provided, auto-downloads mlx-community/Qwen3-VL-2B-Instruct-4bit.
        """
    )

    @Argument(help: "One or more image file paths.")
    var images: [String] = []

    @Option(name: [.customShort("m"), .long], help: "Local model root directory path (optional, auto-downloads if omitted).")
    var model: String?

    @Option(name: [.customShort("o"), .long], help: "Output directory for .txt captions (default: alongside images).")
    var outputDir: String?

    @Option(name: [.long], help: "Caption instruction/prompt (short is best for LoRA).")
    var prompt: String?

    @Option(name: [.customLong("prompt-file")], help: "Read caption instructions from a UTF-8 text file.")
    var promptFile: String?

    @Option(
        name: [.customLong("focus")],
        parsing: .upToNextOption,
        help: "Visible details the caption should prioritize, e.g. --focus \"card border\" \"printed title\"."
    )
    var focus: [String] = []

    @Option(
        name: [.customLong("trigger-token")],
        help: "Prefix each saved caption with this exact LoRA trigger token."
    )
    var triggerToken: String?

    @Option(name: [.long], help: "Max new tokens to generate.")
    var maxTokens: Int = 96

    @Option(name: [.long], help: "Sampling temperature.")
    var temperature: Float = 0.2

    @Option(name: [.long], help: "Top-p sampling.")
    var topP: Float = 0.9

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        guard !images.isEmpty else {
            throw ValidationError("Provide at least one image path.")
        }

        let outDirURL: URL? = outputDir.map { URL(fileURLWithPath: $0).standardizedFileURL }
        if let outDirURL {
            try FileManager.default.createDirectory(at: outDirURL, withIntermediateDirectories: true)
        }

        // Use provided model path or auto-download the default vision-language model.
        let modelURL: URL
        if let model {
            modelURL = URL(fileURLWithPath: model).standardizedFileURL
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw ValidationError("Model path not found: \(modelURL.path)")
            }
        } else {
            let autoCaptioner = Qwen3VLAutoCaptioner()
            if await autoCaptioner.isModelCached() {
                print("No model specified, using cached \(Qwen3VLAutoCaptioner.modelId)...")
            } else {
                print("No model specified, downloading \(Qwen3VLAutoCaptioner.modelId)...")
            }
            modelURL = try await autoCaptioner.ensureReady { progress in
                CLIStdout.write("\r\(progress.status) (\(Int(progress.fraction * 100))%)")
            }
            print()  // newline after progress
        }

        let captioner = try QwenVLCaptioner(modelRoot: modelURL)
        let config = QwenVLCaptioner.ModelConfig(maxNewTokens: maxTokens, temperature: temperature, topP: topP)
        let instruction = try resolvedPromptInstruction()

        for path in images {
            let imageURL = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: imageURL.path) else {
                throw ValidationError("Image not found: \(imageURL.path)")
            }

            let caption = try captioner.caption(imageURL: imageURL, prompt: instruction, config: config)
            let captionURL: URL = {
                if let outDirURL {
                    return outDirURL.appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent + ".txt")
                }
                return imageURL.deletingPathExtension().appendingPathExtension("txt")
            }()
            let out = Self.captionOutput(caption, triggerToken: triggerToken) + "\n"
            try Data(out.utf8).write(to: captionURL)
            print("\(imageURL.path) -> \(captionURL.path)")
        }
    }

    func resolvedPromptInstruction() throws -> String {
        let basePrompt: String
        if let promptFile {
            let url = URL(fileURLWithPath: promptFile).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Prompt file not found: \(url.path)")
            }
            basePrompt = try String(contentsOf: url, encoding: .utf8)
        } else {
            basePrompt = prompt ?? Self.defaultPrompt
        }

        var parts = [basePrompt.trimmingCharacters(in: .whitespacesAndNewlines)]
        if let prompt = prompt?.trimmingCharacters(in: .whitespacesAndNewlines),
           promptFile != nil,
           !prompt.isEmpty {
            parts.append("Additional instruction: \(prompt)")
        }

        let focusTerms = focus
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !focusTerms.isEmpty {
            parts.append("Pay special attention to these visible details: \(focusTerms.joined(separator: "; ")).")
        }

        if let triggerToken = triggerToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !triggerToken.isEmpty {
            parts.append("Do not invent hidden details. The exact trigger token \(triggerToken) will be added separately.")
        }

        return parts
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    static func captionOutput(_ caption: String, triggerToken: String?) -> String {
        let trimmed = caption.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let token = triggerToken?.trimmingCharacters(in: .whitespacesAndNewlines),
              !token.isEmpty else {
            return trimmed
        }
        guard !trimmed.isEmpty else {
            return token
        }
        if trimmed.lowercased().hasPrefix(token.lowercased()) {
            return trimmed
        }
        return "\(token) \(trimmed)"
    }
}
