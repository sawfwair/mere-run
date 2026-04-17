import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Vision Caption Command

struct VisionCaption: AsyncParsableCommand {
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
    var prompt: String = "Write a short, concrete caption describing the image for LoRA training. Avoid fluff."

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
            print("No model specified, downloading \(Qwen3VLAutoCaptioner.modelId)...")
            let autoCaptioner = Qwen3VLAutoCaptioner()
            modelURL = try await autoCaptioner.ensureReady { progress in
                print("\r\(progress.status) (\(Int(progress.fraction * 100))%)", terminator: "")
                fflush(stdout)
            }
            print()  // newline after progress
        }

        let captioner = try QwenVLCaptioner(modelRoot: modelURL)
        let config = QwenVLCaptioner.ModelConfig(maxNewTokens: maxTokens, temperature: temperature, topP: topP)

        for path in images {
            let imageURL = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: imageURL.path) else {
                throw ValidationError("Image not found: \(imageURL.path)")
            }

            let caption = try captioner.caption(imageURL: imageURL, prompt: prompt, config: config)
            let captionURL: URL = {
                if let outDirURL {
                    return outDirURL.appendingPathComponent(imageURL.deletingPathExtension().lastPathComponent + ".txt")
                }
                return imageURL.deletingPathExtension().appendingPathExtension("txt")
            }()
            let out = caption.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
            try out.data(using: .utf8)?.write(to: captionURL)
            print("\(imageURL.path) -> \(captionURL.path)")
        }
    }
}
