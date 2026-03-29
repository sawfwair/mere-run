import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Vision OCR Command

enum OCRBackend: String, ExpressibleByArgument, CaseIterable {
    case lighton
    case glm
}

struct VisionOCR: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr",
        abstract: "Extract text from images using LightOnOCR or GLM-OCR.",
        discussion: """
        Performs OCR on one or more images and outputs the extracted text.

        LightOnOCR model should be installed in the local mere.run model store:
          mere.run model pull vision-ocr-lighton

        GLM-OCR support uses the upstream `glmocr` CLI. Install and configure it per:
          https://github.com/zai-org/GLM-OCR
        """
    )

    @Argument(help: "One or more image file paths.")
    var images: [String] = []

    @Option(name: [.customShort("b"), .long], help: "OCR backend: lighton | glm (default: lighton).")
    var backend: OCRBackend = .lighton

    @Flag(name: [.long], help: "Run both backends and output both results (requires LightOn model).")
    var compare: Bool = false

    @Option(name: [.customShort("m"), .long], help: "Path to LightOnOCR-2-1B model directory (required for lighton/compare).")
    var model: String?

    @Option(name: [.long], help: "Path to `glmocr` CLI (default: glmocr, resolved via PATH).")
    var glmocrCLI: String = "glmocr"

    @Option(name: [.long], help: "Path to GLM-OCR config YAML (optional; forwarded to glmocr --config).")
    var glmConfig: String?

    @Option(name: [.customShort("o"), .long], help: "Output directory for .txt files (default: print to stdout).")
    var outputDir: String?

    @Option(name: [.long], help: "Max new tokens to generate.")
    var maxTokens: Int = 4096

    @Option(name: [.long], help: "Sampling temperature (lower = more deterministic).")
    var temperature: Float = 0.2

    @Flag(name: [.short, .long], help: "Output only the OCR text (no file paths).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        guard !images.isEmpty else {
            throw ValidationError("Provide at least one image path.")
        }

        let needsLightOn = compare || backend == .lighton
        let modelURL: URL?
        if needsLightOn {
            guard let model else {
                throw ValidationError("Missing --model (required for backend=lighton or --compare).")
            }
            let url = URL(fileURLWithPath: model).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Model path not found: \(url.path)")
            }
            modelURL = url
        } else {
            modelURL = nil
        }

        let outDirURL: URL? = outputDir.map { URL(fileURLWithPath: $0).standardizedFileURL }
        if let outDirURL {
            try FileManager.default.createDirectory(at: outDirURL, withIntermediateDirectories: true)
        }

        let generator = LightOnOCRGenerator()
        let config = LightOnOCRGenerator.Config(
            maxNewTokens: maxTokens,
            temperature: temperature,
            logProgress: MereRunRuntimeDebug.isEnabled(["MERERUN_OCR_DEBUG"])
        )
        let glm = GLMOCRCLI(executable: glmocrCLI, configPath: glmConfig)

        for path in images {
            let imageURL = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: imageURL.path) else {
                throw ValidationError("Image not found: \(imageURL.path)")
            }

            if !quiet {
                FileHandle.standardError.write(Data("Processing \(imageURL.lastPathComponent)...\n".utf8))
            }

            let output: OCROutput
            if compare {
                guard let modelURL else {
                    throw ValidationError("Missing --model (required for --compare).")
                }
                let lighton = try await generator.ocr(
                    imageURL: imageURL,
                    modelPath: modelURL.path,
                    config: config
                )
                let glmResult = try glm.ocr(imageURL: imageURL)

                output = .comparison(
                    lighton: lighton.text,
                    glm: glmResult.text,
                    glmJSON: glmResult.json
                )
            } else if backend == .lighton {
                guard let modelURL else {
                    throw ValidationError("Missing --model (required for backend=lighton).")
                }
                let result = try await generator.ocr(
                    imageURL: imageURL,
                    modelPath: modelURL.path,
                    config: config
                )
                output = .single(text: result.text)
            } else {
                let glmResult = try glm.ocr(imageURL: imageURL)
                output = .single(text: glmResult.text)
            }

            if let outDirURL {
                // Write to file
                let outputURL = outDirURL.appendingPathComponent(
                    imageURL.deletingPathExtension().lastPathComponent + ".txt"
                )
                let rendered = output.render()
                try (rendered + "\n").data(using: .utf8)?.write(to: outputURL)
                if !quiet {
                    print("\(imageURL.path) -> \(outputURL.path)")
                }
            } else {
                // Print to stdout
                if !quiet && images.count > 1 {
                    print("--- \(imageURL.lastPathComponent) ---")
                }
                print(output.render().trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
    }
}

private enum OCROutput {
    case single(text: String)
    case comparison(lighton: String, glm: String, glmJSON: String?)

    func render() -> String {
        switch self {
        case .single(let text):
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .comparison(let lighton, let glm, let glmJSON):
            var out = ""
            out += "=== LightOnOCR ===\n"
            out += lighton.trimmingCharacters(in: .whitespacesAndNewlines)
            out += "\n\n=== GLM-OCR ===\n"
            out += glm.trimmingCharacters(in: .whitespacesAndNewlines)
            if let glmJSON, !glmJSON.isEmpty {
                out += "\n\n=== GLM-OCR (json) ===\n"
                out += glmJSON.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return out
        }
    }
}

private struct GLMOCRCLI {
    struct Result {
        let text: String
        let json: String?
    }

    let executable: String
    let configPath: String?

    func ocr(imageURL: URL) throws -> Result {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("glmocr-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        var args = ["parse", imageURL.path, "--output", tempRoot.path]
        if let configPath, !configPath.isEmpty {
            args += ["--config", configPath]
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable] + args

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()
        process.waitUntilExit()

        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            throw ValidationError("GLM-OCR failed (\(process.terminationStatus)). Ensure `glmocr` is installed and configured. Stderr: \(stderrText)")
        }

        let mdURL = findFirstFile(named: "result.md", under: tempRoot)
        let jsonURL = findFirstFile(named: "result.json", under: tempRoot)

        let markdown = mdURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines)
        let json = jsonURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines)

        if let markdown, !markdown.isEmpty {
            return Result(text: markdown, json: json)
        }
        if let json, !json.isEmpty {
            return Result(text: json, json: json)
        }

        // Fallback: show stdout/stderr for debugging
        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stdoutText = String(data: stdoutData, encoding: .utf8) ?? ""
        throw ValidationError("GLM-OCR produced no result.{stdout:\(stdoutText.prefix(500))}{stderr:\(stderrText.prefix(500))}")
    }

    private func findFirstFile(named filename: String, under root: URL) -> URL? {
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
        while let url = enumerator?.nextObject() as? URL {
            if url.lastPathComponent == filename {
                return url
            }
        }
        return nil
    }
}
