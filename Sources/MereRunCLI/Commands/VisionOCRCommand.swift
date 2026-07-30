import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Vision OCR Command

enum OCRBackend: String, ExpressibleByArgument, CaseIterable {
    case lighton
    case glm
    case infinity
}

enum InfinityParserBackend: String, ExpressibleByArgument, CaseIterable {
    case transformers
    case vllmEngine = "vllm-engine"
    case vllmServer = "vllm-server"
}

enum InfinityParserRuntime: String, ExpressibleByArgument, CaseIterable {
    case native
    case external
}

enum InfinityParserTask: String, ExpressibleByArgument, CaseIterable {
    case doc2json
    case doc2md
    case custom
}

enum InfinityParserOutputFormat: String, ExpressibleByArgument, CaseIterable {
    case md
    case json
}

struct VisionOCR: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ocr",
        abstract: "Extract text from images using LightOnOCR, GLM-OCR, or Infinity-Parser2.",
        discussion: """
        Performs OCR on one or more images and outputs the extracted text.

        LightOnOCR uses the managed model id `vision-ocr-lighton` by default.
        You can also pass --model with either a canonical managed id or a local model path.

        GLM-OCR support uses the upstream `glmocr` CLI. Install and configure it per:
          https://github.com/zai-org/GLM-OCR

        Infinity-Parser2 uses the native Qwen-family Swift runtime by default
        with managed model id `vision-ocr-infinity-pro-int8`. The unquantized
        `vision-ocr-infinity-pro` model remains available for heavyweight quality
        evals that can tolerate substantially higher memory use. Use
        --infinity-runtime external only when comparing against the upstream
        `infinity_parser2` Python CLI or an already-started vLLM server.
        """
    )

    @Argument(help: "One or more image file paths.")
    var images: [String] = []

    @Option(name: [.customShort("b"), .long], help: "OCR backend: lighton | glm | infinity (default: lighton).")
    var backend: OCRBackend = .lighton

    @Flag(name: [.long], help: "Compare LightOn against the selected secondary backend; defaults to GLM when --backend lighton.")
    var compare: Bool = false

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the LightOnOCR model directory.")
    var model: String = ModelResolver.ModelID.lightOnOCR.rawValue

    @Option(name: [.long], help: "Path to `glmocr` CLI (default: glmocr, resolved via PATH).")
    var glmocrCLI: String = "glmocr"

    @Option(name: [.long], help: "Path to GLM-OCR config YAML (optional; forwarded to glmocr --config).")
    var glmConfig: String?

    @Option(name: [.long], help: "Infinity-Parser2 runtime: native | external.")
    var infinityRuntime: InfinityParserRuntime = .native

    @Option(name: [.long], help: "Path to Infinity-Parser2 `parser` CLI for --infinity-runtime external.")
    var infinityParserCLI: String = "parser"

    @Option(name: [.long], help: "Infinity-Parser2 managed model id or local path for native runs; upstream model/server id for external runs.")
    var infinityModel: String = Q35Resources.infinityParser2ProInt8ModelId

    @Option(name: [.long], help: "External Infinity-Parser2 backend: transformers | vllm-engine | vllm-server.")
    var infinityBackend: InfinityParserBackend = .vllmServer

    @Option(name: [.customLong("infinity-api-url")], help: "External Infinity-Parser2 vLLM server chat completions URL.")
    var infinityAPIURL: String = "http://localhost:8000/v1/chat/completions"

    @Option(name: [.customLong("infinity-api-key")], help: "Optional external vLLM server API key; upstream accepts EMPTY for local servers.")
    var infinityAPIKey: String = "EMPTY"

    @Option(name: [.long], help: "Infinity-Parser2 task: doc2json | doc2md | custom.")
    var infinityTask: InfinityParserTask = .doc2json

    @Option(name: [.long], help: "Infinity-Parser2 custom prompt, required when --infinity-task custom.")
    var infinityPrompt: String?

    @Option(name: [.long], help: "Infinity-Parser2 output format: md | json.")
    var infinityOutputFormat: InfinityParserOutputFormat = .md

    @Option(name: [.long], help: "External Infinity-Parser2 batch size forwarded to parser.")
    var infinityBatchSize: Int = 1

    @Option(name: [.long], help: "External Infinity-Parser2 model cache directory.")
    var infinityModelCacheDir: String?

    @Option(name: [.long], help: "Infinity-Parser2 minimum image pixels.")
    var infinityMinPixels: Int = 2048

    @Option(name: [.long], help: "Infinity-Parser2 maximum image pixels.")
    var infinityMaxPixels: Int = 16_777_216

    @Option(name: [.customShort("o"), .long], help: "Output directory for .txt files (default: print to stdout).")
    var outputDir: String?

    @Option(name: [.long], help: "Max new tokens to generate.")
    var maxTokens: Int = 4096

    @Option(name: [.long], help: "Sampling temperature (lower = more deterministic).")
    var temperature: Float = 0.2

    @Flag(name: [.short, .long], help: "Output only the OCR text (no file paths).")
    var quiet: Bool = false

    func validate() throws {
        guard infinityBatchSize > 0 else {
            throw ValidationError("--infinity-batch-size must be greater than zero.")
        }
        guard infinityMinPixels > 0 else {
            throw ValidationError("--infinity-min-pixels must be greater than zero.")
        }
        guard infinityMaxPixels >= infinityMinPixels else {
            throw ValidationError("--infinity-max-pixels must be greater than or equal to --infinity-min-pixels.")
        }
        if infinityTask == .custom {
            guard let prompt = infinityPrompt?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !prompt.isEmpty else {
                throw ValidationError("--infinity-prompt is required when --infinity-task custom.")
            }
        }
        if infinityOutputFormat == .json && infinityTask != .doc2json {
            throw ValidationError("--infinity-output-format json is only supported with --infinity-task doc2json.")
        }
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        guard !images.isEmpty else {
            throw ValidationError("Provide at least one image path.")
        }

        let needsLightOn = compare || backend == .lighton
        let modelURL: URL?
        if needsLightOn {
            do {
                let resolved = try await ManagedModelResolver.resolveForRuntime(
                    requestedModel: model,
                    defaultModelID: ModelResolver.ModelID.lightOnOCR.rawValue
                )
                modelURL = resolved.url
            } catch let error as ManagedModelResolver.ResolverError {
                throw ValidationError(error.localizedDescription)
            }
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
        let infinity = InfinityParser2CLI(
            executable: infinityParserCLI,
            model: infinityModel,
            backend: infinityBackend,
            apiURL: infinityAPIURL,
            apiKey: infinityAPIKey,
            task: infinityTask,
            prompt: infinityPrompt,
            outputFormat: infinityOutputFormat,
            batchSize: infinityBatchSize,
            modelCacheDir: infinityModelCacheDir,
            minPixels: infinityMinPixels,
            maxPixels: infinityMaxPixels
        )

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
                guard let modelURL else { throw ValidationError("LightOn model resolution failed.") }
                let lighton = try await generator.ocr(
                    imageURL: imageURL,
                    modelPath: modelURL.path,
                    config: config
                )
                let compareBackend = backend == .lighton ? OCRBackend.glm : backend
                let secondary = try await selectedSecondaryResult(
                    compareBackend,
                    imageURL: imageURL,
                    glm: glm,
                    infinity: infinity
                )
                output = .comparison(
                    lighton: lighton.text,
                    externalName: secondary.name,
                    external: secondary.text,
                    externalJSON: secondary.json
                )
            } else if backend == .lighton {
                guard let modelURL else { throw ValidationError("LightOn model resolution failed.") }
                let result = try await generator.ocr(
                    imageURL: imageURL,
                    modelPath: modelURL.path,
                    config: config
                )
                output = .single(text: result.text)
            } else if backend == .glm {
                let glmResult = try glm.ocr(imageURL: imageURL)
                output = .single(text: glmResult.text)
            } else {
                let infinityResult = try await infinityResult(imageURL: imageURL, externalCLI: infinity)
                output = .single(text: infinityResult.text)
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

    private func selectedSecondaryResult(
        _ backend: OCRBackend,
        imageURL: URL,
        glm: GLMOCRCLI,
        infinity: InfinityParser2CLI
    ) async throws -> NamedOCRResult {
        switch backend {
        case .lighton:
            throw ValidationError("--compare needs a secondary backend; use --backend glm or --backend infinity.")
        case .glm:
            let result = try glm.ocr(imageURL: imageURL)
            return NamedOCRResult(name: "GLM-OCR", text: result.text, json: result.json)
        case .infinity:
            return try await infinityResult(imageURL: imageURL, externalCLI: infinity)
        }
    }

    private func infinityResult(
        imageURL: URL,
        externalCLI: InfinityParser2CLI
    ) async throws -> NamedOCRResult {
        if infinityRuntime == .external {
            let result = try externalCLI.ocr(imageURL: imageURL)
            return NamedOCRResult(name: "Infinity-Parser2 external", text: result.text, json: result.json)
        }

        let resolvedModelID = ModelResolver.ModelID(rawValue: infinityModel)
        let generator = Q35Generator(
            modelId: resolvedModelID?.rawValue ?? Q35Resources.infinityParser2ProInt8ModelId,
            visionMinPixels: infinityMinPixels,
            visionMaxPixels: infinityMaxPixels
        )
        let modelPath = resolvedModelID == nil ? infinityModel : nil
        let request = ChatRequest(
            messages: [
                ChatMessage(
                    role: .user,
                    content: nativeInfinityPrompt(),
                    imageUrl: imageURL.path
                ),
            ],
            maxTokens: maxTokens,
            temperature: Double(temperature),
            topP: 0.95,
            showThinking: false,
            requiresJSON: infinityOutputFormat == .json,
            maxContextTokens: Q35Resources.defaultContextLength
        )
        do {
            let response = try await generator.chat(
                request,
                modelPath: modelPath,
                progressHandler: progressHandler
            )
            let rendered = response.response.trimmingCharacters(in: .whitespacesAndNewlines)
            return NamedOCRResult(
                name: "Infinity-Parser2 native",
                text: rendered,
                json: infinityOutputFormat == .json ? rendered : nil
            )
        } catch let error as Q35Error {
            throw ValidationError(error.localizedDescription)
        }
    }

    private func nativeInfinityPrompt() -> String {
        switch infinityTask {
        case .doc2json:
            return """
            Parse this document image into structured JSON. Preserve reading order, tables, formulas, layout labels, and visible text. Return only valid JSON.
            """
        case .doc2md:
            return """
            Parse this document image into clean Markdown. Preserve reading order, headings, tables, formulas, and visible text. Return only Markdown.
            """
        case .custom:
            return infinityPrompt ?? ""
        }
    }

    private func progressHandler(_ progress: ChatProgress) {
        guard !quiet, let message = progress.message, !message.isEmpty else { return }
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}

private enum OCROutput {
    case single(text: String)
    case comparison(lighton: String, externalName: String, external: String, externalJSON: String?)

    func render() -> String {
        switch self {
        case .single(let text):
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        case .comparison(let lighton, let externalName, let external, let externalJSON):
            var out = ""
            out += "=== LightOnOCR ===\n"
            out += lighton.trimmingCharacters(in: .whitespacesAndNewlines)
            out += "\n\n=== \(externalName) ===\n"
            out += external.trimmingCharacters(in: .whitespacesAndNewlines)
            if let externalJSON, !externalJSON.isEmpty {
                out += "\n\n=== \(externalName) (json) ===\n"
                out += externalJSON.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            return out
        }
    }
}

private struct NamedOCRResult {
    let name: String
    let text: String
    let json: String?
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

private struct InfinityParser2CLI {
    struct Result {
        let text: String
        let json: String?
    }

    let executable: String
    let model: String
    let backend: InfinityParserBackend
    let apiURL: String
    let apiKey: String
    let task: InfinityParserTask
    let prompt: String?
    let outputFormat: InfinityParserOutputFormat
    let batchSize: Int
    let modelCacheDir: String?
    let minPixels: Int
    let maxPixels: Int

    func ocr(imageURL: URL) throws -> Result {
        let fm = FileManager.default
        let tempRoot = fm.temporaryDirectory.appendingPathComponent("infinity-parser2-\(UUID().uuidString)", isDirectory: true)
        try fm.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: tempRoot) }

        var args = [
            imageURL.path,
            "--output-dir", tempRoot.path,
            "--task", task.rawValue,
            "--output-format", outputFormat.rawValue,
            "--batch-size", String(batchSize),
            "--backend", backend.rawValue,
            "--model-name", model,
            "--api-url", apiURL,
            "--api-key", apiKey,
            "--min-pixels", String(minPixels),
            "--max-pixels", String(maxPixels),
        ]
        if let prompt, !prompt.isEmpty {
            args += ["--prompt", prompt]
        }
        if let modelCacheDir, !modelCacheDir.isEmpty {
            args += ["--model-cache-dir", modelCacheDir]
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

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        let stdoutText = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderrText = String(data: stderrData, encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            throw ValidationError("Infinity-Parser2 failed (\(process.terminationStatus)). Ensure `parser` is installed and the selected backend is reachable. Stderr: \(stderrText)")
        }

        let outputFileName = outputFormat == .json ? "result.json" : "result.md"
        let outputURL = findFirstFile(named: outputFileName, under: tempRoot)
        let rendered = outputURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines)
        let jsonURL = outputFormat == .json ? outputURL : findFirstFile(named: "result.json", under: tempRoot)
        let json = jsonURL.flatMap { try? String(contentsOf: $0, encoding: .utf8) }?.trimmingCharacters(in: .whitespacesAndNewlines)

        if let rendered, !rendered.isEmpty {
            return Result(text: rendered, json: json)
        }
        if outputFormat == .json, let json, !json.isEmpty {
            return Result(text: json, json: json)
        }

        throw ValidationError("Infinity-Parser2 produced no result.{stdout:\(stdoutText.prefix(500))}{stderr:\(stderrText.prefix(500))}")
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
