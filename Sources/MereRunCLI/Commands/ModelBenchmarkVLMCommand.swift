import ArgumentParser
import Foundation
import MediaIO
import MereRunCore
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if canImport(Darwin)
import Darwin
#endif

struct ModelBenchmarkVLM: AsyncParsableCommand {
    static let qwenInspectModelID = "vision-inspect-qwen3-vl-2b"

    static let configuration = CommandConfiguration(
        commandName: "vlm",
        abstract: "Compare vision-language chat models on synthetic or lmms-eval datasets."
    )

    @Option(name: [.long], help: "Comma-separated model ids or aliases to compare.")
    var models: String?

    @Option(name: [.long], help: "Benchmark dataset: synthetic-vqa-v1, mathvista-testmini, mmmu-val, chartqa, docvqa-val, or mme.")
    var dataset: VLMBenchmarkDataset = .syntheticVQA

    @Option(name: [.long], help: "Raw lmms-eval task list. Overrides --dataset and uses the external lmms-eval runner.")
    var lmmsTasks: String?

    @Option(name: [.long], help: "Directory for generated benchmark fixture images. Defaults to a temp directory.")
    var fixtureDir: String?

    @Option(name: [.long], help: "Directory for external lmms-eval output. Defaults under .build/vlm-benchmarks.")
    var outputDir: String?

    @Option(name: [.long], help: "Optional lmms-eval source checkout; runs python -m lmms_eval from this directory.")
    var lmmsEvalRoot: String?

    @Option(name: [.long], help: "Python executable used for lmms-eval.")
    var lmmsEvalPython: String = "python3"

    @Flag(name: [.long], help: "Print external lmms-eval commands without starting servers or running them.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Use an already running OpenAI-compatible endpoint instead of starting mere.run api serve.")
    var externalEndpoint: Bool = false

    @Option(name: [.long], help: "OpenAI-compatible base URL for --external-endpoint, or generated server URL otherwise.")
    var baseURL: String?

    @Option(name: [.long], help: "API key passed to mere.run api serve and lmms-eval.")
    var apiKey: String = "mere-run-local-eval"

    @Option(name: [.long], help: "Host for generated mere.run api serve processes.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "First port used for generated mere.run api serve processes.")
    var port: Int = 11934

    @Option(name: [.long], help: "Limit examples per lmms-eval task. Omit for a full benchmark.")
    var limit: String?

    @Flag(name: [.long], help: "Ask lmms-eval to write per-sample logs.")
    var logSamples: Bool = false

    @Option(name: [.long], help: "Maximum output tokens per image question.")
    var maxTokens: Int = 24

    @Option(name: [.long], help: "Context size for managed API-runtime models.")
    var contextSize: Int = 4096

    @Option(name: [.long], help: "Sampling temperature.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p sampling.")
    var topP: Double = 1

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    private var usesExternalBenchmark: Bool {
        if let lmmsTasks = lmmsTasks?.trimmingCharacters(in: .whitespacesAndNewlines), !lmmsTasks.isEmpty {
            return true
        }
        return dataset.lmmsEvalTasks != nil
    }

    func validate() throws {
        guard maxTokens > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        guard contextSize > 0 else {
            throw ValidationError("--context-size must be greater than zero.")
        }
        guard (0...2).contains(temperature), temperature.isFinite else {
            throw ValidationError("--temperature must be finite and between 0 and 2.")
        }
        guard (0...1).contains(topP), topP.isFinite else {
            throw ValidationError("--top-p must be finite and between 0 and 1.")
        }
        guard port > 0, port <= Int(UInt16.max) else {
            throw ValidationError("--port must be between 1 and \(UInt16.max).")
        }
        if let limit {
            guard Double(limit) != nil else {
                throw ValidationError("--limit must be numeric.")
            }
        }
        if externalEndpoint, baseURL?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            throw ValidationError("--external-endpoint requires --base-url.")
        }
        _ = try modelIDs()
    }

    func run() async throws {
        if let external = try externalBenchmarkPlanIfNeeded() {
            try await runExternalBenchmark(external)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)

        let fixtureURL = try resolvedFixtureDirectory()
        let suite = try VLMBenchmarkSuite.writeDefaultFixtures(to: fixtureURL)
        let requestedModels = try modelIDs()
        let pool = RuntimeModelPool(
            defaultModelID: requestedModels.first ?? Gemma4Resources.visionTwelveBModelId,
            defaultEngine: .textChatGemma4,
            startupModelPath: nil
        )

        var modelResults: [VLMBenchmarkModelResult] = []
        modelResults.reserveCapacity(requestedModels.count)

        for modelID in requestedModels {
            if !json {
                CLIStderr.write("Benchmarking \(modelID)...\n")
            }
            do {
                let result: VLMBenchmarkModelResult
                if Self.isQwenInspectModelID(modelID) {
                    result = try await runQwenInspectModel(modelID: modelID, suite: suite)
                } else {
                    result = try await runManagedModel(modelID: modelID, suite: suite, pool: pool)
                    _ = try? await pool.unloadModel(idOrAlias: modelID)
                }
                modelResults.append(result)
            } catch {
                modelResults.append(
                    VLMBenchmarkModelResult.failed(
                        model: modelID,
                        error: error.localizedDescription
                    )
                )
            }
        }

        let report = VLMBenchmarkReport(
            suite: suite.name,
            fixtureDirectory: suite.fixtureDirectory.path,
            cases: suite.cases.map(VLMBenchmarkCaseSummary.init),
            models: modelResults
        )
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func modelIDs() throws -> [String] {
        let defaultModels = usesExternalBenchmark
            ? Gemma4Resources.visionTwelveBModelId
            : "\(Gemma4Resources.visionTwelveBModelId),\(Self.qwenInspectModelID)"
        let parsed = (models ?? defaultModels)
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !parsed.isEmpty else {
            throw ValidationError("--models must contain at least one model id.")
        }
        return parsed
    }

    private func externalBenchmarkPlanIfNeeded() throws -> VLMExternalBenchmarkPlan? {
        if let lmmsTasks = lmmsTasks?.trimmingCharacters(in: .whitespacesAndNewlines), !lmmsTasks.isEmpty {
            return try VLMExternalBenchmarkPlan(
                dataset: "lmms-eval",
                tasks: lmmsTasks,
                outputDirectory: resolvedExternalOutputDirectory(datasetSlug: "lmms-eval")
            )
        }
        guard let tasks = dataset.lmmsEvalTasks else {
            return nil
        }
        return try VLMExternalBenchmarkPlan(
            dataset: dataset.rawValue,
            tasks: tasks,
            outputDirectory: resolvedExternalOutputDirectory(datasetSlug: dataset.rawValue)
        )
    }

    private func resolvedExternalOutputDirectory(datasetSlug: String) throws -> URL {
        let url: URL
        if let outputDir {
            url = URL(fileURLWithPath: outputDir).standardizedFileURL
        } else {
            let timestamp = ISO8601DateFormatter()
                .string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build/vlm-benchmarks/\(datasetSlug)-\(timestamp)", isDirectory: true)
        }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func runExternalBenchmark(_ plan: VLMExternalBenchmarkPlan) async throws {
        let requestedModels = try modelIDs()
        var results: [VLMExternalModelResult] = []
        results.reserveCapacity(requestedModels.count)

        for (offset, modelID) in requestedModels.enumerated() {
            let modelPort = port + offset
            let endpointURL = externalEndpoint
                ? try resolvedExternalBaseURL()
                : "http://\(host):\(modelPort)/v1"
            let serverProcess: Process?
            if dryRun || externalEndpoint {
                serverProcess = nil
            } else {
                serverProcess = try startAPIServer(modelID: modelID, port: modelPort)
                try waitForAPIHealth(host: host, port: modelPort, modelID: modelID)
            }
            defer {
                if let serverProcess, serverProcess.isRunning {
                    serverProcess.terminate()
                    serverProcess.waitUntilExit()
                }
            }

            let invocation = try makeLMMSEvalInvocation(
                modelID: modelID,
                plan: plan,
                baseURL: endpointURL
            )
            if dryRun {
                results.append(
                    VLMExternalModelResult(
                        model: modelID,
                        command: invocation.shellDescription,
                        outputPath: invocation.outputDirectory.path,
                        exitStatus: nil,
                        elapsedSeconds: nil
                    )
                )
                continue
            }

            let start = Date()
            let processResult = try runExternalProcess(invocation)
            results.append(
                VLMExternalModelResult(
                    model: modelID,
                    command: invocation.shellDescription,
                    outputPath: invocation.outputDirectory.path,
                    exitStatus: processResult.status,
                    elapsedSeconds: Date().timeIntervalSince(start)
                )
            )
            guard processResult.status == 0 else {
                throw ValidationError(
                    "lmms-eval failed for \(modelID) with status \(processResult.status). Stderr tail: \(processResult.stderrTail)"
                )
            }
        }

        let report = VLMExternalBenchmarkReport(
            dataset: plan.dataset,
            tasks: plan.tasks,
            outputDirectory: plan.outputDirectory.path,
            dryRun: dryRun,
            models: results
        )
        if json || dryRun {
            print(try report.jsonString())
        } else {
            print(report.renderText())
        }
    }

    private func resolvedExternalBaseURL() throws -> String {
        guard let raw = baseURL?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            throw ValidationError("--external-endpoint requires --base-url.")
        }
        return raw.hasSuffix("/") ? String(raw.dropLast()) : raw
    }

    private func startAPIServer(modelID: String, port: Int) throws -> Process {
        let engine = try apiEngine(for: modelID)
        let process = Process()
        process.executableURL = CurrentExecutable.url()
        process.arguments = [
            "api", "serve",
            "--engine", engine.rawValue,
            "--model", modelID,
            "--host", host,
            "--port", String(port),
            "--context-size", String(contextSize),
            "--api-key", apiKey,
        ]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        forwardPipe(stdout, prefix: "[api serve] ")
        forwardPipe(stderr, prefix: "[api serve] ")
        try process.run()
        return process
    }

    private func apiEngine(for modelID: String) throws -> APIEngine {
        guard let spec = ManagedModelCatalog.spec(for: modelID),
              let engine = spec.defaultRuntimeServingEngine else {
            throw ValidationError("External VLM datasets require an API-servable managed model; '\(modelID)' is not one.")
        }
        switch engine.canonical {
        case .textChatGemma4:
            return .textChatGemma4
        case .textChatLaguna:
            return .textChatLaguna
        case .textChatQ35:
            return .textChatQ35
        case .textChatQ36:
            return .textChatQ36
        case .textChatKlein:
            return .textChatKlein
        case .textChatLFM2:
            return .textChatLFM2
        case .textChatDeepseekV4Flash:
            return .textChatDeepseekV4Flash
        case .textCode:
            return .textCode
        }
    }

    private func waitForAPIHealth(host: String, port: Int, modelID: String) throws {
        guard let healthURL = URL(string: "http://\(host):\(port)/health") else {
            throw ValidationError("Invalid API health URL for \(host):\(port).")
        }
        let deadline = Date().addingTimeInterval(120)
        while Date() < deadline {
            if let data = try? Data(contentsOf: healthURL),
               String(decoding: data, as: UTF8.self).contains("\"ok\"") {
                return
            }
            Thread.sleep(forTimeInterval: 0.5)
        }
        throw ValidationError("Timed out waiting for API server for \(modelID) on \(healthURL.absoluteString).")
    }

    private func makeLMMSEvalInvocation(
        modelID: String,
        plan: VLMExternalBenchmarkPlan,
        baseURL: String
    ) throws -> VLMExternalInvocation {
        let outputDirectory = plan.outputDirectory.appendingPathComponent(safePathComponent(modelID), isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let modelArgs = [
            "model_version=\(modelID)",
            "base_url=\(baseURL)",
            "api_key=\(apiKey)",
        ].joined(separator: ",")
        var arguments = [
            "-m", "lmms_eval",
            "--model", "openai",
            "--model_args", modelArgs,
            "--tasks", plan.tasks,
            "--batch_size", "1",
            "--output_path", outputDirectory.path,
            "--gen_kwargs", "temperature=\(temperature),top_p=\(topP),max_new_tokens=\(maxTokens)",
        ]
        if let limit {
            arguments += ["--limit", limit]
        }
        if logSamples {
            arguments.append("--log_samples")
        }

        let rootURL = lmmsEvalRoot.map { URL(fileURLWithPath: $0).standardizedFileURL }
        return VLMExternalInvocation(
            executable: lmmsEvalPython,
            arguments: arguments,
            currentDirectory: rootURL,
            outputDirectory: outputDirectory
        )
    }

    private func safePathComponent(_ raw: String) -> String {
        raw.replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
    }

    private func runExternalProcess(_ invocation: VLMExternalInvocation) throws -> VLMExternalProcessResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [invocation.executable] + invocation.arguments
        process.currentDirectoryURL = invocation.currentDirectory
        var environment = ProcessInfo.processInfo.environment
        if let currentDirectory = invocation.currentDirectory {
            let existing = environment["PYTHONPATH"].map { ":\($0)" } ?? ""
            environment["PYTHONPATH"] = currentDirectory.path + existing
        }
        process.environment = environment

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        let capture = VLMExternalProcessCapture()
        capture.forward(stdout, prefix: "[lmms-eval] ", storeTail: false)
        capture.forward(stderr, prefix: "[lmms-eval] ", storeTail: true)
        try process.run()
        process.waitUntilExit()
        capture.finish(stdout, storeTail: false)
        capture.finish(stderr, storeTail: true)
        return VLMExternalProcessResult(
            status: process.terminationStatus,
            stderrTail: capture.stderrTail()
        )
    }

    private func forwardPipe(_ pipe: Pipe, prefix: String) {
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if let text = String(data: data, encoding: .utf8) {
                CLIStderr.write(text.linePrefixed(prefix))
            } else {
                FileHandle.standardError.write(data)
            }
        }
    }

    private func resolvedFixtureDirectory() throws -> URL {
        let url: URL
        if let fixtureDir {
            url = URL(fileURLWithPath: fixtureDir).standardizedFileURL
        } else {
            url = FileManager.default.temporaryDirectory
                .appendingPathComponent("mererun-vlm-benchmark-\(UUID().uuidString)", isDirectory: true)
        }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func isQwenInspectModelID(_ modelID: String) -> Bool {
        let normalized = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == qwenInspectModelID
            || normalized == Qwen3VLAutoCaptioner.modelId.lowercased()
    }

    private func runManagedModel(
        modelID: String,
        suite: VLMBenchmarkSuite,
        pool: RuntimeModelPool
    ) async throws -> VLMBenchmarkModelResult {
        let modelStart = Date()
        var caseResults: [VLMBenchmarkCaseResult] = []
        caseResults.reserveCapacity(suite.cases.count)

        for benchmarkCase in suite.cases {
            do {
                caseResults.append(
                    try await runManagedCase(
                        benchmarkCase,
                        modelID: modelID,
                        pool: pool
                    )
                )
            } catch {
                caseResults.append(
                    benchmarkCase.failedResult(
                        model: modelID,
                        error: error.localizedDescription
                    )
                )
            }
        }

        return VLMBenchmarkModelResult(
            model: modelID,
            backend: "managed-api-runtime",
            elapsedSeconds: Date().timeIntervalSince(modelStart),
            cases: caseResults,
            error: nil
        )
    }

    private func runManagedCase(
        _ benchmarkCase: VLMBenchmarkCase,
        modelID: String,
        pool: RuntimeModelPool
    ) async throws -> VLMBenchmarkCaseResult {
        let request = OpenAIChatRequest(
            model: modelID,
            messages: [
                OpenAIChatMessage(
                    role: "user",
                    content: benchmarkCase.prompt,
                    imageURLs: [benchmarkCase.imageURL.path]
                ),
            ],
            temperature: temperature,
            top_p: topP,
            max_tokens: maxTokens,
            stream: false
        )
        let memoryBefore = VLMBenchmarkMemorySnapshot.currentResidentBytes()
        let caseStart = Date()
        let plan = try await pool.makeChatPlan(
            for: request,
            fallbackLoraPath: nil,
            serverContextSize: contextSize
        )
        let response: ChatResponse
        do {
            response = try await plan.lease.chat(plan.request, progressHandler: progressHandler())
        } catch {
            await plan.lease.release()
            throw error
        }
        await plan.lease.release()
        let elapsed = Date().timeIntervalSince(caseStart)
        let memoryAfter = VLMBenchmarkMemorySnapshot.currentResidentBytes()
        return benchmarkCase.result(
            model: modelID,
            response: response.response,
            elapsedSeconds: elapsed,
            promptTokens: response.promptTokens,
            generatedTokens: response.tokensGenerated,
            timing: response.timing,
            residentMemoryBeforeBytes: memoryBefore,
            residentMemoryAfterBytes: memoryAfter
        )
    }

    private func runQwenInspectModel(
        modelID: String,
        suite: VLMBenchmarkSuite
    ) async throws -> VLMBenchmarkModelResult {
        let modelStart = Date()
        let autoCaptioner = Qwen3VLAutoCaptioner()
        let verb = await autoCaptioner.isModelCached() ? "Loading cached" : "Downloading"
        if !json {
            CLIStderr.write("\(verb) \(Qwen3VLAutoCaptioner.modelId)...\n")
        }
        let modelURL = try await autoCaptioner.ensureReady { progress in
            guard !json else { return }
            CLIStderr.write("\r\(progress.status) (\(Int(progress.fraction * 100))%)")
        }
        if !json {
            CLIStderr.write("\n")
        }

        let captioner = try QwenVLCaptioner(modelRoot: modelURL)
        let config = QwenVLCaptioner.ModelConfig(
            maxNewTokens: maxTokens,
            temperature: Float(temperature),
            topP: Float(topP)
        )
        var caseResults: [VLMBenchmarkCaseResult] = []
        caseResults.reserveCapacity(suite.cases.count)

        for benchmarkCase in suite.cases {
            let memoryBefore = VLMBenchmarkMemorySnapshot.currentResidentBytes()
            let caseStart = Date()
            do {
                let response = try captioner.caption(
                    imageURL: benchmarkCase.imageURL,
                    prompt: benchmarkCase.prompt,
                    config: config
                )
                caseResults.append(
                    benchmarkCase.result(
                        model: modelID,
                        response: response,
                        elapsedSeconds: Date().timeIntervalSince(caseStart),
                        promptTokens: nil,
                        generatedTokens: nil,
                        timing: nil,
                        residentMemoryBeforeBytes: memoryBefore,
                        residentMemoryAfterBytes: VLMBenchmarkMemorySnapshot.currentResidentBytes()
                    )
                )
            } catch {
                caseResults.append(
                    benchmarkCase.failedResult(
                        model: modelID,
                        error: error.localizedDescription
                    )
                )
            }
        }

        return VLMBenchmarkModelResult(
            model: modelID,
            backend: Qwen3VLAutoCaptioner.modelId,
            elapsedSeconds: Date().timeIntervalSince(modelStart),
            cases: caseResults,
            error: nil
        )
    }

    private func progressHandler() -> (@Sendable (ChatProgress) -> Void)? {
        guard !json else { return nil }
        return { progress in
            let suffix = progress.message?.isEmpty == false ? ": \(progress.message!)" : ""
            CLIStderr.write("[\(progress.stage.rawValue)]\(suffix)\n")
        }
    }
}

private struct VLMBenchmarkSuite {
    let name: String
    let fixtureDirectory: URL
    let cases: [VLMBenchmarkCase]

    static func writeDefaultFixtures(to directory: URL) throws -> Self {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let redURL = directory.appendingPathComponent("solid-red.png")
        try writeImage(
            width: 160,
            height: 160,
            fill: .white,
            rectangles: [VLMBenchmarkRectangle(x: 0, y: 0, width: 160, height: 160, color: .red)],
            to: redURL
        )

        let cornerURL = directory.appendingPathComponent("blue-top-right.png")
        try writeImage(
            width: 160,
            height: 160,
            fill: .white,
            rectangles: [
                VLMBenchmarkRectangle(x: 16, y: 16, width: 48, height: 48, color: .green),
                VLMBenchmarkRectangle(x: 96, y: 16, width: 48, height: 48, color: .blue),
                VLMBenchmarkRectangle(x: 16, y: 96, width: 48, height: 48, color: .yellow),
                VLMBenchmarkRectangle(x: 96, y: 96, width: 48, height: 48, color: .red),
            ],
            to: cornerURL
        )

        let countURL = directory.appendingPathComponent("three-black-squares.png")
        try writeImage(
            width: 192,
            height: 96,
            fill: .white,
            rectangles: [
                VLMBenchmarkRectangle(x: 18, y: 28, width: 34, height: 34, color: .black),
                VLMBenchmarkRectangle(x: 78, y: 28, width: 34, height: 34, color: .black),
                VLMBenchmarkRectangle(x: 138, y: 28, width: 34, height: 34, color: .black),
            ],
            to: countURL
        )

        return VLMBenchmarkSuite(
            name: "synthetic-vqa-v1",
            fixtureDirectory: directory,
            cases: [
                VLMBenchmarkCase(
                    id: "dominant-red",
                    imageURL: redURL,
                    prompt: "What is the dominant color in this image? Answer with one lowercase color word.",
                    expectedRegex: "\\bred\\b",
                    expectedDescription: "red"
                ),
                VLMBenchmarkCase(
                    id: "blue-corner",
                    imageURL: cornerURL,
                    prompt: "Which corner contains the blue square? Answer with two words, like top left.",
                    expectedRegex: "\\b(top right|upper right)\\b",
                    expectedDescription: "top right"
                ),
                VLMBenchmarkCase(
                    id: "count-black-squares",
                    imageURL: countURL,
                    prompt: "How many black squares are visible? Answer with one digit.",
                    expectedRegex: "\\b(3|three)\\b",
                    expectedDescription: "3"
                ),
            ]
        )
    }

    private static func writeImage(
        width: Int,
        height: Int,
        fill: VLMBenchmarkColor,
        rectangles: [VLMBenchmarkRectangle],
        to url: URL
    ) throws {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                setPixel(x: x, y: y, width: width, color: fill, pixels: &pixels)
            }
        }
        for rectangle in rectangles {
            let minX = max(0, rectangle.x)
            let minY = max(0, rectangle.y)
            let maxX = min(width, rectangle.x + rectangle.width)
            let maxY = min(height, rectangle.y + rectangle.height)
            guard minX < maxX, minY < maxY else { continue }
            for y in minY..<maxY {
                for x in minX..<maxX {
                    setPixel(x: x, y: y, width: width, color: rectangle.color, pixels: &pixels)
                }
            }
        }
        try MediaImageIO.writePNG(try MediaImage(width: width, height: height, rgba8: pixels), to: url)
    }

    private static func setPixel(
        x: Int,
        y: Int,
        width: Int,
        color: VLMBenchmarkColor,
        pixels: inout [UInt8]
    ) {
        let offset = ((y * width) + x) * 4
        pixels[offset] = color.red
        pixels[offset + 1] = color.green
        pixels[offset + 2] = color.blue
        pixels[offset + 3] = color.alpha
    }
}

private struct VLMBenchmarkCase {
    let id: String
    let imageURL: URL
    let prompt: String
    let expectedRegex: String
    let expectedDescription: String

    func result(
        model: String,
        response: String,
        elapsedSeconds: Double,
        promptTokens: Int?,
        generatedTokens: Int?,
        timing: ChatTiming?,
        residentMemoryBeforeBytes: UInt64?,
        residentMemoryAfterBytes: UInt64?
    ) -> VLMBenchmarkCaseResult {
        let normalized = Self.normalized(response)
        let passed = normalized.range(of: expectedRegex, options: .regularExpression) != nil
        return VLMBenchmarkCaseResult(
            id: id,
            imagePath: imageURL.path,
            prompt: prompt,
            expected: expectedDescription,
            expectedRegex: expectedRegex,
            passed: passed,
            response: response.trimmingCharacters(in: .whitespacesAndNewlines),
            normalizedResponse: normalized,
            elapsedSeconds: elapsedSeconds,
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            loadSeconds: timing?.loadSeconds,
            prefillSeconds: timing?.prefillSeconds,
            decodeSeconds: timing?.decodeSeconds,
            firstTokenSeconds: timing?.firstTokenSeconds,
            residentMemoryBeforeBytes: residentMemoryBeforeBytes,
            residentMemoryAfterBytes: residentMemoryAfterBytes,
            error: nil
        )
    }

    func failedResult(model _: String, error: String) -> VLMBenchmarkCaseResult {
        VLMBenchmarkCaseResult(
            id: id,
            imagePath: imageURL.path,
            prompt: prompt,
            expected: expectedDescription,
            expectedRegex: expectedRegex,
            passed: false,
            response: "",
            normalizedResponse: "",
            elapsedSeconds: nil,
            promptTokens: nil,
            generatedTokens: nil,
            loadSeconds: nil,
            prefillSeconds: nil,
            decodeSeconds: nil,
            firstTokenSeconds: nil,
            residentMemoryBeforeBytes: nil,
            residentMemoryAfterBytes: nil,
            error: error
        )
    }

    private static func normalized(_ text: String) -> String {
        text
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
            .lowercased()
            .replacingOccurrences(
                of: "[^a-z0-9]+",
                with: " ",
                options: .regularExpression
            )
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private struct VLMBenchmarkColor {
    let red: UInt8
    let green: UInt8
    let blue: UInt8
    let alpha: UInt8

    static let black = VLMBenchmarkColor(red: 0, green: 0, blue: 0, alpha: 255)
    static let blue = VLMBenchmarkColor(red: 0, green: 84, blue: 255, alpha: 255)
    static let green = VLMBenchmarkColor(red: 0, green: 170, blue: 80, alpha: 255)
    static let red = VLMBenchmarkColor(red: 230, green: 0, blue: 0, alpha: 255)
    static let white = VLMBenchmarkColor(red: 255, green: 255, blue: 255, alpha: 255)
    static let yellow = VLMBenchmarkColor(red: 255, green: 210, blue: 0, alpha: 255)
}

private struct VLMBenchmarkRectangle {
    let x: Int
    let y: Int
    let width: Int
    let height: Int
    let color: VLMBenchmarkColor
}

private struct VLMBenchmarkCaseSummary: Encodable {
    let id: String
    let imagePath: String
    let prompt: String
    let expected: String
    let expectedRegex: String

    init(_ benchmarkCase: VLMBenchmarkCase) {
        id = benchmarkCase.id
        imagePath = benchmarkCase.imageURL.path
        prompt = benchmarkCase.prompt
        expected = benchmarkCase.expectedDescription
        expectedRegex = benchmarkCase.expectedRegex
    }
}

private struct VLMBenchmarkReport: Encodable {
    let suite: String
    let fixtureDirectory: String
    let cases: [VLMBenchmarkCaseSummary]
    let models: [VLMBenchmarkModelResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "VLM benchmark",
            "suite: \(suite)",
            "fixtures: \(fixtureDirectory)",
            "",
        ]
        for model in models {
            lines.append(model.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct VLMBenchmarkModelResult: Encodable {
    let model: String
    let backend: String?
    let elapsedSeconds: Double?
    let cases: [VLMBenchmarkCaseResult]
    let error: String?

    var passedCases: Int {
        cases.filter(\.passed).count
    }

    var completedCases: Int {
        cases.filter { $0.error == nil }.count
    }

    var totalCases: Int {
        cases.count
    }

    var score: Double {
        guard totalCases > 0 else { return 0 }
        return Double(passedCases) / Double(totalCases)
    }

    static func failed(model: String, error: String) -> Self {
        Self(
            model: model,
            backend: nil,
            elapsedSeconds: nil,
            cases: [],
            error: error
        )
    }

    enum CodingKeys: String, CodingKey {
        case model
        case backend
        case elapsedSeconds
        case passedCases
        case completedCases
        case totalCases
        case score
        case cases
        case error
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model, forKey: .model)
        try container.encodeIfPresent(backend, forKey: .backend)
        try container.encodeIfPresent(elapsedSeconds, forKey: .elapsedSeconds)
        try container.encode(passedCases, forKey: .passedCases)
        try container.encode(completedCases, forKey: .completedCases)
        try container.encode(totalCases, forKey: .totalCases)
        try container.encode(score, forKey: .score)
        try container.encode(cases, forKey: .cases)
        try container.encodeIfPresent(error, forKey: .error)
    }

    func renderText() -> String {
        if let error {
            return [
                "model: \(model)",
                "  error: \(error)",
                "",
            ].joined(separator: "\n")
        }

        var lines = [
            "model: \(model)",
            "  backend: \(backend ?? "unknown")",
            "  score: \(passedCases)/\(totalCases) (\(formatPercent(score))) elapsed=\(formatOptional(elapsedSeconds))s",
        ]
        for result in cases {
            lines.append(result.renderText())
        }
        lines.append("")
        return lines.joined(separator: "\n")
    }
}

private struct VLMBenchmarkCaseResult: Encodable {
    let id: String
    let imagePath: String
    let prompt: String
    let expected: String
    let expectedRegex: String
    let passed: Bool
    let response: String
    let normalizedResponse: String
    let elapsedSeconds: Double?
    let promptTokens: Int?
    let generatedTokens: Int?
    let loadSeconds: Double?
    let prefillSeconds: Double?
    let decodeSeconds: Double?
    let firstTokenSeconds: Double?
    let residentMemoryBeforeBytes: UInt64?
    let residentMemoryAfterBytes: UInt64?
    let error: String?

    func renderText() -> String {
        let status = passed ? "PASS" : "FAIL"
        var lines = [
            "  \(id): \(status) elapsed=\(formatOptional(elapsedSeconds))s",
            "    expected: \(expected)",
        ]
        if let error {
            lines.append("    error: \(error)")
            return lines.joined(separator: "\n")
        }
        lines.append("    response: \(oneLine(response))")
        lines.append(
            "    tokens: prompt=\(promptTokens.map(String.init) ?? "n/a") generated=\(generatedTokens.map(String.init) ?? "n/a")"
        )
        lines.append(
            "    timing: load=\(formatOptional(loadSeconds))s prefill=\(formatOptional(prefillSeconds))s decode=\(formatOptional(decodeSeconds))s first_token=\(formatOptional(firstTokenSeconds))s"
        )
        lines.append(
            "    resident_memory: before=\(formatBytes(residentMemoryBeforeBytes)) after=\(formatBytes(residentMemoryAfterBytes))"
        )
        return lines.joined(separator: "\n")
    }

    private func oneLine(_ text: String) -> String {
        let collapsed = text
            .split(whereSeparator: \.isNewline)
            .map(String.init)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard collapsed.count > 180 else { return collapsed }
        return String(collapsed.prefix(177)) + "..."
    }
}

private func formatOptional(_ value: Double?) -> String {
    guard let value else { return "n/a" }
    return String(format: "%.3f", value)
}

private func formatPercent(_ value: Double) -> String {
    String(format: "%.1f%%", value * 100)
}

private func formatBytes(_ value: UInt64?) -> String {
    guard let value else { return "n/a" }
    return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
}

enum VLMBenchmarkDataset: String, ExpressibleByArgument, CaseIterable {
    case syntheticVQA = "synthetic-vqa-v1"
    case mathvistaTestMini = "mathvista-testmini"
    case mmmuVal = "mmmu-val"
    case chartQA = "chartqa"
    case docVQAVal = "docvqa-val"
    case mme

    init?(argument: String) {
        self.init(rawValue: argument)
    }

    var lmmsEvalTasks: String? {
        switch self {
        case .syntheticVQA:
            return nil
        case .mathvistaTestMini:
            return "mathvista_testmini"
        case .mmmuVal:
            return "mmmu_val"
        case .chartQA:
            return "chartqa"
        case .docVQAVal:
            return "docvqa_val"
        case .mme:
            return "mme"
        }
    }
}

private struct VLMExternalBenchmarkPlan {
    let dataset: String
    let tasks: String
    let outputDirectory: URL
}

private struct VLMExternalBenchmarkReport: Encodable {
    let dataset: String
    let tasks: String
    let outputDirectory: String
    let dryRun: Bool
    let models: [VLMExternalModelResult]

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "VLM external benchmark",
            "dataset: \(dataset)",
            "tasks: \(tasks)",
            "output: \(outputDirectory)",
            "",
        ]
        for model in models {
            lines.append(model.renderText())
        }
        return lines.joined(separator: "\n")
    }
}

private struct VLMExternalModelResult: Encodable {
    let model: String
    let command: String
    let outputPath: String
    let exitStatus: Int32?
    let elapsedSeconds: Double?

    func renderText() -> String {
        [
            "model: \(model)",
            "  command: \(command)",
            "  output: \(outputPath)",
            "  exit_status: \(exitStatus.map(String.init) ?? "dry-run")",
            "  elapsed: \(formatOptional(elapsedSeconds))s",
            "",
        ].joined(separator: "\n")
    }
}

private struct VLMExternalInvocation {
    let executable: String
    let arguments: [String]
    let currentDirectory: URL?
    let outputDirectory: URL

    var shellDescription: String {
        let workingDirectory = currentDirectory.map { "cd \(Self.shellEscape($0.path)) && " } ?? ""
        return workingDirectory + ([executable] + arguments)
            .map(Self.shellEscape)
            .joined(separator: " ")
    }

    private static func shellEscape(_ value: String) -> String {
        guard !value.isEmpty else { return "''" }
        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,+-")
        if value.rangeOfCharacter(from: allowed.inverted) == nil {
            return value
        }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

private struct VLMExternalProcessResult {
    let status: Int32
    let stderrTail: String
}

private final class VLMExternalProcessCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stderr = Data()
    private let maxTailBytes = 16 * 1024

    func forward(_ pipe: Pipe, prefix: String, storeTail: Bool) {
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            if storeTail {
                self?.appendStderr(data)
            }
            if let text = String(data: data, encoding: .utf8) {
                CLIStderr.write(text.linePrefixed(prefix))
            } else {
                FileHandle.standardError.write(data)
            }
        }
    }

    func finish(_ pipe: Pipe, storeTail: Bool) {
        pipe.fileHandleForReading.readabilityHandler = nil
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard !data.isEmpty else { return }
        if storeTail {
            appendStderr(data)
        }
    }

    func stderrTail() -> String {
        lock.lock()
        let data = stderr
        lock.unlock()
        return String(decoding: data, as: UTF8.self)
            .split(whereSeparator: \.isNewline)
            .suffix(12)
            .joined(separator: "\n")
    }

    private func appendStderr(_ data: Data) {
        lock.lock()
        stderr.append(data)
        if stderr.count > maxTailBytes {
            stderr.removeFirst(stderr.count - maxTailBytes)
        }
        lock.unlock()
    }
}

private extension String {
    func linePrefixed(_ prefix: String) -> String {
        split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.isEmpty ? String($0) : prefix + $0 }
            .joined(separator: "\n")
    }
}

private enum VLMBenchmarkMemorySnapshot {
    static func currentResidentBytes() -> UInt64? {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return UInt64(info.resident_size)
        #else
        return nil
        #endif
    }
}
