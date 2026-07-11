import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MereRunCore

struct ModelBenchmarkAPIWorkload: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "api-workload",
        abstract: "Replay a chat workload against a running API server and measure runtime cache counters."
    )

    @Option(name: [.long], help: "Local API host to benchmark.")
    var host: String = "127.0.0.1"

    @Option(name: [.long], help: "Local API port to benchmark.")
    var port: Int = 8080

    @Option(name: [.long], help: "Bearer token for API endpoints. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "OpenAI chat model id or runtime alias.")
    var model: String = Gemma4Resources.turboModelId

    @Option(name: [.long], help: "JSONL workload file. Each line may contain {id,user} or {id,messages}.")
    var workloadFile: String?

    @Option(name: [.long], help: "Number of built-in workload turns when --workload-file is omitted.")
    var turns: Int = 8

    @Option(name: [.long], help: "Repeated shared-context rows in the built-in stable prefix.")
    var sharedPrefixRepeat: Int = 32

    @Option(name: [.long], help: "Maximum completion tokens per request.")
    var maxTokens: Int = 64

    @Option(name: [.long], help: "Temperature for each chat request.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p for each chat request.")
    var topP: Double = 1

    @Option(name: [.long], help: "Requests to issue concurrently. Use >1 with api serve --max-active-requests >1.")
    var concurrency: Int = 1

    @Option(name: [.long], help: "HTTP request timeout in seconds.")
    var timeoutSeconds: Double = 300

    @Flag(name: [.long], help: "Print the workload plan without contacting the server.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        guard port > 0 else {
            throw ValidationError("--port must be greater than zero.")
        }
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("--model must not be empty.")
        }
        guard turns > 0 else {
            throw ValidationError("--turns must be greater than zero.")
        }
        guard sharedPrefixRepeat > 0 else {
            throw ValidationError("--shared-prefix-repeat must be greater than zero.")
        }
        guard maxTokens > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        guard (0...2).contains(temperature), temperature.isFinite else {
            throw ValidationError("--temperature must be finite and between 0 and 2.")
        }
        guard (0...1).contains(topP), topP.isFinite else {
            throw ValidationError("--top-p must be finite and between 0 and 1.")
        }
        guard concurrency > 0 else {
            throw ValidationError("--concurrency must be greater than zero.")
        }
        guard timeoutSeconds > 0, timeoutSeconds.isFinite else {
            throw ValidationError("--timeout-seconds must be a positive finite number.")
        }
    }

    func run() async throws {
        let cases = try workloadCases()
        let plan = APIWorkloadBenchmarkPlan(
            baseURL: baseURLString,
            model: model,
            workloadFile: workloadFile,
            requestCount: cases.count,
            concurrency: concurrency,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            streaming: true,
            sharedPrefixRepeat: workloadFile == nil ? sharedPrefixRepeat : nil
        )
        if dryRun {
            try printReport(APIWorkloadBenchmarkReport(
                plan: plan,
                statusBefore: nil,
                statusBeforeDetail: "dry run",
                statusAfter: nil,
                statusAfterDetail: "dry run",
                statusDelta: nil,
                requests: [],
                summary: APIWorkloadBenchmarkSummary(results: [], statusDelta: nil)
            ))
            return
        }

        let statusBefore = await runtimeStatus()
        let workloadStart = Date()
        let results = try await runCases(cases)
        let wallSeconds = Date().timeIntervalSince(workloadStart)
        let statusAfter = await runtimeStatus()
        let delta = APIWorkloadStatusDelta(before: statusBefore.snapshot, after: statusAfter.snapshot)
        let report = APIWorkloadBenchmarkReport(
            plan: plan,
            statusBefore: statusBefore.snapshot,
            statusBeforeDetail: statusBefore.detail,
            statusAfter: statusAfter.snapshot,
            statusAfterDetail: statusAfter.detail,
            statusDelta: delta,
            requests: results.sorted { $0.sequence < $1.sequence },
            summary: APIWorkloadBenchmarkSummary(results: results, wallSeconds: wallSeconds, statusDelta: delta)
        )
        try printReport(report)
    }

    private var baseURLString: String {
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let urlHost: String
        if trimmedHost.contains(":") && !trimmedHost.hasPrefix("[") {
            urlHost = "[\(trimmedHost)]"
        } else {
            urlHost = trimmedHost.isEmpty ? "127.0.0.1" : trimmedHost
        }
        return "http://\(urlHost):\(port)"
    }

    private func resolvedAPIKey() -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines), !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = ProcessInfo.processInfo.environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }

    private func workloadCases() throws -> [APIWorkloadCase] {
        if let workloadFile {
            return try loadJSONLWorkload(path: workloadFile)
        }
        return Self.builtInStablePrefixWorkload(turns: turns, sharedPrefixRepeat: sharedPrefixRepeat)
    }

    private func loadJSONLWorkload(path: String) throws -> [APIWorkloadCase] {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let raw = try String(contentsOf: url, encoding: .utf8)
        let decoder = JSONDecoder()
        let cases = try raw
            .split(whereSeparator: \.isNewline)
            .enumerated()
            .compactMap { index, line -> APIWorkloadCase? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, !trimmed.hasPrefix("#") else { return nil }
                let input = try decoder.decode(APIWorkloadInputCase.self, from: Data(trimmed.utf8))
                return try input.workloadCase(sequence: index)
            }
        guard !cases.isEmpty else {
            throw ValidationError("--workload-file must contain at least one JSONL request.")
        }
        return cases
    }

    static func builtInStablePrefixWorkload(turns: Int, sharedPrefixRepeat: Int) -> [APIWorkloadCase] {
        let sharedContext = (1...sharedPrefixRepeat)
            .map { index in
                "Cache note \(index): local runtime work needs stable prefixes, visible queueing, " +
                    "bounded memory, and measured latency before SSD persistence."
            }
            .joined(separator: "\n")
        let system = """
        You are measuring a local inference runtime. Use the shared context faithfully and answer
        in one concise sentence.

        Shared context:
        \(sharedContext)
        """
        let questions = [
            "Name the runtime feature that should stay in memory before SSD persistence is considered.",
            "What metric should improve before promoting prefix KV reuse?",
            "Why should request admission remain fair and visible?",
            "What does chunked prefill improve for long prompts?",
            "Which cache signal tells us repeated prefixes are actually reused?",
            "How does continuous batching improve concurrent local serving?",
            "What should happen before enabling SSD-backed KV persistence?",
            "Summarize the benchmark decision rule in one sentence.",
        ]

        return (0..<turns).map { index in
            APIWorkloadCase(
                id: "stable-prefix-\(index + 1)",
                sequence: index,
                messages: [
                    OpenAIChatMessage(role: "system", content: system),
                    OpenAIChatMessage(role: "user", content: questions[index % questions.count]),
                ]
            )
        }
    }

    private func runCases(_ cases: [APIWorkloadCase]) async throws -> [APIWorkloadRequestResult] {
        var results: [APIWorkloadRequestResult] = []
        results.reserveCapacity(cases.count)
        var index = 0
        while index < cases.count {
            let batch = Array(cases[index..<min(index + concurrency, cases.count)])
            let batchResults = try await withThrowingTaskGroup(of: APIWorkloadRequestResult.self) { group in
                for benchmarkCase in batch {
                    group.addTask {
                        try await runCase(benchmarkCase)
                    }
                }
                var collected: [APIWorkloadRequestResult] = []
                collected.reserveCapacity(batch.count)
                for try await result in group {
                    collected.append(result)
                }
                return collected
            }
            results.append(contentsOf: batchResults)
            index += batch.count
        }
        return results
    }

    private func runCase(_ benchmarkCase: APIWorkloadCase) async throws -> APIWorkloadRequestResult {
        var chatRequest = OpenAIChatRequest(
            model: model,
            messages: benchmarkCase.messages,
            temperature: temperature,
            top_p: topP,
            max_tokens: maxTokens,
            stream: true
        )
        chatRequest.stream_options = nil
        let body = try JSONEncoder().encode(chatRequest)
        guard let url = URL(string: "\(baseURLString)/v1/chat/completions") else {
            throw ValidationError("Invalid API server URL.")
        }
        var request = URLRequest(url: url, timeoutInterval: timeoutSeconds)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = resolvedAPIKey() {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body

        var firstTokenSeconds: Double?
        var completion = ""
        var streamedChunks = 0
        var completionTokens: Int?
        var errorBody = ""

        let start = Date()
        func consumeLine(_ line: String, statusCode: Int) {
            if statusCode != 200 {
                errorBody += line
                return
            }
            guard line.hasPrefix("data: ") else { return }
            let payload = String(line.dropFirst("data: ".count))
            guard payload != "[DONE]" else { return }
            guard let data = payload.data(using: String.Encoding.utf8),
                  let chunk = try? JSONDecoder().decode(OpenAIChatResponse.self, from: data) else {
                return
            }
            if let usage = chunk.usage {
                completionTokens = usage.completion_tokens
            }
            for choice in chunk.choices {
                if let token = choice.delta?.content, !token.isEmpty {
                    if firstTokenSeconds == nil {
                        firstTokenSeconds = Date().timeIntervalSince(start)
                    }
                    streamedChunks += 1
                    completion += token
                }
            }
        }

        let http: HTTPURLResponse
        #if os(Linux)
        let (responseData, response) = try await URLSession.shared.data(for: request)
        guard let resolvedHTTP = response as? HTTPURLResponse else {
            throw ValidationError("Chat completion returned a non-HTTP response.")
        }
        http = resolvedHTTP
        let responseText = String(data: responseData, encoding: .utf8)
            ?? String(decoding: responseData, as: UTF8.self)
        for line in responseText.split(separator: "\n", omittingEmptySubsequences: false) {
            consumeLine(String(line), statusCode: http.statusCode)
        }
        #else
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let resolvedHTTP = response as? HTTPURLResponse else {
            throw ValidationError("Chat completion returned a non-HTTP response.")
        }
        http = resolvedHTTP
        for try await line in bytes.lines {
            consumeLine(line, statusCode: http.statusCode)
        }
        #endif

        let totalSeconds = Date().timeIntervalSince(start)
        guard http.statusCode == 200 else {
            return APIWorkloadRequestResult(
                id: benchmarkCase.id,
                sequence: benchmarkCase.sequence,
                promptCharacters: benchmarkCase.promptCharacters,
                statusCode: http.statusCode,
                ttftSeconds: firstTokenSeconds,
                totalSeconds: totalSeconds,
                streamedChunks: streamedChunks,
                completionCharacters: completion.count,
                completionTokens: completionTokens,
                error: errorBody.isEmpty ? "HTTP \(http.statusCode)" : errorBody
            )
        }

        return APIWorkloadRequestResult(
            id: benchmarkCase.id,
            sequence: benchmarkCase.sequence,
            promptCharacters: benchmarkCase.promptCharacters,
            statusCode: http.statusCode,
            ttftSeconds: firstTokenSeconds,
            totalSeconds: totalSeconds,
            streamedChunks: streamedChunks,
            completionCharacters: completion.count,
            completionTokens: completionTokens,
            error: nil
        )
    }

    private func runtimeStatus() async -> (snapshot: RuntimeModelPoolStatus?, detail: String?) {
        guard let url = URL(string: "\(baseURLString)/runtime/status") else {
            return (nil, "invalid runtime status URL")
        }
        var request = URLRequest(url: url, timeoutInterval: max(0.1, timeoutSeconds))
        if let apiKey = resolvedAPIKey() {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return (nil, "runtime status returned a non-HTTP response")
            }
            guard http.statusCode == 200 else {
                return (nil, "runtime status returned HTTP \(http.statusCode)")
            }
            return (try JSONDecoder().decode(RuntimeModelPoolStatus.self, from: data), nil)
        } catch {
            return (nil, Self.shortNetworkError(error))
        }
    }

    private func printReport(_ report: APIWorkloadBenchmarkReport) throws {
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(report.renderText())
        }
    }

    private static func shortNetworkError(_ error: Error) -> String {
        if let urlError = error as? URLError {
            return "\(urlError.code)"
        }
        return error.localizedDescription
    }
}

struct APIWorkloadInputCase: Codable {
    let id: String?
    let user: String?
    let messages: [OpenAIChatMessage]?

    func workloadCase(sequence: Int) throws -> APIWorkloadCase {
        if let messages, !messages.isEmpty {
            return APIWorkloadCase(
                id: id ?? "request-\(sequence + 1)",
                sequence: sequence,
                messages: messages
            )
        }
        if let user = user?.trimmingCharacters(in: .whitespacesAndNewlines), !user.isEmpty {
            return APIWorkloadCase(
                id: id ?? "request-\(sequence + 1)",
                sequence: sequence,
                messages: [OpenAIChatMessage(role: "user", content: user)]
            )
        }
        throw ValidationError("JSONL workload case \(id ?? "#\(sequence + 1)") must include messages or user.")
    }
}

struct APIWorkloadCase {
    let id: String
    let sequence: Int
    let messages: [OpenAIChatMessage]

    var promptCharacters: Int {
        messages.reduce(0) { $0 + $1.content.count }
    }
}

struct APIWorkloadBenchmarkPlan: Codable, Equatable {
    let baseURL: String
    let model: String
    let workloadFile: String?
    let requestCount: Int
    let concurrency: Int
    let maxTokens: Int
    let temperature: Double
    let topP: Double
    let streaming: Bool
    let sharedPrefixRepeat: Int?
}

struct APIWorkloadRequestResult: Codable, Equatable {
    let id: String
    let sequence: Int
    let promptCharacters: Int
    let statusCode: Int
    let ttftSeconds: Double?
    let totalSeconds: Double
    let streamedChunks: Int
    let completionCharacters: Int
    let completionTokens: Int?
    let error: String?
}

struct APIWorkloadStatusDelta: Codable, Equatable {
    let prefixKVHits: Int
    let prefixKVMisses: Int
    let prefixKVStoredPrefixes: Int
    let prefixKVReusedTokens: Int
    let decodeBatchedSteps: Int
    let decodeSingleSteps: Int
    let decodeSamePositionBatchedSteps: Int
    let decodeVariablePositionBatchedSteps: Int
    let completedRequests: Int
    let failedRequests: Int
    let generatedTokens: Int
    let ssdKVCacheAvailable: Bool

    init?(before: RuntimeModelPoolStatus?, after: RuntimeModelPoolStatus?) {
        guard let before, let after else { return nil }
        let beforePrefix = before.cacheStats.prefixKVReuse
        let afterPrefix = after.cacheStats.prefixKVReuse
        let beforeBatching = before.cacheStats.decodeBatching
        let afterBatching = after.cacheStats.decodeBatching
        let beforeBenchmark = before.benchmarkStats
        let afterBenchmark = after.benchmarkStats
        prefixKVHits = afterPrefix.hits - beforePrefix.hits
        prefixKVMisses = afterPrefix.misses - beforePrefix.misses
        prefixKVStoredPrefixes = afterPrefix.storedPrefixes - beforePrefix.storedPrefixes
        prefixKVReusedTokens = afterPrefix.reusedTokens - beforePrefix.reusedTokens
        decodeBatchedSteps = afterBatching.batchedDecodeSteps - beforeBatching.batchedDecodeSteps
        decodeSingleSteps = afterBatching.singleDecodeSteps - beforeBatching.singleDecodeSteps
        decodeSamePositionBatchedSteps =
            afterBatching.samePositionBatchedSteps - beforeBatching.samePositionBatchedSteps
        decodeVariablePositionBatchedSteps =
            afterBatching.variablePositionBatchedSteps - beforeBatching.variablePositionBatchedSteps
        completedRequests = (afterBenchmark?.completedRequests ?? 0) - (beforeBenchmark?.completedRequests ?? 0)
        failedRequests = (afterBenchmark?.failedRequests ?? 0) - (beforeBenchmark?.failedRequests ?? 0)
        generatedTokens = (afterBenchmark?.generatedTokens ?? 0) - (beforeBenchmark?.generatedTokens ?? 0)
        ssdKVCacheAvailable = after.cacheStats.ssdKVCache.available
    }
}

struct APIWorkloadBenchmarkSummary: Codable, Equatable {
    let completedRequests: Int
    let failedRequests: Int
    let wallSeconds: Double?
    let averageTTFTSeconds: Double?
    let p50TTFTSeconds: Double?
    let p95TTFTSeconds: Double?
    let averageTotalSeconds: Double?
    let p50TotalSeconds: Double?
    let p95TotalSeconds: Double?
    let requestsPerSecond: Double?
    let prefixKVHits: Int?
    let prefixKVMisses: Int?
    let prefixKVReusedTokens: Int?
    let decodeBatchedSteps: Int?
    let decodeSingleSteps: Int?
    let ssdKVCacheAvailable: Bool?

    init(results: [APIWorkloadRequestResult], wallSeconds: Double? = nil, statusDelta: APIWorkloadStatusDelta?) {
        completedRequests = results.filter { $0.error == nil && $0.statusCode == 200 }.count
        failedRequests = results.count - completedRequests
        self.wallSeconds = wallSeconds
        let ttfts = results.compactMap(\.ttftSeconds).sorted()
        let totals = results.map(\.totalSeconds).sorted()
        averageTTFTSeconds = Self.average(ttfts)
        p50TTFTSeconds = Self.percentile(ttfts, percentile: 0.50)
        p95TTFTSeconds = Self.percentile(ttfts, percentile: 0.95)
        averageTotalSeconds = Self.average(totals)
        p50TotalSeconds = Self.percentile(totals, percentile: 0.50)
        p95TotalSeconds = Self.percentile(totals, percentile: 0.95)
        if let wallSeconds, wallSeconds > 0 {
            requestsPerSecond = Double(completedRequests) / wallSeconds
        } else {
            let serialSeconds = totals.reduce(0, +)
            requestsPerSecond = serialSeconds > 0 ? Double(completedRequests) / serialSeconds : nil
        }
        prefixKVHits = statusDelta?.prefixKVHits
        prefixKVMisses = statusDelta?.prefixKVMisses
        prefixKVReusedTokens = statusDelta?.prefixKVReusedTokens
        decodeBatchedSteps = statusDelta?.decodeBatchedSteps
        decodeSingleSteps = statusDelta?.decodeSingleSteps
        ssdKVCacheAvailable = statusDelta?.ssdKVCacheAvailable
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func percentile(_ sortedValues: [Double], percentile: Double) -> Double? {
        guard !sortedValues.isEmpty else { return nil }
        let index = min(
            sortedValues.count - 1,
            max(0, Int((Double(sortedValues.count - 1) * percentile).rounded(.toNearestOrAwayFromZero)))
        )
        return sortedValues[index]
    }
}

struct APIWorkloadBenchmarkReport: Codable, Equatable {
    let plan: APIWorkloadBenchmarkPlan
    let statusBefore: RuntimeModelPoolStatus?
    let statusBeforeDetail: String?
    let statusAfter: RuntimeModelPoolStatus?
    let statusAfterDetail: String?
    let statusDelta: APIWorkloadStatusDelta?
    let requests: [APIWorkloadRequestResult]
    let summary: APIWorkloadBenchmarkSummary

    func renderText() -> String {
        var lines = [
            "API workload benchmark",
            "server: \(plan.baseURL)",
            "model: \(plan.model)",
            "requests: \(plan.requestCount) concurrency=\(plan.concurrency) max_tokens=\(plan.maxTokens)",
            "",
            "summary:",
            "  completed=\(summary.completedRequests) failed=\(summary.failedRequests)",
            "  wall=\(formatSeconds(summary.wallSeconds)) rps=\(format(summary.requestsPerSecond))",
            "  ttft avg=\(formatSeconds(summary.averageTTFTSeconds)) " +
                "p50=\(formatSeconds(summary.p50TTFTSeconds)) " +
                "p95=\(formatSeconds(summary.p95TTFTSeconds))",
            "  total avg=\(formatSeconds(summary.averageTotalSeconds)) " +
                "p50=\(formatSeconds(summary.p50TotalSeconds)) " +
                "p95=\(formatSeconds(summary.p95TotalSeconds))",
        ]
        if let statusDelta {
            lines += [
                "  prefix KV hits=\(statusDelta.prefixKVHits) " +
                    "misses=\(statusDelta.prefixKVMisses) " +
                    "reused_tokens=\(statusDelta.prefixKVReusedTokens)",
                "  decode batched_steps=\(statusDelta.decodeBatchedSteps) " +
                    "single_steps=\(statusDelta.decodeSingleSteps)",
                "  SSD KV cache available=\(statusDelta.ssdKVCacheAvailable)",
            ]
        } else {
            lines.append("  runtime status delta: unavailable")
            if let statusBeforeDetail {
                lines.append("  before status: \(statusBeforeDetail)")
            }
            if let statusAfterDetail {
                lines.append("  after status: \(statusAfterDetail)")
            }
        }
        lines.append("")
        for result in requests.sorted(by: { $0.sequence < $1.sequence }) {
            lines.append(
                "  \(result.id): status=\(result.statusCode) " +
                    "ttft=\(formatSeconds(result.ttftSeconds)) " +
                    "total=\(formatSeconds(result.totalSeconds)) " +
                    "chunks=\(result.streamedChunks)"
            )
            if let error = result.error {
                lines.append("    error: \(error)")
            }
        }
        return lines.joined(separator: "\n")
    }

    private func format(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.3f", value)
    }

    private func formatSeconds(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return "\(format(value))s"
    }
}
