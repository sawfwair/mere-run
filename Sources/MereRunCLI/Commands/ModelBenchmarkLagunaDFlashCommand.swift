import ArgumentParser
import Foundation
import MereRunCore
import MLX

struct ModelBenchmarkLagunaDFlash: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "laguna-dflash",
        abstract: "Measure Laguna target-only and DFlash decode in one resident process."
    )

    @Option(
        name: [.long],
        help: "Local poolside/Laguna-S-2.1-NVFP4-mlx checkpoint directory."
    )
    var lagunaPath: String

    @Option(
        name: [.long],
        help: "Local poolside/Laguna-S-2.1-DFlash checkpoint directory."
    )
    var lagunaDflashPath: String

    @Option(
        name: [.long],
        help: "Comma-separated exact decode lengths."
    )
    var decodeTokenValues: String = "8,12,16,24,32,48"

    @Option(name: [.long], help: "Measured repetitions per mode and decode length.")
    var repetitions: Int = 3

    @Option(name: [.long], help: "Laguna DFlash speculative tokens per round (1...15).")
    var lagunaDflashTokens: Int = LagunaDFlashRouting.defaultSpeculativeTokens

    @Option(name: [.long], help: "Temperature for generation.")
    var temperature: Double = 0

    @Option(name: [.long], help: "Top-p for generation.")
    var topP: Double = 1

    @Option(name: [.customLong("top-k")], help: "Top-k for generation; zero disables it.")
    var topK: Int = 0

    @Option(name: [.customLong("min-p")], help: "Min-p cutoff relative to the most likely token.")
    var minP: Double = LagunaResources.recommendedMinP

    @Option(name: [.customShort("p"), .long], help: "Benchmark prompt.")
    var prompt: String?

    @Option(name: [.long], help: "Read the benchmark prompt from a UTF-8 file.")
    var promptFile: String?

    @Option(name: [.long], help: "Built-in benchmark fixture.")
    var fixture: LagunaDFlashBenchmarkFixture = .deterministicProse

    @Option(name: [.long], help: "Maximum context tokens passed to Laguna.")
    var contextSize: Int = 4_096

    @Option(
        name: [.long],
        help: "Optional comma-separated resident concurrency levels, for example 1,2,4."
    )
    var concurrencyValues: String?

    @Option(
        name: [.long],
        help: "Unmeasured warmup groups per concurrency level."
    )
    var warmupRepetitions: Int = 1

    @Flag(
        name: [.long],
        help: "Rotate prose, grounded-email, and code fixtures within concurrent groups."
    )
    var mixedFixtures: Bool = false

    @Flag(
        name: [.long],
        help: "Also measure the length-and-acceptance adaptive routing policy."
    )
    var includeAutomatic: Bool = false

    @Flag(name: [.long], help: "Include generated responses in the report.")
    var logResponses: Bool = false

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func validate() throws {
        if prompt != nil, promptFile != nil {
            throw ValidationError("Specify either --prompt or --prompt-file, not both.")
        }
        guard repetitions > 0 else {
            throw ValidationError("--repetitions must be greater than zero.")
        }
        guard (1...15).contains(lagunaDflashTokens) else {
            throw ValidationError("--laguna-dflash-tokens must be between 1 and 15.")
        }
        guard (0...2).contains(temperature), temperature.isFinite else {
            throw ValidationError("--temperature must be finite and between 0 and 2.")
        }
        guard (0...1).contains(topP), topP.isFinite else {
            throw ValidationError("--top-p must be finite and between 0 and 1.")
        }
        guard topK >= 0 else {
            throw ValidationError("--top-k must be zero or greater.")
        }
        guard (0...1).contains(minP), minP.isFinite else {
            throw ValidationError("--min-p must be finite and between 0 and 1.")
        }
        guard contextSize > 0 else {
            throw ValidationError("--context-size must be greater than zero.")
        }
        guard warmupRepetitions >= 0 else {
            throw ValidationError("--warmup-repetitions must be zero or greater.")
        }
        if mixedFixtures, concurrencyValues == nil {
            throw ValidationError("--mixed-fixtures requires --concurrency-values.")
        }
        if mixedFixtures, prompt != nil || promptFile != nil {
            throw ValidationError(
                "--mixed-fixtures cannot be combined with --prompt or --prompt-file."
            )
        }
        _ = try parsedDecodeTokenValues()
        _ = try parsedConcurrencyValues()
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: json)
        let decodeLengths = try parsedDecodeTokenValues()
        let messages = try resolvedMessages()
        let concurrencyLevels = try parsedConcurrencyValues()
        let generator = LagunaGenerator(
            continuousBatchingEnabled: concurrencyLevels != nil,
            dflashModelPath: lagunaDflashPath,
            dflashSpeculativeTokens: lagunaDflashTokens
        )

        do {
            try await generator.prepare(modelPath: lagunaPath)
            var samples: [LagunaDFlashBenchmarkSample] = []
            samples.reserveCapacity(decodeLengths.count * repetitions * 2)
            let fixtureID = prompt != nil || promptFile != nil
                ? "custom"
                : fixture.rawValue

            for (lengthIndex, decodeLength) in decodeLengths.enumerated() {
                let request = ChatRequest(
                    messages: messages,
                    maxTokens: decodeLength,
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    minP: minP,
                    showThinking: false,
                    stopOnEOS: false,
                    maxContextTokens: contextSize
                )
                for repetition in 0..<repetitions {
                    var baseModes: [LagunaDFlashRoutingMode] = [
                        .targetOnly,
                        .dflash,
                    ]
                    if includeAutomatic {
                        baseModes.append(.automatic)
                    }
                    let rotation = (lengthIndex + repetition) % baseModes.count
                    let modes = (0..<baseModes.count).map {
                        baseModes[(rotation + $0) % baseModes.count]
                    }
                    for (order, mode) in modes.enumerated() {
                        samples.append(try await runSample(
                            generator: generator,
                            request: request,
                            decodeLength: decodeLength,
                            repetition: repetition,
                            order: order,
                            mode: mode,
                            workloadID: "\(fixtureID):\(decodeLength)"
                        ))
                    }
                }
            }

            let concurrencyGroups: [LagunaDFlashConcurrentGroup]
            if let concurrencyLevels {
                concurrencyGroups = try await runConcurrentMatrix(
                    generator: generator,
                    decodeLengths: decodeLengths,
                    concurrencyLevels: concurrencyLevels
                )
            } else {
                concurrencyGroups = []
            }
            let device = GPU.deviceInfo()
            let exactness = makeExactnessReport(
                samples: samples,
                concurrencyGroups: concurrencyGroups
            )
            let report = LagunaDFlashBenchmarkReport(
                targetPath: URL(fileURLWithPath: lagunaPath).standardizedFileURL.path,
                dflashPath: URL(fileURLWithPath: lagunaDflashPath).standardizedFileURL.path,
                speculativeTokens: lagunaDflashTokens,
                repetitions: repetitions,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                promptCharacters: messages.map(\.content.count).reduce(0, +),
                hardware: LagunaDFlashBenchmarkHardware(
                    architecture: device.architecture,
                    physicalMemoryBytes: device.memorySize,
                    recommendedWorkingSetBytes: device.maxRecommendedWorkingSetSize
                ),
                samples: samples,
                concurrencyGroups: concurrencyGroups,
                exactness: exactness
            )
            if json {
                print(try report.jsonString())
            } else {
                print(report.renderText())
            }
            await generator.unload()
        } catch {
            await generator.unload()
            throw error
        }
    }

    private func parsedDecodeTokenValues() throws -> [Int] {
        let values = decodeTokenValues
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError("--decode-token-values must contain at least one value.")
        }
        return try values.map { value in
            guard let parsed = Int(value), parsed > 0 else {
                throw ValidationError(
                    "--decode-token-values values must be positive integers."
                )
            }
            return parsed
        }
    }

    private func parsedConcurrencyValues() throws -> [Int]? {
        guard let concurrencyValues else { return nil }
        let values = concurrencyValues
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard !values.isEmpty else {
            throw ValidationError(
                "--concurrency-values must contain at least one value."
            )
        }
        let parsed = try values.map { value in
            guard let concurrency = Int(value), concurrency > 0 else {
                throw ValidationError(
                    "--concurrency-values values must be positive integers."
                )
            }
            return concurrency
        }
        guard Set(parsed).count == parsed.count else {
            throw ValidationError("--concurrency-values cannot contain duplicates.")
        }
        return parsed.sorted()
    }

    private func resolvedMessages() throws -> [ChatMessage] {
        if let prompt {
            return [ChatMessage(role: .user, content: prompt)]
        }
        if let promptFile {
            let contents = try String(
                contentsOf: URL(fileURLWithPath: promptFile).standardizedFileURL,
                encoding: .utf8
            )
            return [ChatMessage(role: .user, content: contents)]
        }
        return resolvedMessages(for: fixture)
    }

    private func resolvedMessages(
        for fixture: LagunaDFlashBenchmarkFixture
    ) -> [ChatMessage] {
        switch fixture {
        case .deterministicProse:
            return [ChatMessage(
                role: .user,
                content: """
                You are measuring deterministic local inference. Explain in one paragraph how \
                speculative decoding verifies draft tokens against a target model while preserving \
                the target distribution. Continue until the requested token budget is exhausted.
                """
            )]
        case .groundedEmail:
            return [
                ChatMessage(
                    role: .system,
                    content: ModelBenchmarkChat.systemPrompt
                ),
                ChatMessage(
                    role: .user,
                    content: ChatBenchmarkCase.mereChatSlice[0].promptText
                ),
            ]
        case .codeCompletion:
            return [
                ChatMessage(
                    role: .system,
                    content: ModelBenchmarkCode.systemPrompt
                ),
                ChatMessage(
                    role: .user,
                    content: CodeBenchmarkTask.humanEvalSlice[0].promptText
                ),
            ]
        }
    }

    private func runSample(
        generator: LagunaGenerator,
        request: ChatRequest,
        decodeLength: Int,
        repetition: Int,
        order: Int,
        mode: LagunaDFlashRoutingMode,
        workloadID: String
    ) async throws -> LagunaDFlashBenchmarkSample {
        GPU.resetPeakMemory()
        let beforeMemory = Memory.snapshot()
        let beforeStats = await generator.dflashStats()
        let start = Date()
        let response = try await generator.chat(
            request,
            modelPath: lagunaPath,
            dflashRouting: mode,
            progressHandler: nil
        )
        let elapsed = Date().timeIntervalSince(start)
        let afterStats = await generator.dflashStats()
        let afterMemory = Memory.snapshot()

        guard response.tokensGenerated == decodeLength else {
            throw ValidationError(
                "\(mode.rawValue) generated \(response.tokensGenerated) tokens, "
                    + "expected \(decodeLength)."
            )
        }
        guard let timing = response.timing else {
            throw ValidationError("\(mode.rawValue) did not return timing data.")
        }

        let draftedTokens = afterStats.draftedTokens - beforeStats.draftedTokens
        let acceptedDraftTokens =
            afterStats.acceptedDraftTokens - beforeStats.acceptedDraftTokens
        return LagunaDFlashBenchmarkSample(
            workloadID: workloadID,
            decodeTokens: decodeLength,
            repetition: repetition,
            order: order,
            mode: mode.rawValue,
            promptTokens: response.promptTokens ?? 0,
            generatedTokens: response.tokensGenerated,
            elapsedSeconds: elapsed,
            loadSeconds: timing.loadSeconds,
            prefillSeconds: timing.prefillSeconds,
            decodeSeconds: timing.decodeSeconds,
            firstTokenSeconds: timing.firstTokenSeconds,
            activeMemoryBeforeBytes: beforeMemory.activeMemory,
            activeMemoryAfterBytes: afterMemory.activeMemory,
            cacheMemoryAfterBytes: afterMemory.cacheMemory,
            peakMemoryBytes: afterMemory.peakMemory,
            dflashRounds: afterStats.rounds - beforeStats.rounds,
            draftedTokens: draftedTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            rejectedDraftTokens:
                afterStats.rejectedDraftTokens - beforeStats.rejectedDraftTokens,
            targetVerificationForwards:
                afterStats.targetVerificationForwards
                - beforeStats.targetVerificationForwards,
            targetRecoveryForwards:
                afterStats.targetRecoveryForwards
                - beforeStats.targetRecoveryForwards,
            targetFallbackForwards:
                afterStats.targetFallbackForwards
                - beforeStats.targetFallbackForwards,
            adaptiveFallbacks:
                afterStats.adaptiveFallbacks - beforeStats.adaptiveFallbacks,
            acceptanceRate: draftedTokens > 0
                ? Double(acceptedDraftTokens) / Double(draftedTokens)
                : nil,
            outputFingerprint: stableFingerprint(response.response),
            output: logResponses ? response.response : nil
        )
    }

    private func runConcurrentMatrix(
        generator: LagunaGenerator,
        decodeLengths: [Int],
        concurrencyLevels: [Int]
    ) async throws -> [LagunaDFlashConcurrentGroup] {
        let fixtures = mixedFixtures
            ? LagunaDFlashBenchmarkFixture.allCases
            : [fixture]
        var groups: [LagunaDFlashConcurrentGroup] = []
        var groupIndex = 0

        for concurrency in concurrencyLevels {
            let warmupLength = min(4, decodeLengths[0])
            let warmupWorkloads = try makeConcurrentWorkloads(
                concurrency: concurrency,
                fixtures: fixtures,
                decodeLengths: [warmupLength],
                lengthIndex: 0,
                repetition: 0
            )
            for warmup in 0..<warmupRepetitions {
                for mode in [LagunaDFlashRoutingMode.targetOnly, .dflash] {
                    _ = try await runConcurrentGroup(
                        generator: generator,
                        workloads: warmupWorkloads,
                        concurrency: concurrency,
                        repetition: -(warmup + 1),
                        order: 0,
                        groupIndex: -1,
                        mode: mode
                    )
                }
            }

            for lengthIndex in decodeLengths.indices {
                for repetition in 0..<repetitions {
                    var baseModes: [LagunaDFlashRoutingMode] = [
                        .targetOnly,
                        .dflash,
                    ]
                    if includeAutomatic {
                        baseModes.append(.automatic)
                    }
                    let rotation =
                        (lengthIndex + repetition + concurrency) % baseModes.count
                    let modes = (0..<baseModes.count).map {
                        baseModes[(rotation + $0) % baseModes.count]
                    }
                    let workloads = try makeConcurrentWorkloads(
                        concurrency: concurrency,
                        fixtures: fixtures,
                        decodeLengths: decodeLengths,
                        lengthIndex: lengthIndex,
                        repetition: repetition
                    )
                    for (order, mode) in modes.enumerated() {
                        groups.append(try await runConcurrentGroup(
                            generator: generator,
                            workloads: workloads,
                            concurrency: concurrency,
                            repetition: repetition,
                            order: order,
                            groupIndex: groupIndex,
                            mode: mode
                        ))
                        groupIndex += 1
                    }
                }
            }
        }
        return groups
    }

    private func makeConcurrentWorkloads(
        concurrency: Int,
        fixtures: [LagunaDFlashBenchmarkFixture],
        decodeLengths: [Int],
        lengthIndex: Int,
        repetition: Int
    ) throws -> [LagunaDFlashConcurrentWorkload] {
        let customMessages = mixedFixtures ? nil : try resolvedMessages()
        return (0..<concurrency).map { rowIndex in
            let selectedFixture = fixtures[
                (lengthIndex + repetition + rowIndex) % fixtures.count
            ]
            let decodeIndex = mixedFixtures
                ? (lengthIndex + rowIndex) % decodeLengths.count
                : lengthIndex
            let decodeTokens = decodeLengths[decodeIndex]
            let messages = customMessages ?? resolvedMessages(for: selectedFixture)
            let workloadFixture = prompt != nil || promptFile != nil
                ? "custom"
                : selectedFixture.rawValue
            return LagunaDFlashConcurrentWorkload(
                rowIndex: rowIndex,
                workloadID: "\(workloadFixture):\(decodeTokens)",
                fixture: workloadFixture,
                request: ChatRequest(
                    messages: messages,
                    maxTokens: decodeTokens,
                    temperature: temperature,
                    topP: topP,
                    topK: topK,
                    minP: minP,
                    showThinking: false,
                    stopOnEOS: false,
                    maxContextTokens: contextSize
                ),
                expectedTokens: decodeTokens
            )
        }
    }

    private func runConcurrentGroup(
        generator: LagunaGenerator,
        workloads: [LagunaDFlashConcurrentWorkload],
        concurrency: Int,
        repetition: Int,
        order: Int,
        groupIndex: Int,
        mode: LagunaDFlashRoutingMode
    ) async throws -> LagunaDFlashConcurrentGroup {
        GPU.resetPeakMemory()
        let beforeMemory = Memory.snapshot()
        let beforeDFlash = await generator.dflashStats()
        let beforeBatching = await generator.continuousBatchingStats()
        let groupStart = Date()
        let targetPath = lagunaPath
        let includeResponses = logResponses

        let rows = try await withThrowingTaskGroup(
            of: LagunaDFlashConcurrentRow.self
        ) { group in
            for workload in workloads {
                group.addTask {
                    let startedAt = Date()
                    let response = try await generator.chat(
                        workload.request,
                        modelPath: targetPath,
                        dflashRouting: mode,
                        progressHandler: nil
                    )
                    let elapsed = Date().timeIntervalSince(startedAt)
                    guard response.tokensGenerated == workload.expectedTokens else {
                        throw ValidationError(
                            "\(mode.rawValue) \(workload.workloadID) generated "
                                + "\(response.tokensGenerated) tokens, expected "
                                + "\(workload.expectedTokens)."
                        )
                    }
                    guard let timing = response.timing else {
                        throw ValidationError(
                            "\(mode.rawValue) \(workload.workloadID) has no timing."
                        )
                    }
                    return LagunaDFlashConcurrentRow(
                        rowIndex: workload.rowIndex,
                        workloadID: workload.workloadID,
                        fixture: workload.fixture,
                        promptTokens: response.promptTokens ?? 0,
                        generatedTokens: response.tokensGenerated,
                        elapsedSeconds: elapsed,
                        prefillSeconds: timing.prefillSeconds,
                        decodeSeconds: timing.decodeSeconds,
                        firstTokenSeconds: timing.firstTokenSeconds,
                        outputFingerprint: stableFingerprint(response.response),
                        output: includeResponses ? response.response : nil
                    )
                }
            }
            var completed: [LagunaDFlashConcurrentRow] = []
            for try await row in group {
                completed.append(row)
            }
            return completed.sorted { $0.rowIndex < $1.rowIndex }
        }

        let wallSeconds = Date().timeIntervalSince(groupStart)
        let afterBatching = await generator.continuousBatchingStats()
        let afterDFlash = await generator.dflashStats()
        let afterMemory = Memory.snapshot()
        let totalTokens = rows.map(\.generatedTokens).reduce(0, +)
        let decodeRates = rows.compactMap { row -> Double? in
            guard row.decodeSeconds > 0 else { return nil }
            return Double(row.generatedTokens) / row.decodeSeconds
        }
        let firstTokenValues = rows.compactMap(\.firstTokenSeconds)
        let draftedTokens =
            afterDFlash.draftedTokens - beforeDFlash.draftedTokens
        let acceptedDraftTokens =
            afterDFlash.acceptedDraftTokens - beforeDFlash.acceptedDraftTokens

        return LagunaDFlashConcurrentGroup(
            groupIndex: groupIndex,
            concurrency: concurrency,
            repetition: repetition,
            order: order,
            mode: mode.rawValue,
            wallSeconds: wallSeconds,
            aggregateTokensPerSecond: wallSeconds > 0
                ? Double(totalTokens) / wallSeconds
                : 0,
            p50LatencySeconds: percentile(rows.map(\.elapsedSeconds), 0.50),
            p95LatencySeconds: percentile(rows.map(\.elapsedSeconds), 0.95),
            p50FirstTokenSeconds: percentile(firstTokenValues, 0.50),
            p95FirstTokenSeconds: percentile(firstTokenValues, 0.95),
            fairnessRatio: fairnessRatio(decodeRates),
            activeMemoryBeforeBytes: beforeMemory.activeMemory,
            activeMemoryAfterBytes: afterMemory.activeMemory,
            cacheMemoryAfterBytes: afterMemory.cacheMemory,
            peakMemoryBytes: afterMemory.peakMemory,
            batchedDecodeSteps:
                afterBatching.batchedDecodeSteps - beforeBatching.batchedDecodeSteps,
            samePositionBatchedSteps:
                afterBatching.samePositionBatchedSteps
                - beforeBatching.samePositionBatchedSteps,
            variablePositionBatchedSteps:
                afterBatching.variablePositionBatchedSteps
                - beforeBatching.variablePositionBatchedSteps,
            singleDecodeSteps:
                afterBatching.singleDecodeSteps - beforeBatching.singleDecodeSteps,
            totalBatchedRows:
                afterBatching.totalBatchedRows - beforeBatching.totalBatchedRows,
            maximumBatchSize: afterBatching.maxBatchSize,
            dflashRounds: afterDFlash.rounds - beforeDFlash.rounds,
            draftedTokens: draftedTokens,
            acceptedDraftTokens: acceptedDraftTokens,
            rejectedDraftTokens:
                afterDFlash.rejectedDraftTokens - beforeDFlash.rejectedDraftTokens,
            targetVerificationForwards:
                afterDFlash.targetVerificationForwards
                - beforeDFlash.targetVerificationForwards,
            targetRecoveryForwards:
                afterDFlash.targetRecoveryForwards
                - beforeDFlash.targetRecoveryForwards,
            targetFallbackForwards:
                afterDFlash.targetFallbackForwards
                - beforeDFlash.targetFallbackForwards,
            adaptiveFallbacks:
                afterDFlash.adaptiveFallbacks - beforeDFlash.adaptiveFallbacks,
            acceptanceRate: draftedTokens > 0
                ? Double(acceptedDraftTokens) / Double(draftedTokens)
                : nil,
            rows: rows
        )
    }

    private func makeExactnessReport(
        samples: [LagunaDFlashBenchmarkSample],
        concurrencyGroups: [LagunaDFlashConcurrentGroup]
    ) -> LagunaDFlashExactnessReport {
        var targetFingerprints: [String: String] = [:]
        var inconsistentTargetWorkloads = Set<String>()

        func recordTarget(workloadID: String, fingerprint: String) {
            if let existing = targetFingerprints[workloadID],
               existing != fingerprint {
                inconsistentTargetWorkloads.insert(workloadID)
            } else {
                targetFingerprints[workloadID] = fingerprint
            }
        }

        for sample in samples where sample.mode == LagunaDFlashRoutingMode.targetOnly.rawValue {
            recordTarget(
                workloadID: sample.workloadID,
                fingerprint: sample.outputFingerprint
            )
        }
        for group in concurrencyGroups
            where group.mode == LagunaDFlashRoutingMode.targetOnly.rawValue {
            for row in group.rows {
                recordTarget(
                    workloadID: row.workloadID,
                    fingerprint: row.outputFingerprint
                )
            }
        }

        var comparedRows = 0
        var mismatches = Set<String>()
        for sample in samples where sample.mode != LagunaDFlashRoutingMode.targetOnly.rawValue {
            guard let target = targetFingerprints[sample.workloadID] else { continue }
            comparedRows += 1
            if target != sample.outputFingerprint {
                mismatches.insert("\(sample.mode):\(sample.workloadID)")
            }
        }
        for group in concurrencyGroups
            where group.mode != LagunaDFlashRoutingMode.targetOnly.rawValue {
            for row in group.rows {
                guard let target = targetFingerprints[row.workloadID] else { continue }
                comparedRows += 1
                if target != row.outputFingerprint {
                    mismatches.insert("\(group.mode):\(row.workloadID)")
                }
            }
        }

        return LagunaDFlashExactnessReport(
            byteExactTargetEquivalent:
                mismatches.isEmpty && inconsistentTargetWorkloads.isEmpty,
            comparedRows: comparedRows,
            mismatchWorkloads: mismatches.sorted(),
            inconsistentTargetWorkloads: inconsistentTargetWorkloads.sorted()
        )
    }

    private func percentile(_ values: [Double], _ quantile: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let index = max(
            0,
            min(sorted.count - 1, Int(ceil(quantile * Double(sorted.count))) - 1)
        )
        return sorted[index]
    }

    private func fairnessRatio(_ values: [Double]) -> Double? {
        guard let minimum = values.min(),
              let maximum = values.max(),
              maximum > 0 else {
            return nil
        }
        return minimum / maximum
    }
}

enum LagunaDFlashBenchmarkFixture: String, CaseIterable, ExpressibleByArgument {
    case deterministicProse = "deterministic-prose"
    case groundedEmail = "grounded-email"
    case codeCompletion = "code-completion"
}

private func stableFingerprint(_ value: String) -> String {
    var hash: UInt64 = 14_695_981_039_346_656_037
    for byte in value.utf8 {
        hash ^= UInt64(byte)
        hash &*= 1_099_511_628_211
    }
    let hex = String(hash, radix: 16)
    return String(repeating: "0", count: max(0, 16 - hex.count)) + hex
}

private struct LagunaDFlashConcurrentWorkload: Sendable {
    let rowIndex: Int
    let workloadID: String
    let fixture: String
    let request: ChatRequest
    let expectedTokens: Int
}

private struct LagunaDFlashBenchmarkReport: Encodable {
    let targetPath: String
    let dflashPath: String
    let speculativeTokens: Int
    let repetitions: Int
    let temperature: Double
    let topP: Double
    let topK: Int
    let minP: Double
    let promptCharacters: Int
    let hardware: LagunaDFlashBenchmarkHardware
    let samples: [LagunaDFlashBenchmarkSample]
    let concurrencyGroups: [LagunaDFlashConcurrentGroup]
    let exactness: LagunaDFlashExactnessReport

    func jsonString() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }

    func renderText() -> String {
        var lines = [
            "Laguna DFlash resident A/B",
            "speculative_tokens: \(speculativeTokens)",
            "repetitions: \(repetitions)",
            "sampling: temperature=\(temperature) top_p=\(topP) "
                + "top_k=\(topK) min_p=\(minP)",
            "hardware: \(hardware.architecture)",
        ]
        for decodeTokens in Set(samples.map(\.decodeTokens)).sorted() {
            let target = samples.filter {
                $0.decodeTokens == decodeTokens && $0.mode == LagunaDFlashRoutingMode.targetOnly.rawValue
            }
            let dflash = samples.filter {
                $0.decodeTokens == decodeTokens && $0.mode == LagunaDFlashRoutingMode.dflash.rawValue
            }
            let targetDecode = median(target.map(\.decodeSeconds))
            let dflashDecode = median(dflash.map(\.decodeSeconds))
            let speedup = targetDecode > 0
                ? (targetDecode / dflashDecode - 1) * 100
                : 0
            let drafted = dflash.map(\.draftedTokens).reduce(0, +)
            let accepted = dflash.map(\.acceptedDraftTokens).reduce(0, +)
            let acceptance = drafted > 0 ? Double(accepted) / Double(drafted) : 0
            lines.append(
                String(
                    format: "tokens=%d target_decode=%.3fs dflash_decode=%.3fs "
                        + "speedup=%+.1f%% acceptance=%.3f",
                    decodeTokens,
                    targetDecode,
                    dflashDecode,
                    speedup,
                    acceptance
                )
            )
            let automatic = samples.filter {
                $0.decodeTokens == decodeTokens
                    && $0.mode == LagunaDFlashRoutingMode.automatic.rawValue
            }
            if !automatic.isEmpty {
                let automaticDecode = median(automatic.map(\.decodeSeconds))
                let automaticSpeedup = targetDecode > 0
                    ? (targetDecode / automaticDecode - 1) * 100
                    : 0
                lines.append(
                    String(
                        format: "tokens=%d automatic_decode=%.3fs speedup=%+.1f%% "
                            + "fallbacks=%d",
                        decodeTokens,
                        automaticDecode,
                        automaticSpeedup,
                        automatic.map(\.adaptiveFallbacks).reduce(0, +)
                    )
                )
            }
        }
        for concurrency in Set(concurrencyGroups.map(\.concurrency)).sorted() {
            for mode in [
                LagunaDFlashRoutingMode.targetOnly.rawValue,
                LagunaDFlashRoutingMode.dflash.rawValue,
                LagunaDFlashRoutingMode.automatic.rawValue,
            ] {
                let matching = concurrencyGroups.filter {
                    $0.concurrency == concurrency && $0.mode == mode
                }
                guard !matching.isEmpty else { continue }
                let aggregateRate = median(matching.map(\.aggregateTokensPerSecond))
                let wall = median(matching.map(\.wallSeconds))
                let fairnessValues = matching.compactMap(\.fairnessRatio)
                let fairness = fairnessValues.isEmpty ? 0 : median(fairnessValues)
                lines.append(String(
                    format: "concurrency=%d mode=%@ wall=%.3fs aggregate=%.3f tok/s "
                        + "fairness=%.3f max_batch=%d",
                    concurrency,
                    mode,
                    wall,
                    aggregateRate,
                    fairness,
                    matching.map(\.maximumBatchSize).max() ?? 0
                ))
            }
        }
        lines.append(
            "byte_exact_target_equivalent: "
                + "\(exactness.byteExactTargetEquivalent) "
                + "compared_rows=\(exactness.comparedRows) "
                + "mismatches=\(exactness.mismatchWorkloads.count) "
                + "target_inconsistencies="
                + "\(exactness.inconsistentTargetWorkloads.count)"
        )
        return lines.joined(separator: "\n")
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }
}

private struct LagunaDFlashBenchmarkHardware: Encodable {
    let architecture: String
    let physicalMemoryBytes: Int
    let recommendedWorkingSetBytes: UInt64
}

private struct LagunaDFlashExactnessReport: Encodable {
    let byteExactTargetEquivalent: Bool
    let comparedRows: Int
    let mismatchWorkloads: [String]
    let inconsistentTargetWorkloads: [String]
}

private struct LagunaDFlashConcurrentGroup: Encodable {
    let groupIndex: Int
    let concurrency: Int
    let repetition: Int
    let order: Int
    let mode: String
    let wallSeconds: Double
    let aggregateTokensPerSecond: Double
    let p50LatencySeconds: Double?
    let p95LatencySeconds: Double?
    let p50FirstTokenSeconds: Double?
    let p95FirstTokenSeconds: Double?
    let fairnessRatio: Double?
    let activeMemoryBeforeBytes: Int
    let activeMemoryAfterBytes: Int
    let cacheMemoryAfterBytes: Int
    let peakMemoryBytes: Int
    let batchedDecodeSteps: Int
    let samePositionBatchedSteps: Int
    let variablePositionBatchedSteps: Int
    let singleDecodeSteps: Int
    let totalBatchedRows: Int
    let maximumBatchSize: Int
    let dflashRounds: Int
    let draftedTokens: Int
    let acceptedDraftTokens: Int
    let rejectedDraftTokens: Int
    let targetVerificationForwards: Int
    let targetRecoveryForwards: Int
    let targetFallbackForwards: Int
    let adaptiveFallbacks: Int
    let acceptanceRate: Double?
    let rows: [LagunaDFlashConcurrentRow]
}

private struct LagunaDFlashConcurrentRow: Encodable, Sendable {
    let rowIndex: Int
    let workloadID: String
    let fixture: String
    let promptTokens: Int
    let generatedTokens: Int
    let elapsedSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let outputFingerprint: String
    let output: String?
}

private struct LagunaDFlashBenchmarkSample: Encodable {
    let workloadID: String
    let decodeTokens: Int
    let repetition: Int
    let order: Int
    let mode: String
    let promptTokens: Int
    let generatedTokens: Int
    let elapsedSeconds: Double
    let loadSeconds: Double
    let prefillSeconds: Double
    let decodeSeconds: Double
    let firstTokenSeconds: Double?
    let activeMemoryBeforeBytes: Int
    let activeMemoryAfterBytes: Int
    let cacheMemoryAfterBytes: Int
    let peakMemoryBytes: Int
    let dflashRounds: Int
    let draftedTokens: Int
    let acceptedDraftTokens: Int
    let rejectedDraftTokens: Int
    let targetVerificationForwards: Int
    let targetRecoveryForwards: Int
    let targetFallbackForwards: Int
    let adaptiveFallbacks: Int
    let acceptanceRate: Double?
    let outputFingerprint: String
    let output: String?
}
