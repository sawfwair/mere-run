import ArgumentParser
import Foundation
import MereRunRelayKit

struct GraphBatch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch",
        abstract: "Distribute independent complete workflow graphs across a relay fleet.",
        subcommands: [GraphBatchList.self, GraphBatchSubmit.self, GraphBatchInspect.self, GraphBatchWatch.self,
                      GraphBatchCancel.self, GraphBatchRetry.self, GraphBatchFetch.self]
    )
}

struct GraphBatchSubmit: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "submit", abstract: "Prepare and submit a durable workflow batch.")

    @Argument(help: "Workflow graph JSON file.") var file: String
    @Option(name: .customLong("items-jsonl"), help: "JSONL file of {key, inputs} rows, or a JSON array file.") var itemsJSONL: String
    @Option(name: .long, help: "Relay executor reference: relay:<profile>.") var executor: String
    @Option(name: .customLong("run-dir"), help: "Durable local batch directory; reuse to resume submission.") var runDirectory: String
    @Option(name: .customLong("max-in-flight"), help: "Maximum active children admitted for this batch (1–20).") var maxInFlight = 8
    @Option(name: .customLong("max-attempts"), help: "Total infrastructure attempts per child (1–4); use retries only for replay-safe workflows.") var maxAttempts = 1
    @Flag(name: .long, help: "Emit JSON (batch commands always emit machine-readable JSON).") var json = false

    func run() async throws {
        try await mapRelayErrors {
            let relay = try batchExecutor(executor)
            let graph = try WorkflowGraphDocument.load(from: URL(fileURLWithPath: file))
            let items = try Self.loadItems(URL(fileURLWithPath: itemsJSONL))
            let root = URL(fileURLWithPath: runDirectory).standardizedFileURL
            _ = try WorkflowBatchMaterializer.materialize(graph: graph, items: items, destination: root,
                                                         environment: .cli, maxInFlight: maxInFlight, maxAttempts: maxAttempts)
            try printBatch(try await relay.submitBatch(directory: root, pluginNodes: WorkflowMaterializationEnvironment.cli.pluginNodes))
        }
    }

    static func loadItems(_ url: URL) throws -> [WorkflowBatchInput] {
        let data = try Data(contentsOf: url)
        let decoder = WorkflowBundleCodec.decoder()
        if String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("[") {
            return try decoder.decode([WorkflowBatchInput].self, from: data)
        }
        return try String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).enumerated().map { index, line in
            do { return try decoder.decode(WorkflowBatchInput.self, from: Data(line.utf8)) }
            catch { throw ValidationError("Invalid batch item at line \(index + 1): \(error)") }
        }
    }
}

struct GraphBatchRemoteOptions: ParsableArguments {
    @Argument(help: "Relay batch UUID.") var batchID: String
    @Option(name: .long, help: "Relay executor reference: relay:<profile>.") var executor: String
    @Flag(name: .long, help: "Emit JSON (batch commands always emit machine-readable JSON).") var json = false
}

struct GraphBatchInspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "inspect", abstract: "Inspect batch progress and item output references.")
    @OptionGroup var options: GraphBatchRemoteOptions
    func run() async throws {
        try await mapRelayErrors { try printBatch(try await batchExecutor(options.executor).batch(batchID: options.batchID)) }
    }
}

struct GraphBatchCancel: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "cancel", abstract: "Stop admission and cancel unfinished batch children.")
    @OptionGroup var options: GraphBatchRemoteOptions
    func run() async throws {
        try await mapRelayErrors { try printBatch(try await batchExecutor(options.executor).cancelBatch(batchID: options.batchID)) }
    }
}

struct GraphBatchRetry: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "retry", abstract: "Retry failed items while retaining completed outputs.")
    @OptionGroup var options: GraphBatchRemoteOptions
    func run() async throws {
        try await mapRelayErrors { try printBatch(try await batchExecutor(options.executor).retryBatch(batchID: options.batchID)) }
    }
}

struct GraphBatchWatch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "watch", abstract: "Watch batch changes as NDJSON until terminal.")
    @OptionGroup var options: GraphBatchRemoteOptions
    @Option(name: .customLong("timeout-seconds"), help: "Stop after this many seconds; zero waits indefinitely.") var timeoutSeconds = 0

    func validate() throws {
        guard timeoutSeconds >= 0 else { throw ValidationError("Timeout must be nonnegative.") }
    }

    func run() async throws {
        try await mapRelayErrors {
            let relay = try batchExecutor(options.executor)
            let started = ContinuousClock.now
            var last: Data?
            while true {
                let batch = try await relay.batch(batchID: options.batchID)
                let data = try WorkflowBundleCodec.lineEncoder().encode(batch)
                if data != last { print(String(decoding: data, as: UTF8.self)); last = data }
                if batch.state.isTerminal {
                    if batch.state != .finished { throw ExitCode.failure }
                    return
                }
                if timeoutSeconds > 0, started.duration(to: .now) >= .seconds(timeoutSeconds) {
                    throw ValidationError("Batch watch timed out; the durable batch continues on Relay.")
                }
                try await Task.sleep(for: .seconds(1))
            }
        }
    }
}

struct GraphBatchFetch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "fetch", abstract: "Fetch verified outputs from completed batch items.")
    @OptionGroup var options: GraphBatchRemoteOptions
    @Option(name: .long, help: "Output directory containing one run per completed item.") var into: String

    func run() async throws {
        try await mapRelayErrors {
            let relay = try batchExecutor(options.executor)
            let batch = try await relay.batch(batchID: options.batchID)
            let root = URL(fileURLWithPath: into).standardizedFileURL
            let indexURL = root.appendingPathComponent("batch-results.json")
            if FileManager.default.fileExists(atPath: indexURL.path) {
                let existing = try WorkflowBundleCodec.decoder().decode(RelayGraphBatch.self, from: Data(contentsOf: indexURL))
                guard existing.batchID == batch.batchID else { throw RelayClientError("Fetch directory belongs to another batch.") }
            }
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            for item in batch.items where item.state == .finished {
                _ = try await relay.fetch(jobID: item.jobID, into: root.appendingPathComponent(String(item.ordinal)),
                                          allArtifacts: false)
            }
            try WorkflowBundleCodec.write(batch, to: root.appendingPathComponent("batch-results.json"))
            try printBatch(batch)
        }
    }
}

private func batchExecutor(_ reference: String) throws -> RelayWorkflowExecutor {
    let parsed = try WorkflowExecutorReference(reference)
    guard parsed.kind == .relay else { throw RelayClientError("Batch execution requires relay:<profile>.") }
    return RelayWorkflowExecutor(profile: try WorkflowExecutorProfileStore.require(reference))
}

private func printBatch(_ batch: RelayGraphBatch) throws {
    print(String(decoding: try WorkflowBundleCodec.encoder().encode(batch), as: UTF8.self))
}

struct GraphBatchList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "list", abstract: "List durable batches in the relay account.")
    @Option(name: .long, help: "Relay executor reference: relay:<profile>.") var executor: String
    @Flag(name: .long, help: "Emit JSON (batch commands always emit machine-readable JSON).") var json = false
    func run() async throws {
        try await mapRelayErrors {
            let batches = try await batchExecutor(executor).listBatches()
            print(String(decoding: try WorkflowBundleCodec.encoder().encode(Result(batches: batches)), as: UTF8.self))
        }
    }
    private struct Result: Encodable { let batches: [RelayGraphBatchSummary] }
}
