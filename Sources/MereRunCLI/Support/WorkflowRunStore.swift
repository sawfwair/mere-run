import ArgumentParser
import Foundation
import MereRunExecution
import MereRunRelayKit

/// Serial run persistence. The execution driver retains the returned lease
/// until every child and terminal record has completed.
struct WorkflowRunStore {
    struct Session {
        let lease: RunDirectoryLease
        let manifest: GraphRunManifest
        let eventCount: Int
    }

    let bundleDirectory: URL
    let runDirectory: URL
    let resume: Bool
    let executor: GraphRunExecutorRecord
    let fileManager: FileManager
    let now: () -> Date
    let eventHandler: ((GraphRunEvent) -> Void)?

    func prepare(graph: WorkflowGraphDocument, job: WorkflowJobManifest, order: [String]) throws -> Session {
        if fileManager.fileExists(atPath: runDirectory.path) {
            guard resume || bundleDirectory == runDirectory else {
                throw ValidationError("Run directory already exists. Pass --resume to reuse it: \(runDirectory.path)")
            }
        } else {
            try fileManager.createDirectory(at: runDirectory, withIntermediateDirectories: true)
        }
        guard let lease = try RunDirectoryLease.acquire(in: runDirectory, filename: ".workflow-run.lock") else {
            throw ValidationError("Workflow run already has an active worker: \(runDirectory.path)")
        }
        let liveChildren = WorkflowChildProcessRegistry.activeProcessIDs(in: runDirectory, fileManager: fileManager)
        guard liveChildren.isEmpty else {
            throw ValidationError("Workflow run still has active child processes: \(liveChildren.map(String.init).joined(separator: ", ")). Wait for them to stop before resuming.")
        }
        let manifest = try initialManifest(graph: graph, job: job, order: order)
        let eventCount = try existingEventCount()
        try initializeDirectory()
        return Session(lease: lease, manifest: manifest, eventCount: eventCount)
    }

    private func initializeDirectory() throws {
        try fileManager.createDirectory(
            at: runDirectory.appendingPathComponent("nodes", isDirectory: true),
            withIntermediateDirectories: true
        )
        let cancellationURL = runDirectory.appendingPathComponent("cancel.request")
        if fileManager.fileExists(atPath: cancellationURL.path) {
            try fileManager.removeItem(at: cancellationURL)
        }
        try WorkflowChildProcessRegistry.clear(in: runDirectory, fileManager: fileManager)
        try fileManager.createDirectory(
            at: runDirectory.appendingPathComponent("outputs", isDirectory: true),
            withIntermediateDirectories: true
        )
        for filename in ["graph.json", "inputs.json", WorkflowAssetManifest.filename, WorkflowJobManifest.filename] {
            let source = bundleDirectory.appendingPathComponent(filename)
            let destination = runDirectory.appendingPathComponent(filename)
            if source != destination, !fileManager.fileExists(atPath: destination.path) {
                try fileManager.copyItem(at: source, to: destination)
            }
        }
        let actionsURL = runDirectory.appendingPathComponent("actions.json")
        if !fileManager.fileExists(atPath: actionsURL.path) {
            try WorkflowBundleCodec.write([DeclarativeAction](), to: actionsURL)
        }
    }

    func initialManifest(
        graph: WorkflowGraphDocument,
        job: WorkflowJobManifest,
        order: [String]
    ) throws -> GraphRunManifest {
        let manifestURL = runDirectory.appendingPathComponent(GraphRunManifest.filename)
        if resume, fileManager.fileExists(atPath: manifestURL.path) {
            var existing = try WorkflowBundleCodec.decoder().decode(
                GraphRunManifest.self,
                from: Data(contentsOf: manifestURL)
            )
            guard existing.contractVersion == GraphRunManifest.contractVersion else {
                throw ValidationError("Cannot resume: unsupported workflow run contract '\(existing.contractVersion)'.")
            }
            guard existing.graphFingerprint == job.graphFingerprint else {
                throw ValidationError("Cannot resume: workflow graph fingerprint changed.")
            }
            guard existing.sourceInputFingerprint == job.sourceInputFingerprint else {
                throw ValidationError("Cannot resume: workflow input fingerprint changed.")
            }
            let previousJob = try WorkflowBundleCodec.decoder().decode(
                WorkflowJobManifest.self,
                from: Data(contentsOf: runDirectory.appendingPathComponent(WorkflowJobManifest.filename))
            )
            guard previousJob.inputFingerprint == job.inputFingerprint else {
                throw ValidationError("Cannot resume: workflow input fingerprint changed.")
            }
            existing.attempt += 1
            existing.state = .planned
            existing.error = nil
            existing.executor = executor
            existing.updatedAt = now()
            return existing
        }
        let nodesByID = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        return GraphRunManifest(
            contractVersion: GraphRunManifest.contractVersion,
            jobID: job.jobID,
            graphName: graph.name,
            graphFingerprint: job.graphFingerprint,
            sourceGraphFingerprint: job.sourceGraphFingerprint,
            sourceInputFingerprint: job.sourceInputFingerprint,
            state: .planned,
            createdAt: now(),
            updatedAt: now(),
            attempt: 1,
            executor: executor,
            nodes: order.compactMap { id in
                nodesByID[id].map {
                    GraphRunNodeRecord(
                        id: id,
                        kind: $0.kind,
                        state: .planned,
                        startedAt: nil,
                        completedAt: nil,
                        exitStatus: nil,
                        attempt: 0,
                        maxAttempts: $0.execution?.resolvedMaxAttempts ?? 1,
                        fingerprint: "",
                        artifacts: [],
                        error: nil
                    )
                }
            },
            outputs: [],
            error: nil
        )
    }

    func persist(_ manifest: GraphRunManifest) throws {
        try WorkflowBundleCodec.write(manifest, to: runDirectory.appendingPathComponent(GraphRunManifest.filename))
    }

    func record(_ event: GraphRunEvent) throws {
        let data = try WorkflowBundleCodec.lineEncoder().encode(event)
        let url = runDirectory.appendingPathComponent("events.jsonl")
        if !fileManager.fileExists(atPath: url.path) {
            try Data().write(to: url)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data + Data("\n".utf8))
        try handle.synchronize()
        eventHandler?(event)
    }

    func existingEventCount() throws -> Int {
        let url = runDirectory.appendingPathComponent("events.jsonl")
        guard fileManager.fileExists(atPath: url.path) else { return 0 }
        return try String(contentsOf: url, encoding: .utf8).split(separator: "\n").count
    }}
