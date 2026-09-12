import ArgumentParser
import Foundation
import MereRunExecution
import MereRunRelayKit

enum WorkflowRunRecovery {
    static func inspect(at directory: URL, now: Date = Date()) throws -> GraphRunManifest {
        let url = directory.appendingPathComponent(GraphRunManifest.filename)
        func read() throws -> GraphRunManifest {
            let manifest = try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: Data(contentsOf: url))
            guard manifest.contractVersion == GraphRunManifest.contractVersion else {
                throw ValidationError("Unsupported workflow run contract '\(manifest.contractVersion)'.")
            }
            return manifest
        }
        let snapshot = try read()
        guard isRecoverable(snapshot) else { return snapshot }
        guard let lease = try RunDirectoryLease.acquire(in: directory, filename: ".workflow-run.lock") else {
            return snapshot
        }
        defer { lease.release() }
        var manifest = try read()
        guard isRecoverable(manifest), WorkflowChildProcessRegistry.activeProcessIDs(in: directory).isEmpty else {
            return manifest
        }
        let sequence = try WorkflowRunStore.repairEventTail(in: directory)
        let message = "Workflow worker stopped before completion. Resume the run to reuse verified node outputs."
        for index in manifest.nodes.indices where manifest.nodes[index].state == .running
            || manifest.nodes[index].state == .preflighting {
            manifest.nodes[index].state = .failed
            manifest.nodes[index].completedAt = now
            manifest.nodes[index].error = message
        }
        manifest.state = .failed
        manifest.interruptedAt = now
        manifest.updatedAt = now
        manifest.error = message
        let store = WorkflowRunStore(
            bundleDirectory: directory, runDirectory: directory, resume: false,
            executor: manifest.executor, fileManager: .default, now: { now }, eventHandler: nil
        )
        try store.record(.init(
            sequence: sequence, createdAt: now, type: "run_interrupted",
            state: .failed, nodeID: nil, message: message
        ))
        try store.persist(manifest)
        return manifest
    }

    private static func isRecoverable(_ manifest: GraphRunManifest) -> Bool {
        ["local", "worker"].contains(manifest.executor.kind)
            && [.planned, .preflighting, .running].contains(manifest.state)
    }
}
