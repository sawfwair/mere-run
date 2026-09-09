import Foundation
import MLX

/// Request-boundary allocator measurements for diagnosing long-running sessions.
enum Q35MemoryTrace {
    private static let enabled = ProcessInfo.processInfo.environment["MERERUN_Q35_DEBUG_MEMORY"] == "1"

    private struct Record: Encodable {
        let model: String
        let memory: Memory.Snapshot
        let memoryLimit: Int
        let cacheLimit: Int
    }

    static func record(modelID: String) {
        guard enabled else { return }
        let record = Record(
            model: modelID,
            memory: Memory.snapshot(),
            memoryLimit: Memory.memoryLimit,
            cacheLimit: Memory.cacheLimit
        )
        guard let data = try? JSONEncoder().encode(record) else { return }
        FileHandle.standardError.write(Data("[q35-memory] ".utf8) + data + Data("\n".utf8))
    }
}
