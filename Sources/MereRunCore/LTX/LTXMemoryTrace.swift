import Foundation
import MereRunEnvironment
import MLX

let ltxMemoryTraceEnvironmentKey = "MERERUN_LTX_MEMORY_TRACE"

func ltxTraceMemory(
    _ phase: String,
    environment: [String: String] = ProcessInfo.processInfo.environment
) {
    guard MereRunEnvironmentSwitch.isEnabled(ltxMemoryTraceEnvironmentKey, default: false, environment: environment) else { return }
    let snapshot = Memory.snapshot()
    let line = ltxMemoryTraceLine(
        phase: phase,
        activeBytes: snapshot.activeMemory,
        cacheBytes: snapshot.cacheMemory,
        peakBytes: snapshot.peakMemory
    )
    FileHandle.standardError.write(Data((line + "\n").utf8))
}

func ltxMemoryTraceLine(
    phase: String,
    activeBytes: Int,
    cacheBytes: Int,
    peakBytes: Int
) -> String {
    "[ltx-memory] phase=\(phase)"
        + " active=\(ltxMemoryGiB(activeBytes))GiB"
        + " cache=\(ltxMemoryGiB(cacheBytes))GiB"
        + " peak=\(ltxMemoryGiB(peakBytes))GiB"
}

private func ltxMemoryGiB(_ bytes: Int) -> String {
    String(format: "%.2f", Double(bytes) / 1_073_741_824.0)
}
