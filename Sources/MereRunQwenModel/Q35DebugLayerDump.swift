import Foundation
import MLX

/// Env-gated reference-parity instrumentation. Set
/// MERERUN_Q35_DEBUG_LAYER_DUMP=<path> to write one JSON line per stage
/// (embeddings, each decoder layer, final norm) with the float32 L2 norm and
/// leading values of the LAST prompt position during a multi-token forward.
/// Decode steps (single-token forwards) are ignored so the dump captures the
/// prompt prefill only. The matching mlx_lm-side script lives in the
/// reference-parity harness; keys and value layout must stay in sync with it.
enum Q35DebugLayerDump {
    static let path: String? = ProcessInfo.processInfo.environment["MERERUN_Q35_DEBUG_LAYER_DUMP"]

    /// Debug-only accumulation on the single generation path; not
    /// concurrency-safe by design (the dump is meaningless under concurrent
    /// generation anyway).
    nonisolated(unsafe) private static var lines: [String] = []

    static var isEnabled: Bool { path != nil }

    static func record(stage: String, _ hidden: MLXArray) {
        guard isEnabled, hidden.ndim == 3, hidden.dim(1) > 1 else { return }
        let last = hidden[0, hidden.dim(1) - 1, 0...].asType(.float32)
        MLX.eval(last)
        let norm = sqrt(MLX.sum(last * last).item(Float.self))
        let head = last[0..<8].asArray(Float.self)
        let headJSON = head.map { String(format: "%.6g", $0) }.joined(separator: ",")
        lines.append("{\"stage\":\"\(stage)\",\"norm\":\(String(format: "%.6g", norm)),\"head\":[\(headJSON)]}")
    }

    static func flush() {
        guard let path, !lines.isEmpty else { return }
        try? (lines.joined(separator: "\n") + "\n").write(
            toFile: path,
            atomically: true,
            encoding: .utf8
        )
        lines.removeAll()
    }
}
