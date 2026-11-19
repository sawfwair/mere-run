import Foundation

/// Submission boundaries change scheduling only. They do not alter the
/// projection kernels, prefill token grid, cache contents, or sampling draws.
package enum Q35DenseSubmissionPolicy {
    @TaskLocal package static var decodeStride = stride(
        ProcessInfo.processInfo.environment["MERERUN_Q38_DECODE_ASYNC_BLOCKS"]
    )
    @TaskLocal package static var prefillStride = stride(
        ProcessInfo.processInfo.environment["MERERUN_Q38_PREFILL_ASYNC_BLOCKS"]
    )

    package static func stride(_ raw: String?) -> Int {
        raw.flatMap(Int.init).flatMap { [1, 4, 8, 16].contains($0) ? $0 : nil } ?? 0
    }
}
