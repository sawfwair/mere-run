import Foundation

/// Chunked concurrent iteration for the remesher and atlas baker. Bodies
/// receive disjoint index ranges; callers hand out disjoint output regions
/// via unsafe buffer pointers.
enum Trellis2Parallel {
    /// Linux's swift-corelibs-dispatch declares `concurrentPerform`'s closure
    /// `@Sendable` (Darwin's SDK does not), so the non-Sendable body must be
    /// smuggled across in a box. Sound because chunk ranges are disjoint and
    /// `concurrentPerform` returns only after every iteration completes.
    private struct UncheckedSendableBody: @unchecked Sendable {
        let call: (Range<Int>) -> Void
    }

    static func chunks(_ count: Int, chunk: Int = 4_096, _ body: (Range<Int>) -> Void) {
        let chunkCount = (count + chunk - 1) / chunk
        if chunkCount <= 1 {
            body(0..<count)
            return
        }
        withoutActuallyEscaping(body) { body in
            let boxed = UncheckedSendableBody(call: body)
            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunkIndex in
                let start = chunkIndex * chunk
                boxed.call(start..<min(start + chunk, count))
            }
        }
    }
}
