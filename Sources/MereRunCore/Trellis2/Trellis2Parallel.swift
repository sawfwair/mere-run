import Foundation

/// Chunked concurrent iteration for the remesher and atlas baker. Bodies
/// receive disjoint index ranges; callers hand out disjoint output regions
/// via unsafe buffer pointers.
enum Trellis2Parallel {
    static func chunks(_ count: Int, chunk: Int = 4_096, _ body: (Range<Int>) -> Void) {
        let chunkCount = (count + chunk - 1) / chunk
        if chunkCount <= 1 {
            body(0..<count)
            return
        }
        DispatchQueue.concurrentPerform(iterations: chunkCount) { chunkIndex in
            let start = chunkIndex * chunk
            body(start..<min(start + chunk, count))
        }
    }
}
