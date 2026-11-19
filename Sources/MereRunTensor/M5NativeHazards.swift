#if os(macOS)
import Metal

/// Dependency tracking for one concurrent compute encoder. Intervals distinguish
/// independent fields in a shared arena. A Metal resource barrier drains all
/// pending accesses to that buffer, including its other intervals.
@available(macOS 26.0, *)
package final class M5NativeHazards {
    package struct Access {
        let buffer: any MTLBuffer
        let range: Range<Int>
        package init(_ buffer: any MTLBuffer, offset: Int = 0, count: Int? = nil) {
            let end = offset + (count ?? (buffer.length - offset))
            precondition(offset >= 0 && end > offset && end <= buffer.length)
            self.buffer = buffer
            range = offset..<end
        }
    }
    private struct Pending {
        let range: Range<Int>
        let writes: Bool
    }
    private var pending: [ObjectIdentifier: [Pending]] = [:]

    package init() {}

    package func prepare(encoder: any MTLComputeCommandEncoder, reads: [Access], writes: [Access]) {
        let accesses = reads.map { ($0, false) } + writes.map { ($0, true) }
        var drained: Set<ObjectIdentifier> = []
        var barriers: [any MTLResource] = []
        for (access, writing) in accesses {
            let key = ObjectIdentifier(access.buffer)
            if pending[key]?.contains(where: { ($0.writes || writing) && $0.range.overlaps(access.range) }) == true,
               drained.insert(key).inserted {
                barriers.append(access.buffer)
            }
        }
        if !barriers.isEmpty {
            encoder.memoryBarrier(resources: barriers)
            for key in drained { pending.removeValue(forKey: key) }
        }
        for (access, writing) in accesses {
            pending[ObjectIdentifier(access.buffer), default: []].append(Pending(range: access.range, writes: writing))
        }
    }
}
#endif
