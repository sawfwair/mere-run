import Foundation
import NIOCore

enum RuntimeChatProxyStream {
    /// Keeps the upstream producer and both request leases attached to the
    /// downstream stream. Cancelling its consumer also cancels upstream reads.
    static func make<Source: AsyncSequence & Sendable>(
        from source: Source, session: RuntimeChatSession
    ) -> AsyncStream<ByteBuffer> where Source.Element == UInt8 {
        AsyncStream { continuation in
            let producer = Task {
                var buffer = Data()
                buffer.reserveCapacity(4_096)
                do {
                    for try await byte in source {
                        try Task.checkCancellation()
                        buffer.append(byte)
                        if byte == 10 || buffer.count >= 4_096 {
                            continuation.yield(ByteBuffer(bytes: buffer))
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    try Task.checkCancellation()
                    if !buffer.isEmpty { continuation.yield(ByteBuffer(bytes: buffer)) }
                    await session.finish()
                } catch {
                    await session.finish(cancelled: Task.isCancelled || error is CancellationError)
                }
                continuation.finish()
            }
            continuation.onTermination = { termination in
                if case .cancelled = termination {
                    session.observeClientDisconnect()
                    producer.cancel()
                }
            }
        }
    }
}
