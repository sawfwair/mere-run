import Foundation
import Hummingbird
import NIOCore

/// Couples request preparation and execution to the connected client without
/// consuming its body or changing HTTP keep-alive. Response producers retain
/// their existing ownership once the handler returns a streaming body.
struct APIRequestCancellationMiddleware: RouterMiddleware {
    func handle(
        _ request: Request,
        context: APIServerRequestContext,
        next: (Request, APIServerRequestContext) async throws -> Response
    ) async throws -> Response {
        try Task.checkCancellation()
        return try await withoutActuallyEscaping(next) { next in
            let cancellation = APIRequestCancellation()
            let handler = APIConnectionCancellationHandler(cancellation: cancellation)
            try await context.channel.pipeline.addHandler(handler, position: .first).get()
            let operation = APIRequestOperation { try await next(request, context) }
            let task = Task {
                try cancellation.checkConnection()
                try Task.checkCancellation()
                return try await operation.run()
            }
            cancellation.install { task.cancel() }
            return try await withTaskCancellationHandler {
                do {
                    let response = try await task.value
                    cancellation.finish()
                    try? await context.channel.pipeline.removeHandler(handler).get()
                    return response
                } catch {
                    cancellation.finish()
                    try? await context.channel.pipeline.removeHandler(handler).get()
                    throw error
                }
            } onCancel: {
                task.cancel()
            }
        }
    }
}

/// Hummingbird's next closure is borrowed, not Sendable. It runs in exactly one
/// task, which is awaited before withoutActuallyEscaping returns on every path.
private struct APIRequestOperation: @unchecked Sendable {
    let run: () async throws -> Response
}

private final class APIRequestCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var disconnected = false
    private var cancel: (@Sendable () -> Void)?

    func install(_ cancel: @escaping @Sendable () -> Void) {
        let closed = lock.withLock {
            self.cancel = cancel
            return disconnected
        }
        if closed { cancel() }
    }

    func checkConnection() throws {
        if lock.withLock({ disconnected }) { throw CancellationError() }
    }

    func disconnect() {
        let callback = lock.withLock {
            disconnected = true
            return cancel
        }
        callback?()
    }

    func finish() { lock.withLock { cancel = nil } }
}

/// Removed when the request task settles, so a persistent connection does not
/// accumulate closeFuture callbacks or retain completed request tasks.
private final class APIConnectionCancellationHandler: ChannelInboundHandler, RemovableChannelHandler, @unchecked Sendable {
    typealias InboundIn = ByteBuffer
    private let cancellation: APIRequestCancellation

    init(cancellation: APIRequestCancellation) { self.cancellation = cancellation }

    func handlerAdded(context: ChannelHandlerContext) {
        if !context.channel.isActive { cancellation.disconnect() }
    }

    func channelInactive(context: ChannelHandlerContext) {
        cancellation.disconnect()
        context.fireChannelInactive()
    }

    func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
        if let event = event as? ChannelEvent, event == .inputClosed {
            cancellation.disconnect()
        }
        context.fireUserInboundEventTriggered(event)
    }

    func errorCaught(context: ChannelHandlerContext, error: Error) {
        cancellation.disconnect()
        context.fireErrorCaught(error)
    }
}
