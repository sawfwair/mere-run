import AudioCore
import Foundation
import Hummingbird
import NIOCore
import XCTest
@testable import MereRunCLI
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

final class APIRequestRecoveryTests: XCTestCase {
    func testDisconnectCancelsActiveWorkAndTheNextRequestSucceeds() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            let client = try APIRecoverySocket(port: port)
            try client.send("hold")
            let started = await waitUntil { probe.started.contains("hold") }
            XCTAssertTrue(started)
            client.close()
            let released = await waitUntil { await admission.snapshot().activeRequests == 0 }
            XCTAssertTrue(released, "The disconnected request still owns admission")
            XCTAssertTrue(probe.cancelled.contains("hold"))
            XCTAssertTrue(probe.cleaned.contains("hold"))
            guard released else { return }

            let retry = try APIRecoverySocket(port: port)
            try retry.send("retry")
            XCTAssertEqual(try retry.responseBody(), "ok:retry")
            let snapshot = await admission.snapshot()
            XCTAssertEqual(snapshot.activeRequests, 0)
            XCTAssertEqual(snapshot.totalCancelledRequests, 1)
            XCTAssertEqual(snapshot.totalCompletedRequests, 1)
        }
    }

    func testDisconnectedQueuedRequestNeverStartsAfterTheActiveRequestStops() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            let active = try APIRecoverySocket(port: port)
            try active.send("hold")
            let started = await waitUntil { probe.started.contains("hold") }
            XCTAssertTrue(started)
            let queued = try APIRecoverySocket(port: port)
            try queued.send("queued")
            let waiting = await waitUntil { await admission.snapshot().queuedRequests == 1 }
            XCTAssertTrue(waiting)
            queued.close()
            let removed = await waitUntil { await admission.snapshot().queuedRequests == 0 }
            XCTAssertTrue(removed, "The disconnected request remains queued")
            XCTAssertFalse(probe.started.contains("queued"))
            active.close()
            let released = await waitUntil { await admission.snapshot().activeRequests == 0 }
            XCTAssertTrue(released)
            XCTAssertFalse(probe.started.contains("queued"))
            guard removed && released else { return }

            let retry = try APIRecoverySocket(port: port)
            try retry.send("retry")
            XCTAssertEqual(try retry.responseBody(), "ok:retry")
            let snapshot = await admission.snapshot()
            XCTAssertEqual(snapshot.totalAdmittedRequests, 2)
        }
    }

    func testSuccessfulRequestsPreserveHTTPKeepAlive() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            let client = try APIRecoverySocket(port: port)
            for name in ["first", "second", "third"] {
                try client.send(name)
                XCTAssertEqual(try client.responseBody(), "ok:\(name)")
            }
            let snapshot = await admission.snapshot()
            XCTAssertEqual(snapshot.totalCompletedRequests, 3)
            XCTAssertEqual(snapshot.totalCancelledRequests, 0)
        }
    }

    func testAdmissionStaysHeldUntilCancelledCleanupFinishes() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            defer { probe.releaseCleanup() }
            let active = try APIRecoverySocket(port: port)
            try active.send("hold-cleanup")
            let started = await waitUntil { probe.started.contains("hold-cleanup") }
            XCTAssertTrue(started)
            active.close()
            let cleaning = await waitUntil { probe.cancelled.contains("hold-cleanup") }
            XCTAssertTrue(cleaning)
            let retry = try APIRecoverySocket(port: port)
            try retry.send("retry")
            let waiting = await waitUntil { await admission.snapshot().queuedRequests == 1 }
            XCTAssertTrue(waiting)
            let duringCleanup = await admission.snapshot()
            XCTAssertEqual(duringCleanup.activeRequests, 1)
            XCTAssertFalse(probe.started.contains("retry"))
            probe.releaseCleanup()
            XCTAssertEqual(try retry.responseBody(), "ok:retry")
            XCTAssertTrue(probe.cleaned.contains("hold-cleanup"))
        }
    }

    func testFailedRequestReleasesAdmissionBeforeRetry() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            let client = try APIRecoverySocket(port: port)
            try client.send("fail")
            _ = try client.responseBody(status: 500)
            XCTAssertTrue(probe.cleaned.contains("fail"))
            let failed = await admission.snapshot()
            XCTAssertEqual(failed.activeRequests, 0)
            try client.send("retry")
            XCTAssertEqual(try client.responseBody(), "ok:retry")
        }
    }

    func testPipelinedRequestsKeepTheirBodiesAndResponseOrder() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let probe = APIRecoveryProbe()
        try await withServer(admission: admission, probe: probe) { port in
            let client = try APIRecoverySocket(port: port)
            try client.send("first")
            try client.send("second")
            XCTAssertEqual(try client.responseBody(), "ok:first")
            XCTAssertEqual(try client.responseBody(), "ok:second")
        }
    }

    /// Mirrors the speech synthesis route: the body is collected, admission is
    /// held, and the operation is awaited inline with no streaming body. The
    /// router's cancellation middleware must reach the executor on disconnect.
    func testDisconnectCancelsSpeechSynthesisAndReleasesAdmission() async throws {
        let admission = RuntimeRequestAdmission(maxActiveRequests: 1)
        let executor = APIRecoverySpeechExecutor()
        let output = FileManager.default.temporaryDirectory
            .appendingPathComponent("api-recovery-speech-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("speech.wav")
        defer { try? FileManager.default.removeItem(at: output.deletingLastPathComponent()) }
        let plan = try SpeechSynthesisPlan(request: TTSRequest(text: "hold", outputURL: output))
        try await withServer(configure: { router in
            router.post("/speech") { request, _ in
                _ = try await request.body.collect(upTo: 1_024)
                return try await withRuntimeRequestAdmission(using: admission) {
                    _ = try await SpeechSynthesisOperation.execute(plan, executor: executor)
                    return "ok"
                }
            }
        }) { port in
            let client = try APIRecoverySocket(port: port)
            try client.send("hold", path: "/speech")
            let started = await waitUntil { executor.started }
            XCTAssertTrue(started)
            client.close()
            let cancelled = await waitUntil { executor.cancelled }
            XCTAssertTrue(cancelled, "The client disconnect did not reach the speech executor")
            let released = await waitUntil { await admission.snapshot().activeRequests == 0 }
            XCTAssertTrue(released, "The disconnected speech request still owns admission")
            let snapshot = await admission.snapshot()
            XCTAssertEqual(snapshot.totalCancelledRequests, 1)
            XCTAssertEqual(snapshot.totalCompletedRequests, 0)
            XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        }
    }

    private func withServer(
        admission: RuntimeRequestAdmission,
        probe: APIRecoveryProbe,
        operation: (Int) async throws -> Void
    ) async throws {
        try await withServer(configure: { router in
            router.post("/work") { request, _ in
                let data = try await request.body.collect(upTo: 1_024)
                let name = String(buffer: data)
                return try await withRuntimeRequestAdmission(using: admission) {
                    probe.begin(name)
                    defer { probe.clean(name) }
                    do {
                        if name == "hold" || name == "queued" || name == "hold-cleanup" {
                            try await Task.sleep(for: .seconds(30))
                        }
                        if name == "fail" { throw HTTPError(.internalServerError) }
                        return "ok:\(name)"
                    } catch {
                        if error is CancellationError { probe.cancel(name) }
                        if name == "hold-cleanup" {
                            await Task.detached {
                                while !probe.cleanupReleased { try? await Task.sleep(for: .milliseconds(10)) }
                            }.value
                        }
                        throw error
                    }
                }
            }
        }, operation: operation)
    }

    private func withServer(
        configure: (Router<APIServerRequestContext>) -> Void,
        operation: (Int) async throws -> Void
    ) async throws {
        let router = Router(context: APIServerRequestContext.self)
        router.middlewares.add(APIRequestCancellationMiddleware())
        configure(router)
        let (ports, continuation) = AsyncStream<Int>.makeStream()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname("127.0.0.1", port: 0)),
            onServerRunning: { channel in
                if let port = channel.localAddress?.port { continuation.yield(port) }
                continuation.finish()
            }
        )
        let server = Task { try await app.runService() }
        var iterator = ports.makeAsyncIterator()
        let receivedPort = await iterator.next()
        let port = try XCTUnwrap(receivedPort)
        do {
            try await operation(port)
            server.cancel()
            _ = try? await server.value
        } catch {
            server.cancel()
            _ = try? await server.value
            throw error
        }
    }

    private func waitUntil(_ predicate: () async -> Bool) async -> Bool {
        for _ in 0..<100 {
            if await predicate() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return false
    }
}

private final class APIRecoveryProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var released = false
    var cleanupReleased: Bool { lock.withLock { released } }
    func releaseCleanup() { lock.withLock { released = true } }
    private var starts = Set<String>()
    private var cancellations = Set<String>()
    private var cleanups = Set<String>()
    var started: Set<String> { lock.withLock { starts } }
    var cancelled: Set<String> { lock.withLock { cancellations } }
    var cleaned: Set<String> { lock.withLock { cleanups } }
    func begin(_ name: String) { _ = lock.withLock { starts.insert(name) } }
    func cancel(_ name: String) { _ = lock.withLock { cancellations.insert(name) } }
    func clean(_ name: String) { _ = lock.withLock { cleanups.insert(name) } }
}

private final class APIRecoverySpeechExecutor: SpeechSynthesisExecutor, @unchecked Sendable {
    private let lock = NSLock()
    private var startedFlag = false
    private var cancelledFlag = false
    var started: Bool { lock.withLock { startedFlag } }
    var cancelled: Bool { lock.withLock { cancelledFlag } }

    func generate(
        _ request: TTSRequest,
        progressHandler: (@Sendable (TTSProgress) -> Void)?
    ) async throws -> AudioWaveform {
        lock.withLock { startedFlag = true }
        do {
            try await Task.sleep(for: .seconds(30))
        } catch {
            lock.withLock { cancelledFlag = true }
            throw error
        }
        return try AudioWaveform(interleaved: [0], channels: 1, sampleRate: 24_000)
    }
}

private final class APIRecoverySocket {
    private var descriptor: Int32
    private var buffered = Data()

    init(port: Int) throws {
        #if canImport(Darwin)
        descriptor = socket(AF_INET, SOCK_STREAM, 0)
        #else
        descriptor = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #endif
        guard descriptor >= 0 else { throw POSIXError(.EIO) }
        var timeout = timeval(tv_sec: 3, tv_usec: 0)
        setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        var address = sockaddr_in()
        #if canImport(Darwin)
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        #endif
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(port).bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard result == 0 else { close(); throw POSIXError(.ECONNREFUSED) }
    }

    deinit { close() }

    func send(_ name: String, path: String = "/work") throws {
        let data = Data("POST \(path) HTTP/1.1\r\nHost: localhost\r\nContent-Length: \(name.utf8.count)\r\n\r\n\(name)".utf8)
        let count = data.withUnsafeBytes { write(descriptor, $0.baseAddress, $0.count) }
        guard count == data.count else { throw POSIXError(.EIO) }
    }

    func responseBody(status: Int = 200) throws -> String {
        while true {
            if let separator = buffered.range(of: Data("\r\n\r\n".utf8)) {
                let header = String(decoding: buffered[..<separator.lowerBound], as: UTF8.self)
                let length = header.components(separatedBy: "\r\n").compactMap { line -> Int? in
                    guard line.lowercased().hasPrefix("content-length:") else { return nil }
                    return Int(line.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces))
                }.first
                guard header.hasPrefix("HTTP/1.1 \(status)"), let length else { throw POSIXError(.EPROTO) }
                let end = separator.upperBound + length
                if buffered.count >= end {
                    let body = String(decoding: buffered[separator.upperBound..<end], as: UTF8.self)
                    buffered.removeSubrange(..<end)
                    return body
                }
            }
            var bytes = [UInt8](repeating: 0, count: 4_096)
            let count = read(descriptor, &bytes, bytes.count)
            guard count > 0 else { throw POSIXError(.ECONNRESET) }
            buffered.append(contentsOf: bytes.prefix(count))
        }
    }

    func close() {
        guard descriptor >= 0 else { return }
        #if canImport(Darwin)
        _ = Darwin.shutdown(descriptor, SHUT_RDWR)
        _ = Darwin.close(descriptor)
        #else
        _ = Glibc.shutdown(descriptor, Int32(SHUT_RDWR))
        _ = Glibc.close(descriptor)
        #endif
        descriptor = -1
    }
}
