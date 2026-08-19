import Foundation
import os
import XCTest

/// Diagnostic: exercises the hub download stack inside the app process on a
/// physical device, in three isolation layers, to pinpoint where on-device
/// model downloads stall. Gated on MERERUN_TEST_DOWNLOAD_DIAG.
final class HubDownloadDeviceTests: XCTestCase {
    private let smallFileURL = URL(
        string: "https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit/resolve/main/manifest.json"
    )!

    private func requireDiag() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_TEST_DOWNLOAD_DIAG"] != nil else {
            throw XCTSkip("Set MERERUN_TEST_DOWNLOAD_DIAG to run download diagnostics.")
        }
    }

    func testV1PlainSharedSessionFetch() async throws {
        try requireDiag()
        let (data, response) = try await URLSession.shared.data(from: smallFileURL)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "Plain fetch failed: \(http.statusCode)")
        XCTAssertGreaterThan(data.count, 10)
    }

    func testV2DownloadWithNoRedirectDelegate() async throws {
        try requireDiag()
        // Mirrors HubSnapshot's exact transfer shape: a fresh session whose
        // delegate refuses redirects, driven through async download(for:).
        final class NoRedirect: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
            var wroteBytes = false
            func urlSession(_: URLSession, task _: URLSessionTask, willPerformHTTPRedirection _: HTTPURLResponse, newRequest _: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
                completionHandler(nil)
            }
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo _: URL) {}
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten _: Int64, totalBytesExpectedToWrite _: Int64) {
                wroteBytes = true
            }
        }
        var currentURL = smallFileURL
        for _ in 0..<8 {
            let delegate = NoRedirect()
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            defer { session.invalidateAndCancel() }
            var request = URLRequest(url: currentURL)
            request.httpMethod = "GET"
            let (tempURL, response) = try await session.download(for: request)
            let http = try XCTUnwrap(response as? HTTPURLResponse)
            if (200..<300).contains(http.statusCode) {
                let size = (try? FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int) ?? 0
                XCTAssertGreaterThan(size, 10, "Downloaded file is empty")
                return
            }
            guard (300..<400).contains(http.statusCode),
                  let location = http.value(forHTTPHeaderField: "Location"),
                  let next = URL(string: location, relativeTo: currentURL) else {
                return XCTFail("Unexpected status \(http.statusCode) with no redirect")
            }
            currentURL = next.absoluteURL
        }
        XCTFail("Too many redirects")
    }


    func testV4HeadProbeLikeResolveRemoteFile() async throws {
        try requireDiag()
        // Mirrors HubSnapshot.resolveRemoteFile: HEAD, identity encoding,
        // redirect-refusing session delegate, async data(for:).
        final class NoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
            func urlSession(_: URLSession, task _: URLSessionTask, willPerformHTTPRedirection _: HTTPURLResponse, newRequest _: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
                completionHandler(nil)
            }
        }
        var request = URLRequest(url: smallFileURL)
        request.httpMethod = "HEAD"
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let delegate = NoRedirect()
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let (_, response) = try await session.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertTrue((200..<400).contains(http.statusCode), "HEAD returned \(http.statusCode)")
        let report = XCTAttachment(string: "HEAD status \(http.statusCode), etag: \(http.value(forHTTPHeaderField: "ETag") ?? "-"), location: \(http.value(forHTTPHeaderField: "Location") ?? "-")")
        report.name = "v4-head"
        report.lifetime = .keepAlways
        add(report)
    }

    func testV5LFSRedirectDownloadProgress() async throws {
        try requireDiag()
        // The LFS path V2 never exercised: refuse the 302 at the session
        // delegate, follow Location manually, then measure whether bytes
        // actually flow from the CDN for a large file.
        final class Probe: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
            private let counter = OSAllocatedUnfairLock(initialState: Int64(0))
            var written: Int64 { counter.withLock { $0 } }
            func urlSession(_: URLSession, task _: URLSessionTask, willPerformHTTPRedirection _: HTTPURLResponse, newRequest _: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
                completionHandler(nil)
            }
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo _: URL) {}
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite _: Int64) {
                counter.withLock { $0 = totalBytesWritten }
            }
        }
        let bigURL = URL(string: "https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit/resolve/main/transformer-packed-mflux/diffusion_pytorch_model.safetensors")!

        // Step 1: capture the redirect the way HubSnapshot does.
        let probe1 = Probe()
        let session1 = URLSession(configuration: .default, delegate: probe1, delegateQueue: nil)
        var request = URLRequest(url: bigURL)
        request.httpMethod = "GET"
        let (temp1, response1) = try await session1.download(for: request)
        try? FileManager.default.removeItem(at: temp1)
        let http1 = try XCTUnwrap(response1 as? HTTPURLResponse)
        var notes = ["step1 status \(http1.statusCode)"]
        guard (300..<400).contains(http1.statusCode),
              let location = http1.value(forHTTPHeaderField: "Location") else {
            notes.append("no redirect; direct status \(http1.statusCode)")
            attach(notes, name: "v5-progress")
            session1.invalidateAndCancel()
            return
        }
        session1.invalidateAndCancel()
        let resolved = URL(string: location, relativeTo: bigURL)?.absoluteURL
        notes.append("location parsed: \(resolved != nil), host: \(resolved?.host ?? "NIL")")
        guard let cdnURL = resolved else {
            attach(notes, name: "v5-progress")
            return XCTFail("Location did not parse into a URL")
        }

        // Step 2: start the CDN download and sample byte progress for 30s.
        let probe2 = Probe()
        let session2 = URLSession(configuration: .default, delegate: probe2, delegateQueue: nil)
        var cdnRequest = URLRequest(url: cdnURL)
        cdnRequest.httpMethod = "GET"
        let task = Task { try await session2.download(for: cdnRequest) }
        for second in stride(from: 5, through: 30, by: 5) {
            try await Task.sleep(nanoseconds: 5_000_000_000)
            notes.append("t+\(second)s: \(probe2.written / 1_048_576) MB")
        }
        task.cancel()
        session2.invalidateAndCancel()
        let finalBytes = probe2.written
        attach(notes, name: "v5-progress")
        XCTAssertGreaterThan(finalBytes, 1_048_576, "CDN transfer moved fewer than 1 MB in 30s: \(notes)")
    }

    private func attach(_ lines: [String], name: String) {
        let attachment = XCTAttachment(string: lines.joined(separator: "\n"))
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testV5bCDNRequestOutcome() async throws {
        try requireDiag()
        // Follow the 302 manually like HubSnapshot, then AWAIT the CDN
        // response so its status code or transport error is captured.
        let bigURL = URL(string: "https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit/resolve/main/transformer-packed-mflux/diffusion_pytorch_model.safetensors")!
        final class NoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
            func urlSession(_: URLSession, task _: URLSessionTask, willPerformHTTPRedirection _: HTTPURLResponse, newRequest _: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
                completionHandler(nil)
            }
        }
        let session1 = URLSession(configuration: .default, delegate: NoRedirect(), delegateQueue: nil)
        defer { session1.invalidateAndCancel() }
        var request = URLRequest(url: bigURL)
        request.httpMethod = "GET"
        let (_, response1) = try await session1.data(for: request)
        let http1 = try XCTUnwrap(response1 as? HTTPURLResponse)
        let location = try XCTUnwrap(http1.value(forHTTPHeaderField: "Location"))
        let cdnURL = try XCTUnwrap(URL(string: location, relativeTo: bigURL)?.absoluteURL)

        var notes = ["cdn url prefix: \(cdnURL.absoluteString.prefix(120))"]
        var cdnRequest = URLRequest(url: cdnURL, timeoutInterval: 25)
        cdnRequest.httpMethod = "GET"
        cdnRequest.setValue("bytes=0-1048575", forHTTPHeaderField: "Range")
        do {
            let (data, response2) = try await URLSession.shared.data(for: cdnRequest)
            let http2 = try XCTUnwrap(response2 as? HTTPURLResponse)
            notes.append("cdn status \(http2.statusCode), bytes \(data.count)")
            if !(200..<300).contains(http2.statusCode) {
                notes.append("body: \(String(decoding: data.prefix(300), as: UTF8.self))")
            }
        } catch {
            notes.append("cdn error: \(error)")
        }
        attach(notes, name: "v5b-outcome")
    }

    func testV6SystemRedirectDownloadProgress() async throws {
        try requireDiag()
        // Let URLSession follow the redirect itself; sample byte progress.
        final class Progress: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
            private let counter = OSAllocatedUnfairLock(initialState: Int64(0))
            var written: Int64 { counter.withLock { $0 } }
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo _: URL) {}
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite _: Int64) {
                counter.withLock { $0 = totalBytesWritten }
            }
        }
        let bigURL = URL(string: "https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit/resolve/main/transformer-packed-mflux/diffusion_pytorch_model.safetensors")!
        let probe = Progress()
        let session = URLSession(configuration: .default, delegate: probe, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let task = Task { try await session.download(for: URLRequest(url: bigURL)) }
        var notes: [String] = []
        for second in stride(from: 5, through: 25, by: 5) {
            try await Task.sleep(nanoseconds: 5_000_000_000)
            notes.append("t+\(second)s: \(probe.written / 1_048_576) MB")
        }
        task.cancel()
        attach(notes, name: "v6-progress")
        XCTAssertGreaterThan(probe.written, 1_048_576, "System-redirect transfer also moved nothing: \(notes)")
    }

    func testV7PerTaskDelegateDeliversProgress() async throws {
        try requireDiag()
        final class Progress: NSObject, URLSessionTaskDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
            private let counter = OSAllocatedUnfairLock(initialState: (bytes: Int64(0), events: 0))
            var state: (bytes: Int64, events: Int) { counter.withLock { $0 } }
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo _: URL) {}
            func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite _: Int64) {
                counter.withLock { $0 = (totalBytesWritten, $0.events + 1) }
            }
        }
        let mediumURL = URL(string: "https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit/resolve/main/tokenizer/tokenizer.json")!

        // A: delegate attached to the session (HubSnapshot's current shape).
        let sessionDelegate = Progress()
        let sessionA = URLSession(configuration: .default, delegate: sessionDelegate, delegateQueue: nil)
        let (tempA, _) = try await sessionA.download(for: URLRequest(url: mediumURL))
        let sizeA = (try? FileManager.default.attributesOfItem(atPath: tempA.path)[.size] as? Int64) ?? 0
        try? FileManager.default.removeItem(at: tempA)
        sessionA.invalidateAndCancel()

        // B: delegate passed per-task to the async API.
        let taskDelegate = Progress()
        let sessionB = URLSession(configuration: .default)
        let (tempB, _) = try await sessionB.download(for: URLRequest(url: mediumURL), delegate: taskDelegate)
        let sizeB = (try? FileManager.default.attributesOfItem(atPath: tempB.path)[.size] as? Int64) ?? 0
        try? FileManager.default.removeItem(at: tempB)
        sessionB.invalidateAndCancel()

        let a = sessionDelegate.state
        let b = taskDelegate.state
        attach([
            "A(session delegate): file \(sizeA) bytes, didWriteData events \(a.events), reported \(a.bytes)",
            "B(per-task delegate): file \(sizeB) bytes, didWriteData events \(b.events), reported \(b.bytes)",
        ], name: "v7-delegates")
        XCTAssertGreaterThan(sizeA, 1000, "Session-delegate download produced no file")
        XCTAssertGreaterThan(sizeB, 1000, "Per-task-delegate download produced no file")
        XCTAssertGreaterThan(b.events, 0, "Per-task delegate never saw progress")
    }
}
