import Foundation

#if !canImport(FoundationNetworking)
/// Owns the single background URL session used for long-running Hub payloads.
///
/// A completed payload is moved out of URLSession's temporary directory before
/// the delegate returns. The regular `HubSnapshot` pipeline subsequently
/// validates its size and pinned Hub identity before the model is installed.
public final class HubBackgroundTransferSession: NSObject, @unchecked Sendable {
    package struct Descriptor: Codable, Equatable, Sendable {
        package let id: String
        package let stagingPath: String
        package let applicationSupportRelativePath: String?

        package init(id: String, stagingURL: URL) {
            self.id = id
            stagingPath = stagingURL.standardizedFileURL.path
            guard let applicationSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first else {
                applicationSupportRelativePath = nil
                return
            }
            let rootPath = applicationSupport.standardizedFileURL.path
            let prefix = rootPath + "/"
            if stagingPath.hasPrefix(prefix) {
                applicationSupportRelativePath = String(stagingPath.dropFirst(prefix.count))
            } else {
                applicationSupportRelativePath = nil
            }
        }

        package var stagingURL: URL {
            if let applicationSupportRelativePath,
               let applicationSupport = FileManager.default.urls(
                   for: .applicationSupportDirectory,
                   in: .userDomainMask
               ).first {
                return applicationSupport.appendingPathComponent(
                    applicationSupportRelativePath
                )
            }
            return URL(fileURLWithPath: stagingPath)
        }
    }

    private struct CompletionRecord: Codable, Sendable {
        let statusCode: Int
        let responseURL: String?
    }

    private struct PendingTransfer {
        let continuation: CheckedContinuation<URL, Error>
        let progress: @Sendable (Int64, Int64) -> Void
    }

    public static let shared = HubBackgroundTransferSession()

    public static var sessionIdentifier: String {
        let bundleID = Bundle.main.bundleIdentifier ?? "run.mere"
        return "\(bundleID).hub-model-downloads"
    }

    private let lock = NSLock()
    private var pending: [String: [PendingTransfer]] = [:]
    private var taskResults: [Int: Result<URL, Error>] = [:]
    private var unclaimedResults: [String: Result<URL, Error>] = [:]
    private var backgroundCompletionHandler: (() -> Void)?

    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.background(
            withIdentifier: Self.sessionIdentifier
        )
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        configuration.waitsForConnectivity = true
        configuration.allowsCellularAccess = true
        configuration.allowsExpensiveNetworkAccess = true
        configuration.allowsConstrainedNetworkAccess = true
        configuration.timeoutIntervalForResource = 7 * 24 * 60 * 60
        return URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }()

    private override init() {
        super.init()
    }

    /// Reconnects the system-owned session after iOS relaunches the app for
    /// completed background events. Returns false for an unrelated session.
    @discardableResult
    public static func handleEvents(
        for identifier: String,
        completionHandler: @escaping () -> Void
    ) -> Bool {
        guard identifier == sessionIdentifier else { return false }
        shared.setBackgroundCompletionHandler(completionHandler)
        shared.reconnect()
        return true
    }

    /// Recreates the session on an ordinary launch so in-flight tasks can
    /// reconnect even though UIKit only calls the app delegate after completion.
    public func reconnect() {
        _ = session
    }

    /// Downloads or rejoins one durable transfer. Cancellation of the caller
    /// intentionally does not cancel the system task; a later install attempt
    /// can rejoin it by transfer ID.
    public func download(
        request: URLRequest,
        transferID: String,
        stagingDirectory: URL,
        progress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> URL {
        try FileManager.default.createDirectory(
            at: stagingDirectory,
            withIntermediateDirectories: true
        )
        let stagingURL = stagingDirectory.appendingPathComponent("\(transferID).download")
        if Self.hasCompletedTransfer(at: stagingURL) {
            discardUnclaimedResult(for: transferID)
            return stagingURL
        }
        Self.removeTransferFiles(at: stagingURL)

        let description = try Self.taskDescription(
            for: Descriptor(id: transferID, stagingURL: stagingURL)
        )
        let existingTask = await session.allTasks.first { task in
            Self.descriptor(from: task.taskDescription)?.id == transferID
        }
        if Self.hasCompletedTransfer(at: stagingURL) {
            discardUnclaimedResult(for: transferID)
            return stagingURL
        }

        return try await withCheckedThrowingContinuation { continuation in
            let pendingTransfer = PendingTransfer(
                continuation: continuation,
                progress: progress
            )
            lock.lock()
            if let completedResult = unclaimedResults.removeValue(forKey: transferID) {
                lock.unlock()
                continuation.resume(with: completedResult)
                return
            }
            pending[transferID, default: []].append(pendingTransfer)
            lock.unlock()

            if let existingTask {
                progress(
                    max(existingTask.countOfBytesReceived, 0),
                    max(existingTask.countOfBytesExpectedToReceive, 0)
                )
                existingTask.resume()
                return
            }

            let task = session.downloadTask(with: request)
            task.taskDescription = description
            task.resume()
        }
    }

    /// Removes the completion sidecar after HubSnapshot has consumed the file.
    public static func finishConsuming(_ stagingURL: URL) {
        try? FileManager.default.removeItem(at: completionRecordURL(for: stagingURL))
    }

    package static func taskDescription(for descriptor: Descriptor) throws -> String {
        try JSONEncoder().encode(descriptor).base64EncodedString()
    }

    package static func descriptor(from taskDescription: String?) -> Descriptor? {
        guard let taskDescription,
              let data = Data(base64Encoded: taskDescription) else {
            return nil
        }
        return try? JSONDecoder().decode(Descriptor.self, from: data)
    }

    package static func hasCompletedTransfer(at stagingURL: URL) -> Bool {
        guard FileManager.default.fileExists(atPath: stagingURL.path),
              let data = try? Data(contentsOf: completionRecordURL(for: stagingURL)),
              let record = try? JSONDecoder().decode(CompletionRecord.self, from: data) else {
            return false
        }
        return (200..<300).contains(record.statusCode)
    }

    private static func completionRecordURL(for stagingURL: URL) -> URL {
        stagingURL.appendingPathExtension("json")
    }

    private static func removeTransferFiles(at stagingURL: URL) {
        try? FileManager.default.removeItem(at: stagingURL)
        try? FileManager.default.removeItem(at: completionRecordURL(for: stagingURL))
    }

    private func setBackgroundCompletionHandler(_ completionHandler: @escaping () -> Void) {
        lock.lock()
        backgroundCompletionHandler = completionHandler
        lock.unlock()
    }

    private func discardUnclaimedResult(for transferID: String) {
        lock.lock()
        unclaimedResults[transferID] = nil
        lock.unlock()
    }

    private func storeResult(_ result: Result<URL, Error>, taskIdentifier: Int) {
        lock.lock()
        taskResults[taskIdentifier] = result
        lock.unlock()
    }

    private func complete(task: URLSessionTask, error: Error?) {
        guard let descriptor = Self.descriptor(from: task.taskDescription) else { return }

        lock.lock()
        let transfers = pending.removeValue(forKey: descriptor.id) ?? []
        let storedResult = taskResults.removeValue(forKey: task.taskIdentifier)
        lock.unlock()

        let result: Result<URL, Error>
        if let error {
            Self.removeTransferFiles(at: descriptor.stagingURL)
            result = .failure(error)
        } else if let storedResult {
            result = storedResult
        } else if Self.hasCompletedTransfer(at: descriptor.stagingURL) {
            result = .success(descriptor.stagingURL)
        } else {
            result = .failure(HubBackgroundTransferError.missingPayload)
        }
        if transfers.isEmpty {
            if case .failure = result {
                lock.lock()
                unclaimedResults[descriptor.id] = result
                lock.unlock()
            }
            return
        }
        for transfer in transfers {
            transfer.continuation.resume(with: result)
        }
    }
}

extension HubBackgroundTransferSession: URLSessionDownloadDelegate {
    public func urlSession(
        _: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection _: HTTPURLResponse,
        newRequest: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        var redirectedRequest = newRequest
        if task.currentRequest?.url?.host != newRequest.url?.host {
            redirectedRequest.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(redirectedRequest)
    }

    public func urlSession(
        _: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let descriptor = Self.descriptor(from: downloadTask.taskDescription),
              let response = downloadTask.response as? HTTPURLResponse else {
            storeResult(
                .failure(HubBackgroundTransferError.missingResponse),
                taskIdentifier: downloadTask.taskIdentifier
            )
            return
        }
        guard (200..<300).contains(response.statusCode) else {
            Self.removeTransferFiles(at: descriptor.stagingURL)
            storeResult(
                .failure(HubBackgroundTransferError.httpStatus(response.statusCode)),
                taskIdentifier: downloadTask.taskIdentifier
            )
            return
        }

        do {
            try FileManager.default.createDirectory(
                at: descriptor.stagingURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            Self.removeTransferFiles(at: descriptor.stagingURL)
            try FileManager.default.moveItem(at: location, to: descriptor.stagingURL)
            let record = CompletionRecord(
                statusCode: response.statusCode,
                responseURL: response.url?.absoluteString
            )
            let data = try JSONEncoder().encode(record)
            try data.write(
                to: Self.completionRecordURL(for: descriptor.stagingURL),
                options: .atomic
            )
            storeResult(
                .success(descriptor.stagingURL),
                taskIdentifier: downloadTask.taskIdentifier
            )
        } catch {
            Self.removeTransferFiles(at: descriptor.stagingURL)
            storeResult(.failure(error), taskIdentifier: downloadTask.taskIdentifier)
        }
    }

    public func urlSession(
        _: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        complete(task: task, error: error)
    }

    public func urlSession(
        _: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData _: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let descriptor = Self.descriptor(from: downloadTask.taskDescription) else { return }
        lock.lock()
        let transfers = pending[descriptor.id] ?? []
        lock.unlock()
        for transfer in transfers {
            transfer.progress(totalBytesWritten, totalBytesExpectedToWrite)
        }
    }

    public func urlSessionDidFinishEvents(forBackgroundURLSession _: URLSession) {
        lock.lock()
        let completionHandler = backgroundCompletionHandler
        backgroundCompletionHandler = nil
        lock.unlock()
        completionHandler?()
    }
}

private enum HubBackgroundTransferError: LocalizedError {
    case httpStatus(Int)
    case missingPayload
    case missingResponse

    var errorDescription: String? {
        switch self {
        case .httpStatus(let status):
            return "Background model download failed with HTTP \(status)."
        case .missingPayload:
            return "Background model download completed without a payload."
        case .missingResponse:
            return "Background model download completed without an HTTP response."
        }
    }
}
#endif
