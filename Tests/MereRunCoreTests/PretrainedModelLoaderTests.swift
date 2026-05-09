import Foundation
import Network
import XCTest
@testable import MereRunCore

final class PretrainedModelLoaderTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDown() {
        unsetenv(MereRunModelSourceConfiguration.baseURLEnvironmentKey)
        MereRunModelPaths.setProcessModelsDirOverride(nil)
        for root in temporaryRoots {
            try? FileManager.default.removeItem(at: root)
        }
        temporaryRoots.removeAll()
        super.tearDown()
    }

    func testStrictSizeMismatchDoesNotLeaveManagedSingleFile() async throws {
        let modelsDir = try makeModelsDirectory()
        let relativePath = "poc-bad.gguf"
        let managedFile = modelsDir.appendingPathComponent(relativePath, isDirectory: false)
        let server = try LoopbackHTTPServer(body: Data("bad".utf8))
        let baseURL = try await server.start()
        defer { server.stop() }
        setenv(MereRunModelSourceConfiguration.baseURLEnvironmentKey, baseURL.absoluteString, 1)

        try await assertSizeMismatch(
            relativePath: relativePath,
            expectedSize: 999,
            managedFile: managedFile
        )

        XCTAssertFalse(FileManager.default.fileExists(atPath: managedFile.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: managedFile.appendingPathExtension("partial").path))

        try await assertSizeMismatch(
            relativePath: relativePath,
            expectedSize: 999,
            managedFile: managedFile
        )
    }

    func testMatchingSingleFileDownloadIsPersistedAndReturned() async throws {
        let payload = Data("good".utf8)
        let modelsDir = try makeModelsDirectory()
        let relativePath = "poc-good.gguf"
        let managedFile = modelsDir.appendingPathComponent(relativePath, isDirectory: false)
        let server = try LoopbackHTTPServer(body: payload)
        let baseURL = try await server.start()
        defer { server.stop() }
        setenv(MereRunModelSourceConfiguration.baseURLEnvironmentKey, baseURL.absoluteString, 1)

        let downloaded = try await resolveSingleFile(
            relativePath: relativePath,
            expectedSize: Int64(payload.count)
        )

        XCTAssertEqual(downloaded.standardizedFileURL, managedFile.standardizedFileURL)
        XCTAssertEqual(try Data(contentsOf: managedFile), payload)

        unsetenv(MereRunModelSourceConfiguration.baseURLEnvironmentKey)
        let cached = try await resolveSingleFile(
            relativePath: relativePath,
            expectedSize: Int64(payload.count)
        )

        XCTAssertEqual(cached.standardizedFileURL, managedFile.standardizedFileURL)
    }

    func testArchiveDownloadRequiresConfiguredSHA256() async throws {
        _ = try makeModelsDirectory()
        setenv(MereRunModelSourceConfiguration.baseURLEnvironmentKey, "http://127.0.0.1:1/", 1)

        do {
            _ = try await PretrainedModelLoader.fromPretrainedArchive(
                modelPath: nil,
                modelId: "poc-archive",
                defaultModelIds: ["poc-archive"],
                storageId: "poc-archive",
                archiveKey: "models/poc-archive.tar.gz",
                archiveSize: 0,
                archiveSHA256: nil,
                hubFallback: nil,
                strictArchiveSize: false,
                validate: { root, fileManager in
                    let expected = root.appendingPathComponent("config.json")
                    return fileManager.fileExists(atPath: expected.path) ? [] : [expected]
                }
            )
            XCTFail("Expected missing archive SHA-256 to throw")
        } catch let error as PretrainedModelLoader.LoadError {
            XCTAssertTrue(error.localizedDescription.contains("Missing required SHA-256 digest"))
            XCTAssertTrue(error.localizedDescription.contains("models/poc-archive.tar.gz"))
        }
    }

    private func assertSizeMismatch(
        relativePath: String,
        expectedSize: Int64,
        managedFile: URL
    ) async throws {
        do {
            _ = try await resolveSingleFile(relativePath: relativePath, expectedSize: expectedSize)
            XCTFail("Expected strict size mismatch to throw")
        } catch let error as PretrainedModelLoader.LoadError {
            XCTAssertTrue(
                error.localizedDescription.contains("File size mismatch"),
                "Unexpected error: \(error.localizedDescription)"
            )
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: managedFile.path))
    }

    private func resolveSingleFile(
        relativePath: String,
        expectedSize: Int64
    ) async throws -> URL {
        try await PretrainedModelLoader.fromPretrainedFile(
            modelPath: nil,
            modelId: "poc-model",
            defaultModelIds: ["poc-model"],
            relativePath: relativePath,
            remoteKey: "models/poc.bin",
            expectedSize: expectedSize,
            expectedSHA256: nil,
            hubFallback: nil,
            strictSizeCheck: true,
            validate: { url, fileManager in
                fileManager.fileExists(atPath: url.path) ? [] : [url]
            }
        )
    }

    private func makeModelsDirectory() throws -> URL {
        let root = try TestFileSystem.makeTempDir(prefix: "mererun-pretrained-loader-tests")
        temporaryRoots.append(root)
        let modelsDir = root.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsDir)
        return modelsDir
    }
}

private final class LoopbackHTTPServer: @unchecked Sendable {
    private enum ServerError: Error {
        case missingPort
    }

    private let body: Data
    private let listener: NWListener
    private let queue = DispatchQueue(label: "mere.run.tests.loopback-http-server")
    private let lock = NSLock()
    private var startContinuation: CheckedContinuation<URL, Error>?
    private var didResume = false

    init(body: Data) throws {
        self.body = body
        self.listener = try NWListener(using: .tcp, on: .any)
    }

    func start() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            startContinuation = continuation
            lock.unlock()

            listener.stateUpdateHandler = { [weak self] state in
                self?.handleState(state)
            }
            listener.newConnectionHandler = { [weak self] connection in
                self?.handleConnection(connection)
            }
            listener.start(queue: queue)
        }
    }

    func stop() {
        listener.cancel()
    }

    private func handleState(_ state: NWListener.State) {
        switch state {
        case .ready:
            guard let port = listener.port else {
                resume(.failure(ServerError.missingPort))
                return
            }
            resume(.success(URL(string: "http://127.0.0.1:\(port.rawValue)/")!))
        case .failed(let error):
            resume(.failure(error))
        default:
            break
        }
    }

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [body] _, _, _, _ in
            let headers = "HTTP/1.1 200 OK\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
            var response = Data(headers.utf8)
            response.append(body)
            connection.send(content: response, completion: .contentProcessed { _ in
                connection.cancel()
            })
        }
    }

    private func resume(_ result: Result<URL, Error>) {
        lock.lock()
        guard !didResume, let continuation = startContinuation else {
            lock.unlock()
            return
        }
        didResume = true
        startContinuation = nil
        lock.unlock()

        switch result {
        case .success(let url):
            continuation.resume(returning: url)
        case .failure(let error):
            continuation.resume(throwing: error)
        }
    }
}
