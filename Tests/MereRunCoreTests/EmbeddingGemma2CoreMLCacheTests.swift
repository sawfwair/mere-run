#if canImport(CoreML) && os(macOS) && arch(arm64)
import Foundation
import XCTest
@testable import MereRunCoreMLModel

final class EmbeddingGemma2CoreMLCacheTests: XCTestCase {
    private actor Compiler {
        var calls = 0
        let root: URL
        init(root: URL) { self.root = root }
        func compile(_ package: URL) async throws -> URL {
            calls += 1
            for _ in 0..<10 { await Task.yield() }
            let output = root.appending(path: UUID().uuidString + ".mlmodelc")
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
            try Data("compiled".utf8).write(to: output.appending(path: "model"))
            return output
        }
    }

    private func fixture() throws -> (URL, URL) {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let package = root.appending(path: "source.mlpackage")
        try FileManager.default.createDirectory(at: package.appending(path: "Data/com.apple.CoreML/weights"), withIntermediateDirectories: true)
        for name in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"] {
            try Data(name.utf8).write(to: package.appending(path: name))
        }
        return (root, package)
    }

    func testConcurrentCompilationPersistsAcrossStoreLifetimesAndInvalidatesOnReplacement() async throws {
        let (root, package) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let compiler = Compiler(root: root)
        let directory = root.appending(path: "cache")
        let store = CompiledModelStore(directory: directory, compile: { try await compiler.compile($0) })
        async let first = store.compiled(package: package)
        async let second = store.compiled(package: package)
        let urls = try await [first, second]
        XCTAssertEqual(urls[0], urls[1])
        let count = await compiler.calls
        XCTAssertEqual(count, 1)
        let reopened = CompiledModelStore(directory: directory, compile: { try await compiler.compile($0) })
        let cached = try await reopened.compiled(package: package)
        XCTAssertEqual(cached, urls[0])
        let alias = root.appending(path: "alias.mlpackage")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: package)
        XCTAssertEqual(try CompiledModelStore.fingerprint(package: alias), try CompiledModelStore.fingerprint(package: package))
        try Data("changed graph".utf8).write(to: package.appending(path: "Data/com.apple.CoreML/model.mlmodel"))
        let changed = try await reopened.compiled(package: package)
        XCTAssertNotEqual(changed, cached)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cached.path))
        let beforeWeights = try CompiledModelStore.fingerprint(package: package)
        let weights = package.appending(path: "Data/com.apple.CoreML/weights/weight.bin")
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSinceNow: 10)], ofItemAtPath: weights.path)
        XCTAssertNotEqual(try CompiledModelStore.fingerprint(package: package), beforeWeights)
        let finalCount = await compiler.calls
        XCTAssertEqual(finalCount, 2)
    }

    func testIndependentWritersPublishOnlyCompleteDirectories() async throws {
        let (root, package) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let compiler = Compiler(root: root)
        let directory = root.appending(path: "cache")
        let first = CompiledModelStore(directory: directory, compile: { try await compiler.compile($0) })
        let second = CompiledModelStore(directory: directory, compile: { try await compiler.compile($0) })
        async let left = first.compiled(package: package)
        async let right = second.compiled(package: package)
        let urls = try await [left, right]
        XCTAssertEqual(urls[0], urls[1])
        XCTAssertEqual(try Data(contentsOf: urls[0].appending(path: "model")), Data("compiled".utf8))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path).count, 1)
    }

    func testFailedCompilationCanBeRetriedWithoutPublishing() async throws {
        let (root, package) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appending(path: "cache")
        let failure = CompiledModelStore(directory: directory, compile: { _ in throw EmbedderError(description: "fixture failure") })
        for _ in 0..<2 {
            do { _ = try await failure.compiled(package: package); XCTFail("Expected compilation failure") }
            catch { XCTAssertEqual(error.localizedDescription, "fixture failure") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }
}
#endif
