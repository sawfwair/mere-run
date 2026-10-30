import Foundation
import XCTest

#if os(macOS)
import CryptoKit
@testable import MereRunCLI

final class MLXBundleSupportTests: XCTestCase {
  private var repositoryRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }

  func testVendoredMetallibMatchesCompiledProvenance() throws {
    let sidecarURL =
      repositoryRoot
      .appendingPathComponent("vendor/mlx-swift_Cmlx.bundle/Contents/Resources")
      .appendingPathComponent("default.metallib.version")
    let contents = try String(contentsOf: sidecarURL, encoding: .utf8)

    XCTAssertEqual(MLXBundleSupport.stampStatus(contents: contents), .matched)
    XCTAssertEqual(MLXBundleSupport.stampStatus(bundleURL: sidecarURL.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()), .matched)
  }

  func testCopiedLegacyStampRejectsAReplacementBinary() throws {
    let bundle = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: bundle) }
    let resources = bundle.appendingPathComponent("Contents/Resources")
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    let originalStamp = repositoryRoot.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib.version")
    try FileManager.default.copyItem(at: originalStamp, to: resources.appendingPathComponent("default.metallib.version"))
    let replacement = Data("replacement kernels".utf8)
    try replacement.write(to: resources.appendingPathComponent("default.metallib"))
    let digest = SHA256.hash(data: replacement).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(MLXBundleSupport.stampStatus(bundleURL: bundle), .mismatched([
      .init(field: "metallib-sha256", stamped: digest, expected: MLXBundleSupport.legacyMetallibSHA256)
    ]))
  }

  func testBinaryBoundStampAcceptsRebuildAndRejectsSameSizeMutation() throws {
    let bundle = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: bundle) }
    let resources = bundle.appendingPathComponent("Contents/Resources")
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    let stampURL = repositoryRoot.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib.version")
    let bytes = Data("kernel A".utf8)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let stamp = try String(contentsOf: stampURL, encoding: .utf8) + "\nmetallib-sha256: \(digest)\n"
    try stamp.write(to: resources.appendingPathComponent("default.metallib.version"), atomically: true, encoding: .utf8)
    try bytes.write(to: resources.appendingPathComponent("default.metallib"))
    XCTAssertEqual(MLXBundleSupport.stampStatus(bundleURL: bundle), .matched)
    let changed = Data("kernel B".utf8)
    XCTAssertEqual(bytes.count, changed.count)
    try changed.write(to: resources.appendingPathComponent("default.metallib"))
    let actual = SHA256.hash(data: changed).map { String(format: "%02x", $0) }.joined()
    XCTAssertEqual(MLXBundleSupport.stampStatus(bundleURL: bundle), .mismatched([
      .init(field: "metallib-sha256", stamped: actual, expected: digest)
    ]))
  }

  func testCompatibilityCopyReplacesDifferentKernelsWithEqualFileSize() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let source = root.appendingPathComponent("source.metallib")
    let destination = root.appendingPathComponent("mlx.metallib")
    try Data("new kernels".utf8).write(to: source)
    try Data("old kernels".utf8).write(to: destination)
    try MLXBundleSupport.refreshCopy(from: source, to: destination)
    XCTAssertEqual(try Data(contentsOf: source), try Data(contentsOf: destination))
  }

  func testFlatResourcesRefreshTheColocatedLibraryThatMLXLoadsFirst() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let resources = root.appendingPathComponent("Resources")
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    let stampURL = repositoryRoot.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib.version")
    let bytes = Data("new kernels".utf8)
    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    let stamp = try String(contentsOf: stampURL, encoding: .utf8) + "\nmetallib-sha256: \(digest)\n"
    try stamp.write(to: resources.appendingPathComponent("default.metallib.version"), atomically: true, encoding: .utf8)
    try bytes.write(to: resources.appendingPathComponent("default.metallib"))
    try Data("old kernels".utf8).write(to: root.appendingPathComponent("mlx.metallib"))
    try MLXBundleSupport.validateFlatResources(resourcesURL: resources, executableDir: root, quiet: true)
    XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("mlx.metallib")), bytes)
    XCTAssertEqual(try Data(contentsOf: resources.appendingPathComponent("mlx.metallib")), bytes)
  }

  func testCompiledSwiftRevisionMatchesPackagePin() throws {
    let packageURL = repositoryRoot.appendingPathComponent("Package.swift")
    let package = try String(contentsOf: packageURL, encoding: .utf8)

    XCTAssertTrue(
      package.contains(#"revision: "\#(MLXBundleSupport.expectedProvenance.swiftRevision)""#)
    )
  }

  func testVendoredMetallibBinaryTargetsMacOS15() throws {
    let library = repositoryRoot.appendingPathComponent(
      "vendor/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
    )
    let process = Process()
    let output = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    process.arguments = ["metal-objdump", "--metallib", "--private-headers", library.path]
    process.standardOutput = output
    process.standardError = output
    try process.run()
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let header = String(decoding: data, as: UTF8.self)

    XCTAssertEqual(process.terminationStatus, 0, header)
    XCTAssertTrue(header.contains("Platform: MACOS\n"), header)
    XCTAssertTrue(header.contains("PlatformMajor: 15\n"), header)
    // The linker header represents minor/patch zero as NUL characters.
    let normalized = header.replacingOccurrences(of: "\u{0000}", with: "0")
    XCTAssertTrue(normalized.contains("PlatformMinor: 0\n"), normalized)
    XCTAssertTrue(normalized.contains("PlatformUpdate: 0\n"), normalized)
  }

  func testMissingRevisionOrKernelHashIsUnstamped() {
    let status = MLXBundleSupport.stampStatus(
      contents: "mlx-core-version: \(MLXBundleSupport.expectedProvenance.coreVersion)\n"
    )

    XCTAssertEqual(
      status,
      .unstamped(missingFields: [
        "mlx-core-revision",
        "mlx-upstream-tag",
        "mlx-upstream-revision",
        "mlx-swift-revision",
        "kernel-sources-sha256",
        "metal-language-version",
        "macos-deployment-target",
      ])
    )
  }

  func testRevisionMismatchIsRejected() {
    let expected = MLXBundleSupport.expectedProvenance
    let status = MLXBundleSupport.stampStatus(
      contents: """
        mlx-core-version: \(expected.coreVersion)
        mlx-core-revision: \(expected.coreRevision)
        mlx-upstream-tag: \(expected.upstreamTag)
        mlx-upstream-revision: \(expected.upstreamRevision)
        mlx-swift-revision: \(String(repeating: "0", count: 40))
        kernel-sources-sha256: \(expected.kernelSourcesSHA256)
        metal-language-version: \(expected.metalLanguageVersion)
        macos-deployment-target: \(expected.macOSDeploymentTarget)
        """
    )

    XCTAssertEqual(
      status,
      .mismatched([
        .init(
          field: "mlx-swift-revision",
          stamped: String(repeating: "0", count: 40),
          expected: expected.swiftRevision
        )
      ])
    )
  }

  func testCoreAndUpstreamRevisionMismatchesAreRejected() {
    let expected = MLXBundleSupport.expectedProvenance
    let wrongCore = String(repeating: "1", count: 40)
    let wrongUpstream = String(repeating: "2", count: 40)
    let status = MLXBundleSupport.stampStatus(
      contents: """
        mlx-core-version: \(expected.coreVersion)
        mlx-core-revision: \(wrongCore)
        mlx-upstream-tag: \(expected.upstreamTag)
        mlx-upstream-revision: \(wrongUpstream)
        mlx-swift-revision: \(expected.swiftRevision)
        kernel-sources-sha256: \(expected.kernelSourcesSHA256)
        metal-language-version: \(expected.metalLanguageVersion)
        macos-deployment-target: \(expected.macOSDeploymentTarget)
        """
    )

    XCTAssertEqual(
      status,
      .mismatched([
        .init(field: "mlx-core-revision", stamped: wrongCore, expected: expected.coreRevision),
        .init(
          field: "mlx-upstream-revision",
          stamped: wrongUpstream,
          expected: expected.upstreamRevision
        ),
      ])
    )
  }

  func testKernelSourceHashMismatchIsRejected() {
    let expected = MLXBundleSupport.expectedProvenance
    let status = MLXBundleSupport.stampStatus(
      contents: """
        mlx-core-version: \(expected.coreVersion)
        mlx-core-revision: \(expected.coreRevision)
        mlx-upstream-tag: \(expected.upstreamTag)
        mlx-upstream-revision: \(expected.upstreamRevision)
        mlx-swift-revision: \(expected.swiftRevision)
        kernel-sources-sha256: \(String(repeating: "0", count: 64))
        metal-language-version: \(expected.metalLanguageVersion)
        macos-deployment-target: \(expected.macOSDeploymentTarget)
        """
    )

    XCTAssertEqual(
      status,
      .mismatched([
        .init(
          field: "kernel-sources-sha256",
          stamped: String(repeating: "0", count: 64),
          expected: expected.kernelSourcesSHA256
        )
      ])
    )
  }

  func testHostSDKMetallibIsRejectedDespiteMatchingKernelSources() throws {
    let resources = repositoryRoot.appendingPathComponent(
      "vendor/mlx-swift_Cmlx.bundle/Contents/Resources"
    )
    let contents = try String(
      contentsOf: resources.appendingPathComponent("default.metallib.version"), encoding: .utf8
    )
    let incompatible = contents
      .replacingOccurrences(of: "metal-language-version: 3.2", with: "metal-language-version: 4.0")
      .replacingOccurrences(of: "macos-deployment-target: 15.0", with: "macos-deployment-target: 26.0")

    XCTAssertEqual(
      MLXBundleSupport.stampStatus(contents: incompatible),
      .mismatched([
        .init(field: "metal-language-version", stamped: "4.0", expected: "3.2"),
        .init(field: "macos-deployment-target", stamped: "26.0", expected: "15.0"),
      ])
    )
  }

  func testStampWithoutCompatibilityFieldsIsUnstamped() throws {
    let resources = repositoryRoot.appendingPathComponent(
      "vendor/mlx-swift_Cmlx.bundle/Contents/Resources"
    )
    let contents = try String(
      contentsOf: resources.appendingPathComponent("default.metallib.version"), encoding: .utf8
    )
    let legacy = contents.split(separator: "\n").filter {
      !$0.hasPrefix("metal-language-version:") && !$0.hasPrefix("macos-deployment-target:")
    }.joined(separator: "\n")

    XCTAssertEqual(
      MLXBundleSupport.stampStatus(contents: legacy),
      .unstamped(missingFields: ["metal-language-version", "macos-deployment-target"])
    )
  }

  func testConcurrentFirstRunBundleInstallIsIdempotent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mere-run-mlx-bundle-race-\(UUID().uuidString)",
      isDirectory: true
    )
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source/mlx-swift_Cmlx.bundle", isDirectory: true)
    let resources = source.appendingPathComponent("Contents/Resources", isDirectory: true)
    try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
    let metallib = Data("fixture-metallib".utf8)
    try metallib.write(to: resources.appendingPathComponent("default.metallib"))
    let destination = root.appendingPathComponent("debug/mlx-swift_Cmlx.bundle", isDirectory: true)
    try FileManager.default.createDirectory(
      at: destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let results = ConcurrentBundleInstallResults()

    DispatchQueue.concurrentPerform(iterations: 8) { _ in
      do {
        try MLXBundleSupport.installBundleIfMissing(from: source, to: destination)
        results.recordSuccess()
      } catch {
        results.recordFailure(error)
      }
    }

    XCTAssertEqual(results.successCount, 8)
    XCTAssertEqual(results.failures, [])
    XCTAssertEqual(
      try Data(contentsOf: destination.appendingPathComponent("Contents/Resources/default.metallib")),
      metallib
    )
  }
}

private final class ConcurrentBundleInstallResults: @unchecked Sendable {
  private let lock = NSLock()
  private var successes = 0
  private var recordedFailures: [String] = []

  var successCount: Int { lock.withLock { successes } }
  var failures: [String] { lock.withLock { recordedFailures } }

  func recordSuccess() {
    lock.withLock { successes += 1 }
  }

  func recordFailure(_ error: Error) {
    lock.withLock { recordedFailures.append(String(describing: error)) }
  }
}
#endif
