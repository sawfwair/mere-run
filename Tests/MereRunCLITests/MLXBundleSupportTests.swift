import Foundation
import XCTest

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
  }

  func testCompiledSwiftRevisionMatchesPackagePin() throws {
    let packageURL = repositoryRoot.appendingPathComponent("Package.swift")
    let package = try String(contentsOf: packageURL, encoding: .utf8)

    XCTAssertTrue(
      package.contains(#"revision: "\#(MLXBundleSupport.expectedProvenance.swiftRevision)""#)
    )
  }

  func testMissingRevisionOrKernelHashIsUnstamped() {
    let status = MLXBundleSupport.stampStatus(
      contents: "mlx-core-version: \(MLXBundleSupport.expectedProvenance.coreVersion)\n"
    )

    XCTAssertEqual(
      status,
      .unstamped(missingFields: ["mlx-swift-revision", "kernel-sources-sha256"])
    )
  }

  func testRevisionMismatchIsRejected() {
    let expected = MLXBundleSupport.expectedProvenance
    let status = MLXBundleSupport.stampStatus(
      contents: """
        mlx-core-version: \(expected.coreVersion)
        mlx-swift-revision: \(String(repeating: "0", count: 40))
        kernel-sources-sha256: \(expected.kernelSourcesSHA256)
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

  func testKernelSourceHashMismatchIsRejected() {
    let expected = MLXBundleSupport.expectedProvenance
    let status = MLXBundleSupport.stampStatus(
      contents: """
        mlx-core-version: \(expected.coreVersion)
        mlx-swift-revision: \(expected.swiftRevision)
        kernel-sources-sha256: \(String(repeating: "0", count: 64))
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
}
