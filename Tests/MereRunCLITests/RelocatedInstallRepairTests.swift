import Foundation
import XCTest

@testable import MereRunCore

/// iOS relocates the app data container on every update; absolute symlinks
/// written by a previous install must be re-pointed at the same suffix under
/// the new container, and recreated relative so they never dangle again.
final class RelocatedInstallRepairTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("relocation-tests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testDanglingAbsoluteLinksAreRebasedAndMadeRelative() throws {
        let fm = FileManager.default
        let oldBase = root.appendingPathComponent("old/Application Support/MereRun", isDirectory: true)
        let newBase = root.appendingPathComponent("new/Application Support/MereRun", isDirectory: true)
        let payload = newBase.appendingPathComponent("hub/snapshots/abc/weights.safetensors")
        let modelsRoot = newBase.appendingPathComponent("models", isDirectory: true)
        let link = modelsRoot.appendingPathComponent("m1/weights.safetensors")
        try fm.createDirectory(at: payload.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: payload)
        try fm.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        // The link a previous container wrote: absolute, into the old base.
        try fm.createSymbolicLink(
            at: link,
            withDestinationURL: oldBase.appendingPathComponent("hub/snapshots/abc/weights.safetensors")
        )
        XCTAssertFalse(fm.fileExists(atPath: link.resolvingSymlinksInPath().path), "Precondition: dangling")

        let repaired = ManagedModelResolver.repairRelocatedInstalls(
            modelsRoot: modelsRoot,
            applicationSupportBase: newBase
        )

        XCTAssertEqual(repaired, 1)
        XCTAssertTrue(fm.fileExists(atPath: link.resolvingSymlinksInPath().path), "Link must resolve after repair")
        let destination = try fm.destinationOfSymbolicLink(atPath: link.path)
        XCTAssertFalse(destination.hasPrefix("/"), "Repaired link must be relative, got \(destination)")
        XCTAssertEqual(
            try Data(contentsOf: link),
            Data("weights".utf8)
        )
    }

    func testHealthyAndUnrepairableLinksAreLeftAlone() throws {
        let fm = FileManager.default
        let base = root.appendingPathComponent("only/Application Support/MereRun", isDirectory: true)
        let modelsRoot = base.appendingPathComponent("models", isDirectory: true)
        let payload = base.appendingPathComponent("hub/blob")
        try fm.createDirectory(at: payload.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("ok".utf8).write(to: payload)
        let healthy = modelsRoot.appendingPathComponent("m1/good")
        try fm.createDirectory(at: healthy.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fm.createSymbolicLink(at: healthy, withDestinationURL: payload)
        // Dangling link whose suffix does not exist under the current base.
        let hopeless = modelsRoot.appendingPathComponent("m1/gone")
        try fm.createSymbolicLink(
            at: hopeless,
            withDestinationURL: root.appendingPathComponent("elsewhere/Application Support/MereRun/hub/missing")
        )

        let repaired = ManagedModelResolver.repairRelocatedInstalls(
            modelsRoot: modelsRoot,
            applicationSupportBase: base
        )

        XCTAssertEqual(repaired, 0)
        XCTAssertEqual(try Data(contentsOf: healthy), Data("ok".utf8))
        XCTAssertFalse(fm.fileExists(atPath: hopeless.resolvingSymlinksInPath().path))
    }

    func testRelocatableSymlinkIsRelativeAndResolves() throws {
        let fm = FileManager.default
        let base = root.appendingPathComponent("store", isDirectory: true)
        let target = base.appendingPathComponent("hub/deep/dir/file.bin")
        let link = base.appendingPathComponent("models/m1/file.bin")
        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("x".utf8).write(to: target)
        try fm.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)

        try ManagedModelResolver.createRelocatableSymlink(at: link, to: target, fileManager: fm)

        let destination = try fm.destinationOfSymbolicLink(atPath: link.path)
        XCTAssertFalse(destination.hasPrefix("/"))
        XCTAssertEqual(try Data(contentsOf: link), Data("x".utf8))

        // The property that matters: the pair survives the store moving.
        let moved = root.appendingPathComponent("moved", isDirectory: true)
        try fm.moveItem(at: base, to: moved)
        let movedLink = moved.appendingPathComponent("models/m1/file.bin")
        XCTAssertEqual(try Data(contentsOf: movedLink), Data("x".utf8))
    }
}
