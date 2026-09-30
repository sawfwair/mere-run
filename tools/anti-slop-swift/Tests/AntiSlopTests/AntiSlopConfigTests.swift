import XCTest

@testable import AntiSlopCore

final class AntiSlopConfigTests: XCTestCase {
    // New instance per test method, so the lazy directory is per-test fresh.
    private lazy var temporaryDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("anti-slop-config-\(UUID().uuidString)")
        .standardizedFileURL

    override func setUpWithError() throws {
        try FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }

    private func writeConfig(_ contents: String, in directory: URL) throws -> String {
        let path = directory.appendingPathComponent(AntiSlopConfig.fileName).path
        try contents.write(toFile: path, atomically: true, encoding: .utf8)
        return path
    }

    func testFindsConfigInStartingDirectory() throws {
        let path = try writeConfig("{\"disabled\": [\"no-force-try\"]}", in: temporaryDirectory)
        XCTAssertEqual(
            AntiSlopConfig.findConfigFile(startingAt: temporaryDirectory.path),
            path
        )
    }

    func testWalksUpToParentDirectory() throws {
        let parent = temporaryDirectory.appendingPathComponent("parent")
        let child = parent.appendingPathComponent("child/grandchild")
        try FileManager.default.createDirectory(at: child, withIntermediateDirectories: true)
        let path = try writeConfig("{}", in: parent)

        XCTAssertEqual(
            AntiSlopConfig.findConfigFile(startingAt: child.path),
            path
        )
    }

    func testReturnsNilWhenNoConfigExists() {
        XCTAssertNil(
            AntiSlopConfig.findConfigFile(startingAt: temporaryDirectory.path)
        )
    }

    func testLoadsDisabledRules() throws {
        let path = try writeConfig(
            #"{"disabled": ["no-force-try", "no-key-value-coding"], "futureKey": 1}"#,
            in: temporaryDirectory
        )
        XCTAssertEqual(
            try AntiSlopConfig.loadDisabledRules(from: path),
            ["no-force-try", "no-key-value-coding"]
        )
    }

    func testUnknownKeysAreIgnored() throws {
        let path = try writeConfig(#"{"other": true}"#, in: temporaryDirectory)
        XCTAssertEqual(try AntiSlopConfig.loadDisabledRules(from: path), [])
    }

    func testMalformedJSONThrows() throws {
        let path = try writeConfig("{not json", in: temporaryDirectory)
        XCTAssertThrowsError(try AntiSlopConfig.loadDisabledRules(from: path))
    }

    func testMissingDisabledArrayThrows() throws {
        let path = try writeConfig(#"{"disabled": "no-force-try"}"#, in: temporaryDirectory)
        XCTAssertThrowsError(try AntiSlopConfig.loadDisabledRules(from: path))
    }
}
