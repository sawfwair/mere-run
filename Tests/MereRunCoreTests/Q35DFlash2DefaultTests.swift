import Foundation
import XCTest
@testable import MereRunCore

final class Q35DFlash2DefaultTests: XCTestCase {
    func testAutomaticDraftRequiresQualifiedModelHardwareAndMemory() {
        let memory: UInt64 = 64 * 1_024 * 1_024 * 1_024
        XCTAssertTrue(Q35Generator.automaticDFlash2Enabled(
            modelID: "vision-chat-q38-27b-4bit", hardwareSupported: true, physicalMemory: memory
        ))
        for (id, hardware, ram) in [("other", true, memory), ("vision-chat-q38-27b-4bit", false, memory),
                                    ("vision-chat-q38-27b-4bit", true, memory / 2)] {
            XCTAssertFalse(Q35Generator.automaticDFlash2Enabled(
                modelID: id, hardwareSupported: hardware, physicalMemory: ram
            ))
        }
    }

    func testInstalledCompanionIsAutomaticAndOptOutWinsOverExplicitPath() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let draft = root.appendingPathComponent("dflash2", isDirectory: true)
        try FileManager.default.createDirectory(at: draft, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func resolve(_ enabled: Bool = true, automatic: Bool = true, path: String? = nil, value: String? = nil) -> URL? {
            Q35Generator.dflash2Root(
                targetRoot: root, configuredPath: path, enabled: enabled, automatic: automatic,
                environment: value.map { ["MERERUN_Q35_DFLASH2": $0] } ?? [:]
            )
        }
        XCTAssertNil(resolve())
        try Data("{}".utf8).write(to: draft.appendingPathComponent("config.json"))
        XCTAssertNil(resolve(), "Partial companion installs keep the existing runtime")
        try Data().write(to: draft.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(resolve(), draft)
        XCTAssertNil(resolve(automatic: false))
        XCTAssertNil(resolve(false), "Benchmark baselines disable automatic discovery")
        XCTAssertEqual(
            resolve(automatic: false, path: "/tmp/explicit-draft"),
            URL(fileURLWithPath: "/tmp/explicit-draft", isDirectory: true)
        )
        for value in ["0", "false", "no", "off", " OFF "] {
            XCTAssertNil(resolve(path: "/tmp/explicit-draft", value: value))
        }
    }
}
