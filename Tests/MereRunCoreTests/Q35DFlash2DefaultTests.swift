import Foundation
import XCTest
@testable import MereRunCore

final class Q35DFlash2DefaultTests: XCTestCase {
    func testLoRADraftingRequiresExplicitOptIn() {
        XCTAssertFalse(Q35Generator.dflash2LoRAEnabled(environment: [:]))
        XCTAssertTrue(Q35Generator.dflash2LoRAEnabled(environment: ["MERERUN_Q35_DFLASH2_LORA": "1"]))
    }

    func testAdaptiveDraftingRequiresExplicitOptIn() {
        XCTAssertFalse(Q35Generator.adaptiveDFlash2Enabled(environment: [:]))
        XCTAssertFalse(Q35Generator.adaptiveDFlash2Enabled(environment: ["MERERUN_Q35_DFLASH2_ADAPTIVE": "0"]))
        XCTAssertTrue(Q35Generator.adaptiveDFlash2Enabled(environment: ["MERERUN_Q35_DFLASH2_ADAPTIVE": "1"]))
        let control = ["MERERUN_Q35_DFLASH2_ADAPTIVE": "calibrate-only"]
        XCTAssertTrue(Q35Generator.dflash2CalibrationEnabled(environment: control))
        XCTAssertFalse(Q35Generator.adaptiveDFlash2Enabled(environment: control))
    }

    func testDraftPrefixReuseDefaultsOnWithLoRAAndRespectsOverrides() {
        let flag = "MERERUN_Q35_DFLASH2_PREFIX_CACHE"
        XCTAssertFalse(Q35Generator.dflash2PrefixCacheEnabled(hasLoRA: false, environment: [:]))
        XCTAssertTrue(Q35Generator.dflash2PrefixCacheEnabled(hasLoRA: true, environment: [:]))
        XCTAssertFalse(Q35Generator.dflash2PrefixCacheEnabled(hasLoRA: true, environment: [flag: "0"]))
        XCTAssertTrue(Q35Generator.dflash2PrefixCacheEnabled(hasLoRA: false, environment: [flag: "1"]))
    }

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
