import XCTest
@testable import MereRunCore

final class Q35M5VerificationPolicyTests: XCTestCase {
    func testMatrixBackendOverridesAreLimitedToQualifiedModel() {
        let modelID = "vision-chat-q38-27b-4bit"
        XCTAssertFalse(Q35Generator.shouldUseM5Verification(modelID: modelID, environment: [:]))
        for value in ["1", "true", "yes", "on", " TRUE "] {
            let environment = ["MERERUN_Q38_M5_VERIFICATION": value]
            XCTAssertTrue(Q35Generator.shouldUseM5Verification(modelID: modelID, environment: environment))
            XCTAssertFalse(Q35Generator.shouldUseM5Verification(modelID: "other-model", environment: environment))
        }
        for value in ["0", "false", "off", "no", ""] {
            XCTAssertFalse(Q35Generator.shouldUseM5Verification(
                modelID: modelID, environment: ["MERERUN_Q38_M5_VERIFICATION": value]
            ))
        }
    }

    func testMatrixDefaultRequiresDraftSupportedHardwareAndMemory() {
        let id = "vision-chat-q38-27b-4bit"
        let memory: UInt64 = 64 * 1_024 * 1_024 * 1_024
        XCTAssertTrue(Q35Generator.shouldUseM5Verification(
            modelID: id, draftAvailable: true, hardwareSupported: true, physicalMemory: memory, environment: [:]
        ))
        for (draft, hardware, ram) in [(false, true, memory), (true, false, memory), (true, true, memory / 2)] {
            XCTAssertFalse(Q35Generator.shouldUseM5Verification(
                modelID: id, draftAvailable: draft, hardwareSupported: hardware, physicalMemory: ram, environment: [:]
            ))
        }
        XCTAssertFalse(Q35Generator.shouldUseM5Verification(
            modelID: id, draftAvailable: true, hardwareSupported: true, physicalMemory: memory,
            environment: ["MERERUN_Q38_M5_VERIFICATION": "0"]
        ))
    }

    func testHeadScopeLeavesTargetBodyOnReferenceBackend() {
        for scope in ["head", " HEAD "] {
            let environment = ["MERERUN_Q38_M5_PROJECTIONS": scope]
            XCTAssertTrue(Q35Generator.shouldUseM5Projection(path: "lm_head", environment: environment))
            XCTAssertFalse(Q35Generator.shouldUseM5Projection(path: "model.layers.0.mlp.gate_proj", environment: environment))
        }
        XCTAssertTrue(Q35Generator.shouldUseM5Projection(path: "model.layers.0.mlp.gate_proj", environment: [:]))
        XCTAssertTrue(Q35Generator.shouldUseM5Projection(
            path: "model.layers.0.mlp.gate_proj", environment: ["MERERUN_Q38_M5_PROJECTIONS": "all"]
        ))
        XCTAssertFalse(Q35Generator.shouldUseM5Projection(
            path: "lm_head", environment: ["MERERUN_Q38_M5_PROJECTIONS": "unknown"]
        ))
    }
}
