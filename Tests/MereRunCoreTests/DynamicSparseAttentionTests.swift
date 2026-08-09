import XCTest
@testable import MereRunCore

final class DynamicSparseAttentionTests: MereRunCoreTestCase {
    func testZImageRuntimeIsEnabledByDefaultAndCanBeDisabled() {
        XCTAssertTrue(DynamicSparseAttentionRuntime.isEnabled(model: .zImage, environment: [:]))
        XCTAssertFalse(DynamicSparseAttentionRuntime.isEnabled(
            model: .zImage,
            environment: ["MERERUN_DYNAMIC_SPARSE_ATTENTION": "0"]
        ))
        XCTAssertEqual(
            DynamicSparseAttentionRuntime.selection(environment: [
                "MERERUN_DYNAMIC_SPARSE_ATTENTION": "z-image",
            ]),
            [.zImage]
        )
        XCTAssertEqual(
            DynamicSparseAttentionRuntime.selection(environment: [
                "MERERUN_DYNAMIC_SPARSE_ATTENTION": "all",
            ]),
            Set(DynamicSparseAttentionModel.allCases)
        )
    }

    func testPolicySupportsModelsWithoutPrefixTokens() throws {
        let policy = DynamicSparseAttentionPolicy(
            minimumSequenceLength: 12_000,
            denseLeadingStepFraction: 0.2,
            denseTrailingStepCount: 1,
            denseLeadingLayerCount: 2
        )
        let request = try XCTUnwrap(policy.request(
            stepIndex: 2,
            stepCount: 4,
            layerIndex: 2,
            sequenceLength: 12_800,
            prefixTokenCount: 0
        ))
        XCTAssertEqual(request.prefixTokenCount, 0)
    }

    func testConfigurationCanLowerThresholdForDiagnosticSmoke() throws {
        let runtime = try XCTUnwrap(DynamicSparseAttentionRuntime.configured(
            model: .zImage,
            environment: [
                "MERERUN_DYNAMIC_SPARSE_ATTENTION": "z-image",
                "MERERUN_DYNAMIC_SPARSE_LOG": "1",
                "MERERUN_DYNAMIC_SPARSE_MIN_TOKENS": "256",
                "MERERUN_DYNAMIC_SPARSE_TAU": "0.75",
            ]
        ))
        XCTAssertEqual(runtime.policy.minimumSequenceLength, 256)
        XCTAssertEqual(runtime.policy.thresholdStandardDeviations, 0.75)
        XCTAssertEqual(runtime.maximumGateRelativeError, 0.015)
        XCTAssertNotNil(runtime.logHandler)
    }
}
