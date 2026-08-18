import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class FusedBenchmarkSuiteTests: XCTestCase {
    func testBundledManifestIsVersionedAndLiteCoversEverySourceFamily() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let comprehensive = manifest.selectedCases(for: .comprehensive)
        let lite = manifest.selectedCases(for: .lite)

        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.version, "1.1.0")
        XCTAssertEqual(comprehensive.count, 110)
        XCTAssertEqual(lite.count, 24)
        XCTAssertEqual(Set(comprehensive.map(\.sourceID)), Set(lite.map(\.sourceID)))
        XCTAssertTrue(comprehensive.allSatisfy { !$0.capabilityTags.isEmpty })
        XCTAssertTrue(comprehensive.allSatisfy { !$0.selectionRationale.isEmpty })
    }

    func testComprehensiveHasBalancedLaneFloors() throws {
        let cases = try FusedBenchmarkManifest.bundled().selectedCases(for: .comprehensive)
        let counts = Dictionary(grouping: cases, by: { $0.adapter.lane }).mapValues(\.count)

        XCTAssertEqual(counts[.chat], 59)
        XCTAssertEqual(counts[.code], 21)
        XCTAssertEqual(counts[.tools], 20)
        XCTAssertEqual(counts[.vision], 10)
        XCTAssertGreaterThanOrEqual(counts[.chat, default: 0], 50)
        XCTAssertGreaterThanOrEqual(counts[.code, default: 0], 20)
        XCTAssertGreaterThanOrEqual(counts[.tools, default: 0], 20)
        XCTAssertGreaterThanOrEqual(counts[.vision, default: 0], 10)
        XCTAssertLessThan(Double(counts.values.max() ?? 0) / Double(cases.count), 0.60)
    }

    func testLiteIsStratifiedAcrossEveryLaneAndSource() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let cases = manifest.selectedCases(for: .lite)
        let counts = Dictionary(grouping: cases, by: { $0.adapter.lane }).mapValues(\.count)

        XCTAssertEqual(counts[.chat], 12)
        XCTAssertEqual(counts[.code], 4)
        XCTAssertEqual(counts[.tools], 5)
        XCTAssertEqual(counts[.vision], 3)
        XCTAssertEqual(Set(cases.map(\.sourceID)), Set(manifest.sources.map(\.id)))
    }

    func testComprehensiveIncludesEveryMereAuthoredChatAndToolCase() throws {
        let cases = try FusedBenchmarkManifest.bundled().selectedCases(for: .comprehensive)
        let selectedChat = Set(cases.filter { $0.adapter == .mereChat }.map(\.sourceCaseID))
        let selectedTools = Set(cases.filter { $0.adapter == .mereTool }.map(\.sourceCaseID))

        XCTAssertEqual(selectedChat, Set(ChatBenchmarkCase.mereChatSlice.map(\.caseID)))
        XCTAssertEqual(selectedTools, Set(ToolBenchmarkCase.mereToolSlice.map(\.caseID)))
    }

    func testChatLaneIncludesConversationalFlavorScenarios() throws {
        let cases = try FusedBenchmarkManifest.bundled().selectedCases(for: .comprehensive)
        let tags = Set(cases.filter { $0.adapter.lane == .chat }.flatMap(\.capabilityTags))

        XCTAssertTrue(tags.isSuperset(of: [
            "empathy",
            "rewriting",
            "creativity",
            "false-premise",
            "tradeoffs",
            "prompt-injection",
            "uncertainty",
        ]))
    }

    func testGeneratedVisionFixturesCarryHashVerifiedImageInputs() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let source = try XCTUnwrap(manifest.source(id: "mere-vision"))
        let descriptors = manifest.cases.filter { $0.adapter == .mereVision }

        XCTAssertEqual(descriptors.count, 10)
        for descriptor in descriptors {
            let fixture = try FusedBenchmarkVisionFixtures.resolve(
                descriptor: descriptor,
                source: source
            )
            let imagePaths = fixture.messages.compactMap(\.imageUrl)
            let imagePath = try XCTUnwrap(imagePaths.first)
            let imageData = try Data(contentsOf: URL(fileURLWithPath: imagePath))

            XCTAssertEqual(fixture.kind, .vision)
            XCTAssertEqual(imagePaths.count, 1)
            XCTAssertEqual(fixture.imageSHA256, FusedBenchmarkHash.sha256(imageData))
            XCTAssertEqual(try fixture.stamped().contentSHA256, fixture.contentSHA256)
        }
    }

    func testVisionFixtureLoaderRejectsMutatedImageBytes() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let source = try XCTUnwrap(manifest.source(id: "mere-vision"))
        let descriptor = try XCTUnwrap(manifest.cases.first { $0.adapter == .mereVision })
        let generated = try FusedBenchmarkVisionFixtures.resolve(
            descriptor: descriptor,
            source: source
        )
        let generatedImage = try XCTUnwrap(generated.messages.compactMap(\.imageUrl).first)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("fused-vision-mutation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let imageURL = directory.appendingPathComponent("fixture.png")
        try FileManager.default.copyItem(at: URL(fileURLWithPath: generatedImage), to: imageURL)
        let messages = generated.messages.map { message in
            var updated = message
            if updated.imageUrl != nil {
                updated.imageUrl = imageURL.path
            }
            return updated
        }
        let fixture = try FusedExternalBenchmarkCase(
            id: generated.id,
            kind: generated.kind,
            sourceVersion: generated.sourceVersion,
            originalID: generated.originalID,
            messages: messages,
            tools: generated.tools,
            requiredPhrases: generated.requiredPhrases,
            forbiddenPhrases: generated.forbiddenPhrases,
            expectedToolName: generated.expectedToolName,
            expectedArguments: generated.expectedArguments,
            entryPoint: generated.entryPoint,
            tests: generated.tests,
            imageSHA256: nil,
            contentSHA256: ""
        ).stamped()
        let fixtureURL = directory.appendingPathComponent("fixture.jsonl")
        try JSONEncoder().encode(fixture).write(to: fixtureURL)
        try Data("mutated-image".utf8).write(to: imageURL)

        XCTAssertThrowsError(try FusedExternalBenchmarkLoader.load(paths: [fixtureURL.path])) { error in
            XCTAssertTrue(error.localizedDescription.contains("image SHA-256"))
        }
    }

    func testNativeQualityProfilesNeverUseGreedySampling() {
        let models = [
            Q35Resources.q38TwentySevenBModelId,
            Q35Resources.q38TwentySevenB4BitModelId,
            NemotronHResources.modelID,
            LagunaResources.xsModelID,
            "fallback-model",
        ]
        for model in models {
            let profiles = FusedBenchmarkSamplingProfile.nativeProfiles(
                modelID: model,
                suite: .comprehensive
            )
            XCTAssertFalse(profiles.isEmpty)
            XCTAssertTrue(profiles.allSatisfy { $0.temperature > 0 })
        }
    }

    func testQwenComprehensiveRunsEveryNativeReasoningTier() {
        let profiles = FusedBenchmarkSamplingProfile.nativeProfiles(
            modelID: Q35Resources.q38TwentySevenB4BitModelId,
            suite: .comprehensive
        )
        XCTAssertEqual(profiles.compactMap(\.reasoningTier), ["low", "medium", "xhigh"])
        XCTAssertTrue(profiles.allSatisfy { $0.temperature == 1 && $0.topP == 0.95 && $0.topK == 20 })
    }

    func testExternalFixtureRequiresMatchingContentHash() throws {
        let base = FusedExternalBenchmarkCase(
            id: "humaneval-plus.0",
            kind: .code,
            sourceVersion: "evalplus-test-pin",
            originalID: "HumanEval/0",
            messages: [ChatMessage(role: .user, content: "Implement f")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: "f",
            tests: "def check(candidate):\n    assert candidate() == 1",
            imageSHA256: nil,
            contentSHA256: ""
        )
        let fixture = FusedExternalBenchmarkCase(
            id: base.id,
            kind: base.kind,
            sourceVersion: base.sourceVersion,
            originalID: base.originalID,
            messages: base.messages,
            tools: base.tools,
            requiredPhrases: base.requiredPhrases,
            forbiddenPhrases: base.forbiddenPhrases,
            expectedToolName: base.expectedToolName,
            expectedArguments: base.expectedArguments,
            entryPoint: base.entryPoint,
            tests: base.tests,
            imageSHA256: base.imageSHA256,
            contentSHA256: try base.computedContentSHA256()
        )
        let encoder = JSONEncoder()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fused-fixture-\(UUID().uuidString).jsonl")
        try encoder.encode(fixture).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let loaded = try FusedExternalBenchmarkLoader.load(paths: [url.path])
        XCTAssertEqual(loaded[fixture.id]?.contentSHA256, fixture.contentSHA256)
    }

    func testExternalFixtureCannotReplaceSelectedUpstreamCase() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(
            manifest.cases.first { $0.id == "humaneval-plus.0" }
        )
        let imported = FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .code,
            sourceVersion: "evalplus-test-pin",
            originalID: "HumanEval/999",
            messages: [ChatMessage(role: .user, content: "Implement f")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: "f",
            tests: "def check(candidate): pass",
            imageSHA256: nil,
            contentSHA256: "unused-by-resolver"
        )

        let source = try XCTUnwrap(manifest.source(id: descriptor.sourceID))
        XCTAssertThrowsError(try FusedExternalBenchmarkContract.validate(
            imported,
            for: descriptor,
            source: source
        )) { error in
            XCTAssertTrue(error.localizedDescription.contains("original id"))
        }
    }

    func testCalibrationSeparatesFragilePassesAndConfidentFailures() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(manifest.cases.first)
        let source = try XCTUnwrap(manifest.source(id: descriptor.sourceID))
        let provenance = FusedBenchmarkProvenance(
            descriptor: descriptor,
            source: source,
            contentSHA256: "abc"
        )
        func result(id: String, passed: Bool, logprob: Double) -> FusedBenchmarkCaseResult {
            let token = ChatTokenLogprob(
                tokenID: 1,
                rawLogprob: logprob,
                policyLogprob: logprob,
                rawEntropy: 1,
                policyEntropy: 1,
                rawTop1Top2Margin: 1,
                policyTop1Top2Margin: 1
            )
            return FusedBenchmarkCaseResult(
                lane: "quality-final-target",
                model: "test",
                profile: "native",
                trial: 1,
                caseID: id,
                provenance: provenance,
                passed: passed,
                score: passed ? 1 : 0,
                generationSeconds: 1,
                executionSeconds: nil,
                tokensGenerated: 1,
                logprobs: FusedBenchmarkLogprobMetrics(ChatLogprobDiagnostics(
                    capture: .tokens,
                    measuredTokens: [token],
                    captureSeconds: 0
                )),
                acceleration: ChatAccelerationDiagnostics(route: "final-target-pipelined"),
                response: nil,
                error: nil
            )
        }

        let calibration = FusedBenchmarkCalibration.calculate(from: [
            result(id: "fragile", passed: true, logprob: -2),
            result(id: "confident", passed: false, logprob: -0.1),
        ])
        XCTAssertEqual(calibration.fragilePasses, ["fragile"])
        XCTAssertEqual(calibration.confidentFailures, ["confident"])
        XCTAssertNotNil(calibration.expectedCalibrationError)
    }

    func testLowConfidenceSpansSplitAtSemanticRegionBoundaries() {
        let tokens = [
            ChatTokenLogprob(
                tokenID: 1,
                region: .visible,
                rawLogprob: -3,
                policyLogprob: -2,
                rawEntropy: 1,
                policyEntropy: 1,
                rawTop1Top2Margin: 0.1,
                policyTop1Top2Margin: 0.1
            ),
            ChatTokenLogprob(
                tokenID: 2,
                region: .code,
                rawLogprob: -3,
                policyLogprob: -2,
                rawEntropy: 1,
                policyEntropy: 1,
                rawTop1Top2Margin: 0.1,
                policyTop1Top2Margin: 0.1
            ),
        ]
        let metrics = FusedBenchmarkLogprobMetrics(ChatLogprobDiagnostics(
            capture: .tokens,
            measuredTokens: tokens,
            captureSeconds: 0
        ))

        XCTAssertEqual(metrics.lowConfidenceSpans.map(\.region), ["visible", "code"])
    }
}
