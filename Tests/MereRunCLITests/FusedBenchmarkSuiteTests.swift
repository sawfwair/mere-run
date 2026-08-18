import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore
import MereRunRelayKit

final class FusedBenchmarkSuiteTests: XCTestCase {
    func testRuntimeInfrastructureFailureLeavesCaseTrialPending() {
        let preparation = FusedBenchmarkRuntimeError.modelPreparation(
            model: "test-model",
            profile: "test-profile",
            trial: 2,
            caseID: "test-case",
            detail: "model unavailable"
        )
        let generation = FusedBenchmarkRuntimeError.generation(
            model: "test-model",
            profile: "test-profile",
            trial: 2,
            caseID: "test-case",
            detail: "runtime interrupted"
        )

        XCTAssertTrue(preparation.localizedDescription.contains("model preparation failed"))
        XCTAssertTrue(preparation.localizedDescription.contains("the row remains pending"))
        XCTAssertTrue(generation.localizedDescription.contains("generation failed"))
        XCTAssertTrue(generation.localizedDescription.contains("the row remains pending"))
    }

    func testBundledManifestIsVersionedAndLiteCoversEverySourceFamily() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let comprehensive = manifest.selectedCases(for: .comprehensive)
        let lite = manifest.selectedCases(for: .lite)

        XCTAssertEqual(manifest.schemaVersion, 1)
        XCTAssertEqual(manifest.version, "1.2.0")
        XCTAssertEqual(comprehensive.count, 110)
        XCTAssertEqual(lite.count, 24)
        XCTAssertEqual(Set(comprehensive.map(\.sourceID)), Set(lite.map(\.sourceID)))
        XCTAssertTrue(comprehensive.allSatisfy { !$0.capabilityTags.isEmpty })
        XCTAssertTrue(comprehensive.allSatisfy { !$0.selectionRationale.isEmpty })
    }

    func testCheckpointRequiresExactPlanAndRejectsDuplicateRows() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(manifest.selectedCases(for: .lite).first)
        let source = try XCTUnwrap(manifest.source(id: descriptor.sourceID))
        let provenance = FusedBenchmarkProvenance(
            descriptor: descriptor,
            source: source,
            contentSHA256: nil
        )
        let profile = FusedBenchmarkSamplingProfile(
            name: "test-native",
            temperature: 1,
            topP: 0.95,
            topK: 20,
            minP: 0,
            reasoningEffort: nil,
            reasoningTier: nil,
            showThinking: false
        )
        let model = FusedBenchmarkPlannedModel(
            id: "test-model",
            profiles: [profile],
            catalogRepository: "example/test-model",
            catalogRevision: "0123456789abcdef",
            installed: true,
            runtimeManifestID: "test-model",
            runtimeManifestSchemaVersion: 3,
            runtimeManifestSHA256: String(repeating: "a", count: 64),
            runtimeManifestSources: nil
        )
        let plannedCase = FusedBenchmarkPlannedCase(
            id: descriptor.id,
            adapter: descriptor.adapter.rawValue,
            lane: descriptor.adapter.lane.rawValue,
            resolved: true,
            unresolvedReason: nil,
            provenance: provenance
        )
        let plan = FusedBenchmarkPlan(
            runnerVersion: MereRunCLIVersion.current,
            runnerExecutableByteCount: 123,
            runnerExecutableSHA256: String(repeating: "b", count: 64),
            host: FusedBenchmarkHost(
                processorName: "test-processor",
                physicalMemoryBytes: 128 * 1_073_741_824,
                architecture: "arm64",
                operatingSystemVersion: "test-os",
                logicalProcessorCount: 12
            ),
            manifestID: manifest.id,
            manifestVersion: manifest.version,
            manifestSHA256: try manifest.contentSHA256(),
            suite: "lite",
            models: [model],
            trials: 1,
            qualityLane: "sampled-final-target; exact-policy; logprobs=summary",
            performanceLane: "none",
            settings: FusedBenchmarkRunSettings(
                maxTokensOverride: nil,
                contextSize: 32_768,
                logprobs: "summary",
                topLogprobs: 5,
                executionTimeout: 5,
                pythonExecutable: "python3",
                pythonResolvedPath: "/usr/bin/python3",
                pythonExecutableSHA256: String(repeating: "c", count: 64),
                sandbox: "auto",
                sandboxBackend: "macos-sandbox-exec",
                allowCodeExecution: true,
                logResponses: false
            ),
            cases: [plannedCase],
            importedFixtureFiles: []
        )
        let result = FusedBenchmarkCaseResult(
            lane: "quality-final-target",
            model: model.id,
            profile: profile.name,
            trial: 1,
            caseID: descriptor.id,
            provenance: provenance,
            passed: true,
            score: 1,
            generationSeconds: 0.1,
            executionSeconds: nil,
            tokensGenerated: 3,
            logprobs: nil,
            acceleration: nil,
            response: nil,
            error: nil
        )
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("fused-checkpoint-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = FusedBenchmarkCheckpointStore(
            url: directory.appendingPathComponent("checkpoint.json")
        )

        try store.initialize(plan: plan)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let initialized = try decoder.decode(
            FusedBenchmarkCheckpoint.self,
            from: Data(contentsOf: store.url)
        )
        XCTAssertEqual(try store.loadResults(validating: plan), [])
        try store.write(plan: plan, results: [result])
        let updated = try decoder.decode(
            FusedBenchmarkCheckpoint.self,
            from: Data(contentsOf: store.url)
        )
        XCTAssertEqual(initialized.createdAt, updated.createdAt)
        XCTAssertGreaterThanOrEqual(updated.updatedAt, initialized.updatedAt)
        XCTAssertEqual(try store.loadResults(validating: plan), [result])
        XCTAssertThrowsError(try store.initialize(plan: plan))

        let changedPlan = FusedBenchmarkPlan(
            runnerVersion: plan.runnerVersion,
            runnerExecutableByteCount: plan.runnerExecutableByteCount,
            runnerExecutableSHA256: plan.runnerExecutableSHA256,
            host: plan.host,
            manifestID: plan.manifestID,
            manifestVersion: plan.manifestVersion,
            manifestSHA256: plan.manifestSHA256,
            suite: "comprehensive",
            models: plan.models,
            trials: plan.trials,
            qualityLane: plan.qualityLane,
            performanceLane: plan.performanceLane,
            settings: plan.settings,
            cases: plan.cases,
            importedFixtureFiles: plan.importedFixtureFiles
        )
        XCTAssertThrowsError(try store.loadResults(validating: changedPlan)) { error in
            XCTAssertTrue(error.localizedDescription.contains("run plan changed"))
        }

        let changedRunnerPlan = FusedBenchmarkPlan(
            runnerVersion: plan.runnerVersion,
            runnerExecutableByteCount: plan.runnerExecutableByteCount,
            runnerExecutableSHA256: String(repeating: "d", count: 64),
            host: plan.host,
            manifestID: plan.manifestID,
            manifestVersion: plan.manifestVersion,
            manifestSHA256: plan.manifestSHA256,
            suite: plan.suite,
            models: plan.models,
            trials: plan.trials,
            qualityLane: plan.qualityLane,
            performanceLane: plan.performanceLane,
            settings: plan.settings,
            cases: plan.cases,
            importedFixtureFiles: plan.importedFixtureFiles
        )
        XCTAssertThrowsError(try store.loadResults(validating: changedRunnerPlan)) { error in
            XCTAssertTrue(error.localizedDescription.contains("run plan changed"))
        }

        let changedSettingsPlan = FusedBenchmarkPlan(
            runnerVersion: plan.runnerVersion,
            runnerExecutableByteCount: plan.runnerExecutableByteCount,
            runnerExecutableSHA256: plan.runnerExecutableSHA256,
            host: plan.host,
            manifestID: plan.manifestID,
            manifestVersion: plan.manifestVersion,
            manifestSHA256: plan.manifestSHA256,
            suite: plan.suite,
            models: plan.models,
            trials: plan.trials,
            qualityLane: plan.qualityLane,
            performanceLane: plan.performanceLane,
            settings: FusedBenchmarkRunSettings(
                maxTokensOverride: plan.settings.maxTokensOverride,
                contextSize: 4_096,
                logprobs: plan.settings.logprobs,
                topLogprobs: plan.settings.topLogprobs,
                executionTimeout: plan.settings.executionTimeout,
                pythonExecutable: plan.settings.pythonExecutable,
                pythonResolvedPath: plan.settings.pythonResolvedPath,
                pythonExecutableSHA256: plan.settings.pythonExecutableSHA256,
                sandbox: plan.settings.sandbox,
                sandboxBackend: plan.settings.sandboxBackend,
                allowCodeExecution: plan.settings.allowCodeExecution,
                logResponses: plan.settings.logResponses
            ),
            cases: plan.cases,
            importedFixtureFiles: plan.importedFixtureFiles
        )
        XCTAssertThrowsError(try store.loadResults(validating: changedSettingsPlan)) { error in
            XCTAssertTrue(error.localizedDescription.contains("run plan changed"))
        }

        let duplicate = try FusedBenchmarkCheckpoint(plan: plan, results: [result, result])
        XCTAssertThrowsError(try duplicate.validatedResults(for: plan)) { error in
            XCTAssertTrue(error.localizedDescription.contains("duplicate completed case-trials"))
        }
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

    func testExternalLongBenchUsesQAF1InsteadOfPhrasePresence() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(
            manifest.cases.first { $0.id == "longbench.hotpotqa-0" }
        )
        let fixture = try FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .chat,
            sourceVersion: "v1",
            originalID: descriptor.sourceCaseID,
            messages: [ChatMessage(role: .user, content: "Question")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: nil,
            tests: nil,
            imageSHA256: nil,
            contentSHA256: "",
            sourceRevision: "dataset@0123456789abcdef",
            textMetric: .qaF1,
            referenceAnswers: ["Miller v. California"],
            passThreshold: 0.5
        ).stamped()
        let resolved = try FusedResolvedBenchmarkCase.resolve(
            descriptor: descriptor,
            manifest: manifest,
            imported: [fixture.id: fixture]
        )
        let scoring = try resolved.score(
            response: ChatResponse(response: "Miller v California", tokensGenerated: 3),
            visibleResponse: "Miller v California",
            allowCodeExecution: false,
            python: "python3",
            sandbox: .none,
            executionTimeout: 1
        )

        XCTAssertEqual(scoring.score, 1)
        XCTAssertEqual(scoring.passed, true)
    }

    func testExternalBFCLRequiresEveryParallelCallWithoutExtras() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(
            manifest.cases.first { $0.id == "bfcl.v3.parallel-0" }
        )
        let fixture = try FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .tool,
            sourceVersion: "v3",
            originalID: descriptor.sourceCaseID,
            messages: [ChatMessage(role: .user, content: "Play both artists")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: nil,
            tests: nil,
            imageSHA256: nil,
            contentSHA256: "",
            sourceRevision: "ea13468e4423454d0c213704fb87cf7cb3990433",
            expectedToolCalls: [
                FusedExternalToolExpectation(
                    name: "spotify.play",
                    arguments: [
                        "artist": ["Taylor Swift"],
                        "duration": ["20"],
                        "options": ["{\"fade\":true,\"volume\":0.5}"],
                    ]
                ),
                FusedExternalToolExpectation(
                    name: "spotify.play",
                    arguments: ["artist": ["Maroon 5"], "duration": ["15"]]
                ),
            ]
        ).stamped()
        let resolved = try FusedResolvedBenchmarkCase.resolve(
            descriptor: descriptor,
            manifest: manifest,
            imported: [fixture.id: fixture]
        )
        let response = ChatResponse(
            response: "",
            tokensGenerated: 2,
            toolCalls: [
                ToolCall(name: "spotify.play", arguments: ["artist": "Maroon 5", "duration": "15"]),
                ToolCall(
                    name: "spotify.play",
                    arguments: [
                        "artist": "Taylor Swift",
                        "duration": "20",
                        "options": "{\"volume\":0.5,\"fade\":true}",
                    ]
                ),
            ]
        )
        let scoring = try resolved.score(
            response: response,
            visibleResponse: "",
            allowCodeExecution: false,
            python: "python3",
            sandbox: .none,
            executionTimeout: 1
        )

        XCTAssertEqual(scoring.passed, true)
        XCTAssertEqual(scoring.score, 1)
    }

    func testExternalLiveCodeBenchStdinHarnessRunsEveryCase() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(
            manifest.cases.first { $0.id == "lcb.release-v5.stratum-0" }
        )
        let fixture = try FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .code,
            sourceVersion: "release_v5",
            originalID: descriptor.sourceCaseID,
            messages: [ChatMessage(role: .user, content: "Echo uppercased input")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: nil,
            tests: nil,
            imageSHA256: nil,
            contentSHA256: "",
            sourceRevision: "dataset@0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
            codeEvaluation: .stdin,
            codeTests: [
                FusedExternalCodeTest(input: "mere\n", output: "MERE\n"),
                FusedExternalCodeTest(input: "run\n", output: "RUN\n"),
            ]
        ).stamped()
        let resolved = try FusedResolvedBenchmarkCase.resolve(
            descriptor: descriptor,
            manifest: manifest,
            imported: [fixture.id: fixture]
        )
        let source = "print(input().upper())"
        let scoring = try resolved.score(
            response: ChatResponse(response: source, tokensGenerated: 4),
            visibleResponse: source,
            allowCodeExecution: true,
            python: "/usr/bin/python3",
            sandbox: .none,
            executionTimeout: 2
        )

        XCTAssertEqual(scoring.passed, true, scoring.error ?? "")
        XCTAssertEqual(scoring.score, 1)
    }

    func testExternalFunctionHarnessPreservesCandidateNamedCheck() throws {
        let manifest = try FusedBenchmarkManifest.bundled()
        let descriptor = try XCTUnwrap(
            manifest.cases.first { $0.id == "mbpp-plus.56" }
        )
        let fixture = try FusedExternalBenchmarkCase(
            id: descriptor.id,
            kind: .code,
            sourceVersion: "v0.2.0",
            originalID: descriptor.sourceCaseID,
            messages: [ChatMessage(role: .user, content: "Implement check")],
            tools: nil,
            requiredPhrases: nil,
            forbiddenPhrases: nil,
            expectedToolName: nil,
            expectedArguments: nil,
            entryPoint: "check",
            tests: "def check(candidate):\n    assert candidate(3) == 6",
            imageSHA256: nil,
            contentSHA256: "",
            sourceRevision: "pinned",
            codeEvaluation: .function
        ).stamped()
        let resolved = try FusedResolvedBenchmarkCase.resolve(
            descriptor: descriptor,
            manifest: manifest,
            imported: [fixture.id: fixture]
        )

        let scoring = try resolved.score(
            response: ChatResponse(
                response: "def check(value):\n    return value * 2",
                tokensGenerated: 8
            ),
            visibleResponse: "def check(value):\n    return value * 2",
            allowCodeExecution: true,
            python: "/usr/bin/python3",
            sandbox: .none,
            executionTimeout: 5
        )

        XCTAssertEqual(scoring.passed, true, scoring.error ?? "")
        XCTAssertNil(scoring.error)
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
