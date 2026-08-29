import Foundation
import MereRunEvaluation
import XCTest
@testable import MereRunCLI

final class EvaluationCommandTests: XCTestCase {
    func testCommandTreeExposesExternalEvaluationLifecycle() {
        let topLevel = Set(MereRunCLI.configuration.subcommands.map { $0.configuration.commandName })
        let evaluation = Set(EvaluationCommand.configuration.subcommands.map {
            $0.configuration.commandName
        })
        let pack = Set(EvaluationPackCommand.configuration.subcommands.map {
            $0.configuration.commandName
        })

        XCTAssertTrue(topLevel.contains("eval"))
        XCTAssertEqual(evaluation, ["pack", "run", "promote"])
        XCTAssertEqual(pack, ["validate"])
    }

    func testRunCommandParsesRepeatableBindingsAndSafeDryRun() throws {
        let fixture = try makePack()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let command = try EvaluationRunCommand.parse([
            fixture.root.path,
            "--model", "base=synthetic-model",
            "--adapter", "candidate=\(fixture.adapter.path)",
            "--trials", "3",
            "--logprobs", "top",
            "--top-logprobs", "7",
            "--dry-run",
            "--json",
        ])

        XCTAssertEqual(command.modelArguments, ["base=synthetic-model"])
        XCTAssertEqual(command.adapterArguments, ["candidate=\(fixture.adapter.path)"])
        XCTAssertEqual(command.trials, 3)
        XCTAssertEqual(command.logprobs, .top)
        XCTAssertEqual(command.topLogprobs, 7)
        XCTAssertTrue(command.dryRun)
        XCTAssertTrue(command.json)
        XCTAssertFalse(command.allowExternalScorer)
    }

    func testPlanIsAdapterAwareAndContainsHashesNotPackContent() throws {
        let fixture = try makePack()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let resolved = try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: nil,
            maxTokens: nil,
            contextSize: nil,
            logprobs: nil,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        )

        XCTAssertEqual(resolved.plan.arms.count, 4)
        XCTAssertEqual(resolved.plan.expectedResultKeys.count, 4)
        XCTAssertEqual(resolved.plan.adapters.count, 1)
        XCTAssertEqual(resolved.plan.adapters[0].sha256.count, 64)
        XCTAssertEqual(resolved.plan.adapters[0].training?.status, "completed")
        XCTAssertEqual(
            resolved.plan.adapters[0].training?.datasetFingerprint,
            String(repeating: "e", count: 64)
        )
        XCTAssertEqual(resolved.plan.pack.packSHA256.count, 64)
        XCTAssertFalse(resolved.plan.models[0].installed)
        XCTAssertFalse(resolved.plan.settings.runtimeSeedControl)
        XCTAssertNil(resolved.plan.settings.responseFormat)
        let encoded = String(
            decoding: try EvaluationJSON.canonicalEncoder.encode(resolved.plan),
            as: UTF8.self
        )
        XCTAssertFalse(encoded.contains("What is two plus two?"))
        XCTAssertFalse(encoded.contains("Answer with one short sentence."))
        XCTAssertFalse(encoded.contains(fixture.root.path))
    }

    func testVisionCaseResolvesOnlyManifestPinnedImageIntoRuntimeMessage() throws {
        let fixture = try makePack(vision: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let benchmarkCase = try XCTUnwrap(loaded.cases.first)
        let arm = EvaluationArmPlan(
            id: "base-neutral",
            modelSlot: "base",
            adapterSlot: nil,
            adapterScale: 1,
            promptSet: nil,
            profileIDs: ["sampled"]
        )

        let messages = try EvaluationRequestBuilder.messages(
            benchmarkCase: benchmarkCase,
            arm: arm,
            pack: loaded
        )

        XCTAssertEqual(messages.count, 1)
        XCTAssertEqual(
            messages[0].imageURLs,
            [fixture.root.appendingPathComponent("images/synthetic.png").path]
        )
    }

    func testReportGatesPromotionAndCheckpointAreDerivedFromExactPlan() throws {
        let fixture = try makePack()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let resolved = try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: 1,
            maxTokens: nil,
            contextSize: nil,
            logprobs: EvaluationLogprobMode.none,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        )
        let rows = try resolved.plan.expectedResultKeys.map { key in
            try makePassingRow(key: key, plan: resolved.plan)
        }
        let report = try EvaluationRunReport(plan: resolved.plan, results: rows)

        XCTAssertTrue(report.progress.complete)
        XCTAssertTrue(report.gates.allSatisfy(\.passed))
        XCTAssertTrue(report.promotionEligible)
        let reportData = try report.jsonData()
        let receipt = try EvaluationPromotionReceipt(
            reportData: reportData,
            report: report,
            createdAt: Date(timeIntervalSince1970: 1)
        )
        XCTAssertEqual(receipt.payloadSHA256.count, 64)
        XCTAssertEqual(receipt.payload.pack.packSHA256, resolved.plan.pack.packSHA256)
        XCTAssertEqual(receipt.payload.adapters[0].sha256, resolved.plan.adapters[0].sha256)

        let checkpointURL = fixture.root.appendingPathComponent("checkpoint.json")
        let store = EvaluationCheckpointStore(url: checkpointURL)
        try store.initialize(plan: resolved.plan)
        try store.write(plan: resolved.plan, results: rows)
        XCTAssertEqual(try store.loadResults(validating: resolved.plan).count, 4)

        var changedSettings = resolved.plan.settings
        changedSettings = EvaluationRunSettings(
            trials: 2,
            maxTokens: changedSettings.maxTokens,
            contextSize: changedSettings.contextSize,
            logprobs: changedSettings.logprobs,
            topLogprobs: changedSettings.topLogprobs,
            responseFormat: changedSettings.responseFormat,
            logResponses: changedSettings.logResponses,
            externalScorerAuthorized: changedSettings.externalScorerAuthorized,
            runtimeSeedControl: changedSettings.runtimeSeedControl
        )
        let changedPlan = EvaluationRunPlan(
            runner: resolved.plan.runner,
            host: resolved.plan.host,
            pack: resolved.plan.pack,
            models: resolved.plan.models,
            adapters: resolved.plan.adapters,
            prompts: resolved.plan.prompts,
            arms: resolved.plan.arms,
            profiles: resolved.plan.profiles,
            cases: resolved.plan.cases,
            gates: resolved.plan.gates,
            settings: changedSettings
        )
        XCTAssertThrowsError(try store.loadResults(validating: changedPlan))
    }

    func testPlanPinsPackRequestedJSONResponseFormat() throws {
        let fixture = try makePack(responseFormat: .jsonObject)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let resolved = try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: nil,
            maxTokens: nil,
            contextSize: nil,
            logprobs: nil,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        )

        XCTAssertEqual(
            resolved.plan.settings.responseFormat,
            EvaluationResponseFormat.jsonObject.rawValue
        )
        XCTAssertThrowsError(try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: nil,
            maxTokens: nil,
            contextSize: nil,
            logprobs: .summary,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        )) { error in
            XCTAssertTrue(String(describing: error).contains("requires logprobs none"))
        }
    }

    func testBuiltInAssertionsProduceTypedMetrics() throws {
        let assertions = [
            EvaluationAssertion(
                id: "contains-four",
                kind: .contains,
                value: "FOUR",
                caseInsensitive: true
            ),
            EvaluationAssertion(
                id: "no-markdown",
                kind: .excludes,
                value: "```"
            ),
        ]
        let score = try XCTUnwrap(EvaluationScoring.scoreAssertions(
            response: "four",
            assertions: assertions
        ))

        XCTAssertTrue(score.passed)
        XCTAssertEqual(score.score, 1)
        XCTAssertEqual(score.metrics, [EvaluationMetric(id: "assertion-pass-rate", value: 1)])
        XCTAssertEqual(score.failedChecks, [])
    }

    func testReportRejectsDuplicateRows() throws {
        let fixture = try makePack()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let resolved = try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: 1,
            maxTokens: nil,
            contextSize: nil,
            logprobs: EvaluationLogprobMode.none,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        )
        let key = try XCTUnwrap(resolved.plan.expectedResultKeys.first)
        let row = try makePassingRow(key: key, plan: resolved.plan)

        XCTAssertThrowsError(try EvaluationRunReport(plan: resolved.plan, results: [row, row]))
    }

    func testRequiredTrainingManifestCannotBeSkipped() throws {
        let fixture = try makePack()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.removeItem(
            at: fixture.root.appendingPathComponent("candidate.manifest.json")
        )
        let loaded = try EvaluationPackLoader.load(from: fixture.root)

        XCTAssertThrowsError(try EvaluationPlanBuilder.build(
            pack: loaded,
            modelReferences: ["base": "synthetic-uninstalled-model"],
            adapterReferences: ["candidate": fixture.adapter.path],
            trials: 1,
            maxTokens: nil,
            contextSize: nil,
            logprobs: EvaluationLogprobMode.none,
            topLogprobs: nil,
            logResponses: false,
            externalScorerAuthorized: false
        ))
    }

    func testExternalScorerRequiresOptInAndUsesPinnedLocalProtocol() throws {
        let fixture = try makePack(externalScorer: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let loaded = try EvaluationPackLoader.load(from: fixture.root)
        let benchmarkCase = try XCTUnwrap(loaded.cases.first)
        let arm = EvaluationArmPlan(
            id: "base-neutral",
            modelSlot: "base",
            adapterSlot: nil,
            adapterScale: 1,
            promptSet: nil,
            profileIDs: ["sampled"]
        )

        XCTAssertThrowsError(try EvaluationScoring.score(
            response: "four",
            benchmarkCase: benchmarkCase,
            pack: loaded,
            arm: arm,
            profileID: "sampled",
            trial: 1,
            modelID: "synthetic-model",
            adapterSHA256: nil,
            allowExternalScorer: false
        ))
        let score = try EvaluationScoring.score(
            response: "four",
            benchmarkCase: benchmarkCase,
            pack: loaded,
            arm: arm,
            profileID: "sampled",
            trial: 1,
            modelID: "synthetic-model",
            adapterSHA256: nil,
            allowExternalScorer: true
        )
        XCTAssertTrue(score.passed)
        XCTAssertEqual(score.metrics, [EvaluationMetric(id: "synthetic-quality", value: 1)])
    }

    private func makePack(
        externalScorer: Bool = false,
        vision: Bool = false,
        responseFormat: EvaluationResponseFormat? = nil
    ) throws -> (
        root: URL,
        adapter: URL
    ) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("synthetic-evaluation-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let prompts = root.appendingPathComponent("prompts", isDirectory: true)
        try FileManager.default.createDirectory(at: prompts, withIntermediateDirectories: true)
        try Data("Answer with one short sentence.\n".utf8).write(
            to: prompts.appendingPathComponent("concise.txt"),
            options: .atomic
        )
        if vision {
            let images = root.appendingPathComponent("images", isDirectory: true)
            try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
            try Data("synthetic image bytes".utf8).write(
                to: images.appendingPathComponent("synthetic.png"),
                options: .atomic
            )
        }
        let scorer: EvaluationScorer
        if externalScorer {
            let scorerURL = root.appendingPathComponent("score.sh")
            let script = """
            #!/bin/sh
            read request
            printf '%s\n' '{"schema_version":1,"passed":true,"score":1,"metrics":[{"id":"synthetic-quality","value":1}],"hard_failures":[]}'
            """
            try Data(script.utf8).write(to: scorerURL, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: scorerURL.path
            )
            scorer = EvaluationScorer(
                kind: .externalProcess,
                executable: "score.sh",
                timeoutSeconds: 5
            )
        } else {
            scorer = EvaluationScorer(kind: .assertions)
        }
        let gate = EvaluationGate(
            id: "all-pass",
            aggregation: .passRate,
            comparator: .greaterThanOrEqual,
            threshold: 1,
            required: true
        )
        let manifest = EvaluationPackManifest(
            id: "synthetic-arithmetic",
            version: "1.0.0",
            description: "Synthetic arithmetic fixture.",
            caseFiles: ["cases.jsonl"],
            imageFiles: vision ? ["images/synthetic.png"] : nil,
            promptSets: [EvaluationPromptSet(
                id: "concise",
                systemPromptFile: "prompts/concise.txt"
            )],
            arms: [
                EvaluationArm(id: "base-neutral", modelSlot: "base"),
                EvaluationArm(id: "base-prompted", modelSlot: "base", promptSet: "concise"),
                EvaluationArm(
                    id: "adapter-neutral",
                    modelSlot: "base",
                    adapterSlot: "candidate"
                ),
                EvaluationArm(
                    id: "adapter-prompted",
                    modelSlot: "base",
                    adapterSlot: "candidate",
                    promptSet: "concise"
                ),
            ],
            samplingProfiles: [EvaluationSamplingProfile(
                id: "sampled",
                temperature: 0.7,
                topP: 0.9
            )],
            scorer: scorer,
            gates: [gate],
            defaults: EvaluationPackDefaults(
                trials: 1,
                maxTokens: 16,
                contextSize: 512,
                logprobs: responseFormat == .jsonObject ? .none : .summary,
                topLogprobs: 5,
                responseFormat: responseFormat
            ),
            adapterRequirements: EvaluationAdapterRequirements(
                requireTrainingManifest: true
            )
        )
        try EvaluationJSON.prettyEncoder.encode(manifest).write(
            to: root.appendingPathComponent("eval-pack.json"),
            options: .atomic
        )
        let assertions = externalScorer ? [] : [EvaluationAssertion(
            id: "contains-four",
            kind: .contains,
            value: "four",
            caseInsensitive: true
        )]
        let benchmarkCase = EvaluationCase(
            id: "synthetic-arithmetic-001",
            split: .heldOut,
            capabilityTags: ["arithmetic"],
            messages: [EvaluationMessage(
                role: .user,
                content: "What is two plus two?",
                imageFile: vision ? "images/synthetic.png" : nil
            )],
            assertions: assertions
        )
        var caseData = try EvaluationJSON.canonicalEncoder.encode(benchmarkCase)
        caseData.append(0x0A)
        try caseData.write(to: root.appendingPathComponent("cases.jsonl"), options: .atomic)
        let adapter = root.appendingPathComponent("candidate.safetensors")
        try Data("synthetic adapter bytes".utf8).write(to: adapter, options: .atomic)
        let trainingManifest = """
        {
          "adapterName": "synthetic-candidate",
          "baseModel": "synthetic-uninstalled-model",
          "createdAt": "2026-01-01T00:00:00Z",
          "evalPromptCount": 1,
          "format": "synthetic.text-lora",
          "lora": {"alpha": 8, "rank": 8, "targetModules": ["q_proj"]},
          "outputFile": "candidate.safetensors",
          "schemaVersion": 1,
          "status": "completed",
          "training": {
            "batchSize": 1,
            "dataset": {
              "averageAssistantCharacters": 4,
              "exampleCount": 1,
              "fingerprint": "\(String(repeating: "e", count: 64))",
              "maxAssistantCharacters": 4,
              "messageCount": 3,
              "sourceCount": 1
            },
            "learningRate": 0.0001,
            "maxSequenceLength": 512,
            "seed": 42,
            "trainingSteps": 1
          }
        }
        """
        try Data(trainingManifest.utf8).write(
            to: root.appendingPathComponent("candidate.manifest.json"),
            options: .atomic
        )
        return (root, adapter)
    }

    private func makePassingRow(
        key: EvaluationResultKey,
        plan: EvaluationRunPlan
    ) throws -> EvaluationResultRow {
        let arm = try XCTUnwrap(plan.arms.first { $0.id == key.armID })
        let model = try XCTUnwrap(plan.models.first { $0.slot == arm.modelSlot })
        let adapter = arm.adapterSlot.flatMap { slot in
            plan.adapters.first { $0.slot == slot }
        }
        let benchmarkCase = try XCTUnwrap(plan.cases.first { $0.id == key.caseID })
        return EvaluationResultRow(
            armID: key.armID,
            modelID: model.id,
            adapterSHA256: adapter?.sha256,
            profileID: key.profileID,
            trial: key.trial,
            caseID: key.caseID,
            caseSHA256: benchmarkCase.contentSHA256,
            split: benchmarkCase.split,
            capabilityTags: benchmarkCase.capabilityTags,
            passed: true,
            score: 1,
            metrics: [EvaluationMetric(id: "synthetic-quality", value: 1)],
            failedChecks: [],
            hardFailures: [],
            generationSeconds: 0.1,
            scoringSeconds: 0.01,
            tokensGenerated: 1,
            logprobs: nil,
            responseSHA256: String(repeating: "d", count: 64),
            response: nil
        )
    }
}
