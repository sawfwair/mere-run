import ArgumentParser
import Foundation
import MereRunCore
import MereRunEvaluation

struct EvaluationCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "Run reproducible evaluations from external, content-addressed packs.",
        subcommands: [
            EvaluationPackCommand.self,
            EvaluationRunCommand.self,
            EvaluationPromoteCommand.self,
        ]
    )
}

struct EvaluationPackCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pack",
        abstract: "Inspect external evaluation packs without running models.",
        subcommands: [EvaluationPackValidateCommand.self]
    )
}

struct EvaluationPackValidateCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "validate",
        abstract: "Validate and hash an external evaluation pack."
    )

    @Argument(help: "Pack directory or eval-pack.json path.")
    var pack: String

    @Flag(name: [.long], help: "Emit machine-readable JSON.")
    var json: Bool = false

    func run() throws {
        let loaded = try EvaluationPackLoader.load(from: pack)
        let output = EvaluationPackValidationOutput(pack: loaded)
        if json {
            print(String(decoding: try EvaluationJSON.prettyEncoder.encode(output), as: UTF8.self))
        } else {
            print(output.renderText())
        }
    }
}

struct EvaluationPackValidationOutput: Codable {
    let schemaVersion: Int
    let valid: Bool
    let id: String
    let version: String
    let manifestSHA256: String
    let packSHA256: String
    let files: [EvaluationPackFilePin]
    let caseCount: Int
    let splits: [String: Int]
    let arms: [String]
    let profiles: [String]
    let modelSlots: [String]
    let adapterSlots: [String]
    let scorerKind: String
    let gateCount: Int

    init(pack: LoadedEvaluationPack) {
        schemaVersion = 1
        valid = true
        id = pack.manifest.id
        version = pack.manifest.version
        manifestSHA256 = pack.manifestSHA256
        packSHA256 = pack.packSHA256
        files = pack.files
        caseCount = pack.cases.count
        splits = Dictionary(grouping: pack.cases, by: { $0.specification.split.rawValue })
            .mapValues(\.count)
        arms = pack.manifest.arms.map(\.id)
        profiles = pack.manifest.samplingProfiles.map(\.id)
        modelSlots = Array(Set(pack.manifest.arms.map(\.modelSlot))).sorted()
        adapterSlots = Array(Set(pack.manifest.arms.compactMap(\.adapterSlot))).sorted()
        scorerKind = pack.manifest.scorer.kind.rawValue
        gateCount = pack.manifest.gates.count
    }

    func renderText() -> String {
        [
            "Valid external evaluation pack \(id)@\(version)",
            "pack sha256: \(packSHA256)",
            "manifest sha256: \(manifestSHA256)",
            "declared files: \(files.count)",
            "cases: \(caseCount)",
            "arms: \(arms.joined(separator: ", "))",
            "profiles: \(profiles.joined(separator: ", "))",
            "model slots: \(modelSlots.joined(separator: ", "))",
            "adapter slots: \(adapterSlots.joined(separator: ", "))",
            "scorer: \(scorerKind)",
            "gates: \(gateCount)",
            "validation only: no models loaded, no inference run, and no scorer executed",
        ].joined(separator: "\n")
    }
}

struct EvaluationRunCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run",
        abstract: "Run a matched, adapter-aware evaluation from an external pack."
    )

    @Argument(help: "Pack directory or eval-pack.json path.")
    var pack: String

    @Option(
        name: [.customLong("model")],
        help: "Model binding as slot=model-id; repeat for multiple slots. A lone model id binds the base slot."
    )
    var modelArguments: [String] = []

    @Option(
        name: [.customLong("adapter")],
        help: "Adapter binding as slot=catalog-id-or-path; repeat for multiple slots. A lone reference binds candidate."
    )
    var adapterArguments: [String] = []

    @Option(name: [.long], help: "Override the pack trial count.")
    var trials: Int?

    @Option(name: [.customLong("max-tokens")], help: "Override the pack generation budget.")
    var maxTokens: Int?

    @Option(name: [.customLong("context-size")], help: "Override the pack context size.")
    var contextSize: Int?

    @Option(name: [.long], help: "Override logprob capture: none, summary, tokens, or top.")
    var logprobs: EvaluationLogprobArgument?

    @Option(name: [.customLong("top-logprobs")], help: "Visible candidates per token for top capture.")
    var topLogprobs: Int?

    @Flag(
        name: [.customLong("allow-external-scorer")],
        help: "Authorize execution of the pack-pinned scorer process."
    )
    var allowExternalScorer: Bool = false

    @Flag(name: [.customLong("log-responses")], help: "Include model responses in the local report.")
    var logResponses: Bool = false

    @Flag(name: [.customLong("dry-run")], help: "Print the complete plan without loading models or executing scorers.")
    var dryRun: Bool = false

    @Option(name: [.long], help: "Atomically checkpoint the plan and every completed row.")
    var checkpoint: String?

    @Flag(name: [.long], help: "Resume a checkpoint whose plan exactly matches this invocation.")
    var resume: Bool = false

    @Option(
        name: [.customLong("case-trial-limit")],
        help: "Stop cleanly after this many new rows; requires --checkpoint."
    )
    var caseTrialLimit: Int?

    @Option(name: [.long], help: "Atomically write the report JSON to this path.")
    var output: String?

    @Flag(name: [.long], help: "Emit the report as machine-readable JSON on stdout.")
    var json: Bool = false

    func validate() throws {
        guard trials == nil || trials! > 0 else {
            throw ValidationError("--trials must be greater than zero.")
        }
        guard maxTokens == nil || maxTokens! > 0 else {
            throw ValidationError("--max-tokens must be greater than zero.")
        }
        guard contextSize == nil || contextSize! > 0 else {
            throw ValidationError("--context-size must be greater than zero.")
        }
        guard topLogprobs == nil || (1...20).contains(topLogprobs!) else {
            throw ValidationError("--top-logprobs must be between 1 and 20.")
        }
        guard !resume || checkpoint != nil else {
            throw ValidationError("--resume requires --checkpoint.")
        }
        guard caseTrialLimit == nil || caseTrialLimit! > 0 else {
            throw ValidationError("--case-trial-limit must be greater than zero.")
        }
        guard caseTrialLimit == nil || checkpoint != nil else {
            throw ValidationError("--case-trial-limit requires --checkpoint.")
        }
        if let checkpoint, let output {
            let checkpointURL = URL(fileURLWithPath: checkpoint).standardizedFileURL
            let outputURL = URL(fileURLWithPath: output).standardizedFileURL
            guard checkpointURL != outputURL else {
                throw ValidationError("--checkpoint and --output must use different paths.")
            }
        }
        let loaded = try EvaluationPackLoader.load(from: pack)
        if loaded.manifest.scorer.kind == .externalProcess,
           !allowExternalScorer,
           !dryRun {
            throw ValidationError(
                "This pack uses an external scorer; inspect it, then pass --allow-external-scorer."
            )
        }
        _ = try EvaluationBindingParser.parse(
            modelArguments,
            defaultSlot: "base",
            kind: "model"
        )
        _ = try EvaluationBindingParser.parse(
            adapterArguments,
            defaultSlot: "candidate",
            kind: "adapter"
        )
    }

    func run() async throws {
        let loadedPack = try EvaluationPackLoader.load(from: pack)
        let modelReferences = try EvaluationBindingParser.parse(
            modelArguments,
            defaultSlot: "base",
            kind: "model"
        )
        let adapterReferences = try EvaluationBindingParser.parse(
            adapterArguments,
            defaultSlot: "candidate",
            kind: "adapter"
        )
        let resolved = try EvaluationPlanBuilder.build(
            pack: loadedPack,
            modelReferences: modelReferences,
            adapterReferences: adapterReferences,
            trials: trials,
            maxTokens: maxTokens,
            contextSize: contextSize,
            logprobs: logprobs?.contractValue,
            topLogprobs: topLogprobs,
            logResponses: logResponses,
            externalScorerAuthorized: allowExternalScorer
        )
        let checkpointStore = checkpoint.map {
            EvaluationCheckpointStore(url: URL(fileURLWithPath: $0).standardizedFileURL)
        }
        var results: [EvaluationResultRow]
        if let checkpointStore {
            if resume {
                results = try checkpointStore.loadResults(validating: resolved.plan)
            } else {
                try checkpointStore.initialize(plan: resolved.plan)
                results = []
            }
        } else {
            results = []
        }

        if !dryRun {
            results = try await execute(
                resolved,
                initialResults: results,
                checkpointStore: checkpointStore
            )
        }
        let report = try EvaluationRunReport(plan: resolved.plan, results: results)
        if let output {
            let outputURL = URL(fileURLWithPath: output).standardizedFileURL
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try report.jsonData().write(to: outputURL, options: .atomic)
        }
        if json {
            print(try report.jsonString())
        } else {
            print(report.renderText(dryRun: dryRun))
            if let output {
                print("report: \(URL(fileURLWithPath: output).standardizedFileURL.path)")
            }
        }
    }

    private func execute(
        _ resolved: ResolvedEvaluationPlan,
        initialResults: [EvaluationResultRow],
        checkpointStore: EvaluationCheckpointStore?
    ) async throws -> [EvaluationResultRow] {
        var results = initialResults
        var completedKeys = Set(results.map(\.key))
        let pending = resolved.plan.expectedResultKeys.subtracting(completedKeys)
        if pending.isEmpty {
            return results
        }
        let modelBySlot = Dictionary(uniqueKeysWithValues: resolved.plan.models.map { ($0.slot, $0) })
        let pendingArmIDs = Set(pending.map(\.armID))
        let unavailable = resolved.plan.arms.filter { pendingArmIDs.contains($0.id) }.compactMap { arm in
            modelBySlot[arm.modelSlot]?.installed == false ? modelBySlot[arm.modelSlot]?.id : nil
        }
        guard unavailable.isEmpty else {
            throw ValidationError(
                "External evaluations never download models. Install or mount: "
                    + Array(Set(unavailable)).sorted().joined(separator: ", ")
            )
        }

        try MLXBundleSupport.ensureAvailable(quiet: json)
        let caseByID = Dictionary(uniqueKeysWithValues: resolved.pack.cases.map {
            ($0.specification.id, $0)
        })
        let profileByID = Dictionary(uniqueKeysWithValues: resolved.pack.manifest.samplingProfiles.map {
            ($0.id, $0)
        })
        let modelIDs = Array(Set(resolved.plan.models.map(\.id))).sorted()
        var completedThisInvocation = 0
        var reachedLimit = false
        for modelID in modelIDs where !reachedLimit {
            let arms = resolved.plan.arms.filter {
                modelBySlot[$0.modelSlot]?.id == modelID && pendingArmIDs.contains($0.id)
            }
            guard !arms.isEmpty else { continue }
            let defaultEngine = ManagedModelCatalog.apiProfile(for: modelID)?.servingEngine
                ?? .textChatQ36
            let pool = RuntimeModelPool(
                defaultModelID: modelID,
                defaultEngine: defaultEngine,
                startupModelPath: nil
            )
            do {
                armLoop: for arm in arms {
                    for profileID in arm.profileIDs {
                        guard let profile = profileByID[profileID] else {
                            throw ValidationError("Evaluation plan references unknown profile \(profileID).")
                        }
                        for trial in 1...resolved.plan.settings.trials {
                            for casePlan in resolved.plan.cases {
                                let key = EvaluationResultKey(
                                    armID: arm.id,
                                    profileID: profileID,
                                    trial: trial,
                                    caseID: casePlan.id
                                )
                                guard !completedKeys.contains(key) else { continue }
                                guard let benchmarkCase = caseByID[casePlan.id] else {
                                    throw ValidationError("Evaluation plan references unknown case \(casePlan.id).")
                                }
                                let row = try await runCase(
                                    benchmarkCase,
                                    arm: arm,
                                    profile: profile,
                                    modelID: modelID,
                                    resolved: resolved,
                                    trial: trial,
                                    pool: pool
                                )
                                results.append(row)
                                completedKeys.insert(row.key)
                                completedThisInvocation += 1
                                try checkpointStore?.write(plan: resolved.plan, results: results)
                                if let checkpointStore {
                                    CLIStderr.write(
                                        "Checkpointed \(results.count)/\(resolved.plan.expectedResultKeys.count) "
                                            + "evaluation rows at \(checkpointStore.url.path).\n"
                                    )
                                }
                                if let caseTrialLimit,
                                   completedThisInvocation >= caseTrialLimit {
                                    reachedLimit = true
                                    break armLoop
                                }
                            }
                        }
                    }
                }
            } catch {
                _ = try? await pool.unloadModel(idOrAlias: modelID)
                throw error
            }
            _ = try? await pool.unloadModel(idOrAlias: modelID)
        }
        return results
    }

    private func runCase(
        _ benchmarkCase: LoadedEvaluationCase,
        arm: EvaluationArmPlan,
        profile: EvaluationSamplingProfile,
        modelID: String,
        resolved: ResolvedEvaluationPlan,
        trial: Int,
        pool: RuntimeModelPool
    ) async throws -> EvaluationResultRow {
        var messages: [OpenAIChatMessage] = []
        if let promptSetID = arm.promptSet,
           let prompt = resolved.pack.promptSet(id: promptSetID)?.systemPrompt {
            messages.append(OpenAIChatMessage(role: "system", content: prompt))
        }
        messages.append(contentsOf: benchmarkCase.specification.messages.map {
            OpenAIChatMessage(role: $0.role.rawValue, content: $0.content)
        })
        let capture = resolved.plan.settings.logprobCapture
        let openAIRequest = OpenAIChatRequest(
            model: modelID,
            messages: messages,
            temperature: profile.temperature,
            top_p: profile.topP,
            min_p: profile.minP,
            max_tokens: benchmarkCase.specification.maxTokens ?? resolved.plan.settings.maxTokens,
            stream: false,
            logprobs: capture != .none,
            top_logprobs: resolved.plan.settings.logprobs == EvaluationLogprobMode.top.rawValue
                ? resolved.plan.settings.topLogprobs
                : nil
        )
        let adapter = arm.adapterSlot.flatMap { resolved.adapters[$0] }
        let runtimePlan: RuntimeChatPlan
        do {
            runtimePlan = try await pool.makeChatPlan(
                for: openAIRequest,
                fallbackLoraPath: adapter?.path,
                serverContextSize: resolved.plan.settings.contextSize
            )
        } catch {
            throw EvaluationRuntimeError.modelPreparation(
                model: modelID,
                arm: arm.id,
                profile: profile.id,
                trial: trial,
                caseID: benchmarkCase.specification.id,
                detail: String(describing: error)
            )
        }
        var request = runtimePlan.request
        request.temperature = profile.temperature
        request.topP = profile.topP
        request.topK = profile.topK == 0 ? nil : profile.topK
        request.minP = profile.minP
        request.reasoningEffort = profile.reasoningEffort
        request.showThinking = profile.showThinking
        request.logprobCapture = capture
        if let adapter {
            request.lora = .local(path: adapter.path, scale: arm.adapterScale)
        }
        let generationStarted = Date()
        let response: ChatResponse
        do {
            response = try await runtimePlan.lease.chat(request, progressHandler: nil)
            await runtimePlan.lease.release()
        } catch {
            await runtimePlan.lease.release()
            throw EvaluationRuntimeError.generation(
                model: modelID,
                arm: arm.id,
                profile: profile.id,
                trial: trial,
                caseID: benchmarkCase.specification.id,
                detail: String(describing: error)
            )
        }
        let generationSeconds = Date().timeIntervalSince(generationStarted)
        let visible = ChatReasoningMarkup.splitThinkBlocks(in: response.response).visibleContent
        let scoringStarted = Date()
        let scoring = try EvaluationScoring.score(
            response: visible,
            benchmarkCase: benchmarkCase,
            pack: resolved.pack,
            arm: arm,
            profileID: profile.id,
            trial: trial,
            modelID: modelID,
            adapterSHA256: adapter?.binding.sha256,
            allowExternalScorer: allowExternalScorer
        )
        let scoringSeconds = Date().timeIntervalSince(scoringStarted)
        return EvaluationResultRow(
            armID: arm.id,
            modelID: modelID,
            adapterSHA256: adapter?.binding.sha256,
            profileID: profile.id,
            trial: trial,
            caseID: benchmarkCase.specification.id,
            caseSHA256: benchmarkCase.contentSHA256,
            split: benchmarkCase.specification.split.rawValue,
            capabilityTags: benchmarkCase.specification.capabilityTags,
            passed: scoring.passed,
            score: scoring.score,
            metrics: scoring.metrics,
            failedChecks: scoring.failedChecks,
            hardFailures: scoring.hardFailures,
            generationSeconds: generationSeconds,
            scoringSeconds: scoringSeconds,
            tokensGenerated: response.tokensGenerated,
            logprobs: response.logprobs.map(FusedBenchmarkLogprobMetrics.init),
            responseSHA256: FusedBenchmarkHash.sha256(visible),
            response: logResponses ? visible : nil
        )
    }
}

enum EvaluationLogprobArgument: String, CaseIterable, ExpressibleByArgument {
    case none
    case summary
    case tokens
    case top

    var contractValue: EvaluationLogprobMode {
        EvaluationLogprobMode(rawValue: rawValue) ?? .none
    }
}

private extension EvaluationRunSettings {
    var logprobCapture: ChatLogprobCapture {
        switch EvaluationLogprobMode(rawValue: logprobs) ?? .none {
        case .none:
            return .none
        case .summary:
            return .summary
        case .tokens:
            return .tokens
        case .top:
            return .top(topLogprobs)
        }
    }
}

struct ResolvedEvaluationPlan {
    let pack: LoadedEvaluationPack
    let plan: EvaluationRunPlan
    let adapters: [String: ResolvedEvaluationAdapter]
}

enum EvaluationPlanBuilder {
    static func build(
        pack: LoadedEvaluationPack,
        modelReferences: [String: String],
        adapterReferences: [String: String],
        trials: Int?,
        maxTokens: Int?,
        contextSize: Int?,
        logprobs: EvaluationLogprobMode?,
        topLogprobs: Int?,
        logResponses: Bool,
        externalScorerAuthorized: Bool,
        fileManager: FileManager = .default
    ) throws -> ResolvedEvaluationPlan {
        let requiredModelSlots = Set(pack.manifest.arms.map(\.modelSlot))
        let requiredAdapterSlots = Set(pack.manifest.arms.compactMap(\.adapterSlot))
        try requireBindings(
            required: requiredModelSlots,
            supplied: Set(modelReferences.keys),
            kind: "model"
        )
        try requireBindings(
            required: requiredAdapterSlots,
            supplied: Set(adapterReferences.keys),
            kind: "adapter"
        )
        let models = try requiredModelSlots.sorted().map { slot in
            try EvaluationModelBinding.resolve(
                slot: slot,
                id: modelReferences[slot]!,
                fileManager: fileManager
            )
        }
        let modelBySlot = Dictionary(uniqueKeysWithValues: models.map { ($0.slot, $0) })
        var resolvedAdapters: [String: ResolvedEvaluationAdapter] = [:]
        for slot in requiredAdapterSlots.sorted() {
            guard let reference = adapterReferences[slot] else { continue }
            let modelIDs = Set(pack.manifest.arms.filter { $0.adapterSlot == slot }.compactMap {
                modelBySlot[$0.modelSlot]?.id
            })
            guard let firstModelID = modelIDs.sorted().first else {
                throw ValidationError("Adapter slot \(slot) is not referenced by an evaluation arm.")
            }
            if let spec = ManagedAdapterCatalog.spec(for: reference) {
                for modelID in modelIDs where !spec.supports(baseModelID: modelID) {
                    throw ManagedAdapterResolutionError.incompatibleBaseModel(
                        adapterID: spec.id,
                        expected: spec.compatibleBaseModelIDs.sorted().joined(separator: " or "),
                        actual: modelID
                    )
                }
            }
            guard let path = try ManagedAdapterArgumentResolver.resolve(
                reference,
                baseModelID: firstModelID
            ) else {
                throw ValidationError("Adapter slot \(slot) resolved to an empty path.")
            }
            let unresolvedURL = URL(fileURLWithPath: path).standardizedFileURL
            let values = try unresolvedURL.resourceValues(forKeys: [
                .isRegularFileKey,
                .isSymbolicLinkKey,
            ])
            let adapterURL = unresolvedURL.resolvingSymlinksInPath()
            guard values.isRegularFile == true,
                  fileManager.fileExists(atPath: adapterURL.path) else {
                throw ValidationError("Adapter for slot \(slot) is not a regular file: \(path)")
            }
            let trainingManifestURL = TextLoRATrainingManifest.url(nextTo: adapterURL)
            let training = try resolveTrainingBinding(
                manifestURL: trainingManifestURL,
                adapterURL: adapterURL,
                modelIDs: modelIDs,
                requirements: pack.manifest.adapterRequirements,
                fileManager: fileManager
            )
            let binding = EvaluationAdapterBinding(
                slot: slot,
                catalogID: ManagedAdapterCatalog.spec(for: reference)?.id,
                byteCount: try ModelArtifactPin.fileByteCount(adapterURL),
                sha256: try ModelArtifactPin.fileSHA256(adapterURL),
                training: training
            )
            resolvedAdapters[slot] = ResolvedEvaluationAdapter(
                binding: binding,
                path: adapterURL.path
            )
        }

        let profiles = pack.manifest.samplingProfiles
        let allProfileIDs = profiles.map(\.id)
        let arms = pack.manifest.arms.map { arm in
            EvaluationArmPlan(
                id: arm.id,
                modelSlot: arm.modelSlot,
                adapterSlot: arm.adapterSlot,
                adapterScale: arm.adapterScale,
                promptSet: arm.promptSet,
                profileIDs: arm.profileIDs ?? allProfileIDs
            )
        }
        let scorerPin = pack.manifest.scorer.executable.flatMap { executable in
            pack.files.first { $0.relativePath == executable }
        }
        let settings = EvaluationRunSettings(
            trials: trials ?? pack.manifest.defaults.trials,
            maxTokens: maxTokens ?? pack.manifest.defaults.maxTokens,
            contextSize: contextSize ?? pack.manifest.defaults.contextSize,
            logprobs: (logprobs ?? pack.manifest.defaults.logprobs).rawValue,
            topLogprobs: topLogprobs ?? pack.manifest.defaults.topLogprobs,
            logResponses: logResponses,
            externalScorerAuthorized: externalScorerAuthorized,
            runtimeSeedControl: false
        )
        let plan = EvaluationRunPlan(
            runner: try FusedBenchmarkRunnerIdentity.current(),
            host: FusedBenchmarkHost.current(),
            pack: EvaluationPackIdentity(
                id: pack.manifest.id,
                version: pack.manifest.version,
                manifestSHA256: pack.manifestSHA256,
                packSHA256: pack.packSHA256,
                files: pack.files,
                scorerKind: pack.manifest.scorer.kind.rawValue,
                scorerExecutableSHA256: scorerPin?.sha256
            ),
            models: models,
            adapters: resolvedAdapters.values.map(\.binding).sorted { $0.slot < $1.slot },
            prompts: pack.promptSets.map {
                EvaluationPromptBinding(id: $0.specification.id, sha256: $0.contentSHA256)
            }.sorted { $0.id < $1.id },
            arms: arms,
            profiles: profiles,
            cases: pack.cases.map {
                EvaluationCasePlan(
                    id: $0.specification.id,
                    split: $0.specification.split.rawValue,
                    capabilityTags: $0.specification.capabilityTags,
                    contentSHA256: $0.contentSHA256,
                    sourceFile: $0.sourceFile,
                    sourceLine: $0.sourceLine
                )
            },
            gates: pack.manifest.gates,
            settings: settings
        )
        return ResolvedEvaluationPlan(pack: pack, plan: plan, adapters: resolvedAdapters)
    }

    private static func resolveTrainingBinding(
        manifestURL: URL,
        adapterURL: URL,
        modelIDs: Set<String>,
        requirements: EvaluationAdapterRequirements?,
        fileManager: FileManager
    ) throws -> EvaluationTrainingBinding? {
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            guard requirements?.requireTrainingManifest != true else {
                throw ValidationError(
                    "Adapter \(adapterURL.lastPathComponent) requires its native training manifest."
                )
            }
            return nil
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: TextLoRATrainingManifest
        do {
            manifest = try EvaluationJSON.decoder.decode(TextLoRATrainingManifest.self, from: data)
        } catch {
            throw ValidationError(
                "Cannot decode training manifest beside \(adapterURL.lastPathComponent): \(error)"
            )
        }
        guard manifest.schemaVersion == TextLoRATrainingManifest.schemaVersion else {
            throw ValidationError("Unsupported native training manifest schema version.")
        }
        guard manifest.outputFile == adapterURL.lastPathComponent else {
            throw ValidationError("Training manifest output does not match the evaluated adapter.")
        }
        if requirements?.requireCompletedTraining == true,
           manifest.status != "completed" {
            throw ValidationError("Evaluated adapter training status is \(manifest.status), not completed.")
        }
        if requirements?.requireBaseModelMatch == true,
           !modelIDs.contains(manifest.baseModel) {
            throw ValidationError(
                "Training manifest base model \(manifest.baseModel) does not match slot models: "
                    + modelIDs.sorted().joined(separator: ", ")
            )
        }
        return EvaluationTrainingBinding(
            manifestSHA256: FusedBenchmarkHash.sha256(data),
            schemaVersion: manifest.schemaVersion,
            format: manifest.format,
            baseModel: manifest.baseModel,
            datasetFingerprint: manifest.training.dataset.fingerprint,
            seed: manifest.training.seed,
            status: manifest.status
        )
    }

    private static func requireBindings(
        required: Set<String>,
        supplied: Set<String>,
        kind: String
    ) throws {
        let missing = required.subtracting(supplied).sorted()
        guard missing.isEmpty else {
            throw ValidationError(
                "Missing --\(kind) bindings for slots: \(missing.joined(separator: ", "))."
            )
        }
        let unused = supplied.subtracting(required).sorted()
        guard unused.isEmpty else {
            throw ValidationError(
                "Unused --\(kind) bindings: \(unused.joined(separator: ", "))."
            )
        }
    }
}

enum EvaluationBindingParser {
    static func parse(
        _ arguments: [String],
        defaultSlot: String,
        kind: String
    ) throws -> [String: String] {
        var bindings: [String: String] = [:]
        for argument in arguments {
            let pieces = argument.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let slot: String
            let value: String
            if pieces.count == 1 {
                guard arguments.count == 1 else {
                    throw ValidationError(
                        "Unqualified --\(kind) is only valid when one binding is supplied."
                    )
                }
                slot = defaultSlot
                value = String(pieces[0])
            } else {
                slot = String(pieces[0])
                value = String(pieces[1])
            }
            guard !slot.isEmpty, !value.isEmpty else {
                throw ValidationError("--\(kind) bindings must use slot=value.")
            }
            guard bindings[slot] == nil else {
                throw ValidationError("Duplicate --\(kind) binding for slot \(slot).")
            }
            bindings[slot] = value
        }
        return bindings
    }
}

enum EvaluationRuntimeError: LocalizedError {
    case modelPreparation(model: String, arm: String, profile: String, trial: Int, caseID: String, detail: String)
    case generation(model: String, arm: String, profile: String, trial: Int, caseID: String, detail: String)

    var errorDescription: String? {
        switch self {
        case .modelPreparation(let model, let arm, let profile, let trial, let caseID, let detail):
            return Self.message(
                stage: "model preparation",
                model: model,
                arm: arm,
                profile: profile,
                trial: trial,
                caseID: caseID,
                detail: detail
            )
        case .generation(let model, let arm, let profile, let trial, let caseID, let detail):
            return Self.message(
                stage: "generation",
                model: model,
                arm: arm,
                profile: profile,
                trial: trial,
                caseID: caseID,
                detail: detail
            )
        }
    }

    private static func message(
        stage: String,
        model: String,
        arm: String,
        profile: String,
        trial: Int,
        caseID: String,
        detail: String
    ) -> String {
        "Evaluation \(stage) failed for \(model) [\(arm)/\(profile)] trial \(trial), case \(caseID); "
            + "the row remains pending: \(detail)"
    }
}

struct EvaluationPromoteCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "promote",
        abstract: "Create a content-addressed receipt for a complete, gate-passing report."
    )

    @Argument(help: "Completed evaluation report JSON path.")
    var report: String

    @Option(name: [.long], help: "Promotion receipt output path.")
    var output: String?

    @Flag(name: [.long], help: "Emit the receipt as machine-readable JSON on stdout.")
    var json: Bool = false

    func run() throws {
        let reportURL = URL(fileURLWithPath: report).standardizedFileURL
        let reportData = try Data(contentsOf: reportURL)
        let decoded = try EvaluationJSON.decoder.decode(EvaluationRunReport.self, from: reportData)
        guard decoded.schemaVersion == EvaluationRunReport.currentSchemaVersion else {
            throw ValidationError("Unsupported evaluation report schema version.")
        }
        let verified = try EvaluationRunReport(plan: decoded.plan, results: decoded.results)
        guard decoded.planSHA256 == verified.planSHA256,
              decoded.progress == verified.progress,
              decoded.gates == verified.gates,
              decoded.calibration == verified.calibration,
              decoded.promotionEligible == verified.promotionEligible else {
            throw ValidationError("Evaluation report derived fields do not match its plan and results.")
        }
        let receipt = try EvaluationPromotionReceipt(reportData: reportData, report: verified)
        let receiptData = try EvaluationJSON.prettyEncoder.encode(receipt)
        let outputURL = output.map { URL(fileURLWithPath: $0).standardizedFileURL }
            ?? reportURL.deletingPathExtension().appendingPathExtension("promotion.json")
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try receiptData.write(to: outputURL, options: .atomic)
        if json {
            print(String(decoding: receiptData, as: UTF8.self))
        } else {
            print(outputURL.path)
        }
    }
}
