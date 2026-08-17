import ArgumentParser
import MereRunRelayKit
import Foundation
import MereRunCore

enum ImageWorkflowRunPlan: Equatable {
    case trainLoRA(LoRATrainingRunPlan)
    case generate(ImageGenerationRunPlan)

    private struct Header: Decodable {
        let kind: String
    }

    static func decode(from url: URL) throws -> ImageWorkflowRunPlan {
        let data = try Data(contentsOf: url)
        let header = try JSONDecoder().decode(Header.self, from: data)
        switch header.kind {
        case LoRATrainingRunPlan.kind:
            return .trainLoRA(try LoRATrainingRunPlan.decode(from: url))
        case ImageGenerationRunPlan.kind:
            return .generate(try ImageGenerationRunPlan.decode(from: url))
        default:
            throw ValidationError("Unsupported image run plan kind '\(header.kind)'.")
        }
    }
}

struct ImageRunPlan: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "run-plan",
        abstract: "Run a saved image workflow plan."
    )

    @Argument(help: "Path to an image workflow plan JSON file.")
    var file: String

    @Flag(name: [.customLong("preflight")], help: "Preflight the saved plan without running it.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "Emit structured JSON when used with --preflight or --materialize.")
    var json: Bool = false

    @Option(
        name: [.customLong("materialize")],
        help: ArgumentHelp("Create a durable run directory from the saved plan.", valueName: "run-directory")
    )
    var materializeRunDirectory: String?

    func run() async throws {
        let plan = try loadWorkflowPlan()
        if let materializeRunDirectory {
            guard !preflight else {
                throw ValidationError("--materialize cannot be combined with --preflight.")
            }
            switch plan {
            case .trainLoRA(let trainPlan):
                let envelope = try materializeEnvelope(
                    plan: trainPlan,
                    runDirectory: materializeRunDirectory
                )
                if json {
                    print(try StructuredRunOutput.encode(envelope))
                } else {
                    print(envelope.summary)
                }
            case .generate(let generatePlan):
                let envelope = try materializeEnvelope(
                    plan: generatePlan,
                    runDirectory: materializeRunDirectory
                )
                if json {
                    print(try StructuredRunOutput.encode(envelope))
                } else {
                    print(envelope.summary)
                }
            }
            return
        }

        guard preflight || !json else {
            throw ValidationError("--json is only supported with --preflight or --materialize for image run-plan.")
        }

        switch plan {
        case .trainLoRA(let trainPlan):
            var command = try makeTrainLoRACommand(from: trainPlan)
            command.preflight = preflight
            command.json = json
            try await runInPlanCWD(trainPlan.cwd) {
                try await command.run()
            }
        case .generate(let generatePlan):
            var command = try makeGenerateCommand(from: generatePlan)
            command.preflight = preflight
            command.json = json
            try await runInPlanCWD(generatePlan.cwd) {
                try await command.run()
            }
        }
    }

    func loadPlan(fileManager: FileManager = .default) throws -> LoRATrainingRunPlan {
        let url = URL(fileURLWithPath: file).standardizedFileURL
        guard fileManager.fileExists(atPath: url.path) else {
            throw ValidationError("Plan file not found: \(url.path)")
        }
        return try LoRATrainingRunPlan.decode(from: url)
    }

    func loadWorkflowPlan(fileManager: FileManager = .default) throws -> ImageWorkflowRunPlan {
        let url = URL(fileURLWithPath: file).standardizedFileURL
        guard fileManager.fileExists(atPath: url.path) else {
            throw ValidationError("Plan file not found: \(url.path)")
        }
        return try ImageWorkflowRunPlan.decode(from: url)
    }

    func makeTrainLoRACommand(from plan: LoRATrainingRunPlan) throws -> ImageTrainLoRA {
        try plan.validateExecutable()
        return try ImageTrainLoRA.parse(plan.arguments.trainLoRAArguments())
    }

    func makeGenerateCommand(from plan: ImageGenerationRunPlan) throws -> ImageGenerate {
        try plan.validateExecutable()
        return try ImageGenerate.parse(plan.arguments.generateArguments())
    }

    private func runInPlanCWD(
        _ cwd: String,
        operation: () async throws -> Void
    ) async throws {
        let originalCWD = FileManager.default.currentDirectoryPath
        if !cwd.isEmpty {
            guard FileManager.default.changeCurrentDirectoryPath(cwd) else {
                throw ValidationError("Plan working directory not found: \(cwd)")
            }
        }
        defer {
            FileManager.default.changeCurrentDirectoryPath(originalCWD)
        }
        try await operation()
    }

    func materializeEnvelope(
        plan: LoRATrainingRunPlan,
        runDirectory: String,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) throws -> LoRATrainingRunMaterializationEnvelope {
        try plan.validateExecutable()

        let runDirectoryURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        let outputName = URL(fileURLWithPath: plan.arguments.output).lastPathComponent
        guard !outputName.isEmpty else {
            throw ValidationError("Plan output path is empty.")
        }
        let outputURL = runDirectoryURL.appendingPathComponent(outputName, isDirectory: false)
        let materializedPlan = plan.relocatingOutput(to: outputURL.path)
        let planURL = runDirectoryURL.appendingPathComponent("plan.json", isDirectory: false)
        let actionsURL = runDirectoryURL.appendingPathComponent("actions.json", isDirectory: false)
        let runManifestURL = runDirectoryURL.appendingPathComponent(LoRATrainingRunManifest.filename, isDirectory: false)
        let eventsURL = LoRATrainingRunEvent.url(nextTo: outputURL)

        try prepareRunDirectory(
            runDirectoryURL,
            targetFiles: [planURL, actionsURL, runManifestURL, eventsURL, outputURL],
            fileManager: fileManager
        )

        let actions = materializedActions(planURL: planURL, runDirectoryURL: runDirectoryURL)
        let encodedPlan = try StructuredRunOutput.encode(materializedPlan)
        try encodedPlan.write(to: planURL, atomically: true, encoding: .utf8)
        try StructuredRunOutput.encode(actions).write(to: actionsURL, atomically: true, encoding: .utf8)
        try writePlannedEvent(
            to: eventsURL,
            outputURL: outputURL,
            planURL: planURL,
            actionsURL: actionsURL,
            now: now
        )
        try writeRunManifest(
            for: materializedPlan,
            outputURL: outputURL,
            planURL: planURL,
            actionsURL: actionsURL,
            eventsURL: eventsURL,
            runManifestURL: runManifestURL,
            encodedPlan: encodedPlan,
            now: now
        )

        let result = LoRATrainingRunMaterializationResult(
            runDirectory: runDirectoryURL.path,
            planPath: planURL.path,
            actionsPath: actionsURL.path,
            runManifestPath: runManifestURL.path,
            eventsPath: eventsURL.path,
            outputPath: outputURL.path,
            originalOutputPath: plan.arguments.output
        )

        return LoRATrainingRunMaterializationEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["image", "run-plan"],
            mode: .materialize,
            status: .ok,
            createdAt: now(),
            cwd: fileManager.currentDirectoryPath,
            summary: "Materialized \(plan.kind) run at \(runDirectoryURL.path).",
            request: LoRATrainingRunMaterializationRequest(
                planFile: URL(fileURLWithPath: file).standardizedFileURL.path,
                runDirectory: runDirectoryURL.path
            ),
            result: result,
            diagnostics: [
                PreflightDiagnostic(
                    id: "output_relocated",
                    severity: .note,
                    title: "Output relocated",
                    message: "The materialized plan writes its output inside the run directory.",
                    locations: [.init(kind: "file", path: outputURL.path)]
                ),
            ],
            actions: actions
        )
    }

    func materializeEnvelope(
        plan: ImageGenerationRunPlan,
        runDirectory: String,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) throws -> ImageGenerationRunMaterializationEnvelope {
        try plan.validateExecutable()

        let runDirectoryURL = URL(fileURLWithPath: runDirectory).standardizedFileURL
        let outputName = URL(fileURLWithPath: plan.arguments.output).lastPathComponent
        guard !outputName.isEmpty else {
            throw ValidationError("Plan output path is empty.")
        }
        let outputURL = runDirectoryURL.appendingPathComponent(outputName, isDirectory: false)
        let structuredPromptOutputURL = materializedStructuredPromptOutputURL(
            for: plan,
            outputURL: outputURL,
            runDirectoryURL: runDirectoryURL
        )
        let materializedPlan = plan.relocatingOutputs(
            output: outputURL.path,
            structuredPromptOutput: structuredPromptOutputURL?.path
        )
        let planURL = runDirectoryURL.appendingPathComponent("plan.json", isDirectory: false)
        let actionsURL = runDirectoryURL.appendingPathComponent("actions.json", isDirectory: false)
        let runManifestURL = runDirectoryURL.appendingPathComponent(LoRATrainingRunManifest.filename, isDirectory: false)
        let eventsURL = LoRATrainingRunEvent.url(nextTo: outputURL)

        var targetFiles = [planURL, actionsURL, runManifestURL, eventsURL, outputURL]
        if let structuredPromptOutputURL {
            targetFiles.append(structuredPromptOutputURL)
        }
        try prepareRunDirectory(
            runDirectoryURL,
            targetFiles: targetFiles,
            fileManager: fileManager
        )

        let actions = materializedGenerationActions(planURL: planURL, runDirectoryURL: runDirectoryURL)
        let encodedPlan = try StructuredRunOutput.encode(materializedPlan)
        try encodedPlan.write(to: planURL, atomically: true, encoding: .utf8)
        try StructuredRunOutput.encode(actions).write(to: actionsURL, atomically: true, encoding: .utf8)
        try writePlannedEvent(
            to: eventsURL,
            outputURL: outputURL,
            planURL: planURL,
            actionsURL: actionsURL,
            now: now
        )
        try writeRunManifest(
            for: materializedPlan,
            outputURL: outputURL,
            structuredPromptOutputURL: structuredPromptOutputURL,
            planURL: planURL,
            actionsURL: actionsURL,
            eventsURL: eventsURL,
            runManifestURL: runManifestURL,
            encodedPlan: encodedPlan,
            now: now
        )

        let result = ImageGenerationRunMaterializationResult(
            runDirectory: runDirectoryURL.path,
            planPath: planURL.path,
            actionsPath: actionsURL.path,
            runManifestPath: runManifestURL.path,
            eventsPath: eventsURL.path,
            outputPath: outputURL.path,
            originalOutputPath: plan.arguments.output,
            structuredPromptOutputPath: structuredPromptOutputURL?.path
        )
        var diagnostics = [
            PreflightDiagnostic(
                id: "output_relocated",
                severity: .note,
                title: "Output relocated",
                message: "The materialized plan writes its output inside the run directory.",
                locations: [.init(kind: "file", path: outputURL.path)]
            ),
        ]
        if let structuredPromptOutputURL {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "structured_prompt_output_relocated",
                    severity: .note,
                    title: "Structured prompt output relocated",
                    message: "The materialized plan writes the structured prompt sidecar inside the run directory.",
                    locations: [.init(kind: "file", path: structuredPromptOutputURL.path)]
                )
            )
        }

        return ImageGenerationRunMaterializationEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["image", "run-plan"],
            mode: .materialize,
            status: .ok,
            createdAt: now(),
            cwd: fileManager.currentDirectoryPath,
            summary: "Materialized \(plan.kind) run at \(runDirectoryURL.path).",
            request: ImageGenerationRunMaterializationRequest(
                planFile: URL(fileURLWithPath: file).standardizedFileURL.path,
                runDirectory: runDirectoryURL.path
            ),
            result: result,
            diagnostics: diagnostics,
            actions: actions
        )
    }

    private func prepareRunDirectory(
        _ runDirectoryURL: URL,
        targetFiles: [URL],
        fileManager: FileManager
    ) throws {
        let targetPaths = targetFiles.map(\.standardizedFileURL.path)
        guard Set(targetPaths).count == targetPaths.count else {
            throw ValidationError("Run materialization target paths must be distinct.")
        }
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: runDirectoryURL.path, isDirectory: &isDirectory), !isDirectory.boolValue {
            throw ValidationError("--materialize path is not a directory: \(runDirectoryURL.path)")
        }
        try fileManager.createDirectory(at: runDirectoryURL, withIntermediateDirectories: true)
        for name in ["artifacts", "samples", "checkpoints", "logs"] {
            try fileManager.createDirectory(
                at: runDirectoryURL.appendingPathComponent(name, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
        for url in targetFiles where fileManager.fileExists(atPath: url.path) {
            throw ValidationError("Run materialization would overwrite existing file: \(url.path)")
        }
    }

    private func materializedStructuredPromptOutputURL(
        for plan: ImageGenerationRunPlan,
        outputURL: URL,
        runDirectoryURL: URL
    ) -> URL? {
        guard plan.arguments.structuredPromptOutput != nil else {
            return nil
        }
        let originalName = URL(fileURLWithPath: plan.arguments.structuredPromptOutput ?? "").lastPathComponent
        let fallbackName = "\(outputURL.deletingPathExtension().lastPathComponent)-prompt.json"
        let filename = originalName.isEmpty ? fallbackName : originalName
        return runDirectoryURL.appendingPathComponent(filename, isDirectory: false)
    }

    private func materializedActions(
        planURL: URL,
        runDirectoryURL: URL
    ) -> [DeclarativeAction] {
        [
            DeclarativeAction(
                id: "start-training",
                label: "Start training",
                kind: .command,
                style: .primary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "image", "run-plan", planURL.path],
                    cwd: runDirectoryURL.path,
                    commandPath: ["image", "run-plan"]
                ),
                requires: ["plan.materialized"]
            ),
            DeclarativeAction(
                id: "preflight-plan",
                label: "Preflight plan",
                kind: .command,
                style: .secondary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "image", "run-plan", planURL.path, "--preflight", "--json"],
                    cwd: runDirectoryURL.path,
                    commandPath: ["image", "run-plan"]
                )
            ),
            DeclarativeAction(
                id: "visualize-run",
                label: "Visualize run",
                kind: .command,
                style: .secondary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "image", "visualize-run", runDirectoryURL.path],
                    cwd: runDirectoryURL.path,
                    commandPath: ["image", "visualize-run"]
                )
            ),
            DeclarativeAction(
                id: "open-run-directory",
                label: "Open run directory",
                kind: .openDirectory,
                style: .link,
                path: runDirectoryURL.path
            ),
        ]
    }

    private func materializedGenerationActions(
        planURL: URL,
        runDirectoryURL: URL
    ) -> [DeclarativeAction] {
        [
            DeclarativeAction(
                id: "start-generation",
                label: "Start generation",
                kind: .command,
                style: .primary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "image", "run-plan", planURL.path],
                    cwd: runDirectoryURL.path,
                    commandPath: ["image", "run-plan"]
                ),
                requires: ["plan.materialized"]
            ),
            DeclarativeAction(
                id: "preflight-plan",
                label: "Preflight plan",
                kind: .command,
                style: .secondary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "image", "run-plan", planURL.path, "--preflight", "--json"],
                    cwd: runDirectoryURL.path,
                    commandPath: ["image", "run-plan"]
                )
            ),
            DeclarativeAction(
                id: "open-run-directory",
                label: "Open run directory",
                kind: .openDirectory,
                style: .link,
                path: runDirectoryURL.path
            ),
        ]
    }

    private func writePlannedEvent(
        to eventsURL: URL,
        outputURL: URL,
        planURL: URL,
        actionsURL: URL,
        now: () -> Date
    ) throws {
        let event = LoRATrainingRunEvent(
            sequence: 0,
            createdAt: now(),
            type: "run_planned",
            stage: "planned",
            message: "Run plan materialized.",
            step: 0,
            totalSteps: nil,
            fraction: 0,
            path: outputURL.path,
            metadata: [
                "plan_file": planURL.lastPathComponent,
                "actions_file": actionsURL.lastPathComponent,
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(event)
        data.append(contentsOf: "\n".utf8)
        try data.write(to: eventsURL, options: [.atomic])
    }

    private func writeRunManifest(
        for plan: LoRATrainingRunPlan,
        outputURL: URL,
        planURL: URL,
        actionsURL: URL,
        eventsURL: URL,
        runManifestURL: URL,
        encodedPlan: String,
        now: () -> Date
    ) throws {
        let runDirectoryURL = outputURL.deletingLastPathComponent()
        let manifest = LoRATrainingRunManifest(
            createdAt: now(),
            format: plan.kind,
            model: plan.arguments.model,
            isEdit: false,
            dataRoot: plan.arguments.data,
            dataRootRelative: LoRATrainingRunManifest.relativePath(
                from: runDirectoryURL,
                to: plan.arguments.data
            ),
            dataFingerprint: nil,
            checkpointFiles: [
                "plan": planURL.lastPathComponent,
                "actions": actionsURL.lastPathComponent,
                "events": eventsURL.lastPathComponent,
                "lora_adapter": outputURL.lastPathComponent,
            ],
            step: 0,
            totalSteps: plan.arguments.trainingSteps,
            seed: plan.arguments.seed,
            rngState: nil,
            datasetFingerprint: nil,
            configFingerprint: LoRATrainingFingerprint.sha256Hex(encodedPlan),
            configSnapshot: [
                "kind": plan.kind,
                "output": outputURL.path,
                "source_recipe": plan.arguments.sourceRecipe ?? "",
            ]
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        try data.write(to: runManifestURL, options: [.atomic])
    }

    private func writeRunManifest(
        for plan: ImageGenerationRunPlan,
        outputURL: URL,
        structuredPromptOutputURL: URL?,
        planURL: URL,
        actionsURL: URL,
        eventsURL: URL,
        runManifestURL: URL,
        encodedPlan: String,
        now: () -> Date
    ) throws {
        var checkpointFiles = [
            "plan": planURL.lastPathComponent,
            "actions": actionsURL.lastPathComponent,
            "events": eventsURL.lastPathComponent,
            "output_image": outputURL.lastPathComponent,
        ]
        if let structuredPromptOutputURL {
            checkpointFiles["structured_prompt"] = structuredPromptOutputURL.lastPathComponent
        }

        var configSnapshot = [
            "kind": plan.kind,
            "output": outputURL.path,
            "prompt": plan.arguments.prompt,
            "width": String(plan.arguments.width),
            "height": String(plan.arguments.height),
            "input_mode": plan.resolved.inputMode,
            "structured_prompt": String(plan.arguments.structuredPrompt),
        ]
        if let input = plan.arguments.input {
            configSnapshot["input"] = input
        }
        if !plan.arguments.referenceImages.isEmpty {
            configSnapshot["reference_images"] = plan.arguments.referenceImages.joined(separator: "\n")
        }
        if let lora = plan.arguments.lora {
            configSnapshot["lora"] = lora
            configSnapshot["lora_scale"] = String(plan.arguments.loraScale)
        }

        let manifest = LoRATrainingRunManifest(
            createdAt: now(),
            format: plan.kind,
            model: plan.arguments.model,
            isEdit: plan.arguments.input != nil || !plan.arguments.referenceImages.isEmpty,
            dataRoot: nil,
            dataRootRelative: nil,
            dataFingerprint: nil,
            checkpointFiles: checkpointFiles,
            step: 0,
            totalSteps: plan.resolved.effectiveSteps ?? plan.arguments.steps ?? 0,
            seed: plan.arguments.seed ?? 0,
            rngState: nil,
            datasetFingerprint: nil,
            configFingerprint: LoRATrainingFingerprint.sha256Hex(encodedPlan),
            configSnapshot: configSnapshot
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(manifest)
        try data.write(to: runManifestURL, options: [.atomic])
    }
}
