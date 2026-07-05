import Foundation
import MereRunCore

struct SFXVideoPreflightInput {
    let prompt: String
    let inputURL: URL
    let outputURL: URL
    let model: String
    let synchformerModel: String
    let durationSeconds: Float
    let steps: Int?
    let guidanceScale: Float?
    let seed: UInt64?
    let renoise: String?
    let syncBatchSize: Int
    let generationArgv: [String]
    let cwd: String
}

struct SFXVideoPreflightRequest: Codable, Equatable {
    let prompt: String
    let input: String
    let output: String
    let model: String
    let synchformerModel: String
    let durationSeconds: Float
    let steps: Int?
    let guidanceScale: Float?
    let seed: UInt64?
    let renoise: String?
    let syncBatchSize: Int

    enum CodingKeys: String, CodingKey {
        case prompt
        case input
        case output
        case model
        case synchformerModel = "synchformer_model"
        case durationSeconds = "duration_seconds"
        case steps
        case guidanceScale = "guidance_scale"
        case seed
        case renoise
        case syncBatchSize = "sync_batch_size"
    }
}

struct SFXVideoPreflightResult: Codable, Equatable {
    let input: SFXVideoInputPreflightSummary
    let output: SFXVideoOutputPreflightSummary
    let model: SFXVideoModelPreflightSummary
    let synchformer: SFXVideoModelPreflightSummary?
    let plan: SFXVideoPlanPreflightSummary
}

struct SFXVideoInputPreflightSummary: Codable, Equatable {
    let requested: String
    let path: String
    let exists: Bool
    let isDirectory: Bool
    let kind: String
    let requiresSynchformer: Bool

    enum CodingKeys: String, CodingKey {
        case requested
        case path
        case exists
        case isDirectory = "is_directory"
        case kind
        case requiresSynchformer = "requires_synchformer"
    }
}

struct SFXVideoOutputPreflightSummary: Codable, Equatable {
    let path: String
    let parentDirectory: String
    let parentExists: Bool
    let parentWillBeCreated: Bool
    let exists: Bool
    let expectedExtension: String
    let extensionValid: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentDirectory = "parent_directory"
        case parentExists = "parent_exists"
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case expectedExtension = "expected_extension"
        case extensionValid = "extension_valid"
    }
}

struct SFXVideoModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let id: String?
    let variant: String?
    let supportedForVideo: Bool?
    let missingFiles: [String]
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case id
        case variant
        case supportedForVideo = "supported_for_video"
        case missingFiles = "missing_files"
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

struct SFXVideoPlanPreflightSummary: Codable, Equatable {
    let inputKind: String
    let durationSeconds: Float
    let requestedSteps: Int?
    let effectiveSteps: Int
    let requestedGuidanceScale: Float?
    let effectiveGuidanceScale: Float
    let seed: UInt64?
    let renoiseSchedule: [Float]
    let syncBatchSize: Int
    let sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case inputKind = "input_kind"
        case durationSeconds = "duration_seconds"
        case requestedSteps = "requested_steps"
        case effectiveSteps = "effective_steps"
        case requestedGuidanceScale = "requested_guidance_scale"
        case effectiveGuidanceScale = "effective_guidance_scale"
        case seed
        case renoiseSchedule = "renoise_schedule"
        case syncBatchSize = "sync_batch_size"
        case sampleRate = "sample_rate"
    }
}

typealias SFXVideoPreflightEnvelope = StructuredRunEnvelope<
    SFXVideoPreflightRequest,
    SFXVideoPreflightResult
>

struct SFXVideoPreflightAnalyzer {
    let input: SFXVideoPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: SFXVideoPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> SFXVideoPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        validateStaticOptions(diagnostics: &diagnostics)
        let inputSummary = inputSummary(diagnostics: &diagnostics)
        let output = outputSummary(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let plan = planSummary(model: model, diagnostics: &diagnostics)
        let synchformer = inputSummary.requiresSynchformer
            ? synchformerSummary(diagnostics: &diagnostics)
            : nil
        let status = StructuredRunOutput.status(for: diagnostics)

        return SFXVideoPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["sfx", "video", "generate"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: SFXVideoPreflightResult(
                input: inputSummary,
                output: output,
                model: model,
                synchformer: synchformer,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions(status: status, inputSummary: inputSummary, output: output, model: model, synchformer: synchformer)
        )
    }

    private func request() -> SFXVideoPreflightRequest {
        SFXVideoPreflightRequest(
            prompt: input.prompt,
            input: input.inputURL.path,
            output: input.outputURL.path,
            model: input.model,
            synchformerModel: input.synchformerModel,
            durationSeconds: input.durationSeconds,
            steps: input.steps,
            guidanceScale: input.guidanceScale,
            seed: input.seed,
            renoise: input.renoise,
            syncBatchSize: input.syncBatchSize
        )
    }

    private func validateStaticOptions(diagnostics: inout [PreflightDiagnostic]) {
        if input.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_empty",
                    severity: .blocker,
                    title: "Prompt is empty",
                    message: "Provide a non-empty sound-effect prompt."
                )
            )
        }
        if input.durationSeconds <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_invalid",
                    severity: .blocker,
                    title: "Duration is invalid",
                    message: "--duration must be > 0."
                )
            )
        }
        if let steps = input.steps, steps <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "steps_invalid",
                    severity: .blocker,
                    title: "Step count is invalid",
                    message: "--steps must be >= 1."
                )
            )
        }
        if let guidanceScale = input.guidanceScale, guidanceScale < 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "cfg_invalid",
                    severity: .blocker,
                    title: "Guidance scale is invalid",
                    message: "--cfg must be >= 0."
                )
            )
        }
        if input.syncBatchSize <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "sync_batch_size_invalid",
                    severity: .blocker,
                    title: "Sync batch size is invalid",
                    message: "--sync-batch-size must be >= 1."
                )
            )
        }
    }

    private func inputSummary(diagnostics: inout [PreflightDiagnostic]) -> SFXVideoInputPreflightSummary {
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: input.inputURL.path, isDirectory: &isDirectory)
        let kind = input.inputURL.pathExtension.lowercased() == "npy" ? "features" : "video"
        if !exists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "input_missing",
                    severity: .blocker,
                    title: "Input missing",
                    message: "Input not found: \(input.inputURL.path)",
                    locations: [.init(kind: "file", path: input.inputURL.path)]
                )
            )
        } else if isDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "input_is_directory",
                    severity: .blocker,
                    title: "Input is a directory",
                    message: "Input path is a directory: \(input.inputURL.path)",
                    locations: [.init(kind: "directory", path: input.inputURL.path)]
                )
            )
        }
        if kind == "features" {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "feature_shape_unverified",
                    severity: .note,
                    title: "Feature shape is checked at runtime",
                    message: ".npy feature inputs are not loaded during preflight; runtime still validates shape [frames, 768] or [1, frames, 768]."
                )
            )
        }
        return SFXVideoInputPreflightSummary(
            requested: input.inputURL.path,
            path: input.inputURL.path,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue,
            kind: kind,
            requiresSynchformer: kind != "features"
        )
    }

    private func outputSummary(diagnostics: inout [PreflightDiagnostic]) -> SFXVideoOutputPreflightSummary {
        let parent = input.outputURL.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: input.outputURL.path)
        let extensionValid = input.outputURL.pathExtension.lowercased() == "wav"

        if parentExists, !parentIsDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_parent_not_directory",
                    severity: .blocker,
                    title: "Output parent is not a directory",
                    message: "Output parent is not a directory: \(parent.path)",
                    locations: [.init(kind: "file", path: parent.path)]
                )
            )
        }
        if outputExists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_exists",
                    severity: .warning,
                    title: "Output exists",
                    message: "Output already exists and may be overwritten: \(input.outputURL.path)",
                    locations: [.init(kind: "file", path: input.outputURL.path)]
                )
            )
        }
        if !extensionValid {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_extension_unusual",
                    severity: .warning,
                    title: "Output extension is not WAV",
                    message: "SFX video generation writes WAV data; use a .wav output path for clarity.",
                    locations: [.init(kind: "file", path: input.outputURL.path)]
                )
            )
        }

        return SFXVideoOutputPreflightSummary(
            path: input.outputURL.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            expectedExtension: "wav",
            extensionValid: extensionValid
        )
    }

    private func modelSummary(diagnostics: inout [PreflightDiagnostic]) -> SFXVideoModelPreflightSummary {
        wooshModelSummary(
            requested: input.model,
            role: "model",
            missingDiagnosticID: "model_missing",
            unsupportedDiagnosticID: "model_not_video_to_audio",
            suggestedActionID: "pull-model",
            requireVideoVariant: true,
            diagnostics: &diagnostics
        )
    }

    private func synchformerSummary(diagnostics: inout [PreflightDiagnostic]) -> SFXVideoModelPreflightSummary {
        let requested = input.synchformerModel.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveRequested = requested.isEmpty ? ModelResolver.ModelID.wooshSynchformer.rawValue : requested
        let explicitURL = URL(fileURLWithPath: effectiveRequested).standardizedFileURL
        if fileManager.fileExists(atPath: explicitURL.path) {
            let resources = WooshSynchformerResources(rootURL: explicitURL)
            let missing = resources.missingFiles(fileManager: fileManager)
            if !missing.isEmpty {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "synchformer_missing_files",
                        severity: .blocker,
                        title: "Synchformer files missing",
                        message: "Synchformer resource is missing \(missing.count) required file(s).",
                        locations: missing.map { .init(kind: "file", path: $0.path) }
                    )
                )
            }
            return modelResult(
                requested: effectiveRequested,
                kind: "local_path",
                installed: missing.isEmpty,
                path: explicitURL.path,
                missingFiles: missing.map(\.path)
            )
        }

        guard let modelID = ModelResolver.ModelID(rawValue: effectiveRequested),
              let spec = ManagedModelCatalog.spec(for: effectiveRequested) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "synchformer_unknown",
                    severity: .blocker,
                    title: "Unknown Synchformer model",
                    message: "Synchformer path not found and not a known model id: \(effectiveRequested)."
                )
            )
            return modelResult(requested: effectiveRequested, kind: "unknown", installed: false)
        }

        if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
            let resources = WooshSynchformerResources(rootURL: resolution.rootURL)
            let missing = resources.missingFiles(fileManager: fileManager)
            if !missing.isEmpty {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "synchformer_missing_files",
                        severity: .blocker,
                        title: "Synchformer files missing",
                        message: "Installed Synchformer resource is missing \(missing.count) required file(s).",
                        locations: missing.map { .init(kind: "file", path: $0.path) }
                    )
                )
            }
            return modelResult(
                requested: effectiveRequested,
                kind: "managed_model",
                installed: missing.isEmpty,
                path: resolution.rootURL.path,
                id: modelID.rawValue,
                missingFiles: missing.map(\.path),
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "synchformer_missing",
                severity: .blocker,
                title: "Synchformer model missing",
                message: "Raw video input requires \(effectiveRequested). Pull it before generation, or pass precomputed .npy features.",
                suggestedActionIDs: ["pull-synchformer"]
            )
        )
        return modelResult(
            requested: effectiveRequested,
            kind: "managed_model",
            installed: false,
            id: modelID.rawValue,
            upstreamRepoID: spec.upstreamRepoId,
            estimatedDownloadBytes: spec.estimatedDownloadBytes
        )
    }

    private func wooshModelSummary(
        requested: String,
        role: String,
        missingDiagnosticID: String,
        unsupportedDiagnosticID: String,
        suggestedActionID: String,
        requireVideoVariant: Bool,
        diagnostics: inout [PreflightDiagnostic]
    ) -> SFXVideoModelPreflightSummary {
        let requested = requested.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveRequested = requested.isEmpty ? ModelResolver.ModelID.wooshDVFlow8s.rawValue : requested
        let explicitURL = URL(fileURLWithPath: effectiveRequested).standardizedFileURL
        if fileManager.fileExists(atPath: explicitURL.path) {
            return localWooshSummary(
                requested: effectiveRequested,
                root: explicitURL,
                role: role,
                unsupportedDiagnosticID: unsupportedDiagnosticID,
                requireVideoVariant: requireVideoVariant,
                diagnostics: &diagnostics
            )
        }

        guard let modelID = ModelResolver.ModelID(rawValue: effectiveRequested),
              let spec = ManagedModelCatalog.spec(for: effectiveRequested) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "\(role)_unknown",
                    severity: .blocker,
                    title: "Unknown \(role)",
                    message: "\(role) path not found and not a known model id: \(effectiveRequested)."
                )
            )
            return modelResult(requested: effectiveRequested, kind: "unknown", installed: false)
        }

        let requestedVariant = WooshVariant.resolve(model: effectiveRequested, fileManager: fileManager)
        if requireVideoVariant, !isVideoVariant(requestedVariant) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: unsupportedDiagnosticID,
                    severity: .blocker,
                    title: "Model is not a video-to-audio Woosh model",
                    message: "sfx video generate requires \(ModelResolver.ModelID.wooshVFlow8s.rawValue) or \(ModelResolver.ModelID.wooshDVFlow8s.rawValue)."
                )
            )
        }

        if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
            let checkpointsRoot = WooshResources.normalizeRoot(resolution.rootURL, fileManager: fileManager)
            return installedWooshSummary(
                requested: effectiveRequested,
                kind: "managed_model",
                root: checkpointsRoot,
                id: modelID.rawValue,
                spec: spec,
                unsupportedDiagnosticID: unsupportedDiagnosticID,
                requireVideoVariant: requireVideoVariant,
                diagnostics: &diagnostics
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: missingDiagnosticID,
                severity: .blocker,
                title: "\(role.capitalized) missing",
                message: "Model \(effectiveRequested) is not installed. Pull it before generation.",
                suggestedActionIDs: [suggestedActionID]
            )
        )
        return modelResult(
            requested: effectiveRequested,
            kind: "managed_model",
            installed: false,
            id: modelID.rawValue,
            variant: requestedVariant?.rawValue,
            supportedForVideo: isVideoVariant(requestedVariant),
            upstreamRepoID: spec.upstreamRepoId,
            estimatedDownloadBytes: spec.estimatedDownloadBytes
        )
    }

    private func localWooshSummary(
        requested: String,
        root: URL,
        role: String,
        unsupportedDiagnosticID: String,
        requireVideoVariant: Bool,
        diagnostics: inout [PreflightDiagnostic]
    ) -> SFXVideoModelPreflightSummary {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "\(role)_path_not_directory",
                    severity: .blocker,
                    title: "\(role.capitalized) path is not a directory",
                    message: "\(role.capitalized) path is not a directory: \(root.path)",
                    locations: [.init(kind: "file", path: root.path)]
                )
            )
            return modelResult(requested: requested, kind: "local_path", installed: false, path: root.path)
        }
        let checkpointsRoot = WooshResources.normalizeRoot(root, fileManager: fileManager)
        return installedWooshSummary(
            requested: requested,
            kind: "local_path",
            root: checkpointsRoot,
            id: nil,
            spec: nil,
            unsupportedDiagnosticID: unsupportedDiagnosticID,
            requireVideoVariant: requireVideoVariant,
            diagnostics: &diagnostics
        )
    }

    private func installedWooshSummary(
        requested: String,
        kind: String,
        root: URL,
        id: String?,
        spec: ManagedModelSpec?,
        unsupportedDiagnosticID: String,
        requireVideoVariant: Bool,
        diagnostics: inout [PreflightDiagnostic]
    ) -> SFXVideoModelPreflightSummary {
        guard let variant = WooshVariant.resolve(model: requested, rootURL: root, fileManager: fileManager) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_variant_unknown",
                    severity: .blocker,
                    title: "Woosh variant not found",
                    message: "Woosh checkpoints were not found under \(root.path).",
                    locations: [.init(kind: "directory", path: root.path)]
                )
            )
            return modelResult(
                requested: requested,
                kind: kind,
                installed: false,
                path: root.path,
                id: id,
                upstreamRepoID: spec?.upstreamRepoId,
                estimatedDownloadBytes: spec?.estimatedDownloadBytes
            )
        }

        if requireVideoVariant, !isVideoVariant(variant) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: unsupportedDiagnosticID,
                    severity: .blocker,
                    title: "Model is not a video-to-audio Woosh model",
                    message: "Found \(variant.rawValue), but sfx video generate requires \(WooshVariant.vflow8s.rawValue) or \(WooshVariant.dvflow8s.rawValue).",
                    locations: [.init(kind: "directory", path: root.path)]
                )
            )
        }

        let resources = WooshModelResources(checkpointsRootURL: root, variant: variant)
        let missing = resources.missingFiles(fileManager: fileManager)
        if !missing.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_missing_files",
                    severity: .blocker,
                    title: "Model files missing",
                    message: "Woosh model is missing \(missing.count) required file(s).",
                    locations: missing.map { .init(kind: "file", path: $0.path) }
                )
            )
        }

        return modelResult(
            requested: requested,
            kind: kind,
            installed: missing.isEmpty,
            path: root.path,
            id: id,
            variant: variant.rawValue,
            supportedForVideo: isVideoVariant(variant),
            missingFiles: missing.map(\.path),
            upstreamRepoID: spec?.upstreamRepoId,
            estimatedDownloadBytes: spec?.estimatedDownloadBytes
        )
    }

    private func isVideoVariant(_ variant: WooshVariant?) -> Bool {
        variant == .vflow8s || variant == .dvflow8s
    }

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        id: String? = nil,
        variant: String? = nil,
        supportedForVideo: Bool? = nil,
        missingFiles: [String] = [],
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil
    ) -> SFXVideoModelPreflightSummary {
        SFXVideoModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            id: id,
            variant: variant,
            supportedForVideo: supportedForVideo,
            missingFiles: missingFiles,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes
        )
    }

    private func planSummary(
        model: SFXVideoModelPreflightSummary,
        diagnostics: inout [PreflightDiagnostic]
    ) -> SFXVideoPlanPreflightSummary {
        let variant = model.variant.flatMap(WooshVariant.init(rawValue:))
            ?? WooshVariant.resolve(model: input.model, fileManager: fileManager)
            ?? .dvflow8s
        let stepCount = input.steps ?? variant.defaultSteps
        let cfg = input.guidanceScale ?? (variant == .dvflow8s ? 3.0 : variant.defaultGuidanceScale)
        let renoiseSchedule = parseRenoiseSchedule(steps: stepCount, diagnostics: &diagnostics)

        return SFXVideoPlanPreflightSummary(
            inputKind: input.inputURL.pathExtension.lowercased() == "npy" ? "features" : "video",
            durationSeconds: input.durationSeconds,
            requestedSteps: input.steps,
            effectiveSteps: stepCount,
            requestedGuidanceScale: input.guidanceScale,
            effectiveGuidanceScale: cfg,
            seed: input.seed,
            renoiseSchedule: renoiseSchedule,
            syncBatchSize: input.syncBatchSize,
            sampleRate: WooshResources.sampleRate
        )
    }

    private func parseRenoiseSchedule(
        steps: Int,
        diagnostics: inout [PreflightDiagnostic]
    ) -> [Float] {
        guard let renoise = input.renoise, !renoise.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }
        let values = renoise
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !values.isEmpty else {
            return []
        }
        var parsed: [Float] = []
        for value in values {
            guard let number = Float(value) else {
                diagnostics.append(renoiseDiagnostic("--renoise must be a float or comma-separated floats."))
                return []
            }
            guard (0...1).contains(number) else {
                diagnostics.append(renoiseDiagnostic("--renoise values must be between 0 and 1."))
                return []
            }
            parsed.append(number)
        }
        if parsed.count != 1 && parsed.count != steps {
            diagnostics.append(renoiseDiagnostic("--renoise must contain one value or exactly --steps values."))
            return parsed
        }
        return parsed
    }

    private func renoiseDiagnostic(_ message: String) -> PreflightDiagnostic {
        PreflightDiagnostic(
            id: "renoise_invalid",
            severity: .blocker,
            title: "Renoise schedule is invalid",
            message: message
        )
    }

    private func actions(
        status: StructuredRunStatus,
        inputSummary: SFXVideoInputPreflightSummary,
        output: SFXVideoOutputPreflightSummary,
        model: SFXVideoModelPreflightSummary,
        synchformer: SFXVideoModelPreflightSummary?
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-sfx-video-generation",
                label: "Start SFX video generation",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.generationArgv,
                    cwd: input.cwd,
                    commandPath: ["sfx", "video", "generate"]
                ),
                requires: ["preflight.passed"]
            )
        )

        if model.kind == "managed_model", !model.installed, model.supportedForVideo != false {
            actions.append(pullAction(id: "pull-model", label: "Pull SFX model", target: model.requested))
        }
        if let synchformer, synchformer.kind == "managed_model", !synchformer.installed {
            actions.append(pullAction(id: "pull-synchformer", label: "Pull Synchformer", target: synchformer.requested))
        }

        actions.append(
            DeclarativeAction(
                id: "reveal-input",
                label: "Reveal input",
                kind: .revealFile,
                style: .link,
                enabled: inputSummary.exists && !inputSummary.isDirectory,
                path: inputSummary.path
            )
        )
        actions.append(
            DeclarativeAction(
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: output.parentExists,
                disabledReason: output.parentExists ? nil : "Output directory will be created when generation starts.",
                path: output.parentDirectory
            )
        )
        return actions
    }

    private func pullAction(id: String, label: String, target: String) -> DeclarativeAction {
        DeclarativeAction(
            id: id,
            label: label,
            kind: .command,
            style: .secondary,
            command: DeclarativeCommand(
                argv: ["mere.run", "model", "pull", target],
                cwd: input.cwd,
                commandPath: ["model", "pull"]
            )
        )
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        switch status {
        case .ok:
            return "SFX video generation preflight passed."
        case .warning:
            return "SFX video generation preflight found \(diagnostics.count) warning(s) or note(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "SFX video generation preflight blocked by \(blockers) issue(s)."
        default:
            return "SFX video generation preflight status: \(status.rawValue)."
        }
    }
}
