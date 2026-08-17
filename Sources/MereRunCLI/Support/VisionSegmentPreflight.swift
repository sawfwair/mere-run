import Foundation
import MereRunRelayKit
import MereRunCore

struct VisionSegmentPreflightInput {
    let imageURL: URL
    let outputImageURL: URL
    let outputJSONURL: URL
    let maskOutputDirectoryURL: URL?
    let prompt: [String]
    let box: [String]
    let point: [String]
    let model: String?
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let multimask: Bool
    let segmentArgv: [String]
    let cwd: String
}

struct VisionSegmentPreflightRequest: Codable, Equatable {
    let image: String
    let output: String
    let jsonOutput: String
    let maskOutputDir: String?
    let prompt: [String]
    let box: [String]
    let point: [String]
    let model: String?
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let multimask: Bool

    enum CodingKeys: String, CodingKey {
        case image
        case output
        case jsonOutput = "json_output"
        case maskOutputDir = "mask_output_dir"
        case prompt
        case box
        case point
        case model
        case threshold
        case resolution
        case showBoxes = "show_boxes"
        case multimask
    }
}

struct VisionSegmentPreflightResult: Codable, Equatable {
    let model: VisionTrackModelPreflightSummary
    let image: VisionTrackPathPreflightSummary
    let prompts: VisionTrackPromptPreflightSummary
    let outputs: VisionSegmentOutputsPreflightSummary
    let plan: VisionSegmentPlanPreflightSummary
}

struct VisionSegmentOutputsPreflightSummary: Codable, Equatable {
    let annotatedImage: VisionTrackOutputPreflightSummary
    let segmentationJSON: VisionTrackOutputPreflightSummary
    let maskDirectory: VisionTrackDirectoryPreflightSummary?

    enum CodingKeys: String, CodingKey {
        case annotatedImage = "annotated_image"
        case segmentationJSON = "segmentation_json"
        case maskDirectory = "mask_directory"
    }
}

struct VisionSegmentPlanPreflightSummary: Codable, Equatable {
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let multimask: Bool

    enum CodingKeys: String, CodingKey {
        case threshold
        case resolution
        case showBoxes = "show_boxes"
        case multimask
    }
}

typealias VisionSegmentPreflightEnvelope = StructuredRunEnvelope<
    VisionSegmentPreflightRequest,
    VisionSegmentPreflightResult
>

struct VisionSegmentPreflightAnalyzer {
    let input: VisionSegmentPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: VisionSegmentPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> VisionSegmentPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let image = imageSummary(diagnostics: &diagnostics)
        validateStaticOptions(diagnostics: &diagnostics)
        let prompts = promptSummary(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let outputs = outputSummaries(diagnostics: &diagnostics)
        let plan = VisionSegmentPlanPreflightSummary(
            threshold: input.threshold,
            resolution: input.resolution,
            showBoxes: input.showBoxes,
            multimask: input.multimask
        )
        let status = StructuredRunOutput.status(for: diagnostics)

        return VisionSegmentPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["vision", "segment"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: VisionSegmentPreflightResult(
                model: model,
                image: image,
                prompts: prompts,
                outputs: outputs,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions(status: status, model: model, image: image, outputs: outputs)
        )
    }

    private func request() -> VisionSegmentPreflightRequest {
        VisionSegmentPreflightRequest(
            image: input.imageURL.path,
            output: input.outputImageURL.path,
            jsonOutput: input.outputJSONURL.path,
            maskOutputDir: input.maskOutputDirectoryURL?.path,
            prompt: input.prompt,
            box: input.box,
            point: input.point,
            model: input.model,
            threshold: input.threshold,
            resolution: input.resolution,
            showBoxes: input.showBoxes,
            multimask: input.multimask
        )
    }

    private func imageSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionTrackPathPreflightSummary {
        let summary = pathSummary(requested: input.imageURL.path)
        if !summary.exists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "image_missing",
                    severity: .blocker,
                    title: "Image missing",
                    message: "Image not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                )
            )
        } else if summary.isDirectory {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "image_is_directory",
                    severity: .blocker,
                    title: "Image path is a directory",
                    message: "Image path is a directory: \(summary.path)",
                    locations: [.init(kind: "directory", path: summary.path)]
                )
            )
        }
        return summary
    }

    private func validateStaticOptions(diagnostics: inout [PreflightDiagnostic]) {
        if input.prompt.isEmpty, input.box.isEmpty, input.point.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_set_empty",
                    severity: .blocker,
                    title: "Prompt set is empty",
                    message: "Provide at least one --prompt, --box, or --point value."
                )
            )
        }
        if !(0.0...1.0).contains(input.threshold) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "threshold_invalid",
                    severity: .blocker,
                    title: "Threshold is invalid",
                    message: "--threshold must be between 0 and 1."
                )
            )
        }
        if input.resolution <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "resolution_invalid",
                    severity: .blocker,
                    title: "Resolution is invalid",
                    message: "--resolution must be greater than 0."
                )
            )
        }
    }

    private func promptSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionTrackPromptPreflightSummary {
        let normalizedTextPrompts = VisionSegment.normalizedTextPrompts(input.prompt)
        do {
            _ = try input.box.map(VisionSegment.parseBoxPrompt)
            _ = try input.point.map(VisionSegment.parsePointPrompt)
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_parse_failed",
                    severity: .blocker,
                    title: "Prompt parse failed",
                    message: error.localizedDescription
                )
            )
        }
        if normalizedTextPrompts.isEmpty,
           input.box.isEmpty,
           input.point.isEmpty,
           !input.prompt.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "text_prompts_empty_after_normalization",
                    severity: .blocker,
                    title: "Text prompts are empty",
                    message: "Text prompts must include object words after normalization."
                )
            )
        }

        return VisionTrackPromptPreflightSummary(
            textPrompts: input.prompt,
            normalizedTextPrompts: normalizedTextPrompts,
            boxPrompts: input.box,
            pointPrompts: input.point,
            textCount: normalizedTextPrompts.count,
            boxCount: input.box.count,
            pointCount: input.point.count,
            totalCount: normalizedTextPrompts.count + input.box.count + input.point.count
        )
    }

    private func modelSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionTrackModelPreflightSummary {
        let requested = input.model?.trimmingCharacters(in: .whitespacesAndNewlines)
        let effectiveRequested = requested?.isEmpty == false
            ? requested!
            : VisionSegment.defaultManagedModelID.rawValue

        if let requested, !requested.isEmpty {
            let url = URL(fileURLWithPath: requested).standardizedFileURL
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
                if !isDirectory.boolValue {
                    diagnostics.append(
                        PreflightDiagnostic(
                            id: "model_path_not_directory",
                            severity: .blocker,
                            title: "Model path is not a directory",
                            message: "Model path is not a directory: \(url.path)",
                            locations: [.init(kind: "file", path: url.path)]
                        )
                    )
                    return modelResult(
                        requested: requested,
                        kind: "local_path",
                        installed: false,
                        path: url.path
                    )
                }
                return modelResult(
                    requested: requested,
                    kind: "local_path",
                    installed: true,
                    path: url.path
                )
            }
        }

        guard let modelID = ModelResolver.ModelID(rawValue: effectiveRequested),
              let spec = ManagedModelCatalog.spec(for: effectiveRequested) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_unknown",
                    severity: .blocker,
                    title: "Unknown model",
                    message: "Model path not found and not a known model id: \(effectiveRequested)."
                )
            )
            return modelResult(requested: effectiveRequested, kind: "unknown", installed: false)
        }

        if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
            return modelResult(
                requested: effectiveRequested,
                kind: "managed_model",
                installed: true,
                path: resolution.rootURL.path,
                id: modelID.rawValue,
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "model_missing",
                severity: .blocker,
                title: "Model missing",
                message: "Model \(effectiveRequested) is not installed. Pull it before segmentation.",
                suggestedActionIDs: ["pull-model"]
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

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        id: String? = nil,
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil
    ) -> VisionTrackModelPreflightSummary {
        VisionTrackModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            id: id,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes
        )
    }

    private func outputSummaries(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionSegmentOutputsPreflightSummary {
        VisionSegmentOutputsPreflightSummary(
            annotatedImage: outputSummary(
                url: input.outputImageURL,
                expectedExtension: "png",
                role: "annotated image",
                diagnosticPrefix: "output_image",
                diagnostics: &diagnostics
            ),
            segmentationJSON: outputSummary(
                url: input.outputJSONURL,
                expectedExtension: "json",
                role: "segmentation JSON",
                diagnosticPrefix: "json_output",
                diagnostics: &diagnostics
            ),
            maskDirectory: input.maskOutputDirectoryURL.map {
                maskDirectorySummary(url: $0, diagnostics: &diagnostics)
            }
        )
    }

    private func outputSummary(
        url: URL,
        expectedExtension: String,
        role: String,
        diagnosticPrefix: String,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionTrackOutputPreflightSummary {
        let parent = url.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: url.path)
        let extensionValid = url.pathExtension.lowercased() == expectedExtension

        if parentExists, !parentIsDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "\(diagnosticPrefix)_parent_not_directory",
                    severity: .blocker,
                    title: "Output parent is not a directory",
                    message: "\(role) parent is not a directory: \(parent.path)",
                    locations: [.init(kind: "file", path: parent.path)]
                )
            )
        }
        if outputExists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "\(diagnosticPrefix)_exists",
                    severity: .warning,
                    title: "Output exists",
                    message: "\(role) already exists and may be overwritten: \(url.path)",
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
        }
        if !extensionValid {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "\(diagnosticPrefix)_extension_unusual",
                    severity: .warning,
                    title: "Output extension is unusual",
                    message: "\(role) should use a .\(expectedExtension) output path for clarity.",
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
        }

        return VisionTrackOutputPreflightSummary(
            path: url.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            expectedExtension: expectedExtension,
            extensionValid: extensionValid
        )
    }

    private func maskDirectorySummary(
        url: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VisionTrackDirectoryPreflightSummary {
        let parent = url.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)

        if parentExists, !parentIsDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "mask_output_parent_not_directory",
                    severity: .blocker,
                    title: "Mask output parent is not a directory",
                    message: "Mask output parent is not a directory: \(parent.path)",
                    locations: [.init(kind: "file", path: parent.path)]
                )
            )
        }
        if exists, !isDirectory.boolValue {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "mask_output_not_directory",
                    severity: .blocker,
                    title: "Mask output path is not a directory",
                    message: "Mask output path is not a directory: \(url.path)",
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
        }

        return VisionTrackDirectoryPreflightSummary(
            path: url.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func pathSummary(requested: String) -> VisionTrackPathPreflightSummary {
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return VisionTrackPathPreflightSummary(
            requested: requested,
            path: url.path,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func actions(
        status: StructuredRunStatus,
        model: VisionTrackModelPreflightSummary,
        image: VisionTrackPathPreflightSummary,
        outputs: VisionSegmentOutputsPreflightSummary
    ) -> [DeclarativeAction] {
        var actions = [
            DeclarativeAction(
                id: "start-segmentation",
                label: "Start segmentation",
                kind: .command,
                style: .primary,
                enabled: status != .blocked,
                disabledReason: status == .blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.segmentArgv,
                    cwd: input.cwd,
                    commandPath: ["vision", "segment"]
                ),
                requires: ["preflight.passed"]
            )
        ]

        if model.kind == "managed_model", !model.installed {
            actions.append(
                DeclarativeAction(
                    id: "pull-model",
                    label: "Pull model",
                    kind: .command,
                    style: .secondary,
                    command: DeclarativeCommand(
                        argv: ["mere.run", "model", "pull", model.requested],
                        cwd: input.cwd,
                        commandPath: ["model", "pull"]
                    )
                )
            )
        }

        actions.append(
            DeclarativeAction(
                id: "reveal-input-image",
                label: "Reveal input image",
                kind: .revealFile,
                style: .link,
                enabled: image.exists && !image.isDirectory,
                path: image.path
            )
        )
        actions.append(
            DeclarativeAction(
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: outputs.annotatedImage.parentExists,
                disabledReason: outputs.annotatedImage.parentExists
                    ? nil
                    : "Output directory will be created when segmentation starts.",
                path: outputs.annotatedImage.parentDirectory
            )
        )
        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        switch status {
        case .ok:
            return "Vision segmentation preflight passed."
        case .warning:
            return "Vision segmentation preflight found \(diagnostics.count) warning(s) or note(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "Vision segmentation preflight blocked by \(blockers) issue(s)."
        default:
            return "Vision segmentation preflight status: \(status.rawValue)."
        }
    }
}
