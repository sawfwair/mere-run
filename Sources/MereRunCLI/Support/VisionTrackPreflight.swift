import Foundation
import MereRunRelayKit
import MereRunCore

struct VisionTrackPreflightInput {
    let videoURL: URL
    let outputVideoURL: URL
    let outputJSONURL: URL
    let maskOutputDirectoryURL: URL?
    let prompt: [String]
    let box: [String]
    let point: [String]
    let model: String?
    let initFrame: Int
    let endFrame: Int?
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let showLabels: Bool
    let trackArgv: [String]
    let cwd: String
}

struct VisionTrackPreflightRequest: Codable, Equatable {
    let video: String
    let output: String
    let jsonOutput: String
    let maskOutputDir: String?
    let prompt: [String]
    let box: [String]
    let point: [String]
    let model: String?
    let initFrame: Int
    let endFrame: Int?
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let showLabels: Bool

    enum CodingKeys: String, CodingKey {
        case video
        case output
        case jsonOutput = "json_output"
        case maskOutputDir = "mask_output_dir"
        case prompt
        case box
        case point
        case model
        case initFrame = "init_frame"
        case endFrame = "end_frame"
        case threshold
        case resolution
        case showBoxes = "show_boxes"
        case showLabels = "show_labels"
    }
}

struct VisionTrackPreflightResult: Codable, Equatable {
    let model: VisionTrackModelPreflightSummary
    let video: VisionTrackPathPreflightSummary
    let prompts: VisionTrackPromptPreflightSummary
    let outputs: VisionTrackOutputsPreflightSummary
    let plan: VisionTrackPlanPreflightSummary
}

struct VisionTrackModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let id: String?
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case id
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

struct VisionTrackPathPreflightSummary: Codable, Equatable {
    let requested: String
    let path: String
    let exists: Bool
    let isDirectory: Bool

    enum CodingKeys: String, CodingKey {
        case requested
        case path
        case exists
        case isDirectory = "is_directory"
    }
}

struct VisionTrackPromptPreflightSummary: Codable, Equatable {
    let textPrompts: [String]
    let normalizedTextPrompts: [String]
    let boxPrompts: [String]
    let pointPrompts: [String]
    let textCount: Int
    let boxCount: Int
    let pointCount: Int
    let totalCount: Int

    enum CodingKeys: String, CodingKey {
        case textPrompts = "text_prompts"
        case normalizedTextPrompts = "normalized_text_prompts"
        case boxPrompts = "box_prompts"
        case pointPrompts = "point_prompts"
        case textCount = "text_count"
        case boxCount = "box_count"
        case pointCount = "point_count"
        case totalCount = "total_count"
    }
}

struct VisionTrackOutputsPreflightSummary: Codable, Equatable {
    let annotatedVideo: VisionTrackOutputPreflightSummary
    let trackingJSON: VisionTrackOutputPreflightSummary
    let maskDirectory: VisionTrackDirectoryPreflightSummary?

    enum CodingKeys: String, CodingKey {
        case annotatedVideo = "annotated_video"
        case trackingJSON = "tracking_json"
        case maskDirectory = "mask_directory"
    }
}

struct VisionTrackOutputPreflightSummary: Codable, Equatable {
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

struct VisionTrackDirectoryPreflightSummary: Codable, Equatable {
    let path: String
    let parentDirectory: String
    let parentExists: Bool
    let parentWillBeCreated: Bool
    let exists: Bool
    let isDirectory: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentDirectory = "parent_directory"
        case parentExists = "parent_exists"
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case isDirectory = "is_directory"
    }
}

struct VisionTrackPlanPreflightSummary: Codable, Equatable {
    let initFrame: Int
    let endFrame: Int?
    let threshold: Double
    let resolution: Int
    let showBoxes: Bool
    let showLabels: Bool

    enum CodingKeys: String, CodingKey {
        case initFrame = "init_frame"
        case endFrame = "end_frame"
        case threshold
        case resolution
        case showBoxes = "show_boxes"
        case showLabels = "show_labels"
    }
}

typealias VisionTrackPreflightEnvelope = StructuredRunEnvelope<
    VisionTrackPreflightRequest,
    VisionTrackPreflightResult
>

struct VisionTrackPreflightAnalyzer {
    let input: VisionTrackPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: VisionTrackPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> VisionTrackPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let video = videoSummary(diagnostics: &diagnostics)
        validateStaticOptions(diagnostics: &diagnostics)
        let prompts = promptSummary(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let outputs = outputSummaries(diagnostics: &diagnostics)
        let plan = VisionTrackPlanPreflightSummary(
            initFrame: input.initFrame,
            endFrame: input.endFrame,
            threshold: input.threshold,
            resolution: input.resolution,
            showBoxes: input.showBoxes,
            showLabels: input.showLabels
        )
        let status = StructuredRunOutput.status(for: diagnostics)

        return VisionTrackPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["vision", "track"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: VisionTrackPreflightResult(
                model: model,
                video: video,
                prompts: prompts,
                outputs: outputs,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions(status: status, model: model, video: video, outputs: outputs)
        )
    }

    private func request() -> VisionTrackPreflightRequest {
        VisionTrackPreflightRequest(
            video: input.videoURL.path,
            output: input.outputVideoURL.path,
            jsonOutput: input.outputJSONURL.path,
            maskOutputDir: input.maskOutputDirectoryURL?.path,
            prompt: input.prompt,
            box: input.box,
            point: input.point,
            model: input.model,
            initFrame: input.initFrame,
            endFrame: input.endFrame,
            threshold: input.threshold,
            resolution: input.resolution,
            showBoxes: input.showBoxes,
            showLabels: input.showLabels
        )
    }

    private func videoSummary(diagnostics: inout [PreflightDiagnostic]) -> VisionTrackPathPreflightSummary {
        let summary = pathSummary(requested: input.videoURL.path)
        if !summary.exists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "video_missing",
                    severity: .blocker,
                    title: "Video missing",
                    message: "Video not found: \(summary.path)",
                    locations: [.init(kind: "file", path: summary.path)]
                )
            )
        } else if summary.isDirectory {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "video_is_directory",
                    severity: .blocker,
                    title: "Video path is a directory",
                    message: "Video path is a directory: \(summary.path)",
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
        if input.initFrame < 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "init_frame_invalid",
                    severity: .blocker,
                    title: "Init frame is invalid",
                    message: "--init-frame must be greater than or equal to 0."
                )
            )
        }
        if let endFrame = input.endFrame, endFrame < input.initFrame {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "end_frame_before_init_frame",
                    severity: .blocker,
                    title: "End frame is before init frame",
                    message: "--end-frame must be greater than or equal to --init-frame."
                )
            )
        }
    }

    private func promptSummary(diagnostics: inout [PreflightDiagnostic]) -> VisionTrackPromptPreflightSummary {
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
        if normalizedTextPrompts.isEmpty, input.box.isEmpty, input.point.isEmpty, !input.prompt.isEmpty {
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

    private func modelSummary(diagnostics: inout [PreflightDiagnostic]) -> VisionTrackModelPreflightSummary {
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
                    return modelResult(requested: requested, kind: "local_path", installed: false, path: url.path)
                }
                return modelResult(requested: requested, kind: "local_path", installed: true, path: url.path)
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
                message: "Model \(effectiveRequested) is not installed. Pull it before tracking.",
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

    private func outputSummaries(diagnostics: inout [PreflightDiagnostic]) -> VisionTrackOutputsPreflightSummary {
        VisionTrackOutputsPreflightSummary(
            annotatedVideo: outputSummary(
                url: input.outputVideoURL,
                expectedExtension: "mp4",
                role: "annotated video",
                diagnosticPrefix: "output_video",
                diagnostics: &diagnostics
            ),
            trackingJSON: outputSummary(
                url: input.outputJSONURL,
                expectedExtension: "json",
                role: "tracking JSON",
                diagnosticPrefix: "json_output",
                diagnostics: &diagnostics
            ),
            maskDirectory: input.maskOutputDirectoryURL.map { maskDirectorySummary(url: $0, diagnostics: &diagnostics) }
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
        video: VisionTrackPathPreflightSummary,
        outputs: VisionTrackOutputsPreflightSummary
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-tracking",
                label: "Start tracking",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.trackArgv,
                    cwd: input.cwd,
                    commandPath: ["vision", "track"]
                ),
                requires: ["preflight.passed"]
            )
        )

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
                id: "reveal-input-video",
                label: "Reveal input video",
                kind: .revealFile,
                style: .link,
                enabled: video.exists && !video.isDirectory,
                path: video.path
            )
        )
        actions.append(
            DeclarativeAction(
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: outputs.annotatedVideo.parentExists,
                disabledReason: outputs.annotatedVideo.parentExists ? nil : "Output directory will be created when tracking starts.",
                path: outputs.annotatedVideo.parentDirectory
            )
        )
        if let maskDirectory = outputs.maskDirectory {
            actions.append(
                DeclarativeAction(
                    id: "open-mask-directory",
                    label: "Open mask directory",
                    kind: .openDirectory,
                    style: .link,
                    enabled: maskDirectory.exists && maskDirectory.isDirectory,
                    disabledReason: maskDirectory.exists ? nil : "Mask directory will be created when tracking starts.",
                    path: maskDirectory.path
                )
            )
        }

        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        switch status {
        case .ok:
            return "Vision tracking preflight passed."
        case .warning:
            return "Vision tracking preflight found \(diagnostics.count) warning(s) or note(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "Vision tracking preflight blocked by \(blockers) issue(s)."
        default:
            return "Vision tracking preflight status: \(status.rawValue)."
        }
    }
}
