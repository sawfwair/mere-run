import Foundation
import MereRunCore

struct VideoGenerationPreflightInput {
    let prompt: String
    let outputURL: URL
    let model: String
    let variant: LTXVideoVariant
    let modelRoot: String?
    let width: Int
    let height: Int
    let numFrames: Int
    let duration: Double?
    let fps: Int
    let seed: Int?
    let image: String?
    let imageStrength: Float
    let endImage: String?
    let endImageStrength: Float
    let generationArgv: [String]
    let cwd: String
}

struct VideoGenerationPreflightRequest: Codable, Equatable {
    let prompt: String
    let output: String
    let model: String
    let variant: String
    let modelRoot: String?
    let width: Int
    let height: Int
    let numFrames: Int
    let duration: Double?
    let fps: Int
    let seed: Int?
    let image: String?
    let imageStrength: Float
    let endImage: String?
    let endImageStrength: Float

    enum CodingKeys: String, CodingKey {
        case prompt
        case output
        case model
        case variant
        case modelRoot = "model_root"
        case width
        case height
        case numFrames = "num_frames"
        case duration
        case fps
        case seed
        case image
        case imageStrength = "image_strength"
        case endImage = "end_image"
        case endImageStrength = "end_image_strength"
    }
}

struct VideoGenerationPreflightResult: Codable, Equatable {
    let model: VideoGenerationModelPreflightSummary
    let output: VideoGenerationOutputPreflightSummary
    let inputs: VideoGenerationInputPreflightSummary
    let plan: VideoGenerationPlanPreflightSummary
}

struct VideoGenerationModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let id: String?
    let layout: String?
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?
    let companionModelIDs: [String]

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case id
        case layout
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
        case companionModelIDs = "companion_model_ids"
    }
}

struct VideoGenerationOutputPreflightSummary: Codable, Equatable {
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

struct VideoGenerationInputPreflightSummary: Codable, Equatable {
    let mode: String
    let sourceImage: VideoGenerationPathPreflightSummary?
    let endImage: VideoGenerationPathPreflightSummary?
    let missingCount: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case sourceImage = "source_image"
        case endImage = "end_image"
        case missingCount = "missing_count"
    }
}

struct VideoGenerationPathPreflightSummary: Codable, Equatable {
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

struct VideoGenerationPlanPreflightSummary: Codable, Equatable {
    let variant: String
    let inputMode: String
    let requestedWidth: Int
    let requestedHeight: Int
    let resolvedWidth: Int
    let resolvedHeight: Int
    let requestedNumFrames: Int
    let requestedDurationSeconds: Double?
    let fps: Int
    let resolvedNumFrames: Int
    let resolvedDurationSeconds: Double?
    let seed: Int
    let writesAudio: Bool

    enum CodingKeys: String, CodingKey {
        case variant
        case inputMode = "input_mode"
        case requestedWidth = "requested_width"
        case requestedHeight = "requested_height"
        case resolvedWidth = "resolved_width"
        case resolvedHeight = "resolved_height"
        case requestedNumFrames = "requested_num_frames"
        case requestedDurationSeconds = "requested_duration_seconds"
        case fps
        case resolvedNumFrames = "resolved_num_frames"
        case resolvedDurationSeconds = "resolved_duration_seconds"
        case seed
        case writesAudio = "writes_audio"
    }
}

typealias VideoGenerationPreflightEnvelope = StructuredRunEnvelope<
    VideoGenerationPreflightRequest,
    VideoGenerationPreflightResult
>

struct VideoGenerationPreflightAnalyzer {
    let input: VideoGenerationPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: VideoGenerationPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    private var usesWanGeometry: Bool {
        if input.model.trimmingCharacters(in: .whitespacesAndNewlines) == Wan2Resources.modelID {
            return true
        }
        let candidate = input.modelRoot ?? input.model
        let url = URL(fileURLWithPath: candidate).standardizedFileURL
        let resources = Wan2Resources(rootURL: url)
        return resources.validate().isEmpty && (try? resources.loadConfiguration()) != nil
    }

    func envelope() -> VideoGenerationPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        validateStaticOptions(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let output = outputSummary(diagnostics: &diagnostics)
        let inputs = inputSummary(diagnostics: &diagnostics)
        let plan = planSummary(inputs: inputs, diagnostics: &diagnostics)
        let status = StructuredRunOutput.status(for: diagnostics)

        return VideoGenerationPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["video", "generate"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary(status: status, diagnostics: diagnostics),
            request: request(),
            result: VideoGenerationPreflightResult(
                model: model,
                output: output,
                inputs: inputs,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions(status: status, model: model, output: output, inputs: inputs)
        )
    }

    private func request() -> VideoGenerationPreflightRequest {
        VideoGenerationPreflightRequest(
            prompt: input.prompt,
            output: input.outputURL.path,
            model: input.model,
            variant: input.variant.rawValue,
            modelRoot: input.modelRoot,
            width: input.width,
            height: input.height,
            numFrames: input.numFrames,
            duration: input.duration,
            fps: input.fps,
            seed: input.seed,
            image: input.image,
            imageStrength: input.imageStrength,
            endImage: input.endImage,
            endImageStrength: input.endImageStrength
        )
    }

    private func validateStaticOptions(diagnostics: inout [PreflightDiagnostic]) {
        if input.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "prompt_empty",
                    severity: .blocker,
                    title: "Prompt is empty",
                    message: "Provide a non-empty video prompt."
                )
            )
        }
        if input.fps < 1 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "fps_invalid",
                    severity: .blocker,
                    title: "FPS is invalid",
                    message: "--fps must be >= 1."
                )
            )
        }
        if let duration = input.duration, duration <= 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_invalid",
                    severity: .blocker,
                    title: "Duration is invalid",
                    message: "--duration must be > 0."
                )
            )
        }
        let minimumSpatialDimension = usesWanGeometry ? 32 : 64
        let minimumFrameCount = usesWanGeometry ? 5 : 9
        if input.width < minimumSpatialDimension {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "width_too_small",
                    severity: .blocker,
                    title: "Width is too small",
                    message: "--width must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.height < minimumSpatialDimension {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "height_too_small",
                    severity: .blocker,
                    title: "Height is too small",
                    message: "--height must be >= \(minimumSpatialDimension)."
                )
            )
        }
        if input.numFrames < minimumFrameCount {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "num_frames_too_small",
                    severity: .blocker,
                    title: "Frame count is too small",
                    message: "--num-frames must be >= \(minimumFrameCount)."
                )
            )
        }
        if !(0...1).contains(input.imageStrength) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "image_strength_invalid",
                    severity: .blocker,
                    title: "Image strength is invalid",
                    message: "--image-strength must be between 0 and 1."
                )
            )
        }
        if !(0...1).contains(input.endImageStrength) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "end_image_strength_invalid",
                    severity: .blocker,
                    title: "End image strength is invalid",
                    message: "--end-image-strength must be between 0 and 1."
                )
            )
        }
        if input.endImage != nil, input.image == nil {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "end_image_requires_source_image",
                    severity: .blocker,
                    title: "End keyframe needs a source image",
                    message: "--end-image requires --image so the start keyframe is anchored."
                )
            )
        }
        if input.variant == .unifiedAV, input.fps > 0, input.fps != 24 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "unified_av_fps_unusual",
                    severity: .warning,
                    title: "Unified AV is tuned for 24 fps",
                    message: "LTX unified AV is trained around 24 fps; --fps \(input.fps) can make motion look time-stretched relative to audio."
                )
            )
        }
    }

    private func modelSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        if let modelRoot = input.modelRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !modelRoot.isEmpty {
            return localModelSummary(
                requested: modelRoot,
                kind: "model_root",
                diagnostics: &diagnostics
            )
        }

        let trimmedModel = input.model.trimmingCharacters(in: .whitespacesAndNewlines)
        let requested = trimmedModel.isEmpty ? ModelResolver.ModelID.ltxVideo23AVMLX.rawValue : trimmedModel
        let localURL = URL(fileURLWithPath: requested).standardizedFileURL
        if fileManager.fileExists(atPath: localURL.path) {
            return localModelSummary(
                requested: requested,
                kind: "local_path",
                diagnostics: &diagnostics
            )
        }

        guard let modelID = ModelResolver.ModelID(rawValue: requested),
              let spec = ManagedModelCatalog.spec(for: requested) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_unknown",
                    severity: .blocker,
                    title: "Unknown model",
                    message: "Model path not found and not a known model id: \(requested)."
                )
            )
            return modelResult(requested: requested, kind: "unknown", installed: false)
        }

        if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
            return installedManagedModelSummary(
                requested: requested,
                spec: spec,
                path: resolution.rootURL,
                diagnostics: &diagnostics
            )
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "model_missing",
                severity: .blocker,
                title: "Model missing",
                message: "Model \(requested) is not installed. Pull it before video generation.",
                suggestedActionIDs: ["pull-model"]
            )
        )
        return modelResult(
            requested: requested,
            kind: "managed_model",
            installed: false,
            id: requested,
            upstreamRepoID: spec.upstreamRepoId,
            estimatedDownloadBytes: spec.estimatedDownloadBytes,
            companionModelIDs: spec.companionModelIDs
        )
    }

    private func localModelSummary(
        requested: String,
        kind: String,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_missing",
                    severity: .blocker,
                    title: "Model root missing",
                    message: "Model root not found: \(url.path)",
                    locations: [.init(kind: "directory", path: url.path)]
                )
            )
            return modelResult(requested: requested, kind: kind, installed: false, path: url.path)
        }
        guard isDirectory.boolValue else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_not_directory",
                    severity: .blocker,
                    title: "Model root is not a directory",
                    message: "Model root is not a directory: \(url.path)",
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
            return modelResult(requested: requested, kind: kind, installed: false, path: url.path)
        }

        do {
            try validateNativeModelRoot(url)
            return modelResult(
                requested: requested,
                kind: kind,
                installed: true,
                path: url.path,
                layout: videoLayout(at: url)
            )
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_invalid",
                    severity: .blocker,
                    title: "Model root is invalid",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: url.path)]
                )
            )
            return modelResult(
                requested: requested,
                kind: kind,
                installed: false,
                path: url.path,
                layout: videoLayout(at: url)
            )
        }
    }

    private func installedManagedModelSummary(
        requested: String,
        spec: ManagedModelSpec,
        path: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationModelPreflightSummary {
        do {
            try validateNativeModelRoot(path)
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: true,
                path: path.path,
                id: spec.id,
                layout: videoLayout(at: path),
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes,
                companionModelIDs: spec.companionModelIDs
            )
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_root_invalid",
                    severity: .blocker,
                    title: "Installed model root is invalid",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: path.path)]
                )
            )
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: false,
                path: path.path,
                id: spec.id,
                layout: videoLayout(at: path),
                upstreamRepoID: spec.upstreamRepoId,
                estimatedDownloadBytes: spec.estimatedDownloadBytes,
                companionModelIDs: spec.companionModelIDs
            )
        }
    }

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        id: String? = nil,
        layout: String? = nil,
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil,
        companionModelIDs: [String] = []
    ) -> VideoGenerationModelPreflightSummary {
        VideoGenerationModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            id: id,
            layout: layout,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes,
            companionModelIDs: companionModelIDs
        )
    }

    private func videoLayout(at url: URL) -> String? {
        let resources = Wan2Resources(rootURL: url)
        if resources.validate().isEmpty, (try? resources.loadConfiguration()) != nil {
            return "wan22_ti2v_mlx"
        }
        return isLTX23SplitModelRoot(url, fileManager: fileManager) ? "ltx23_split" : "ltx_merged"
    }

    private func outputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationOutputPreflightSummary {
        let parent = input.outputURL.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: input.outputURL.path)
        let extensionValid = input.outputURL.pathExtension.lowercased() == "mp4"

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
                    title: "Output extension is not MP4",
                    message: "Video generation writes MP4 data; use a .mp4 output path for clarity.",
                    locations: [.init(kind: "file", path: input.outputURL.path)]
                )
            )
        }

        return VideoGenerationOutputPreflightSummary(
            path: input.outputURL.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            expectedExtension: "mp4",
            extensionValid: extensionValid
        )
    }

    private func inputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationInputPreflightSummary {
        let sourceImage = input.image.map { pathSummary(requested: $0) }
        let endImage = input.endImage.map { pathSummary(requested: $0) }
        for (summary, prefix) in [(sourceImage, "source_image"), (endImage, "end_image")] {
            guard let summary else { continue }
            if !summary.exists {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "\(prefix)_missing",
                        severity: .blocker,
                        title: "Input image missing",
                        message: "Input image not found: \(summary.path)",
                        locations: [.init(kind: "file", path: summary.path)]
                    )
                )
            } else if summary.isDirectory {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "\(prefix)_is_directory",
                        severity: .blocker,
                        title: "Input image is a directory",
                        message: "Input image path is a directory: \(summary.path)",
                        locations: [.init(kind: "directory", path: summary.path)]
                    )
                )
            }
        }

        let allInputs = [sourceImage, endImage].compactMap { $0 }
        return VideoGenerationInputPreflightSummary(
            mode: sourceImage == nil ? "text_to_video" : (endImage == nil ? "image_to_video" : "directed_image_to_video"),
            sourceImage: sourceImage,
            endImage: endImage,
            missingCount: allInputs.filter { !$0.exists }.count
        )
    }

    private func pathSummary(requested: String) -> VideoGenerationPathPreflightSummary {
        let url = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        let exists = fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return VideoGenerationPathPreflightSummary(
            requested: requested,
            path: url.path,
            exists: exists,
            isDirectory: exists && isDirectory.boolValue
        )
    }

    private func planSummary(
        inputs: VideoGenerationInputPreflightSummary,
        diagnostics: inout [PreflightDiagnostic]
    ) -> VideoGenerationPlanPreflightSummary {
        let spatialMultiple = usesWanGeometry ? 32 : 64
        let temporalMultiple = usesWanGeometry ? 4 : 8
        let minimumFrames = usesWanGeometry ? 5 : 9
        let resolvedWidth = max(spatialMultiple, (input.width / spatialMultiple) * spatialMultiple)
        let resolvedHeight = max(spatialMultiple, (input.height / spatialMultiple) * spatialMultiple)
        let requestedFrames = input.duration.map {
            usesWanGeometry
                ? nearestWanFrameCount(duration: $0, fps: input.fps)
                : nearestLTXFrameCount(duration: $0, fps: input.fps)
        } ?? input.numFrames
        let resolvedFrames = max(
            minimumFrames,
            ((requestedFrames - 1) / temporalMultiple) * temporalMultiple + 1
        )

        if input.width >= spatialMultiple,
           input.height >= spatialMultiple,
           resolvedWidth != input.width || resolvedHeight != input.height {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dimensions_will_be_adjusted",
                    severity: .note,
                    title: "Dimensions will be adjusted",
                    message: "Video dimensions will be snapped from \(input.width)x\(input.height) to \(resolvedWidth)x\(resolvedHeight)."
                )
            )
        }
        if input.numFrames >= minimumFrames, input.duration == nil, resolvedFrames != input.numFrames {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "num_frames_will_be_adjusted",
                    severity: .note,
                    title: "Frame count will be adjusted",
                    message: "Frame count will be snapped from \(input.numFrames) to \(resolvedFrames) to satisfy \(temporalMultiple)n+1."
                )
            )
        }
        if let duration = input.duration, input.fps > 0, duration > 0 {
            let resolvedSeconds = Double(resolvedFrames) / Double(input.fps)
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duration_resolved_to_frame_count",
                    severity: .note,
                    title: "Duration resolved to frame count",
                    message: String(
                        format: "Duration %.2fs resolves to %d frames at %d fps (~%.2fs).",
                        duration,
                        resolvedFrames,
                        input.fps,
                        resolvedSeconds
                    )
                )
            )
        }

        return VideoGenerationPlanPreflightSummary(
            variant: usesWanGeometry ? "wan22-ti2v" : input.variant.rawValue,
            inputMode: inputs.mode,
            requestedWidth: input.width,
            requestedHeight: input.height,
            resolvedWidth: resolvedWidth,
            resolvedHeight: resolvedHeight,
            requestedNumFrames: input.numFrames,
            requestedDurationSeconds: input.duration,
            fps: input.fps,
            resolvedNumFrames: resolvedFrames,
            resolvedDurationSeconds: input.fps > 0 ? Double(resolvedFrames) / Double(input.fps) : nil,
            seed: input.seed ?? 42,
            writesAudio: !usesWanGeometry && input.variant == .unifiedAV
        )
    }

    private func actions(
        status: StructuredRunStatus,
        model: VideoGenerationModelPreflightSummary,
        output: VideoGenerationOutputPreflightSummary,
        inputs: VideoGenerationInputPreflightSummary
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-video-generation",
                label: "Start video generation",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.generationArgv,
                    cwd: input.cwd,
                    commandPath: ["video", "generate"]
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
                id: "open-output-directory",
                label: "Open output directory",
                kind: .openDirectory,
                style: .link,
                enabled: output.parentExists,
                disabledReason: output.parentExists ? nil : "Output directory will be created when generation starts.",
                path: output.parentDirectory
            )
        )

        if let sourceImage = inputs.sourceImage {
            actions.append(
                DeclarativeAction(
                    id: "reveal-source-image",
                    label: "Reveal source image",
                    kind: .revealFile,
                    style: .link,
                    enabled: sourceImage.exists && !sourceImage.isDirectory,
                    path: sourceImage.path
                )
            )
        }
        if let endImage = inputs.endImage {
            actions.append(
                DeclarativeAction(
                    id: "reveal-end-image",
                    label: "Reveal end image",
                    kind: .revealFile,
                    style: .link,
                    enabled: endImage.exists && !endImage.isDirectory,
                    path: endImage.path
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
            return "Video generation preflight passed."
        case .warning:
            return "Video generation preflight found \(diagnostics.count) warning(s) or note(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "Video generation preflight blocked by \(blockers) issue(s)."
        default:
            return "Video generation preflight status: \(status.rawValue)."
        }
    }
}
