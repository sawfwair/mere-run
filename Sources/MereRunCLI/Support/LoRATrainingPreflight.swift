import Foundation
import MereRunCore

struct LoRATrainingPreflightInput {
    let data: String?
    let output: String
    let recipe: String?
    let excludePreviewImages: Bool
    let syntheticSamples: Int?
    let options: ImageTrainLoRA.ResolvedLoRATrainingOptions
    let trainingArgv: [String]
    let cwd: String
}

struct LoRATrainingPreflightRequest: Codable, Equatable {
    let data: String?
    let output: String
    let model: String
    let recipe: String?
    let trainingSteps: Int
    let width: Int
    let height: Int
    let rank: Int
    let alpha: Float?
    let learningRate: Float
    let captionDropout: Float

    enum CodingKeys: String, CodingKey {
        case data
        case output
        case model
        case recipe
        case trainingSteps = "training_steps"
        case width
        case height
        case rank
        case alpha
        case learningRate = "learning_rate"
        case captionDropout = "caption_dropout"
    }
}

struct LoRATrainingPreflightResult: Codable, Equatable {
    let dataset: LoRATrainingDatasetPreflightSummary
    let model: LoRATrainingModelPreflightSummary
    let output: LoRATrainingOutputPreflightSummary
    let plan: LoRATrainingPlanPreflightSummary
}

struct LoRATrainingDatasetPreflightSummary: Codable, Equatable {
    let directory: String?
    let mode: String
    let imageCount: Int
    let captionCount: Int
    let usablePairCount: Int
    let missingCaptionCount: Int
    let emptyCaptionCount: Int
    let duplicateCaptionGroupCount: Int
    let duplicateCaptionCount: Int
    let excludedPreviewImageCount: Int
    let placeholderCaptionCount: Int
    let syntheticSampleCount: Int?

    enum CodingKeys: String, CodingKey {
        case directory
        case mode
        case imageCount = "image_count"
        case captionCount = "caption_count"
        case usablePairCount = "usable_pair_count"
        case missingCaptionCount = "missing_caption_count"
        case emptyCaptionCount = "empty_caption_count"
        case duplicateCaptionGroupCount = "duplicate_caption_group_count"
        case duplicateCaptionCount = "duplicate_caption_count"
        case excludedPreviewImageCount = "excluded_preview_image_count"
        case placeholderCaptionCount = "placeholder_caption_count"
        case syntheticSampleCount = "synthetic_sample_count"
    }
}

struct LoRATrainingModelPreflightSummary: Codable, Equatable {
    let requested: String
    let kind: String
    let installed: Bool
    let path: String?
    let family: String?
    let upstreamRepoID: String?
    let estimatedDownloadBytes: Int64?

    enum CodingKeys: String, CodingKey {
        case requested
        case kind
        case installed
        case path
        case family
        case upstreamRepoID = "upstream_repo_id"
        case estimatedDownloadBytes = "estimated_download_bytes"
    }
}

struct LoRATrainingOutputPreflightSummary: Codable, Equatable {
    let path: String
    let parentDirectory: String
    let parentExists: Bool
    let parentWillBeCreated: Bool
    let exists: Bool
    let extensionValid: Bool

    enum CodingKeys: String, CodingKey {
        case path
        case parentDirectory = "parent_directory"
        case parentExists = "parent_exists"
        case parentWillBeCreated = "parent_will_be_created"
        case exists
        case extensionValid = "extension_valid"
    }
}

struct LoRATrainingPlanPreflightSummary: Codable, Equatable {
    let recipe: String?
    let trainingSteps: Int
    let width: Int
    let height: Int
    let rank: Int
    let alpha: Float?
    let learningRate: Float
    let captionDropout: Float
    let checkpointInterval: Int?
    let expectedCheckpointCount: Int
    let maxResolution: Int?
    let lowRam: Bool
    let noCompile: Bool
    let loraTargetPreset: String?
    let lrWarmupSteps: Int?
    let useCosineScheduler: Bool?
    let lrMinFactor: Float?

    enum CodingKeys: String, CodingKey {
        case recipe
        case trainingSteps = "training_steps"
        case width
        case height
        case rank
        case alpha
        case learningRate = "learning_rate"
        case captionDropout = "caption_dropout"
        case checkpointInterval = "checkpoint_interval"
        case expectedCheckpointCount = "expected_checkpoint_count"
        case maxResolution = "max_resolution"
        case lowRam = "low_ram"
        case noCompile = "no_compile"
        case loraTargetPreset = "lora_target_preset"
        case lrWarmupSteps = "lr_warmup_steps"
        case useCosineScheduler = "use_cosine_scheduler"
        case lrMinFactor = "lr_min_factor"
    }
}

typealias LoRATrainingPreflightEnvelope = StructuredRunEnvelope<
    LoRATrainingPreflightRequest,
    LoRATrainingPreflightResult
>

struct LoRATrainingPreflightAnalyzer {
    let input: LoRATrainingPreflightInput
    let fileManager: FileManager
    let now: () -> Date

    init(
        input: LoRATrainingPreflightInput,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.input = input
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> LoRATrainingPreflightEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let dataset = datasetSummary(diagnostics: &diagnostics)
        let model = modelSummary(diagnostics: &diagnostics)
        let output = outputSummary(diagnostics: &diagnostics)
        let plan = planSummary()
        let status = StructuredRunOutput.status(for: diagnostics)
        let actions = actions(status: status, model: model)
        let summary = summary(status: status, dataset: dataset, diagnostics: diagnostics)

        return LoRATrainingPreflightEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["image", "train-lora"],
            mode: .preflight,
            status: status,
            createdAt: now(),
            cwd: input.cwd,
            summary: summary,
            request: LoRATrainingPreflightRequest(
                data: input.data,
                output: input.output,
                model: input.options.model ?? ImageTrainLoRA.defaultManagedModelID.rawValue,
                recipe: input.recipe,
                trainingSteps: input.options.trainingSteps,
                width: input.options.width,
                height: input.options.height,
                rank: input.options.rank,
                alpha: input.options.alpha,
                learningRate: input.options.learningRate,
                captionDropout: input.options.captionDropout
            ),
            result: LoRATrainingPreflightResult(
                dataset: dataset,
                model: model,
                output: output,
                plan: plan
            ),
            diagnostics: diagnostics,
            actions: actions
        )
    }

    private func datasetSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> LoRATrainingDatasetPreflightSummary {
        if let syntheticSamples = input.syntheticSamples {
            return LoRATrainingDatasetPreflightSummary(
                directory: nil,
                mode: "synthetic",
                imageCount: 0,
                captionCount: 0,
                usablePairCount: syntheticSamples,
                missingCaptionCount: 0,
                emptyCaptionCount: 0,
                duplicateCaptionGroupCount: 0,
                duplicateCaptionCount: 0,
                excludedPreviewImageCount: 0,
                placeholderCaptionCount: 0,
                syntheticSampleCount: syntheticSamples
            )
        }

        guard let data = input.data?.trimmingCharacters(in: .whitespacesAndNewlines), !data.isEmpty else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "data_required",
                    severity: .blocker,
                    title: "Dataset required",
                    message: "--data is required unless --synthetic-samples is set."
                )
            )
            return emptyDatasetSummary(directory: nil, mode: "missing")
        }

        let directory = URL(fileURLWithPath: data).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dataset_not_found",
                    severity: .blocker,
                    title: "Dataset not found",
                    message: "Dataset directory not found: \(directory.path)",
                    locations: [.init(kind: "directory", path: directory.path)]
                )
            )
            return emptyDatasetSummary(directory: directory.path, mode: "missing")
        }

        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dataset_unreadable",
                    severity: .blocker,
                    title: "Dataset unreadable",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: directory.path)]
                )
            )
            return emptyDatasetSummary(directory: directory.path, mode: "unreadable")
        }

        let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp"]
        let regularFiles = contents.filter { url in
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            return values?.isRegularFile == true
        }
        let allImages = regularFiles
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let previewImages = allImages.filter { $0.deletingPathExtension().lastPathComponent.hasPrefix("preview") }
        let images = input.excludePreviewImages
            ? allImages.filter { !previewImages.contains($0) }
            : allImages

        if images.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "no_training_images",
                    severity: .blocker,
                    title: "No training images",
                    message: "No PNG, JPG, JPEG, or WEBP training images were found in \(directory.path).",
                    locations: [.init(kind: "directory", path: directory.path)]
                )
            )
        }

        let editOutputs = images.filter { $0.deletingPathExtension().lastPathComponent.hasSuffix("_out") }
        if !editOutputs.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "edit_dataset_not_supported",
                    severity: .blocker,
                    title: "Edit dataset detected",
                    message: "image train-lora currently expects image + .txt caption pairs, not *_in/*_out edit pairs.",
                    locations: editOutputs.prefix(5).map { .init(kind: "file", path: $0.path) }
                )
            )
        }

        if !previewImages.isEmpty, !input.excludePreviewImages {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "preview_images_included",
                    severity: .warning,
                    title: "Preview images included",
                    message: "\(previewImages.count) preview image(s) will be treated as training images. Use --exclude-preview-images if they are generated previews.",
                    locations: previewImages.prefix(5).map { .init(kind: "file", path: $0.path) }
                )
            )
        }

        var captionCount = 0
        var usablePairCount = 0
        var missingCaptions: [URL] = []
        var emptyCaptions: [URL] = []
        var placeholderCaptions: [URL] = []
        var captionsByText: [String: [URL]] = [:]

        for image in images {
            let captionURL = image.deletingPathExtension().appendingPathExtension("txt")
            guard fileManager.fileExists(atPath: captionURL.path) else {
                missingCaptions.append(captionURL)
                continue
            }
            captionCount += 1
            let captionData: Data
            do {
                captionData = try Data(contentsOf: captionURL)
            } catch {
                emptyCaptions.append(captionURL)
                continue
            }
            let caption = String(decoding: captionData, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !caption.isEmpty else {
                emptyCaptions.append(captionURL)
                continue
            }
            usablePairCount += 1
            captionsByText[caption, default: []].append(captionURL)
            if Self.isPlaceholderCaption(caption) {
                placeholderCaptions.append(captionURL)
            }
        }

        if !missingCaptions.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "missing_captions",
                    severity: .blocker,
                    title: "Missing captions",
                    message: "\(missingCaptions.count) image file(s) do not have matching .txt captions.",
                    locations: missingCaptions.prefix(10).map { .init(kind: "file", path: $0.path) }
                )
            )
        }
        if !emptyCaptions.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "empty_captions",
                    severity: .blocker,
                    title: "Empty captions",
                    message: "\(emptyCaptions.count) caption file(s) are empty.",
                    locations: emptyCaptions.prefix(10).map { .init(kind: "file", path: $0.path) }
                )
            )
        }
        if !placeholderCaptions.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "placeholder_captions",
                    severity: .warning,
                    title: "Placeholder captions",
                    message: "\(placeholderCaptions.count) caption file(s) look like placeholders.",
                    locations: placeholderCaptions.prefix(10).map { .init(kind: "file", path: $0.path) }
                )
            )
        }

        let duplicateGroups = captionsByText.values.filter { $0.count > 1 }
        let duplicateCaptionCount = duplicateGroups.reduce(0) { $0 + $1.count }
        if !duplicateGroups.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "duplicate_captions",
                    severity: .warning,
                    title: "Repeated captions",
                    message: "\(duplicateCaptionCount) caption file(s) are part of \(duplicateGroups.count) exact duplicate group(s).",
                    locations: duplicateGroups.flatMap { $0 }.prefix(10).map { .init(kind: "file", path: $0.path) }
                )
            )
        }

        return LoRATrainingDatasetPreflightSummary(
            directory: directory.path,
            mode: editOutputs.isEmpty ? "image_caption" : "edit_detected",
            imageCount: images.count,
            captionCount: captionCount,
            usablePairCount: usablePairCount,
            missingCaptionCount: missingCaptions.count,
            emptyCaptionCount: emptyCaptions.count,
            duplicateCaptionGroupCount: duplicateGroups.count,
            duplicateCaptionCount: duplicateCaptionCount,
            excludedPreviewImageCount: input.excludePreviewImages ? previewImages.count : 0,
            placeholderCaptionCount: placeholderCaptions.count,
            syntheticSampleCount: nil
        )
    }

    private func modelSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> LoRATrainingModelPreflightSummary {
        let requested = input.options.model ?? ImageTrainLoRA.defaultManagedModelID.rawValue
        let localURL = URL(fileURLWithPath: requested).standardizedFileURL
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: localURL.path, isDirectory: &isDirectory) {
            guard isDirectory.boolValue else {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "model_path_not_directory",
                        severity: .blocker,
                        title: "Model path is not a directory",
                        message: "Model path is not a directory: \(localURL.path)",
                        locations: [.init(kind: "file", path: localURL.path)]
                    )
                )
                return modelResult(requested: requested, kind: "local_path", installed: false, path: localURL.path)
            }
            let family = loadManifestFamily(at: localURL, diagnostics: &diagnostics)
            return modelResult(
                requested: requested,
                kind: "local_path",
                installed: family != nil,
                path: localURL.path,
                family: family
            )
        }

        if let modelID = ModelResolver.ModelID(rawValue: requested) {
            let spec = ManagedModelCatalog.spec(for: modelID.rawValue)
            if let resolution = ModelResolver(fileManager: fileManager).resolveIfPresent(modelID) {
                let family = loadManifestFamily(at: resolution.rootURL, diagnostics: &diagnostics)
                return modelResult(
                    requested: requested,
                    kind: "managed_model",
                    installed: family != nil,
                    path: resolution.rootURL.path,
                    family: family,
                    upstreamRepoID: spec?.upstreamRepoId,
                    estimatedDownloadBytes: spec?.estimatedDownloadBytes
                )
            }

            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_missing",
                    severity: .blocker,
                    title: "Model missing",
                    message: "Model \(modelID.rawValue) is not installed. Pull it before training.",
                    suggestedActionIDs: ["pull-model"]
                )
            )
            return modelResult(
                requested: requested,
                kind: "managed_model",
                installed: false,
                upstreamRepoID: spec?.upstreamRepoId,
                estimatedDownloadBytes: spec?.estimatedDownloadBytes
            )
        }

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

    private func outputSummary(
        diagnostics: inout [PreflightDiagnostic]
    ) -> LoRATrainingOutputPreflightSummary {
        let outputURL = URL(fileURLWithPath: input.output).standardizedFileURL
        let parent = outputURL.deletingLastPathComponent()
        var parentIsDirectory: ObjCBool = false
        let parentExists = fileManager.fileExists(atPath: parent.path, isDirectory: &parentIsDirectory)
        let outputExists = fileManager.fileExists(atPath: outputURL.path)
        let extensionValid = outputURL.pathExtension.lowercased() == "safetensors"

        if !extensionValid {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "output_extension_invalid",
                    severity: .blocker,
                    title: "Invalid output extension",
                    message: "--output must end in .safetensors.",
                    locations: [.init(kind: "file", path: outputURL.path)]
                )
            )
        }
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
                    message: "Output already exists and may be overwritten: \(outputURL.path)",
                    locations: [.init(kind: "file", path: outputURL.path)]
                )
            )
        }

        return LoRATrainingOutputPreflightSummary(
            path: outputURL.path,
            parentDirectory: parent.path,
            parentExists: parentExists && parentIsDirectory.boolValue,
            parentWillBeCreated: !parentExists,
            exists: outputExists,
            extensionValid: extensionValid
        )
    }

    private func planSummary() -> LoRATrainingPlanPreflightSummary {
        let checkpointCount: Int
        if let checkpointInterval = input.options.checkpointInterval {
            checkpointCount = input.options.trainingSteps / checkpointInterval
        } else {
            checkpointCount = 0
        }
        return LoRATrainingPlanPreflightSummary(
            recipe: input.recipe,
            trainingSteps: input.options.trainingSteps,
            width: input.options.width,
            height: input.options.height,
            rank: input.options.rank,
            alpha: input.options.alpha,
            learningRate: input.options.learningRate,
            captionDropout: input.options.captionDropout,
            checkpointInterval: input.options.checkpointInterval,
            expectedCheckpointCount: checkpointCount,
            maxResolution: input.options.maxResolution,
            lowRam: input.options.lowRam,
            noCompile: input.options.noCompile,
            loraTargetPreset: input.options.loraTargetPreset,
            lrWarmupSteps: input.options.lrWarmupSteps,
            useCosineScheduler: input.options.useCosineScheduler,
            lrMinFactor: input.options.lrMinFactor
        )
    }

    private func actions(
        status: StructuredRunStatus,
        model: LoRATrainingModelPreflightSummary
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        let blocked = status == .blocked
        actions.append(
            DeclarativeAction(
                id: "start-training",
                label: "Start training",
                kind: .command,
                style: .primary,
                enabled: !blocked,
                disabledReason: blocked ? "Resolve hard blockers first." : nil,
                command: DeclarativeCommand(
                    argv: input.trainingArgv,
                    cwd: input.cwd,
                    commandPath: ["image", "train-lora"]
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

        if let data = input.data {
            let dataURL = URL(fileURLWithPath: data).standardizedFileURL
            actions.append(
                DeclarativeAction(
                    id: "open-dataset",
                    label: "Open dataset",
                    kind: .openDirectory,
                    style: .link,
                    enabled: fileManager.fileExists(atPath: dataURL.path),
                    path: dataURL.path
                )
            )
        }

        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        dataset: LoRATrainingDatasetPreflightSummary,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        let blockerCount = diagnostics.filter { $0.severity == .blocker }.count
        let warningCount = diagnostics.filter { $0.severity == .warning }.count
        switch status {
        case .blocked:
            return "\(dataset.usablePairCount) usable pair(s), \(blockerCount) blocker(s), \(warningCount) warning(s)."
        case .warning:
            return "\(dataset.usablePairCount) usable pair(s), \(warningCount) warning(s), ready to train."
        default:
            return "\(dataset.usablePairCount) usable pair(s), ready to train."
        }
    }

    private func emptyDatasetSummary(directory: String?, mode: String) -> LoRATrainingDatasetPreflightSummary {
        LoRATrainingDatasetPreflightSummary(
            directory: directory,
            mode: mode,
            imageCount: 0,
            captionCount: 0,
            usablePairCount: 0,
            missingCaptionCount: 0,
            emptyCaptionCount: 0,
            duplicateCaptionGroupCount: 0,
            duplicateCaptionCount: 0,
            excludedPreviewImageCount: 0,
            placeholderCaptionCount: 0,
            syntheticSampleCount: nil
        )
    }

    private func modelResult(
        requested: String,
        kind: String,
        installed: Bool,
        path: String? = nil,
        family: String? = nil,
        upstreamRepoID: String? = nil,
        estimatedDownloadBytes: Int64? = nil
    ) -> LoRATrainingModelPreflightSummary {
        LoRATrainingModelPreflightSummary(
            requested: requested,
            kind: kind,
            installed: installed,
            path: path,
            family: family,
            upstreamRepoID: upstreamRepoID,
            estimatedDownloadBytes: estimatedDownloadBytes
        )
    }

    private func loadManifestFamily(
        at modelRoot: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> String? {
        do {
            let manifest = try MereRunModelManifest.loadRequired(from: modelRoot, fileManager: fileManager)
            let family = manifest.family?.rawValue ?? "unknown"
            if manifest.family != .krea && manifest.family != .klein {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "model_family_unsupported",
                        severity: .blocker,
                        title: "Unsupported model family",
                        message: "Unsupported LoRA training model family: \(family). Use a Krea 2 Raw or FLUX.2 Klein base model.",
                        locations: [.init(kind: "directory", path: modelRoot.path)]
                    )
                )
            }
            return family
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "model_manifest_unreadable",
                    severity: .blocker,
                    title: "Model manifest unreadable",
                    message: error.localizedDescription,
                    locations: [.init(kind: "directory", path: modelRoot.path)]
                )
            )
            return nil
        }
    }

    private static func isPlaceholderCaption(_ caption: String) -> Bool {
        let normalized = caption.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return [
            "todo",
            "tbd",
            "caption",
            "description",
            "image",
            "placeholder",
        ].contains(normalized)
    }
}
