import Foundation
import MereRunCore

struct LoRATrainingDatasetInspectionResult: Equatable {
    let summary: LoRATrainingDatasetPreflightSummary
    let diagnostics: [PreflightDiagnostic]
}

struct LoRATrainingDatasetDiscoveryRequest: Codable, Equatable {
    let root: String
    let maxDepth: Int
    let excludePreviewImages: Bool
    let minUsablePairs: Int
    let trainingOutputRoot: String?
    let trainingModel: String?
    let trainingRecipe: String?

    enum CodingKeys: String, CodingKey {
        case root
        case maxDepth = "max_depth"
        case excludePreviewImages = "exclude_preview_images"
        case minUsablePairs = "min_usable_pairs"
        case trainingOutputRoot = "training_output_root"
        case trainingModel = "training_model"
        case trainingRecipe = "training_recipe"
    }
}

struct LoRATrainingDatasetDiscoveryResult: Codable, Equatable {
    let root: String
    let scannedDirectoryCount: Int
    let candidateCount: Int
    let trainableCandidateCount: Int
    let candidates: [LoRATrainingDatasetDiscoveryCandidate]

    enum CodingKeys: String, CodingKey {
        case root
        case scannedDirectoryCount = "scanned_directory_count"
        case candidateCount = "candidate_count"
        case trainableCandidateCount = "trainable_candidate_count"
        case candidates
    }
}

struct LoRATrainingDatasetDiscoveryCandidate: Codable, Equatable {
    let id: String
    let name: String
    let path: String
    let relativePath: String
    let depth: Int
    let status: StructuredRunStatus
    let trainable: Bool
    let imageCount: Int
    let captionCount: Int
    let usablePairCount: Int
    let missingCaptionCount: Int
    let emptyCaptionCount: Int
    let duplicateCaptionGroupCount: Int
    let placeholderCaptionCount: Int
    let diagnostics: [PreflightDiagnostic]

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case path
        case relativePath = "relative_path"
        case depth
        case status
        case trainable
        case imageCount = "image_count"
        case captionCount = "caption_count"
        case usablePairCount = "usable_pair_count"
        case missingCaptionCount = "missing_caption_count"
        case emptyCaptionCount = "empty_caption_count"
        case duplicateCaptionGroupCount = "duplicate_caption_group_count"
        case placeholderCaptionCount = "placeholder_caption_count"
        case diagnostics
    }
}

typealias LoRATrainingDatasetDiscoveryEnvelope = StructuredRunEnvelope<
    LoRATrainingDatasetDiscoveryRequest,
    LoRATrainingDatasetDiscoveryResult
>

struct LoRATrainingDatasetInspector {
    let fileManager: FileManager

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
    }

    func inspect(
        directory: URL,
        excludePreviewImages: Bool
    ) -> LoRATrainingDatasetInspectionResult {
        let directory = directory.standardizedFileURL
        var diagnostics: [PreflightDiagnostic] = []
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
            return .init(
                summary: Self.emptyDatasetSummary(directory: directory.path, mode: "missing"),
                diagnostics: diagnostics
            )
        }

        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectoryResolvingSymlinks(
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
            return .init(
                summary: Self.emptyDatasetSummary(directory: directory.path, mode: "unreadable"),
                diagnostics: diagnostics
            )
        }

        let regularFiles = contents.filter { url in
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            return values?.isRegularFile == true
        }
        let allImages = regularFiles
            .filter { Self.imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let previewImages = allImages.filter { $0.deletingPathExtension().lastPathComponent.hasPrefix("preview") }
        let images = excludePreviewImages
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

        if !previewImages.isEmpty, !excludePreviewImages {
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

        return .init(
            summary: LoRATrainingDatasetPreflightSummary(
                directory: directory.path,
                mode: editOutputs.isEmpty ? "image_caption" : "edit_detected",
                imageCount: images.count,
                captionCount: captionCount,
                usablePairCount: usablePairCount,
                missingCaptionCount: missingCaptions.count,
                emptyCaptionCount: emptyCaptions.count,
                duplicateCaptionGroupCount: duplicateGroups.count,
                duplicateCaptionCount: duplicateCaptionCount,
                excludedPreviewImageCount: excludePreviewImages ? previewImages.count : 0,
                placeholderCaptionCount: placeholderCaptions.count,
                syntheticSampleCount: nil
            ),
            diagnostics: diagnostics
        )
    }

    static func emptyDatasetSummary(directory: String?, mode: String) -> LoRATrainingDatasetPreflightSummary {
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

    private static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp"]

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

struct LoRATrainingDatasetDiscoveryAnalysis {
    let result: LoRATrainingDatasetDiscoveryResult
    let diagnostics: [PreflightDiagnostic]
    let actions: [DeclarativeAction]
    let status: StructuredRunStatus
    let summary: String
}

struct LoRATrainingDatasetDiscoveryAnalyzer {
    let root: String
    let maxDepth: Int
    let excludePreviewImages: Bool
    let minUsablePairs: Int
    let trainingOutputRoot: String?
    let trainingModel: String?
    let trainingRecipe: String?
    let fileManager: FileManager
    let cwd: String
    let now: () -> Date

    init(
        root: String,
        maxDepth: Int = 4,
        excludePreviewImages: Bool = false,
        minUsablePairs: Int = 1,
        trainingOutputRoot: String? = nil,
        trainingModel: String? = nil,
        trainingRecipe: String? = nil,
        fileManager: FileManager = .default,
        cwd: String? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.root = root
        self.maxDepth = maxDepth
        self.excludePreviewImages = excludePreviewImages
        self.minUsablePairs = minUsablePairs
        self.trainingOutputRoot = trainingOutputRoot
        self.trainingModel = trainingModel
        self.trainingRecipe = trainingRecipe
        self.fileManager = fileManager
        self.cwd = cwd ?? fileManager.currentDirectoryPath
        self.now = now
    }

    func envelope() -> LoRATrainingDatasetDiscoveryEnvelope {
        let analysis = analyze()
        return LoRATrainingDatasetDiscoveryEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["image", "dataset", "discover"],
            mode: .inspection,
            status: analysis.status,
            createdAt: now(),
            cwd: cwd,
            summary: analysis.summary,
            request: LoRATrainingDatasetDiscoveryRequest(
                root: root,
                maxDepth: maxDepth,
                excludePreviewImages: excludePreviewImages,
                minUsablePairs: minUsablePairs,
                trainingOutputRoot: trainingOutputRoot.map { URL(fileURLWithPath: $0).standardizedFileURL.path },
                trainingModel: trainingModel,
                trainingRecipe: trainingRecipe
            ),
            result: analysis.result,
            diagnostics: analysis.diagnostics,
            actions: analysis.actions
        )
    }

    func analyze() -> LoRATrainingDatasetDiscoveryAnalysis {
        let rootURL = URL(fileURLWithPath: root).standardizedFileURL
        var diagnostics: [PreflightDiagnostic] = []
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDirectory) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dataset_root_not_found",
                    severity: .blocker,
                    title: "Dataset root not found",
                    message: "Dataset root not found: \(rootURL.path)",
                    locations: [.init(kind: "directory", path: rootURL.path)]
                )
            )
            return emptyAnalysis(root: rootURL.path, diagnostics: diagnostics)
        }
        guard isDirectory.boolValue else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dataset_root_not_directory",
                    severity: .blocker,
                    title: "Dataset root is not a directory",
                    message: "Dataset root is not a directory: \(rootURL.path)",
                    locations: [.init(kind: "file", path: rootURL.path)]
                )
            )
            return emptyAnalysis(root: rootURL.path, diagnostics: diagnostics)
        }

        var scannedDirectoryCount = 0
        var candidates: [LoRATrainingDatasetDiscoveryCandidate] = []
        var queue: [(url: URL, depth: Int)] = [(rootURL, 0)]
        var index = 0
        let inspector = LoRATrainingDatasetInspector(fileManager: fileManager)

        while index < queue.count {
            let current = queue[index]
            index += 1
            scannedDirectoryCount += 1

            let inspection = inspector.inspect(
                directory: current.url,
                excludePreviewImages: excludePreviewImages
            )
            if Self.isDatasetCandidate(inspection.summary) {
                candidates.append(
                    candidate(
                        for: current.url,
                        root: rootURL,
                        depth: current.depth,
                        inspection: inspection
                    )
                )
            }

            guard current.depth < maxDepth else { continue }
            let childDirectories: [URL]
            do {
                childDirectories = try fileManager.contentsOfDirectoryResolvingSymlinks(
                    at: current.url,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                )
                .filter { url in
                    let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
                    return values?.isDirectory == true
                }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            } catch {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "dataset_directory_unreadable",
                        severity: .warning,
                        title: "Dataset directory unreadable",
                        message: error.localizedDescription,
                        locations: [.init(kind: "directory", path: current.url.path)]
                    )
                )
                continue
            }
            queue.append(contentsOf: childDirectories.map { ($0, current.depth + 1) })
        }

        let trainableCandidateCount = candidates.filter(\.trainable).count
        if candidates.isEmpty {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "no_datasets_found",
                    severity: .warning,
                    title: "No datasets found",
                    message: "No image-caption dataset candidates were found under \(rootURL.path).",
                    locations: [.init(kind: "directory", path: rootURL.path)]
                )
            )
        } else if trainableCandidateCount == 0 {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "no_trainable_datasets_found",
                    severity: .warning,
                    title: "No trainable datasets found",
                    message: "\(candidates.count) candidate dataset(s) were found, but none have enough usable captioned image pairs.",
                    locations: candidates.prefix(10).map { .init(kind: "directory", path: $0.path) }
                )
            )
        } else if candidates.contains(where: { $0.status == .warning || $0.status == .blocked }) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "dataset_candidates_need_review",
                    severity: .warning,
                    title: "Some dataset candidates need review",
                    message: "Some discovered dataset candidates have warnings or blockers.",
                    locations: candidates
                        .filter { $0.status == .warning || $0.status == .blocked }
                        .prefix(10)
                        .map { .init(kind: "directory", path: $0.path) }
                )
            )
        }

        let result = LoRATrainingDatasetDiscoveryResult(
            root: rootURL.path,
            scannedDirectoryCount: scannedDirectoryCount,
            candidateCount: candidates.count,
            trainableCandidateCount: trainableCandidateCount,
            candidates: candidates
        )
        let status = StructuredRunOutput.status(for: diagnostics)
        return LoRATrainingDatasetDiscoveryAnalysis(
            result: result,
            diagnostics: diagnostics,
            actions: actions(root: rootURL.path, candidates: candidates),
            status: status,
            summary: summary(
                candidateCount: candidates.count,
                trainableCandidateCount: trainableCandidateCount,
                scannedDirectoryCount: scannedDirectoryCount
            )
        )
    }

    private func emptyAnalysis(
        root: String,
        diagnostics: [PreflightDiagnostic]
    ) -> LoRATrainingDatasetDiscoveryAnalysis {
        let result = LoRATrainingDatasetDiscoveryResult(
            root: root,
            scannedDirectoryCount: 0,
            candidateCount: 0,
            trainableCandidateCount: 0,
            candidates: []
        )
        let status = StructuredRunOutput.status(for: diagnostics)
        return LoRATrainingDatasetDiscoveryAnalysis(
            result: result,
            diagnostics: diagnostics,
            actions: actions(root: root, candidates: []),
            status: status,
            summary: "0 dataset candidate(s), 0 trainable."
        )
    }

    private func candidate(
        for directory: URL,
        root: URL,
        depth: Int,
        inspection: LoRATrainingDatasetInspectionResult
    ) -> LoRATrainingDatasetDiscoveryCandidate {
        let relativePath = Self.relativePath(for: directory, root: root)
        let status = StructuredRunOutput.status(for: inspection.diagnostics)
        let trainable = status != .blocked && inspection.summary.usablePairCount >= minUsablePairs
        return LoRATrainingDatasetDiscoveryCandidate(
            id: Self.candidateID(for: relativePath),
            name: relativePath == "." ? directory.lastPathComponent : directory.lastPathComponent,
            path: directory.path,
            relativePath: relativePath,
            depth: depth,
            status: status,
            trainable: trainable,
            imageCount: inspection.summary.imageCount,
            captionCount: inspection.summary.captionCount,
            usablePairCount: inspection.summary.usablePairCount,
            missingCaptionCount: inspection.summary.missingCaptionCount,
            emptyCaptionCount: inspection.summary.emptyCaptionCount,
            duplicateCaptionGroupCount: inspection.summary.duplicateCaptionGroupCount,
            placeholderCaptionCount: inspection.summary.placeholderCaptionCount,
            diagnostics: inspection.diagnostics
        )
    }

    private static func isDatasetCandidate(_ summary: LoRATrainingDatasetPreflightSummary) -> Bool {
        summary.usablePairCount > 0 || summary.captionCount > 0
    }

    private func actions(
        root: String,
        candidates: [LoRATrainingDatasetDiscoveryCandidate]
    ) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = [
            DeclarativeAction(
                id: "open-root",
                label: "Open root",
                kind: .openDirectory,
                style: .link,
                enabled: fileManager.fileExists(atPath: root),
                path: root
            )
        ]
        let trainableCandidates = candidates.filter(\.trainable)
        if !trainableCandidates.isEmpty {
            actions.append(
                DeclarativeAction(
                    id: "choose-dataset",
                    label: "Choose dataset",
                    kind: .select,
                    style: .primary,
                    candidates: trainableCandidates.map { candidate in
                        Self.actionCandidate(
                            candidate,
                            command: trainingCommand(for: candidate),
                            patches: Self.datasetSelectionPatches(for: candidate)
                        )
                    }
                )
            )
        }
        return actions
    }

    private func summary(
        candidateCount: Int,
        trainableCandidateCount: Int,
        scannedDirectoryCount: Int
    ) -> String {
        "\(candidateCount) dataset candidate(s), \(trainableCandidateCount) trainable, scanned \(scannedDirectoryCount) directories."
    }

    static func actionCandidate(
        _ candidate: LoRATrainingDatasetDiscoveryCandidate,
        command: DeclarativeCommand? = nil,
        patches: [DeclarativePatch] = []
    ) -> DeclarativeActionCandidate {
        DeclarativeActionCandidate(
            id: candidate.id,
            label: candidate.relativePath,
            value: candidate.path,
            path: candidate.path,
            status: candidate.status.rawValue,
            description: "\(candidate.usablePairCount) usable pair(s)",
            command: command,
            patches: patches
        )
    }

    static func datasetSelectionPatches(
        for candidate: LoRATrainingDatasetDiscoveryCandidate
    ) -> [DeclarativePatch] {
        [
            DeclarativePatch(
                op: .replace,
                path: "request.data",
                value: candidate.path
            ),
            DeclarativePatch(
                op: .replace,
                path: "run_plan.arguments.data",
                value: candidate.path
            ),
        ]
    }

    private func trainingCommand(
        for candidate: LoRATrainingDatasetDiscoveryCandidate
    ) -> DeclarativeCommand? {
        guard let trainingOutputRoot else {
            return nil
        }
        let outputRootURL = URL(fileURLWithPath: trainingOutputRoot).standardizedFileURL
        let outputURL = outputRootURL.appendingPathComponent("\(candidate.id).safetensors", isDirectory: false)
        var argv = [
            "mere.run",
            "image",
            "train-lora",
            "--data",
            candidate.path,
            "--output",
            outputURL.path,
        ]
        if let trainingModel {
            argv += ["--model", trainingModel]
        }
        if let trainingRecipe {
            argv += ["--recipe", trainingRecipe]
        }
        argv += ["--preflight", "--json"]
        return DeclarativeCommand(
            argv: argv,
            cwd: cwd,
            commandPath: ["image", "train-lora"]
        )
    }

    private static func relativePath(for url: URL, root: URL) -> String {
        let rootPath = root.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        if path == rootPath {
            return "."
        }
        let prefix = rootPath.hasSuffix("/") ? rootPath : "\(rootPath)/"
        if path.hasPrefix(prefix) {
            return String(path.dropFirst(prefix.count))
        }
        return path
    }

    private static func candidateID(for relativePath: String) -> String {
        var output = ""
        var previousWasDash = false
        for scalar in relativePath.lowercased().unicodeScalars {
            let isAllowed = CharacterSet.alphanumerics.contains(scalar)
            if isAllowed {
                output.unicodeScalars.append(scalar)
                previousWasDash = false
            } else if !previousWasDash {
                output.append("-")
                previousWasDash = true
            }
        }
        let trimmed = output.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return trimmed.isEmpty ? "dataset" : trimmed
    }
}
