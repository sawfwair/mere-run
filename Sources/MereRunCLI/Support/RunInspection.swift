import Foundation
import MereRunCore

struct RunInspectionRequest: Codable, Equatable {
    let path: String

    enum CodingKeys: String, CodingKey {
        case path
    }
}

struct RunInspectionResult: Codable, Equatable {
    let kind: String
    let path: String
    let runDirectory: RunDirectoryInspectionSummary?
    let report: StructuredReportInspectionSummary?
    let plan: RunPlanInspectionSummary?

    enum CodingKeys: String, CodingKey {
        case kind
        case path
        case runDirectory = "run_directory"
        case report
        case plan
    }
}

struct RunDirectoryInspectionSummary: Codable, Equatable {
    let path: String
    let status: String
    let manifestPath: String?
    let planPath: String?
    let actionsPath: String?
    let eventPaths: [String]
    let manifest: RunManifestInspectionSummary?
    let legacyManifest: LegacyRunManifestInspectionSummary?
    let events: RunEventsInspectionSummary
    let actions: [RunActionInspectionSummary]
    let artifacts: [RunArtifactInspectionSummary]
    let metrics: RunMetricsInspectionSummary

    enum CodingKeys: String, CodingKey {
        case path
        case status
        case manifestPath = "manifest_path"
        case planPath = "plan_path"
        case actionsPath = "actions_path"
        case eventPaths = "event_paths"
        case manifest
        case legacyManifest = "legacy_manifest"
        case events
        case actions
        case artifacts
        case metrics
    }
}

struct RunManifestInspectionSummary: Codable, Equatable {
    let createdAt: Date
    let format: String
    let model: String
    let step: Int
    let totalSteps: Int
    let progress: Double?
    let seed: UInt64
    let isEdit: Bool
    let dataRoot: String?
    let checkpointFiles: [String: String]

    enum CodingKeys: String, CodingKey {
        case createdAt = "created_at"
        case format
        case model
        case step
        case totalSteps = "total_steps"
        case progress
        case seed
        case isEdit = "is_edit"
        case dataRoot = "data_root"
        case checkpointFiles = "checkpoint_files"
    }
}

struct LegacyRunManifestInspectionSummary: Codable, Equatable {
    let contractVersion: String?
    let runID: String?
    let status: String?
    let createdAt: Date?
    let updatedAt: Date?
    let pluginName: String?
    let provider: String?
    let gpu: String?
    let datasetPath: String?
    let datasetPairCount: Int?
    let recipeID: String?
    let trainModel: String?
    let applyModel: String?
    let command: [String]

    enum CodingKeys: String, CodingKey {
        case contractVersion = "contract_version"
        case runID = "run_id"
        case status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case pluginName = "plugin_name"
        case provider
        case gpu
        case datasetPath = "dataset_path"
        case datasetPairCount = "dataset_pair_count"
        case recipeID = "recipe_id"
        case trainModel = "train_model"
        case applyModel = "apply_model"
        case command
    }
}

struct RunEventsInspectionSummary: Codable, Equatable {
    let count: Int
    let latest: LoRATrainingRunEvent?
    let types: [String]

    enum CodingKeys: String, CodingKey {
        case count
        case latest
        case types
    }
}

struct RunActionInspectionSummary: Codable, Equatable {
    let id: String
    let label: String
    let kind: String
    let enabled: Bool
    let commandPath: [String]?

    enum CodingKeys: String, CodingKey {
        case id
        case label
        case kind
        case enabled
        case commandPath = "command_path"
    }
}

struct RunArtifactInspectionSummary: Codable, Equatable {
    let kind: String
    let name: String
    let path: String
    let exists: Bool
    let sizeBytes: Int64?
    let isImage: Bool

    enum CodingKeys: String, CodingKey {
        case kind
        case name
        case path
        case exists
        case sizeBytes = "size_bytes"
        case isImage = "is_image"
    }
}

struct RunMetricsInspectionSummary: Codable, Equatable {
    let lossPath: String?
    let lossPointCount: Int
    let firstStep: Int?
    let latestStep: Int?
    let firstLoss: Float?
    let latestLoss: Float?
    let minLoss: Float?
    let maxLoss: Float?
    let sampleImageCount: Int
    let checkpointCount: Int
    let adapterCount: Int

    enum CodingKeys: String, CodingKey {
        case lossPath = "loss_path"
        case lossPointCount = "loss_point_count"
        case firstStep = "first_step"
        case latestStep = "latest_step"
        case firstLoss = "first_loss"
        case latestLoss = "latest_loss"
        case minLoss = "min_loss"
        case maxLoss = "max_loss"
        case sampleImageCount = "sample_image_count"
        case checkpointCount = "checkpoint_count"
        case adapterCount = "adapter_count"
    }
}

struct StructuredReportInspectionSummary: Codable, Equatable {
    let schemaVersion: Int
    let command: [String]
    let mode: StructuredRunMode
    let status: StructuredRunStatus
    let createdAt: Date?
    let summary: String
    let diagnosticCount: Int
    let actionCount: Int

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case command
        case mode
        case status
        case createdAt = "created_at"
        case summary
        case diagnosticCount = "diagnostic_count"
        case actionCount = "action_count"
    }
}

struct RunPlanInspectionSummary: Codable, Equatable {
    let schemaVersion: Int
    let kind: String
    let command: [String]
    let createdAt: Date?
    let cwd: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case command
        case createdAt = "created_at"
        case cwd
    }
}

typealias RunInspectionEnvelope = StructuredRunEnvelope<
    RunInspectionRequest,
    RunInspectionResult
>

struct RunListRequest: Codable, Equatable {
    let root: String
    let maxDepth: Int

    enum CodingKeys: String, CodingKey {
        case root
        case maxDepth = "max_depth"
    }
}

struct RunListResult: Codable, Equatable {
    let root: String
    let scannedDirectoryCount: Int
    let entryCount: Int
    let entries: [RunListEntry]

    enum CodingKeys: String, CodingKey {
        case root
        case scannedDirectoryCount = "scanned_directory_count"
        case entryCount = "entry_count"
        case entries
    }
}

struct RunListEntry: Codable, Equatable {
    let id: String
    let kind: String
    let path: String
    let relativePath: String
    let depth: Int
    let status: StructuredRunStatus
    let state: String?
    let createdAt: Date?
    let updatedAt: Date?
    let summary: String
    let command: [String]?
    let format: String?
    let eventCount: Int?
    let artifactCount: Int?
    let diagnosticCount: Int
    let blockerCount: Int
    let actions: [DeclarativeAction]

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case path
        case relativePath = "relative_path"
        case depth
        case status
        case state
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case summary
        case command
        case format
        case eventCount = "event_count"
        case artifactCount = "artifact_count"
        case diagnosticCount = "diagnostic_count"
        case blockerCount = "blocker_count"
        case actions
    }
}

typealias RunListEnvelope = StructuredRunEnvelope<
    RunListRequest,
    RunListResult
>

struct RunListAnalysis {
    let result: RunListResult
    let diagnostics: [PreflightDiagnostic]
    let actions: [DeclarativeAction]
    let status: StructuredRunStatus
    let summary: String
}

struct RunInspectionAnalyzer {
    let path: String
    let fileManager: FileManager
    let now: () -> Date

    init(
        path: String,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) {
        self.path = path
        self.fileManager = fileManager
        self.now = now
    }

    func envelope() -> RunInspectionEnvelope {
        var diagnostics: [PreflightDiagnostic] = []
        let targetURL = URL(fileURLWithPath: path).standardizedFileURL
        let result = inspect(targetURL, diagnostics: &diagnostics)
        let status = StructuredRunOutput.status(for: diagnostics)
        return RunInspectionEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["run", "inspect"],
            mode: .inspection,
            status: status,
            createdAt: now(),
            cwd: fileManager.currentDirectoryPath,
            summary: summary(status: status, result: result, diagnostics: diagnostics),
            request: RunInspectionRequest(path: targetURL.path),
            result: result,
            diagnostics: diagnostics,
            actions: actions(for: result, cwd: fileManager.currentDirectoryPath)
        )
    }

    private func inspect(
        _ targetURL: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> RunInspectionResult {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: targetURL.path, isDirectory: &isDirectory) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "path_missing",
                    severity: .blocker,
                    title: "Path missing",
                    message: "Run inspection path not found: \(targetURL.path)",
                    locations: [.init(kind: "file", path: targetURL.path)]
                )
            )
            return RunInspectionResult(kind: "missing", path: targetURL.path, runDirectory: nil, report: nil, plan: nil)
        }
        if isDirectory.boolValue {
            return RunInspectionResult(
                kind: "run_directory",
                path: targetURL.path,
                runDirectory: inspectRunDirectory(targetURL, diagnostics: &diagnostics),
                report: nil,
                plan: nil
            )
        }

        if let report = try? inspectReportFile(targetURL) {
            return RunInspectionResult(kind: "report_file", path: targetURL.path, runDirectory: nil, report: report, plan: nil)
        }
        if let plan = try? inspectPlanFile(targetURL) {
            return RunInspectionResult(kind: "plan_file", path: targetURL.path, runDirectory: nil, report: nil, plan: plan)
        }

        diagnostics.append(
            PreflightDiagnostic(
                id: "unsupported_file",
                severity: .blocker,
                title: "Unsupported file",
                message: "Expected a run directory, structured report JSON, or run plan JSON.",
                locations: [.init(kind: "file", path: targetURL.path)]
            )
        )
        return RunInspectionResult(kind: "unsupported_file", path: targetURL.path, runDirectory: nil, report: nil, plan: nil)
    }

    private func inspectRunDirectory(
        _ runDirectoryURL: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> RunDirectoryInspectionSummary {
        let manifestURL = runDirectoryURL.appendingPathComponent(LoRATrainingRunManifest.filename, isDirectory: false)
        let manifestExists = fileManager.fileExists(atPath: manifestURL.path)
        let manifestResult = loadManifest(at: manifestURL, diagnostics: &diagnostics)
        let manifest = manifestResult.native
        let legacyManifest = manifestResult.legacy
        let planPath = manifest?.checkpointFiles["plan"].map {
            runDirectoryURL.appendingPathComponent($0, isDirectory: false).path
        } ?? existingPath(runDirectoryURL.appendingPathComponent("plan.json", isDirectory: false))
        let actionsPath = manifest?.checkpointFiles["actions"].map {
            runDirectoryURL.appendingPathComponent($0, isDirectory: false).path
        } ?? existingPath(runDirectoryURL.appendingPathComponent("actions.json", isDirectory: false))
        let eventPaths = eventPaths(in: runDirectoryURL, manifest: manifest)
        var seenEvents: Set<LoRATrainingRunEvent> = []
        let events = eventPaths.flatMap { (try? LoRATrainingRunEvent.load(from: URL(fileURLWithPath: $0))) ?? [] }
            .filter { seenEvents.insert($0).inserted }
            .sorted { lhs, rhs in
                if lhs.sequence == rhs.sequence {
                    return lhs.createdAt < rhs.createdAt
                }
                return lhs.sequence < rhs.sequence
            }
        let actions = loadActions(path: actionsPath, diagnostics: &diagnostics)
        let artifacts = artifactSummaries(runDirectoryURL: runDirectoryURL, manifest: manifest)
        let metrics = metricsSummary(runDirectoryURL: runDirectoryURL, artifacts: artifacts)
        if manifest == nil && !manifestExists {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "run_manifest_missing",
                    severity: .warning,
                    title: "Run manifest missing",
                    message: "No run.json manifest found in \(runDirectoryURL.path).",
                    locations: [.init(kind: "file", path: manifestURL.path)]
                )
            )
        }

        return RunDirectoryInspectionSummary(
            path: runDirectoryURL.path,
            status: status(manifest: manifest, legacyManifest: legacyManifest, events: events),
            manifestPath: existingPath(manifestURL),
            planPath: planPath,
            actionsPath: actionsPath,
            eventPaths: eventPaths,
            manifest: manifest.map(manifestSummary),
            legacyManifest: legacyManifest,
            events: eventsSummary(events),
            actions: actions,
            artifacts: artifacts,
            metrics: metrics
        )
    }

    private func loadManifest(
        at manifestURL: URL,
        diagnostics: inout [PreflightDiagnostic]
    ) -> (native: LoRATrainingRunManifest?, legacy: LegacyRunManifestInspectionSummary?) {
        guard fileManager.fileExists(atPath: manifestURL.path) else { return (nil, nil) }
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "run_manifest_decode_failed",
                    severity: .blocker,
                    title: "Run manifest decode failed",
                    message: error.localizedDescription,
                    locations: [.init(kind: "file", path: manifestURL.path)]
                )
            )
            return (nil, nil)
        }

        do {
            return (try LoRATrainingRunManifest.decode(from: data), nil)
        } catch {
            if let legacy = try? decodeLegacyManifest(from: data) {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "legacy_run_manifest_detected",
                        severity: .warning,
                        title: "Legacy run manifest",
                        message: "run.json is a legacy/plugin manifest, not a native mere.run training manifest.",
                        locations: [.init(kind: "file", path: manifestURL.path)]
                    )
                )
                return (nil, legacy)
            }
            diagnostics.append(
                PreflightDiagnostic(
                    id: "run_manifest_decode_failed",
                    severity: .blocker,
                    title: "Run manifest decode failed",
                    message: error.localizedDescription,
                    locations: [.init(kind: "file", path: manifestURL.path)]
                )
            )
            return (nil, nil)
        }
    }

    private func loadActions(
        path: String?,
        diagnostics: inout [PreflightDiagnostic]
    ) -> [RunActionInspectionSummary] {
        guard let path else { return [] }
        let url = URL(fileURLWithPath: path)
        guard fileManager.fileExists(atPath: url.path) else { return [] }
        do {
            let decoded = try JSONDecoder().decode([DeclarativeAction].self, from: Data(contentsOf: url))
            return decoded.map {
                RunActionInspectionSummary(
                    id: $0.id,
                    label: $0.label,
                    kind: $0.kind.rawValue,
                    enabled: $0.enabled,
                    commandPath: $0.command?.commandPath
                )
            }
        } catch {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "actions_decode_failed",
                    severity: .warning,
                    title: "Actions decode failed",
                    message: error.localizedDescription,
                    locations: [.init(kind: "file", path: url.path)]
                )
            )
            return []
        }
    }

    private func inspectReportFile(_ url: URL) throws -> StructuredReportInspectionSummary {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let report = try decoder.decode(StructuredReportHeader.self, from: data)
        return StructuredReportInspectionSummary(
            schemaVersion: report.schemaVersion,
            command: report.command,
            mode: report.mode,
            status: report.status,
            createdAt: report.createdAt,
            summary: report.summary,
            diagnosticCount: report.diagnostics.count,
            actionCount: report.actions.count
        )
    }

    private func inspectPlanFile(_ url: URL) throws -> RunPlanInspectionSummary {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let plan = try decoder.decode(RunPlanHeader.self, from: data)
        return RunPlanInspectionSummary(
            schemaVersion: plan.schemaVersion,
            kind: plan.kind,
            command: plan.command,
            createdAt: plan.createdAt,
            cwd: plan.cwd
        )
    }

    private func decodeLegacyManifest(from data: Data) throws -> LegacyRunManifestInspectionSummary {
        let decoder = JSONDecoder()
        let manifest = try decoder.decode(LegacyRunManifestHeader.self, from: data)
        guard manifest.contractVersion != nil ||
            manifest.runID != nil ||
            manifest.plugin?.name != nil ||
            manifest.remote?.provider != nil else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Missing legacy run identity fields."
                )
            )
        }
        return LegacyRunManifestInspectionSummary(
            contractVersion: manifest.contractVersion,
            runID: manifest.runID,
            status: manifest.status,
            createdAt: Self.parseLegacyDate(manifest.createdAt),
            updatedAt: Self.parseLegacyDate(manifest.updatedAt),
            pluginName: manifest.plugin?.name,
            provider: manifest.remote?.provider,
            gpu: manifest.remote?.gpu,
            datasetPath: manifest.dataset?.path,
            datasetPairCount: manifest.dataset?.pairCount,
            recipeID: manifest.recipe?.id,
            trainModel: manifest.recipe?.trainModel,
            applyModel: manifest.recipe?.applyModel,
            command: manifest.command ?? []
        )
    }

    private func eventPaths(
        in runDirectoryURL: URL,
        manifest: LoRATrainingRunManifest?
    ) -> [String] {
        var paths: Set<String> = []
        if let eventFile = manifest?.checkpointFiles["events"] {
            paths.insert(
                runDirectoryURL.appendingPathComponent(eventFile, isDirectory: false)
                    .standardizedFileURL.path
            )
        }
        let urls = (try? fileManager.contentsOfDirectory(
            at: runDirectoryURL,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        for url in urls where url.lastPathComponent.hasSuffix(LoRATrainingRunEvent.filenameSuffix) {
            paths.insert(url.standardizedFileURL.path)
        }
        return paths.sorted()
    }

    private func artifactSummaries(
        runDirectoryURL: URL,
        manifest: LoRATrainingRunManifest?
    ) -> [RunArtifactInspectionSummary] {
        let manifestArtifacts = manifest?.checkpointFiles.map { key, value in
            artifactSummary(kind: key, url: runDirectoryURL.appendingPathComponent(value, isDirectory: false))
        } ?? []
        let rootArtifacts = rootArtifactSummaries(runDirectoryURL: runDirectoryURL)
        let commonDirectories = ["artifacts", "samples", "early-samples", "checkpoints", "logs"].flatMap { name -> [RunArtifactInspectionSummary] in
            let directoryURL = runDirectoryURL.appendingPathComponent(name, isDirectory: true)
            return artifactSummaries(in: directoryURL, kind: name, maxDepth: 2)
        }
        var seenPaths: Set<String> = []
        return (manifestArtifacts + rootArtifacts + commonDirectories)
            .filter { seenPaths.insert($0.path).inserted }
            .sorted { lhs, rhs in
                if lhs.kind == rhs.kind {
                    return lhs.name < rhs.name
                }
                return lhs.kind < rhs.kind
            }
    }

    private func rootArtifactSummaries(runDirectoryURL: URL) -> [RunArtifactInspectionSummary] {
        let excluded = Set([
            LoRATrainingRunManifest.filename,
            "actions.json",
            "plan.json",
        ])
        let urls = (try? fileManager.contentsOfDirectory(
            at: runDirectoryURL,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls.compactMap { url in
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true,
                  !excluded.contains(url.lastPathComponent),
                  Self.artifactExtensions.contains(url.pathExtension.lowercased()) else {
                return nil
            }
            return artifactSummary(kind: "root", url: url)
        }
    }

    private func artifactSummaries(
        in directoryURL: URL,
        kind: String,
        maxDepth: Int
    ) -> [RunArtifactInspectionSummary] {
        guard maxDepth >= 0 else { return [] }
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directoryURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return []
        }
        let urls = (try? fileManager.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls.flatMap { url -> [RunArtifactInspectionSummary] in
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                return artifactSummaries(in: url, kind: kind, maxDepth: maxDepth - 1)
            }
            guard values?.isRegularFile == true,
                  Self.artifactExtensions.contains(url.pathExtension.lowercased()) else {
                return []
            }
            return [artifactSummary(kind: kind, url: url)]
        }
    }

    private func artifactSummary(kind: String, url: URL) -> RunArtifactInspectionSummary {
        let standardizedURL = url.standardizedFileURL
        let exists = fileManager.fileExists(atPath: standardizedURL.path)
        let values = try? standardizedURL.resourceValues(forKeys: [.fileSizeKey])
        return RunArtifactInspectionSummary(
            kind: kind,
            name: standardizedURL.lastPathComponent,
            path: standardizedURL.path,
            exists: exists,
            sizeBytes: values?.fileSize.map(Int64.init),
            isImage: ["png", "jpg", "jpeg", "webp"].contains(standardizedURL.pathExtension.lowercased())
        )
    }

    private func metricsSummary(
        runDirectoryURL: URL,
        artifacts: [RunArtifactInspectionSummary]
    ) -> RunMetricsInspectionSummary {
        let lossURL = firstLossCSVURL(runDirectoryURL: runDirectoryURL, artifacts: artifacts)
        let points = lossURL.flatMap { try? LoRATrainingMetricsLogger.loadPoints(from: $0) } ?? []
        let sortedPoints = points.sorted { lhs, rhs in
            lhs.step < rhs.step
        }
        let losses = sortedPoints.map(\.loss)
        return RunMetricsInspectionSummary(
            lossPath: lossURL?.path,
            lossPointCount: sortedPoints.count,
            firstStep: sortedPoints.first?.step,
            latestStep: sortedPoints.last?.step,
            firstLoss: sortedPoints.first?.loss,
            latestLoss: sortedPoints.last?.loss,
            minLoss: losses.min(),
            maxLoss: losses.max(),
            sampleImageCount: artifacts.filter { Self.isSampleArtifact($0) && $0.isImage }.count,
            checkpointCount: artifacts.filter(Self.isCheckpointArtifact).count,
            adapterCount: artifacts.filter { $0.path.lowercased().hasSuffix(".safetensors") }.count
        )
    }

    private func firstLossCSVURL(
        runDirectoryURL: URL,
        artifacts: [RunArtifactInspectionSummary]
    ) -> URL? {
        if let path = artifacts
            .filter({ $0.name.hasSuffix(".loss.csv") || $0.kind == "loss_csv" })
            .sorted(by: { $0.name < $1.name })
            .first?
            .path {
            return URL(fileURLWithPath: path)
        }
        let rootCandidates = (try? fileManager.contentsOfDirectory(
            at: runDirectoryURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return rootCandidates
            .filter { url in
                let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
                return values?.isRegularFile == true && url.lastPathComponent.hasSuffix(".loss.csv")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first
    }

    private func manifestSummary(_ manifest: LoRATrainingRunManifest) -> RunManifestInspectionSummary {
        let progress: Double?
        if manifest.totalSteps > 0 {
            progress = min(1, max(0, Double(manifest.step) / Double(manifest.totalSteps)))
        } else {
            progress = nil
        }
        return RunManifestInspectionSummary(
            createdAt: manifest.createdAt,
            format: manifest.format,
            model: manifest.model,
            step: manifest.step,
            totalSteps: manifest.totalSteps,
            progress: progress,
            seed: manifest.seed,
            isEdit: manifest.isEdit,
            dataRoot: manifest.dataRoot,
            checkpointFiles: manifest.checkpointFiles
        )
    }

    private func eventsSummary(_ events: [LoRATrainingRunEvent]) -> RunEventsInspectionSummary {
        RunEventsInspectionSummary(
            count: events.count,
            latest: events.last,
            types: events.map(\.type)
        )
    }

    private func status(
        manifest: LoRATrainingRunManifest?,
        legacyManifest: LegacyRunManifestInspectionSummary?,
        events: [LoRATrainingRunEvent]
    ) -> String {
        let latest = events.last
        if latest?.type == "run_failed" { return "failed" }
        if latest?.type == "run_finished" { return "finished" }
        if let manifest, manifest.totalSteps > 0, manifest.step >= manifest.totalSteps { return "finished" }
        if latest?.type == "run_planned" { return "planned" }
        if !events.isEmpty { return "running" }
        if let legacyStatus = legacyManifest?.status { return legacyStatus }
        return manifest == nil ? "unknown" : "idle"
    }

    private func existingPath(_ url: URL) -> String? {
        fileManager.fileExists(atPath: url.path) ? url.path : nil
    }

    private static let artifactExtensions: Set<String> = [
        "csv",
        "html",
        "jpeg",
        "jpg",
        "json",
        "log",
        "png",
        "safetensors",
        "webp",
        "zip",
    ]

    private static func parseLegacyDate(_ value: String?) -> Date? {
        guard let value else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: value) {
            return date
        }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: value)
    }

    private static func isSampleArtifact(_ artifact: RunArtifactInspectionSummary) -> Bool {
        if ["samples", "early-samples"].contains(artifact.kind) {
            return true
        }
        let lowercasedName = artifact.name.lowercased()
        return artifact.isImage &&
            (lowercasedName.contains("sample") || lowercasedName.hasPrefix("step"))
    }

    private static func isCheckpointArtifact(_ artifact: RunArtifactInspectionSummary) -> Bool {
        if artifact.kind == "checkpoints" {
            return true
        }
        let lowercasedName = artifact.name.lowercased()
        return lowercasedName.contains("checkpoint") ||
            lowercasedName.contains("optimizer") ||
            lowercasedName.contains("state")
    }

    private func actions(for result: RunInspectionResult, cwd: String) -> [DeclarativeAction] {
        var actions: [DeclarativeAction] = []
        if result.kind == "run_directory" {
            actions.append(
                DeclarativeAction(
                    id: "open-run-directory",
                    label: "Open run directory",
                    kind: .openDirectory,
                    style: .link,
                    path: result.path
                )
            )
        }
        if let planPath = result.runDirectory?.planPath {
            actions.append(
                DeclarativeAction(
                    id: "preflight-plan",
                    label: "Preflight plan",
                    kind: .command,
                    style: .secondary,
                    command: DeclarativeCommand(
                        argv: ["mere.run", "image", "run-plan", planPath, "--preflight", "--json"],
                        cwd: cwd,
                        commandPath: ["image", "run-plan"]
                    )
                )
            )
        }
        return actions
    }

    private func summary(
        status: StructuredRunStatus,
        result: RunInspectionResult,
        diagnostics: [PreflightDiagnostic]
    ) -> String {
        switch status {
        case .ok:
            return "Inspected \(result.kind) at \(result.path)."
        case .warning:
            return "Inspected \(result.kind) with \(diagnostics.count) warning(s)."
        case .blocked:
            let blockers = diagnostics.filter { $0.severity == .blocker }.count
            return "Run inspection blocked by \(blockers) issue(s)."
        default:
            return "Run inspection status: \(status.rawValue)."
        }
    }
}

struct RunListAnalyzer {
    let root: String
    let maxDepth: Int
    let fileManager: FileManager
    let cwd: String
    let now: () -> Date

    init(
        root: String,
        maxDepth: Int = 4,
        fileManager: FileManager = .default,
        cwd: String? = nil,
        now: @escaping () -> Date = Date.init
    ) {
        self.root = root
        self.maxDepth = maxDepth
        self.fileManager = fileManager
        self.cwd = cwd ?? fileManager.currentDirectoryPath
        self.now = now
    }

    func envelope() -> RunListEnvelope {
        let analysis = analyze()
        return RunListEnvelope(
            schemaVersion: 1,
            mereRunVersion: MereRunCLIVersion.current,
            command: ["run", "list"],
            mode: .inspection,
            status: analysis.status,
            createdAt: now(),
            cwd: cwd,
            summary: analysis.summary,
            request: RunListRequest(root: analysis.result.root, maxDepth: maxDepth),
            result: analysis.result,
            diagnostics: analysis.diagnostics,
            actions: analysis.actions
        )
    }

    func analyze() -> RunListAnalysis {
        let rootURL = URL(fileURLWithPath: root).standardizedFileURL
        var diagnostics: [PreflightDiagnostic] = []
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDirectory) else {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "run_list_root_not_found",
                    severity: .blocker,
                    title: "Run list root not found",
                    message: "Run list root not found: \(rootURL.path)",
                    locations: [.init(kind: "file", path: rootURL.path)]
                )
            )
            return emptyAnalysis(root: rootURL.path, diagnostics: diagnostics)
        }

        if !isDirectory.boolValue {
            let entry = entry(for: rootURL, root: rootURL, depth: 0)
            if entry == nil {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "run_list_file_unsupported",
                        severity: .blocker,
                        title: "Unsupported run list file",
                        message: "Expected a structured report JSON file or run-plan JSON file.",
                        locations: [.init(kind: "file", path: rootURL.path)]
                    )
                )
            }
            let entries = entry.map { [$0] } ?? []
            return analysis(root: rootURL.path, scannedDirectoryCount: 0, entries: entries, diagnostics: diagnostics)
        }

        var scannedDirectoryCount = 0
        var entries: [RunListEntry] = []
        var queue: [(url: URL, depth: Int)] = [(rootURL, 0)]
        var index = 0

        while index < queue.count {
            let current = queue[index]
            index += 1
            scannedDirectoryCount += 1

            let contents: [URL]
            do {
                contents = try fileManager.contentsOfDirectory(
                    at: current.url,
                    includingPropertiesForKeys: [.isDirectoryKey, .isRegularFileKey],
                    options: [.skipsHiddenFiles]
                )
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            } catch {
                diagnostics.append(
                    PreflightDiagnostic(
                        id: "run_list_directory_unreadable",
                        severity: .warning,
                        title: "Run list directory unreadable",
                        message: error.localizedDescription,
                        locations: [.init(kind: "directory", path: current.url.path)]
                    )
                )
                continue
            }

            if isRunDirectoryCandidate(current.url, contents: contents),
               let entry = entry(for: current.url, root: rootURL, depth: current.depth) {
                entries.append(entry)
                continue
            }

            entries.append(contentsOf: fileEntries(in: contents, root: rootURL, depth: current.depth))

            guard current.depth < maxDepth else { continue }
            let childDirectories = contents.filter { url in
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey])
                return values?.isDirectory == true
            }
            queue.append(contentsOf: childDirectories.map { ($0.standardizedFileURL, current.depth + 1) })
        }

        return analysis(
            root: rootURL.path,
            scannedDirectoryCount: scannedDirectoryCount,
            entries: entries.sorted { lhs, rhs in
                if lhs.kind == rhs.kind {
                    return lhs.relativePath < rhs.relativePath
                }
                return lhs.kind < rhs.kind
            },
            diagnostics: diagnostics
        )
    }

    private func analysis(
        root: String,
        scannedDirectoryCount: Int,
        entries: [RunListEntry],
        diagnostics: [PreflightDiagnostic]
    ) -> RunListAnalysis {
        var diagnostics = diagnostics
        if entries.isEmpty, diagnostics.allSatisfy({ $0.severity != .blocker }) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "no_runs_found",
                    severity: .warning,
                    title: "No runs found",
                    message: "No run directories, structured reports, or run plans were found under \(root).",
                    locations: [.init(kind: "directory", path: root)]
                )
            )
        } else if entries.contains(where: { $0.status == .warning || $0.status == .blocked }) {
            diagnostics.append(
                PreflightDiagnostic(
                    id: "run_entries_need_review",
                    severity: .warning,
                    title: "Some run entries need review",
                    message: "Some discovered run entries have warnings or blockers.",
                    locations: entries
                        .filter { $0.status == .warning || $0.status == .blocked }
                        .prefix(10)
                        .map { .init(kind: $0.kind, path: $0.path) }
                )
            )
        }

        let result = RunListResult(
            root: root,
            scannedDirectoryCount: scannedDirectoryCount,
            entryCount: entries.count,
            entries: entries
        )
        let status = StructuredRunOutput.status(for: diagnostics)
        return RunListAnalysis(
            result: result,
            diagnostics: diagnostics,
            actions: actions(root: root, entries: entries),
            status: status,
            summary: "\(entries.count) run artifact(s), scanned \(scannedDirectoryCount) directories."
        )
    }

    private func emptyAnalysis(root: String, diagnostics: [PreflightDiagnostic]) -> RunListAnalysis {
        analysis(root: root, scannedDirectoryCount: 0, entries: [], diagnostics: diagnostics)
    }

    private func fileEntries(
        in contents: [URL],
        root: URL,
        depth: Int
    ) -> [RunListEntry] {
        contents.compactMap { url in
            let values = try? url.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true, isJSONCandidateFile(url) else {
                return nil
            }
            return entry(for: url.standardizedFileURL, root: root, depth: depth)
        }
    }

    private func entry(for url: URL, root: URL, depth: Int) -> RunListEntry? {
        let envelope = RunInspectionAnalyzer(
            path: url.path,
            fileManager: fileManager,
            now: now
        ).envelope()
        guard ["run_directory", "report_file", "plan_file"].contains(envelope.result.kind) else {
            return nil
        }
        let relativePath = Self.relativePath(for: url, root: root)
        let runDirectory = envelope.result.runDirectory
        let report = envelope.result.report
        let plan = envelope.result.plan
        let createdAt = runDirectory?.manifest?.createdAt ??
            runDirectory?.legacyManifest?.createdAt ??
            report?.createdAt ??
            plan?.createdAt
        let updatedAt = runDirectory?.events.latest?.createdAt ??
            runDirectory?.legacyManifest?.updatedAt ??
            runDirectory?.legacyManifest?.createdAt ??
            runDirectory?.manifest?.createdAt ??
            report?.createdAt ??
            plan?.createdAt
        return RunListEntry(
            id: Self.entryID(for: relativePath, kind: envelope.result.kind),
            kind: envelope.result.kind,
            path: url.path,
            relativePath: relativePath,
            depth: depth,
            status: envelope.status,
            state: runDirectory?.status ?? report?.status.rawValue,
            createdAt: createdAt,
            updatedAt: updatedAt,
            summary: envelope.summary,
            command: report?.command ?? plan?.command,
            format: runDirectory?.manifest?.format ?? plan?.kind,
            eventCount: runDirectory?.events.count,
            artifactCount: runDirectory?.artifacts.count,
            diagnosticCount: envelope.diagnostics.count,
            blockerCount: envelope.diagnostics.filter { $0.severity == .blocker }.count,
            actions: entryActions(path: url.path, kind: envelope.result.kind)
        )
    }

    private func isRunDirectoryCandidate(_ url: URL, contents: [URL]) -> Bool {
        let names = Set(contents.map(\.lastPathComponent))
        if names.contains(LoRATrainingRunManifest.filename) {
            return true
        }
        let hasPlan = names.contains("plan.json")
        let hasActions = names.contains("actions.json")
        let hasEvents = names.contains(where: { name in
            name.hasSuffix(LoRATrainingRunEvent.filenameSuffix)
        })
        let hasArtifactDirectory = !names.intersection(["artifacts", "samples", "checkpoints", "logs"]).isEmpty
        return hasPlan && (hasActions || hasEvents || hasArtifactDirectory || url.lastPathComponent == "run")
    }

    private func isJSONCandidateFile(_ url: URL) -> Bool {
        guard url.pathExtension.lowercased() == "json" else {
            return false
        }
        let name = url.lastPathComponent
        return name != LoRATrainingRunManifest.filename && name != "actions.json"
    }

    private func actions(root: String, entries: [RunListEntry]) -> [DeclarativeAction] {
        var isDirectory: ObjCBool = false
        let rootExists = fileManager.fileExists(atPath: root, isDirectory: &isDirectory)
        var actions: [DeclarativeAction] = [
            DeclarativeAction(
                id: "open-root",
                label: isDirectory.boolValue ? "Open root" : "Reveal root",
                kind: isDirectory.boolValue ? .openDirectory : .revealFile,
                style: .link,
                enabled: rootExists,
                path: root
            ),
        ]
        if !entries.isEmpty {
            actions.append(
                DeclarativeAction(
                    id: "inspect-selected",
                    label: "Inspect selected run artifact",
                    kind: .select,
                    style: .primary,
                    candidates: entries.map { entry in
                        DeclarativeActionCandidate(
                            id: entry.id,
                            label: entry.relativePath,
                            value: entry.path,
                            path: entry.path,
                            status: entry.state ?? entry.status.rawValue,
                            description: entry.kind
                        )
                    }
                )
            )
        }
        return actions
    }

    private func entryActions(path: String, kind: String) -> [DeclarativeAction] {
        var actions = [
            DeclarativeAction(
                id: "inspect",
                label: "Inspect",
                kind: .command,
                style: .primary,
                command: DeclarativeCommand(
                    argv: ["mere.run", "run", "inspect", path, "--json"],
                    cwd: cwd,
                    commandPath: ["run", "inspect"]
                )
            ),
        ]
        if kind == "run_directory" {
            actions.append(
                DeclarativeAction(
                    id: "open",
                    label: "Open directory",
                    kind: .openDirectory,
                    style: .link,
                    path: path
                )
            )
        } else {
            actions.append(
                DeclarativeAction(
                    id: "reveal",
                    label: "Reveal file",
                    kind: .revealFile,
                    style: .link,
                    path: path
                )
            )
        }
        return actions
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

    private static func entryID(for relativePath: String, kind: String) -> String {
        var output = "\(kind.replacingOccurrences(of: "_", with: "-"))-"
        var previousWasDash = true
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
        return trimmed.isEmpty ? kind : trimmed
    }
}

private struct StructuredReportHeader: Decodable {
    struct DiagnosticHeader: Decodable {}
    struct ActionHeader: Decodable {}

    let schemaVersion: Int
    let command: [String]
    let mode: StructuredRunMode
    let status: StructuredRunStatus
    let createdAt: Date?
    let summary: String
    let diagnostics: [DiagnosticHeader]
    let actions: [ActionHeader]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case command
        case mode
        case status
        case createdAt = "created_at"
        case summary
        case diagnostics
        case actions
    }
}

private struct LegacyRunManifestHeader: Decodable {
    struct Dataset: Decodable {
        let path: String?
        let pairCount: Int?
    }

    struct Plugin: Decodable {
        let name: String?
    }

    struct Recipe: Decodable {
        let id: String?
        let trainModel: String?
        let applyModel: String?
    }

    struct Remote: Decodable {
        let provider: String?
        let gpu: String?
    }

    let contractVersion: String?
    let runID: String?
    let status: String?
    let createdAt: String?
    let updatedAt: String?
    let command: [String]?
    let dataset: Dataset?
    let plugin: Plugin?
    let recipe: Recipe?
    let remote: Remote?

    enum CodingKeys: String, CodingKey {
        case contractVersion
        case runID = "runId"
        case status
        case createdAt
        case updatedAt
        case command
        case dataset
        case plugin
        case recipe
        case remote
    }
}

private struct RunPlanHeader: Decodable {
    let schemaVersion: Int
    let kind: String
    let command: [String]
    let createdAt: Date?
    let cwd: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case command
        case createdAt = "created_at"
        case cwd
    }
}
