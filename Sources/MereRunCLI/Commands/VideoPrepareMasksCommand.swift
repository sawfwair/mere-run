import ArgumentParser
import Foundation
import MereRunCore

struct SCAIL2MaskPreflightReport: Codable, Equatable {
    let status: String
    let plan: String
    let outputDirectory: String
    let previewFrame: Int?
    let model: String
    let modelRoot: String?
    let modelInstalled: Bool
    let missingInputFiles: [String]
    let subjectCount: Int
    let frameWidth: Int
    let frameHeight: Int
    let fps: Double

    enum CodingKeys: String, CodingKey {
        case status
        case plan
        case outputDirectory = "output_directory"
        case previewFrame = "preview_frame"
        case model
        case modelRoot = "model_root"
        case modelInstalled = "model_installed"
        case missingInputFiles = "missing_input_files"
        case subjectCount = "subject_count"
        case frameWidth = "frame_width"
        case frameHeight = "frame_height"
        case fps
    }
}

struct VideoPrepareMasks: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "prepare-masks",
        abstract: "Prepare reviewable, palette-safe SCAIL-2 masks with native SAM 3.1.",
        discussion: """
        Decodes a typed SCAIL2MaskPlan, normalizes the driving video, segments
        each reference, tracks each subject independently, applies immutable
        keyframe corrections, and writes a canonical artifact manifest.

        --preview-frame runs only reference and selected-frame segmentation.
        """
    )

    @Option(name: [.customLong("plan")], help: "Path to a schema_version 1 SCAIL2MaskPlan JSON file.")
    var plan: String

    @Option(name: [.customLong("output-dir")], help: "New directory for immutable mask artifacts.")
    var outputDirectory: String

    @Option(name: [.customLong("preview-frame")], help: "Only segment this normalized driving frame.")
    var previewFrame: Int?

    @Option(name: [.customShort("m"), .long], help: "Managed SAM 3.1 model id or local model root.")
    var model: String = ModelResolver.ModelID.visionSegmentSAM31.rawValue

    @Flag(name: [.customLong("preflight")], help: "Validate the plan, files, model, and output without loading SAM.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "Emit structured JSON for preflight or completed preparation.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Suppress diagnostics on stderr.")
    var quiet: Bool = false

    func run() async throws {
        let planURL = URL(fileURLWithPath: plan).standardizedFileURL
        guard FileManager.default.fileExists(atPath: planURL.path) else {
            throw ValidationError("Mask plan not found: \(planURL.path)")
        }
        let resolvedPlan = try SCAIL2MaskPreparer.decodePlan(at: planURL)
        let outputURL = URL(fileURLWithPath: outputDirectory).standardizedFileURL
        let modelRoot = resolveInstalledModelRoot()
        let missingInputs = inputURLs(for: resolvedPlan).filter {
            !FileManager.default.fileExists(atPath: $0.path)
        }
        let manifestExists = FileManager.default.fileExists(
            atPath: outputURL.appendingPathComponent("manifest.json").path
        )

        if preflight {
            let report = SCAIL2MaskPreflightReport(
                status: missingInputs.isEmpty && modelRoot != nil && !manifestExists ? "ok" : "blocked",
                plan: planURL.path,
                outputDirectory: outputURL.path,
                previewFrame: previewFrame,
                model: model,
                modelRoot: modelRoot?.path,
                modelInstalled: modelRoot != nil,
                missingInputFiles: missingInputs.map(\.path),
                subjectCount: resolvedPlan.subjects.count,
                frameWidth: resolvedPlan.width,
                frameHeight: resolvedPlan.height,
                fps: resolvedPlan.fps
            )
            try emit(report)
            return
        }

        let resolvedModel = try VisionSegment.resolveModelRoot(model)
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let segmenter = try SAM31ImageSegmenter(
            modelRootURL: resolvedModel.rootURL,
            expectedModelID: resolvedModel.isManaged ? resolvedModel.modelID : nil
        )
        defer { segmenter.unload() }
        if !quiet {
            CLIStderr.write("Engine: native Swift/MLX SAM 3.1\n")
            CLIStderr.write("Mode: \(previewFrame == nil ? "full tracking" : "preview")\n")
            CLIStderr.write("Subjects: \(resolvedPlan.subjects.count)\n")
        }
        let result = try SCAIL2MaskPreparer(segmenter: segmenter).prepare(
            plan: resolvedPlan,
            outputDirectoryURL: outputURL,
            previewFrame: previewFrame,
            modelRevision: Self.modelRevision(for: resolvedModel)
        )
        if json {
            try emit(result)
        } else {
            if !quiet {
                CLIStderr.write("Status: \(result.status)\n")
                CLIStderr.write("Warnings: \(result.warnings)\n")
            }
            print(result.manifestPath)
        }
    }

    private func resolveInstalledModelRoot() -> URL? {
        let explicit = URL(fileURLWithPath: model).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicit.path) {
            return explicit
        }
        return ManagedModelResolver.resolveInstalledModel(id: model)
    }

    static func modelRevision(for resolvedModel: VisionSegment.ResolvedModel) -> String {
        guard resolvedModel.isManaged else { return "local-unpinned" }
        return ManagedModelCatalog.spec(for: resolvedModel.modelID)?.upstreamRevision
            ?? "managed-unpinned"
    }

    private func inputURLs(for plan: SCAIL2MaskPlan) -> [URL] {
        [URL(fileURLWithPath: plan.drivingVideo)]
            + plan.subjects.map { URL(fileURLWithPath: $0.referenceImage) }
            + plan.corrections.compactMap {
                $0.paintedBinaryCorrectionPNG.map { URL(fileURLWithPath: $0) }
            }
    }

    private func emit<Value: Encodable>(_ value: Value) throws {
        if json {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            print(String(decoding: try encoder.encode(value), as: UTF8.self))
        } else if let report = value as? SCAIL2MaskPreflightReport {
            if !quiet {
                CLIStderr.write("SCAIL-2 mask preflight: \(report.status)\n")
                for path in report.missingInputFiles {
                    CLIStderr.write("Missing input: \(path)\n")
                }
            }
            print(report.status)
        }
    }
}
