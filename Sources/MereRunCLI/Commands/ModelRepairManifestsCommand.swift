import ArgumentParser
import Foundation
import MereRunCore

struct ModelRepairManifests: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "repair-manifests",
        abstract: "Write missing mererun_model.json for known models in the local mere.run model store."
    )

    @Flag(name: [.long], help: "Print what would change without writing files.")
    var dryRun: Bool = false

    @Flag(name: [.long], help: "Emit a structured JSON repair report.")
    var json: Bool = false

    @Flag(
        name: [.customLong("accept-model-license")],
        help: "Record that you reviewed and accept the listed third-party model terms when writing a missing manifest."
    )
    var acceptModelLicense: Bool = false

    @Option(name: [.long], help: "Repair only this canonical model ID.")
    var model: String?

    func run() throws {
        if let requested = normalizedRequestedModel,
           ModelResolver.ModelID(rawValue: requested) == nil {
            throw ValidationError("Unknown canonical model id: \(requested)")
        }
        let modelDirs = resolveCandidateModelDirs()
        guard !modelDirs.isEmpty else {
            throw ValidationError("Could not locate any model directories to repair.")
        }

        let report = makeReport(modelDirs: modelDirs)
        if json {
            print(try StructuredRunOutput.encode(report))
            return
        }

        for entry in report.entries {
            switch entry.status {
            case .ok:
                print("[ok] \(entry.modelID): \(entry.path ?? "")")
            case .wouldWrite:
                print("[dry-run] would write \(entry.modelID) -> \(entry.path ?? "")")
            case .wrote:
                print("[wrote] \(entry.modelID): \(entry.path ?? "")")
            case .skipped:
                print("[skip] \(entry.modelID): \(entry.message ?? "unknown error")")
            }
        }

        print("")
        print(
            "Summary: wrote=\(report.wroteCount) already=\(report.alreadyCount) "
                + "skipped=\(report.skippedCount)"
        )
        if dryRun {
            print("Run again without `--dry-run` to apply.")
        }
    }

    func makeReport(
        modelDirs: [URL],
        fileManager: FileManager = .default
    ) -> ModelRepairManifestsReport {
        var entries: [ModelRepairManifestEntry] = []

        let modelIDs = normalizedRequestedModel.flatMap(ModelResolver.ModelID.init(rawValue:))
            .map { [$0] } ?? ModelResolver.ModelID.allCases
        for modelID in modelIDs {
            var foundAnyModel = false
            for modelsDir in modelDirs {
                let modelRoot = modelsDir.appendingPathComponent(modelID.rawValue, isDirectory: true)
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: modelRoot.path, isDirectory: &isDirectory),
                      isDirectory.boolValue else {
                    continue
                }
                foundAnyModel = true

                let manifestURL = MereRunModelManifest.url(in: modelRoot)
                if fileManager.fileExists(atPath: manifestURL.path) {
                    entries.append(.init(modelID: modelID.rawValue, status: .ok, path: manifestURL.path))
                    continue
                }
                if dryRun {
                    entries.append(.init(modelID: modelID.rawValue, status: .wouldWrite, path: manifestURL.path))
                    continue
                }
                do {
                    _ = try MereRunModelManifest.writeTemplateIfKnown(
                        modelId: modelID.rawValue,
                        to: modelRoot,
                        usageTermsAcknowledged: acceptModelLicense
                    )
                    entries.append(.init(modelID: modelID.rawValue, status: .wrote, path: manifestURL.path))
                } catch {
                    entries.append(
                        .init(
                            modelID: modelID.rawValue,
                            status: .skipped,
                            path: manifestURL.path,
                            message: error.localizedDescription
                        )
                    )
                }
            }

            if !foundAnyModel {
                entries.append(
                    .init(modelID: modelID.rawValue, status: .skipped, message: "model directory not found")
                )
            }
        }

        return ModelRepairManifestsReport(
            mode: dryRun ? "preview" : "apply",
            wroteCount: entries.filter { $0.status == .wouldWrite || $0.status == .wrote }.count,
            alreadyCount: entries.filter { $0.status == .ok }.count,
            skippedCount: entries.filter { $0.status == .skipped }.count,
            entries: entries
        )
    }

    private func resolveCandidateModelDirs() -> [URL] {
        [MereRunModelPaths.modelsDir]
    }

    private var normalizedRequestedModel: String? {
        guard let model else { return nil }
        let normalized = model.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized.isEmpty ? nil : normalized
    }
}

struct ModelRepairManifestsReport: Codable, Equatable {
    let mode: String
    let wroteCount: Int
    let alreadyCount: Int
    let skippedCount: Int
    let entries: [ModelRepairManifestEntry]

    enum CodingKeys: String, CodingKey {
        case mode
        case wroteCount = "wrote_count"
        case alreadyCount = "already_count"
        case skippedCount = "skipped_count"
        case entries
    }
}

struct ModelRepairManifestEntry: Codable, Equatable {
    enum Status: String, Codable {
        case ok
        case wouldWrite = "would_write"
        case wrote
        case skipped
    }

    let modelID: String
    let status: Status
    let path: String?
    let message: String?

    init(modelID: String, status: Status, path: String? = nil, message: String? = nil) {
        self.modelID = modelID
        self.status = status
        self.path = path
        self.message = message
    }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case status
        case path
        case message
    }
}
