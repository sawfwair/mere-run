import Foundation
import MereRunCore

struct ResumeLoRABootstrap: Sendable {
    let datasetRoot: String?
    let modelPath: String?
}

enum ResumeLoRABootstrapLoader {
    static func load(from resumeURL: URL?) throws -> ResumeLoRABootstrap? {
        guard let resumeURL else { return nil }

        let resolved = try LoRACheckpointResolver.resolve(resumeURL)
        defer { resolved.cleanup() }

        let sidecar = try LoRATrainingCheckpointState.load(nextTo: resolved.checkpointURL)
        let runManifest = loadRunManifestLenient(nextTo: resolved.checkpointURL)
        let checkpointConfig = loadCheckpointConfigHints(from: resolved.configStateURL)
        let runSnapshot = runManifest?.configSnapshot
        let sidecarSnapshot = sidecar?.configSnapshot

        let checkpointDirectory = resolved.checkpointURL.deletingLastPathComponent()
        let datasetRootRaw: String? = {
            if let runManifest {
                return runSnapshot?["dataset_root"] ?? runManifest.dataRoot
            }
            return sidecarSnapshot?["dataset_root"] ?? checkpointConfig?.datasetRoot
        }()
        let datasetRoot = normalizedDatasetPath(
            datasetRootRaw,
            relativeFallback: runManifest?.dataRootRelative,
            relativeTo: checkpointDirectory
        )
        let modelRaw: String? = {
            if let runManifest {
                return runSnapshot?["model"] ?? runManifest.model
            }
            return sidecarSnapshot?["model"] ?? checkpointConfig?.model
        }()
        let modelPath = normalizedModelReference(modelRaw)

        if datasetRoot == nil, modelPath == nil {
            return nil
        }

        return ResumeLoRABootstrap(
            datasetRoot: datasetRoot,
            modelPath: modelPath
        )
    }

    private struct CheckpointConfigHints: Sendable {
        let datasetRoot: String?
        let model: String?
    }

    private static func loadRunManifestLenient(nextTo checkpointURL: URL) -> LoRATrainingRunManifest? {
        do {
            return try LoRATrainingRunManifest.load(nextTo: checkpointURL)
        } catch {
            return nil
        }
    }

    private static func loadCheckpointConfigHints(from configURL: URL?) -> CheckpointConfigHints? {
        guard let configURL,
              let data = try? Data(contentsOf: configURL),
              let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        let model: String? = {
            guard let rawValue = raw["model"] as? String else { return nil }
            let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }()

        let datasetRoot: String? = {
            if let rawValue = raw["data"] as? String {
                let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
            if let dataObject = raw["data"] as? [String: Any],
               let path = dataObject["path"] as? String {
                let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
            return nil
        }()

        if datasetRoot == nil, model == nil {
            return nil
        }
        return CheckpointConfigHints(datasetRoot: datasetRoot, model: model)
    }

    private static func normalizedDatasetPath(
        _ rawValue: String?,
        relativeFallback: String?,
        relativeTo baseDirectory: URL
    ) -> String? {
        if let direct = normalizedFilePath(rawValue, relativeTo: baseDirectory) {
            return direct
        }
        return normalizedFilePath(relativeFallback, relativeTo: baseDirectory)
    }

    private static func normalizedFilePath(
        _ rawValue: String?,
        relativeTo baseDirectory: URL
    ) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if trimmed.hasPrefix("/") || trimmed.hasPrefix("~") {
            return URL(fileURLWithPath: trimmed).standardizedFileURL.path
        }
        return baseDirectory
            .appendingPathComponent(trimmed, isDirectory: false)
            .standardizedFileURL
            .path
    }

    private static func normalizedModelReference(_ rawValue: String?) -> String? {
        guard let rawValue else { return nil }
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if trimmed.hasPrefix("/") || trimmed.hasPrefix("~") {
            return URL(fileURLWithPath: trimmed).standardizedFileURL.path
        }
        return trimmed
    }
}
