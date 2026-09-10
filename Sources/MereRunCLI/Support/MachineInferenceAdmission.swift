import Foundation
import MereRunCore
#if os(Linux)
import Glibc
#else
import Darwin
#endif

extension MachineInferenceCoordinator {
    static var shared: MachineInferenceCoordinator {
        MachineInferenceCoordinator(
            stateDirectory: MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true),
            hostSnapshot: { currentHostSnapshot() }
        )
    }

    static func currentHostSnapshot() -> MachineInferenceHostSnapshot {
        currentHostSnapshot(diskURL: MereRunModelPaths.applicationSupportBase.deletingLastPathComponent())
    }
}

enum CLIInferenceAdmissionClassifier {
    static func request(arguments: [String]) -> MachineInferenceRequest? {
        let tokens = Array(arguments.dropFirst())
        guard !tokens.contains("--help"),
              !tokens.contains("-h"),
              !tokens.contains("--version"),
              !tokens.contains("--dry-run") else {
            return nil
        }
        let commandTokens = commandPath(tokens)
        guard let topLevel = commandTokens.first else { return nil }
        // Guides only read bundled text, even when a large model is selected.
        guard topLevel != "guide" else { return nil }
        let subcommand = commandTokens.dropFirst().first
        let nestedSubcommand = commandTokens.dropFirst(2).first
        let label = [topLevel, subcommand].compactMap { $0 }.joined(separator: " ")
        // These preflights report resource blockers without reserving inference
        // permits. Other preflight implementations retain their existing gate.
        if tokens.contains("--preflight"), ["image generate", "text chat"].contains(label) {
            return nil
        }

        // These orchestration commands either acquire a workload-specific
        // lease internally or spawn a child process that does. Classifying
        // their --model argument here would make the parent hold permits while
        // waiting for the child to acquire the same permits.
        if ["api", "open-webui", "run", "graph", "executor", "gate", "setup", "agent", "plugin"]
            .contains(topLevel) {
            return nil
        }

        if tokens.contains(where: { token in
            let normalized = token.lowercased()
            return normalized.contains("deepseek-v4") || normalized.contains("ds4")
        }) {
            return MachineInferenceRequest(label: label, resourceClass: .large)
        }
        let selectedModelIdentifiers = modelIdentifiers(in: tokens)
        if selectedModelIdentifiers.contains(where: isLargeModel) {
            return MachineInferenceRequest(label: label, resourceClass: .large)
        }

        switch topLevel {
        case "speech", "audio":
            return MachineInferenceRequest(label: label, resourceClass: .small)
        case "image":
            let large = [
                "reconstruct-3d",
                "reconstruct-3d-multiview",
                "reconstruct-3d-trellis2",
                "image-to-3d",
                "image-to-3d-multiview",
                "train-lora",
            ].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "text":
            let large = subcommand == "train-lora"
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "vision":
            let large = [
                "geometry",
                "geometry-multiview",
                "image-to-3d",
                "image-to-3d-trellis2",
                "depth-video",
            ].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "music":
            let large = ["train-adapter"].contains(subcommand)
            let resourceClass = large
                ? MachineInferenceClass.large
                : musicResourceClass(modelIdentifiers: selectedModelIdentifiers)
            return MachineInferenceRequest(label: label, resourceClass: resourceClass)
        case "sfx":
            let large = ["train-adapter"].contains(subcommand)
            return MachineInferenceRequest(label: label, resourceClass: large ? .large : .standard)
        case "video", "world":
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "geo":
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "model" where subcommand == "benchmark":
            if nestedSubcommand == "fused-fixture" {
                return nil
            }
            return MachineInferenceRequest(label: label, resourceClass: .large)
        case "adapter" where subcommand != "list" && subcommand != "pull":
            return MachineInferenceRequest(label: label, resourceClass: .standard)
        default:
            return nil
        }
    }

    static func speechTranscriptionRequest(modelID: String) -> MachineInferenceRequest {
        MachineInferenceRequest(label: "speech transcribe", resourceClass: isLargeModel(modelID) ? .large : .small)
    }

    static func imageGenerationRequest(modelID: String) -> MachineInferenceRequest {
        MachineInferenceRequest(label: "image generate", estimatedModelBytes: estimatedModelBytes(modelID))
    }

    static func apiServerRequest(engine: APIEngine, modelID: String? = nil) -> MachineInferenceRequest {
        MachineInferenceRequest(
            label: "api serve \(engine.rawValue)",
            estimatedModelBytes: estimatedModelBytes(modelID),
            requiresExclusive: engine == .textChatDeepseekV4Flash
        )
    }

    private static func modelIdentifiers(in tokens: [String]) -> [String] {
        var identifiers: [String] = []
        for (index, token) in tokens.enumerated() {
            if token == "--model" || token == "-m" {
                if tokens.indices.contains(index + 1) {
                    identifiers.append(tokens[index + 1])
                }
            } else if token.hasPrefix("--model=") {
                identifiers.append(String(token.dropFirst("--model=".count)))
            }
        }
        return identifiers
    }

    private static func isLargeModel(_ identifier: String?) -> Bool {
        .forModel(estimatedBytes: estimatedModelBytes(identifier)) == MachineInferenceClass.large
    }

    private static func musicResourceClass(modelIdentifiers: [String]) -> MachineInferenceClass {
        let identifiers = modelIdentifiers.isEmpty
            ? [ModelResolver.ModelID.aceStep.rawValue]
            : modelIdentifiers
        return identifiers.allSatisfy(isSmallModel) ? .small : .standard
    }

    private static func isSmallModel(_ identifier: String) -> Bool {
        .forModel(estimatedBytes: estimatedModelBytes(identifier), minimum: .small) == MachineInferenceClass.small
    }

    private static func estimatedModelBytes(_ identifier: String?) -> Int64? {
        guard let identifier, !identifier.isEmpty else { return nil }
        if let estimatedBytes = ManagedModelCatalog.spec(for: identifier)?.estimatedDownloadBytes {
            return estimatedBytes
        }
        let fileURL = URL(fileURLWithPath: identifier)
        guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let fileSize = values.fileSize else {
            return nil
        }
        return Int64(fileSize)
    }

    private static func commandPath(_ tokens: [String]) -> [String] {
        var result: [String] = []
        var skipNext = false
        for token in tokens {
            if skipNext {
                skipNext = false
                continue
            }
            if token == "--models-root" {
                skipNext = true
                continue
            }
            if token.hasPrefix("--models-root=") || token.hasPrefix("-") {
                continue
            }
            result.append(token)
            if result.count == 3 {
                break
            }
        }
        return result
    }
}

private func releaseCLIProcessAdmission() {
    CLIProcessAdmissionBootstrap.release()
}

enum CLIProcessAdmissionBootstrap {
    private final class Storage: @unchecked Sendable {
        let lock = NSLock()
        var lease: MachineInferenceLease?
        var registeredExitHandler = false
    }

    private static let storage = Storage()

    static func acquireIfNeeded(arguments: [String]) throws {
        guard let request = CLIInferenceAdmissionClassifier.request(arguments: arguments) else {
            return
        }
        storage.lock.lock()
        defer { storage.lock.unlock() }
        guard storage.lease == nil else { return }
        var waited = false
        let lease = try MachineInferenceCoordinator.shared.acquireBlocking(request) { snapshot in
            waited = true
            CLIStderr.write(
                "Queued by machine admission: \(request.label) "
                    + "(\(snapshot.activePermits)/\(snapshot.capacityPermits) permits active, "
                    + "\(snapshot.queued.count) queued).\n"
            )
        }
        if waited {
            CLIStderr.write("Machine admission granted: \(request.label).\n")
        }
        storage.lease = lease
        if !storage.registeredExitHandler {
            atexit(releaseCLIProcessAdmission)
            storage.registeredExitHandler = true
        }
    }

    static func release() {
        storage.lock.lock()
        let lease = storage.lease
        storage.lease = nil
        storage.lock.unlock()
        lease?.release()
    }
}
