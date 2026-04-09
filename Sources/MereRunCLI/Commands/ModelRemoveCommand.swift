import ArgumentParser
import Foundation
import MereRunCore

struct ModelRemove: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Remove a model from the local model store."
    )

    @Argument(help: "Canonical model id (for example: image-klein-nano).")
    var target: String

    @Flag(name: [.long], help: "Skip confirmation prompt.")
    var force: Bool = false

    func run() throws {
        let id = resolveID(target)
        guard let id else {
            throw ValidationError("Unknown canonical model id: \(target)")
        }

        let resolver = ModelResolver()
        let modelID = ModelResolver.ModelID(rawValue: id)
        let installURL: URL
        if let modelID, let resolved = resolver.resolveIfPresent(modelID) {
            installURL = resolved.rootURL
        } else {
            if let aliasFallback = gemmaAliasInstallURL(for: id) {
                installURL = aliasFallback
            } else {
                let flatDir = MereRunModelPaths.modelDir(id)
                var isDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: flatDir.path, isDirectory: &isDir), isDir.boolValue else {
                    throw ValidationError("\(id) is not installed.")
                }
                installURL = flatDir
            }
        }

        let bytes = FileSystemHelper.directorySize(at: installURL)
        let sizeStr = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)

        if !force {
            print("Remove \(id)?")
            print("  Path: \(installURL.path)")
            print("  Size: \(sizeStr)")
            print("")
            print("Confirm? [y/N] ", terminator: "")
            guard let answer = readLine()?.lowercased(), answer == "y" || answer == "yes" else {
                print("Aborted.")
                return
            }
        }

        try FileManager.default.removeItem(at: installURL)
        print("Removed \(id) (\(sizeStr))")
    }

    /// Resolve a user-supplied string to a canonical model id.
    private func resolveID(_ raw: String) -> String? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let registryID = R2ModelRegistry.entry(for: normalized).map(\.id) {
            return registryID
        }
        return ModelResolver.ModelID(rawValue: normalized)?.rawValue
    }

    private func gemmaAliasInstallURL(for id: String) -> URL? {
        guard id == Gemma4Resources.defaultModelId else {
            return nil
        }

        let candidates = [
            MereRunModelPaths.modelDir(Gemma4Resources.maxModelId),
            MereRunModelPaths.modelDir(Gemma4Resources.nanoModelId),
        ]
        for candidate in candidates {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir), isDir.boolValue {
                return candidate
            }
        }
        return nil
    }
}
