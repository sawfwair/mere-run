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

        let flatDir = MereRunModelPaths.modelDir(id)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: flatDir.path, isDirectory: &isDir), isDir.boolValue else {
            throw ValidationError("\(id) is not installed.")
        }

        let bytes = FileSystemHelper.directorySize(at: flatDir)
        let sizeStr = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)

        if !force {
            print("Remove \(id)?")
            print("  Path: \(flatDir.path)")
            print("  Size: \(sizeStr)")
            print("")
            print("Confirm? [y/N] ", terminator: "")
            guard let answer = readLine()?.lowercased(), answer == "y" || answer == "yes" else {
                print("Aborted.")
                return
            }
        }

        try FileManager.default.removeItem(at: flatDir)
        print("Removed \(id) (\(sizeStr))")
    }

    /// Resolve a user-supplied string to a canonical model id.
    private func resolveID(_ raw: String) -> String? {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return R2ModelRegistry.entry(for: normalized).map(\.id)
    }
}
