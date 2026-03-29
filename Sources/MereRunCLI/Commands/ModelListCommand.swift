import ArgumentParser
import Foundation
import MereRunCore

struct ModelList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List all known models with install status."
    )

    func run() throws {
        let resolver = ModelResolver()

        printRow("ID", "Category", "Status", "Size")
        print(String(repeating: "-", count: 72))

        for entry in R2ModelRegistry.allEntries {
            let status: String
            let size: String

            // For image gen models that have a ModelResolver.ModelID, check via resolver too
            let modelID = ModelResolver.ModelID(rawValue: entry.id)
            let resolvedViaResolver = modelID.flatMap { resolver.resolveIfPresent($0) }

            let flatDir = MereRunModelPaths.modelDir(entry.id)
            let flatInstalled = isNonEmptyDirectory(flatDir)

            if let resolution = resolvedViaResolver {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: resolution.rootURL)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else if flatInstalled {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: flatDir)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else {
                status = "missing"
                size = "—"
            }

            printRow(entry.id, entry.category, status, size)
        }
    }

    private func isNonEmptyDirectory(_ url: URL) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return (try? fm.contentsOfDirectory(atPath: url.path))?.isEmpty == false
    }

    private func printRow(_ id: String, _ category: String, _ status: String, _ size: String) {
        let row = id.padding(toLength: 30, withPad: " ", startingAt: 0)
            + category.padding(toLength: 14, withPad: " ", startingAt: 0)
            + status.padding(toLength: 12, withPad: " ", startingAt: 0)
            + size
        print(row)
    }
}
