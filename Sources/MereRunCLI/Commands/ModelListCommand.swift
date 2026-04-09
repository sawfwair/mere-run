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
        let knownEntries = Dictionary(uniqueKeysWithValues: R2ModelRegistry.allEntries.map { ($0.id, $0) })
        var idsInOrder = R2ModelRegistry.allEntries.map(\.id)
        for modelID in ModelResolver.ModelID.allCases.map(\.rawValue) where knownEntries[modelID] == nil {
            idsInOrder.append(modelID)
        }

        printRow("ID", "Category", "Status", "Size")
        print(String(repeating: "-", count: 72))

        for id in idsInOrder {
            let entry = knownEntries[id]
            let status: String
            let size: String

            // For image gen models that have a ModelResolver.ModelID, check via resolver too
            let modelID = ModelResolver.ModelID(rawValue: id)
            let resolvedViaResolver = modelID.flatMap { resolver.resolveIfPresent($0) }

            let flatDir = MereRunModelPaths.modelDir(id)
            let flatInstalled = isNonEmptyDirectory(flatDir)
            let gemmaAliasInstall = gemmaAliasInstallURL(for: id)

            if let resolution = resolvedViaResolver {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: resolution.rootURL)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else if let gemmaAliasInstall {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: gemmaAliasInstall)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else if flatInstalled {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: flatDir)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else {
                status = "missing"
                size = "—"
            }

            printRow(id, entry?.category ?? inferCategory(for: id), status, size)
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

    private func gemmaAliasInstallURL(for id: String) -> URL? {
        guard id == Gemma4Resources.defaultModelId else {
            return nil
        }

        let maxURL = MereRunModelPaths.modelDir(Gemma4Resources.maxModelId)
        if isNonEmptyDirectory(maxURL) {
            return maxURL
        }

        let nanoURL = MereRunModelPaths.modelDir(Gemma4Resources.nanoModelId)
        if isNonEmptyDirectory(nanoURL) {
            return nanoURL
        }

        return nil
    }

    private func printRow(_ id: String, _ category: String, _ status: String, _ size: String) {
        let row = id.padding(toLength: 30, withPad: " ", startingAt: 0)
            + category.padding(toLength: 14, withPad: " ", startingAt: 0)
            + status.padding(toLength: 12, withPad: " ", startingAt: 0)
            + size
        print(row)
    }

    private func inferCategory(for id: String) -> String {
        if id.hasPrefix("text-chat-") { return "text-chat" }
        if id.hasPrefix("text-code-") { return "text-code" }
        if id.hasPrefix("text-embed-") { return "text-embed" }
        if id.hasPrefix("image-") { return "image" }
        if id.hasPrefix("speech-tts-") { return "speech-tts" }
        if id.hasPrefix("speech-asr-") { return "speech-asr" }
        if id.hasPrefix("vision-ocr-") { return "vision-ocr" }
        if id.hasPrefix("vision-segment-") { return "vision-segment" }
        if id.hasPrefix("music-") { return "music" }
        if id.hasPrefix("video-") { return "video" }
        return "other"
    }
}
