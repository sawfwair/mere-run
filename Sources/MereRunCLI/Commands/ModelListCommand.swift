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
        let specs = ManagedModelCatalog.allSpecs
        let knownSpecs = Dictionary(uniqueKeysWithValues: specs.map { ($0.id, $0) })
        let idsInOrder = specs.map(\.id)
        var rows: [ModelListRow] = []
        rows.reserveCapacity(idsInOrder.count)

        for id in idsInOrder {
            let spec = knownSpecs[id]
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
            } else if flatInstalled || spec?.managedRuntimeURL() != nil {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: flatDir)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else {
                status = "missing"
                size = "—"
            }

            rows.append(
                ModelListRow(
                    id: id,
                    category: spec?.category.rawValue ?? inferCategory(for: id),
                    status: status,
                    size: size
                )
            )
        }

        let widths = ModelListColumnWidths(rows: rows)
        printRow("ID", "Category", "Status", "Size", widths: widths)
        print(String(repeating: "-", count: widths.totalWidth))
        for row in rows {
            printRow(row.id, row.category, row.status, row.size, widths: widths)
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

    private func printRow(_ id: String, _ category: String, _ status: String, _ size: String, widths: ModelListColumnWidths) {
        let row = [
            id.padding(toLength: widths.id, withPad: " ", startingAt: 0),
            category.padding(toLength: widths.category, withPad: " ", startingAt: 0),
            status.padding(toLength: widths.status, withPad: " ", startingAt: 0),
            size
        ].joined(separator: "  ")
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
        if id.hasPrefix("vision-ground-") { return "vision-ground" }
        if id.hasPrefix("music-") { return "music" }
        if id.hasPrefix("video-") { return "video" }
        return "other"
    }
}

private struct ModelListRow {
    let id: String
    let category: String
    let status: String
    let size: String
}

private struct ModelListColumnWidths {
    let id: Int
    let category: Int
    let status: Int

    init(rows: [ModelListRow]) {
        self.id = max("ID".count, rows.map(\.id.count).max() ?? 0)
        self.category = max("Category".count, rows.map(\.category.count).max() ?? 0)
        self.status = max("Status".count, rows.map(\.status.count).max() ?? 0)
    }

    var totalWidth: Int {
        id + category + status + "Size".count + 6
    }
}
