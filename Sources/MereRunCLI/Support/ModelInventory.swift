import Foundation
import MereRunCore

struct ModelInventoryRow: Equatable {
    let id: String
    let category: String
    let status: String
    let size: String

    var isInstalled: Bool {
        status == "installed"
    }
}

enum ModelInventory {
    static func rows(fileManager: FileManager = .default) -> [ModelInventoryRow] {
        let resolver = ModelResolver(fileManager: fileManager)
        let specs = ManagedModelCatalog.allSpecs
        let knownSpecs = Dictionary(uniqueKeysWithValues: specs.map { ($0.id, $0) })
        let idsInOrder = specs.map(\.id)

        return idsInOrder.map { id in
            let spec = knownSpecs[id]
            let status: String
            let size: String

            let modelID = ModelResolver.ModelID(rawValue: id)
            let resolvedViaResolver = modelID.flatMap { resolver.resolveIfPresent($0) }

            let flatDir = MereRunModelPaths.modelDir(id)
            let flatInstalled = isNonEmptyDirectory(flatDir, fileManager: fileManager)
            let gemmaAliasInstall = gemmaAliasInstallURL(for: id, fileManager: fileManager)

            if let spec, spec.usesPinnedGeometryArtifacts {
                if let resolution = resolvedViaResolver {
                    status = "installed"
                    let bytes = FileSystemHelper.directorySize(at: resolution.rootURL)
                    size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
                } else if spec.requiresManagedConversion,
                          ManagedModelResolver.isManagedInstallComplete(
                              spec: spec,
                              at: flatDir,
                              fileManager: fileManager
                          ) {
                    status = "conversion-required"
                    let bytes = FileSystemHelper.directorySize(at: flatDir)
                    size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
                } else if flatInstalled {
                    status = "invalid"
                    let bytes = FileSystemHelper.directorySize(at: flatDir)
                    size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
                } else {
                    status = "missing"
                    size = "—"
                }
            } else if let resolution = resolvedViaResolver {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: resolution.rootURL)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else if let gemmaAliasInstall {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: gemmaAliasInstall)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else if flatInstalled || spec?.managedRuntimeURL(fileManager: fileManager) != nil {
                status = "installed"
                let bytes = FileSystemHelper.directorySize(at: flatDir)
                size = ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
            } else {
                status = "missing"
                size = "—"
            }

            return ModelInventoryRow(
                id: id,
                category: spec?.category.rawValue ?? inferCategory(for: id),
                status: status,
                size: size
            )
        }
    }

    private static func isNonEmptyDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        return (try? fileManager.contentsOfDirectory(atPath: url.path))?.isEmpty == false
    }

    private static func gemmaAliasInstallURL(for id: String, fileManager: FileManager) -> URL? {
        guard id == Gemma4Resources.defaultModelId else {
            return nil
        }

        let maxURL = MereRunModelPaths.modelDir(Gemma4Resources.maxModelId)
        if isNonEmptyDirectory(maxURL, fileManager: fileManager) {
            return maxURL
        }

        let nanoURL = MereRunModelPaths.modelDir(Gemma4Resources.nanoModelId)
        if isNonEmptyDirectory(nanoURL, fileManager: fileManager) {
            return nanoURL
        }

        return nil
    }

    private static func inferCategory(for id: String) -> String {
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
        if id.hasPrefix("sfx-") { return "sfx" }
        if id.hasPrefix("video-") { return "video" }
        return "other"
    }
}
