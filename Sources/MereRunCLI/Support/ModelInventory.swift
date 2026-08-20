import Foundation
import MereRunCore

enum ModelInventoryMode: String, Codable, Equatable {
    case fast
    case verified
    case measured
}

enum ModelInventoryVerification: String, Codable, Equatable {
    case checked
    case notChecked = "not_checked"
}

struct ModelInventorySnapshot: Equatable {
    let rows: [ModelInventoryRow]
    let mode: ModelInventoryMode
    let complete: Bool
    let durationMs: Int
}

struct ModelInventoryRow: Equatable {
    let id: String
    let category: String
    let status: String
    let size: String?
    let manifestPresent: Bool
    let runtimeAvailable: Bool?
    let verification: ModelInventoryVerification

    var isInstalled: Bool {
        status == "installed"
    }
}

enum ModelInventory {
    /// Compatibility entry point for callers that explicitly expect measured sizes.
    static func rows(fileManager: FileManager = .default) -> [ModelInventoryRow] {
        snapshot(mode: .measured, fileManager: fileManager).rows
    }

    static func snapshot(
        mode: ModelInventoryMode,
        fileManager: FileManager = .default,
        locations: ModelLocationSnapshot? = nil
    ) -> ModelInventorySnapshot {
        let startedAt = Date()
        let resolver = ModelResolver(fileManager: fileManager, locations: locations)
        let specs = ManagedModelCatalog.allSpecs
        let knownSpecs = Dictionary(uniqueKeysWithValues: specs.map { ($0.id, $0) })
        let idsInOrder = specs.map(\.id)
        var complete = true

        let rows = idsInOrder.map { id in
            let spec = knownSpecs[id]
            if mode == .fast {
                let result = fastRow(
                    id: id,
                    spec: spec,
                    resolver: resolver,
                    fileManager: fileManager
                )
                complete = complete && result.complete
                return result.row
            }
            return checkedRow(
                id: id,
                spec: spec,
                resolver: resolver,
                measureSize: mode == .measured,
                usesDefaultLocations: locations == nil,
                fileManager: fileManager
            )
        }

        let durationMs = max(0, Int(Date().timeIntervalSince(startedAt) * 1_000))
        return ModelInventorySnapshot(
            rows: rows,
            mode: mode,
            complete: complete,
            durationMs: durationMs
        )
    }

    private static func fastRow(
        id: String,
        spec: ManagedModelSpec?,
        resolver: ModelResolver,
        fileManager: FileManager
    ) -> (row: ModelInventoryRow, complete: Bool) {
        let candidateIDs = [id] + (spec?.resolutionFallbackIDs ?? [])
        var foundInvalid = false
        var foundManifest = false
        var complete = true
        var registeredBindingExists = false

        for candidateID in candidateIDs {
            guard let modelID = ModelResolver.ModelID(rawValue: candidateID) else { continue }
            for candidate in resolver.locationCandidates(for: modelID) {
                if candidate.kind == .registeredBinding {
                    registeredBindingExists = true
                }

                switch shallowDirectoryState(candidate.rootURL, fileManager: fileManager) {
                case .missing, .empty:
                    continue
                case .unavailable:
                    complete = false
                    continue
                case .available:
                    break
                }

                let manifestURL = candidate.rootURL.appendingPathComponent(MereRunModelManifest.filename)
                let manifestPresent = fileManager.fileExists(atPath: manifestURL.path)
                foundManifest = foundManifest || manifestPresent

                if manifestPresent {
                    guard let manifest = try? MereRunModelManifest.loadIfPresent(
                        from: candidate.rootURL,
                        fileManager: fileManager
                    ), manifest.id == candidateID else {
                        foundInvalid = true
                        continue
                    }
                } else if candidate.kind == .registeredSearchRoot {
                    continue
                }

                if candidate.kind.isExternallyManaged,
                   spec?.usageRestriction != nil,
                   !candidate.usageTermsAcknowledged {
                    foundInvalid = true
                    continue
                }

                return (
                    row: ModelInventoryRow(
                        id: id,
                        category: spec?.category.rawValue ?? inferCategory(for: id),
                        status: "installed",
                        size: nil,
                        manifestPresent: manifestPresent,
                        runtimeAvailable: true,
                        verification: .notChecked
                    ),
                    complete: complete
                )
            }
        }

        let status: String
        if foundInvalid {
            status = "invalid"
        } else if registeredBindingExists {
            status = "offline"
        } else {
            status = "missing"
        }
        return (
            row: ModelInventoryRow(
                id: id,
                category: spec?.category.rawValue ?? inferCategory(for: id),
                status: status,
                size: nil,
                manifestPresent: foundManifest,
                runtimeAvailable: false,
                verification: .notChecked
            ),
            complete: complete
        )
    }

    private static func checkedRow(
        id: String,
        spec: ManagedModelSpec?,
        resolver: ModelResolver,
        measureSize: Bool,
        usesDefaultLocations: Bool,
        fileManager: FileManager
    ) -> ModelInventoryRow {
        let status: String
        let measuredRoot: URL?

        let modelID = ModelResolver.ModelID(rawValue: id)
        let resolvedViaResolver = modelID.flatMap { resolver.resolveIfPresent($0) }
        let registeredBindings = (modelID.map { resolver.locationCandidates(for: $0) } ?? [])
            .filter { $0.kind == .registeredBinding }

        let flatDir = modelID.flatMap { modelID in
            resolver.locationCandidates(for: modelID)
                .first { $0.kind == .primaryStore }?
                .rootURL
        } ?? MereRunModelPaths.modelDir(id)
        let flatInstalled = isNonEmptyDirectory(flatDir, fileManager: fileManager)
        let hasManagedManifest = fileManager.fileExists(
            atPath: flatDir.appendingPathComponent(MereRunModelManifest.filename).path
        )
        let gemmaAliasInstall = gemmaAliasInstallURL(for: id, fileManager: fileManager)

        if let spec, spec.usesPinnedGeometryArtifacts {
            if let resolution = resolvedViaResolver {
                status = "installed"
                measuredRoot = resolution.rootURL
            } else if spec.requiresManagedConversion,
                      ManagedModelResolver.isManagedInstallComplete(
                          spec: spec,
                          at: flatDir,
                          fileManager: fileManager
                      ) {
                status = "conversion-required"
                measuredRoot = flatDir
            } else if flatInstalled {
                status = "invalid"
                measuredRoot = flatDir
            } else {
                status = "missing"
                measuredRoot = nil
            }
        } else if let resolution = resolvedViaResolver {
            status = "installed"
            measuredRoot = resolution.rootURL
        } else if let gemmaAliasInstall {
            status = "installed"
            measuredRoot = gemmaAliasInstall
        } else if let spec,
                  ManagedModelResolver.isManagedInstallComplete(
                      spec: spec,
                      at: flatDir,
                      fileManager: fileManager
                  ) || (usesDefaultLocations && spec.managedRuntimeURL(fileManager: fileManager) != nil) {
            status = "installed"
            measuredRoot = flatDir
        } else if flatInstalled {
            // Preserve manifest-less legacy roots, but never let a managed
            // manifest paper over broken links or missing runtime assets.
            status = hasManagedManifest ? "invalid" : "installed"
            measuredRoot = flatDir
        } else if !registeredBindings.isEmpty {
            let availableBinding = registeredBindings.first {
                isNonEmptyDirectory($0.rootURL, fileManager: fileManager)
            }
            status = availableBinding == nil ? "offline" : "invalid"
            measuredRoot = availableBinding?.rootURL
        } else {
            status = "missing"
            measuredRoot = nil
        }

        let size = measureSize ? formattedSize(of: measuredRoot) : nil
        let manifestRoot = resolvedViaResolver?.rootURL ?? measuredRoot
        let manifestPresent = manifestRoot.map {
            fileManager.fileExists(
                atPath: $0.appendingPathComponent(MereRunModelManifest.filename).path
            )
        } ?? false
        return ModelInventoryRow(
            id: id,
            category: spec?.category.rawValue ?? inferCategory(for: id),
            status: status,
            size: size,
            manifestPresent: manifestPresent,
            runtimeAvailable: status == "installed",
            verification: .checked
        )
    }

    private static func formattedSize(of url: URL?) -> String? {
        guard let url else { return nil }
        let bytes = FileSystemHelper.directorySize(at: url)
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    private enum ShallowDirectoryState {
        case missing
        case empty
        case available
        case unavailable
    }

    private static func shallowDirectoryState(
        _ url: URL,
        fileManager: FileManager
    ) -> ShallowDirectoryState {
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return .missing
        }
        do {
            let contents = try fileManager.contentsOfDirectoryResolvingSymlinks(at: url)
            return contents.isEmpty ? .empty : .available
        } catch {
            return .unavailable
        }
    }

    private static func isNonEmptyDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        shallowDirectoryState(url, fileManager: fileManager) == .available
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
