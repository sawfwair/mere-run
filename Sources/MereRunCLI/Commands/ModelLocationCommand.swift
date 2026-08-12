import ArgumentParser
import Foundation
import MereRunCore

struct ModelLocation: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "location",
        abstract: "Manage read-only model catalog locations.",
        discussion: """
        The primary model store remains the only location mere.run writes to. Registered
        roots and bindings are read-only: removing a registration never deletes its payload.
        """,
        subcommands: [
            ModelLocationList.self,
            ModelLocationAdd.self,
            ModelLocationRemove.self,
            ModelLocationBind.self,
            ModelLocationUnbind.self,
        ],
        defaultSubcommand: ModelLocationList.self
    )
}

struct ModelLocationList: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List the writable store, search roots, and explicit bindings."
    )

    @Flag(name: [.long], help: "Emit structured JSON.")
    var json: Bool = false

    func run() throws {
        let fileManager = FileManager.default
        let registry = try ModelLocationRegistry.load(fileManager: fileManager)
        let output = ModelLocationListOutput(
            primaryStore: .init(
                path: MereRunModelPaths.modelsDir.path,
                available: Self.isDirectory(MereRunModelPaths.modelsDir, fileManager: fileManager)
            ),
            searchRoots: registry.searchRoots.map { path in
                let url = URL(fileURLWithPath: path, isDirectory: true)
                return .init(path: url.path, available: Self.isDirectory(url, fileManager: fileManager))
            },
            bindings: registry.bindings.map { binding in
                let url = URL(fileURLWithPath: binding.path, isDirectory: true)
                return .init(
                    modelID: binding.modelID,
                    path: url.path,
                    available: Self.isDirectory(url, fileManager: fileManager),
                    usageTermsAcknowledged: binding.usageTermsAcknowledged
                )
            }
        )

        if json {
            print(try ModelStorageCommandOutput.encode(output))
            return
        }

        print("Primary store (writable)")
        print("  \(output.primaryStore.path)\(output.primaryStore.available ? "" : " (unavailable)")")
        print("\nSearch roots (read-only)")
        if output.searchRoots.isEmpty {
            print("  (none)")
        } else {
            for root in output.searchRoots {
                print("  \(root.path)\(root.available ? "" : " (offline)")")
            }
        }
        print("\nExplicit bindings (read-only)")
        if output.bindings.isEmpty {
            print("  (none)")
        } else {
            for binding in output.bindings {
                print("  \(binding.modelID) -> \(binding.path)\(binding.available ? "" : " (offline)")")
            }
        }
    }

    private static func isDirectory(_ url: URL, fileManager: FileManager) -> Bool {
        var isDirectory: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }
}

struct ModelLocationAdd: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "add",
        abstract: "Register a read-only root containing directories named for canonical model IDs."
    )

    @Argument(help: "Existing directory to search for canonical model subdirectories.")
    var path: String

    func run() throws {
        let url = try ModelLocationSupport.existingDirectory(path)
        guard url != MereRunModelPaths.modelsDir.standardizedFileURL else {
            throw ValidationError("That directory is already the writable primary model store.")
        }
        var registry = try ModelLocationRegistry.load()
        registry.addSearchRoot(url)
        try registry.save()
        print(url.path)
    }
}

struct ModelLocationRemove: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Unregister a search root without deleting its files."
    )

    @Argument(help: "Registered search-root path.")
    var path: String

    func run() throws {
        let url = ModelLocationSupport.normalizedURL(path)
        var registry = try ModelLocationRegistry.load()
        guard registry.removeSearchRoot(url) else {
            throw ValidationError("Search root is not registered: \(url.path)")
        }
        try registry.save()
        print(url.path)
    }
}

struct ModelLocationBind: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bind",
        abstract: "Bind a canonical model id to an arbitrary read-only directory."
    )

    @Argument(help: "Canonical model id.")
    var modelID: String

    @Argument(help: "Existing model directory.")
    var path: String

    @Flag(
        name: [.customLong("accept-model-license")],
        help: "Confirm acceptance of this model's listed third-party terms for the binding."
    )
    var acceptModelLicense: Bool = false

    func run() throws {
        let normalizedID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let spec = ManagedModelCatalog.spec(for: normalizedID) else {
            throw ValidationError("Unknown canonical model id: \(modelID)")
        }
        let url = try ModelLocationSupport.existingDirectory(path)
        let primaryPath = MereRunModelPaths.modelsDir.standardizedFileURL.path
        guard url.path != primaryPath, !url.path.hasPrefix(primaryPath + "/") else {
            throw ValidationError(
                "Bindings are for external read-only directories; this path is inside the writable primary store."
            )
        }
        let manifest = try MereRunModelManifest.loadIfPresent(from: url)
        if let manifest, manifest.id != spec.id {
            throw ValidationError(
                "Manifest identifies \(manifest.id), so it cannot be bound as \(spec.id)."
            )
        }
        let termsAcknowledged = acceptModelLicense || manifest?.usageTermsAcknowledged == true
        if let restriction = spec.usageRestriction, !termsAcknowledged {
            let terms = restriction.terms.map {
                "- \($0.component): \($0.license)\n  \($0.licenseURL)"
            }.joined(separator: "\n")
            throw ValidationError(
                """
                Model \(spec.id) has third-party usage terms: \(restriction.summary)
                \(terms)
                Pass --accept-model-license after reviewing them and confirming your acceptance.
                """
            )
        }

        let report = MereRunModelValidator.validateRegisteredBinding(
            modelRoot: url,
            expectedModelID: spec.id,
            usageTermsAcknowledged: termsAcknowledged
        )
        guard report.isValid else {
            throw MereRunModelValidator.ValidationError.invalidModelRoot(
                url,
                details: report.errors
            )
        }

        var registry = try ModelLocationRegistry.load()
        registry.addBinding(
            modelID: spec.id,
            url: url,
            usageTermsAcknowledged: termsAcknowledged
        )
        try registry.save()
        print("\(spec.id)\t\(url.path)")
    }
}

struct ModelLocationUnbind: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "unbind",
        abstract: "Remove explicit bindings without deleting model files."
    )

    @Argument(help: "Canonical model id.")
    var modelID: String

    @Argument(help: "Optional exact bound path; omit to remove every binding for the id.")
    var path: String?

    func run() throws {
        let normalizedID = modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard ManagedModelCatalog.spec(for: normalizedID) != nil else {
            throw ValidationError("Unknown canonical model id: \(modelID)")
        }
        let url = path.map(ModelLocationSupport.normalizedURL)
        var registry = try ModelLocationRegistry.load()
        let removed = registry.removeBindings(modelID: normalizedID, url: url)
        guard removed > 0 else {
            throw ValidationError("No matching binding is registered for \(normalizedID).")
        }
        try registry.save()
        print("\(normalizedID)\t\(removed)")
    }
}

private enum ModelLocationSupport {
    static func normalizedURL(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
    }

    static func existingDirectory(_ path: String, fileManager: FileManager = .default) throws -> URL {
        let url = normalizedURL(path)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ValidationError("Directory does not exist: \(url.path)")
        }
        guard fileManager.isReadableFile(atPath: url.path) else {
            throw ValidationError("Directory is not readable: \(url.path)")
        }
        return url
    }
}

struct ModelLocationListOutput: Codable, Equatable {
    struct Root: Codable, Equatable {
        let path: String
        let available: Bool
    }

    struct Binding: Codable, Equatable {
        let modelID: String
        let path: String
        let available: Bool
        let usageTermsAcknowledged: Bool

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case path
            case available
            case usageTermsAcknowledged = "usage_terms_acknowledged"
        }
    }

    let primaryStore: Root
    let searchRoots: [Root]
    let bindings: [Binding]

    enum CodingKeys: String, CodingKey {
        case primaryStore = "primary_store"
        case searchRoots = "search_roots"
        case bindings
    }
}
