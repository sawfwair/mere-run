import Foundation

/// Persisted read-only model locations that augment mere.run's writable primary store.
///
/// Search roots use the canonical `<root>/<model-id>` layout. Bindings map a canonical
/// model id to an arbitrary existing directory without copying files or writing a
/// manifest into externally managed storage.
public struct ModelLocationRegistry: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public struct Binding: Codable, Equatable, Hashable, Sendable {
        public let modelID: String
        public let path: String
        public let usageTermsAcknowledged: Bool

        public init(modelID: String, path: String, usageTermsAcknowledged: Bool = false) {
            self.modelID = modelID
            self.path = path
            self.usageTermsAcknowledged = usageTermsAcknowledged
        }

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case path
            case usageTermsAcknowledged = "usage_terms_acknowledged"
        }
    }

    public var schemaVersion: Int
    public var searchRoots: [String]
    public var bindings: [Binding]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        searchRoots: [String] = [],
        bindings: [Binding] = []
    ) {
        self.schemaVersion = schemaVersion
        self.searchRoots = searchRoots
        self.bindings = bindings
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case searchRoots = "search_roots"
        case bindings
    }

    public static var fileURL: URL {
        MereRunModelPaths.applicationSupportBase
            .appendingPathComponent("model_locations.json", isDirectory: false)
    }

    public static func load(from url: URL = fileURL, fileManager: FileManager = .default) throws -> Self {
        guard fileManager.fileExists(atPath: url.path) else {
            return Self()
        }
        let data = try Data(contentsOf: url)
        let registry = try JSONDecoder().decode(Self.self, from: data)
        guard registry.schemaVersion == currentSchemaVersion else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return registry.normalized()
    }

    public func save(to url: URL = fileURL, fileManager: FileManager = .default) throws {
        let normalized = normalized()
        try fileManager.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(normalized).write(to: url, options: .atomic)
        try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    public mutating func addSearchRoot(_ url: URL) {
        let path = Self.pathKey(url)
        searchRoots.removeAll { Self.pathKey(URL(fileURLWithPath: $0)) == path }
        searchRoots.append(path)
    }

    @discardableResult
    public mutating func removeSearchRoot(_ url: URL) -> Bool {
        let path = Self.pathKey(url)
        let oldCount = searchRoots.count
        searchRoots.removeAll { Self.pathKey(URL(fileURLWithPath: $0)) == path }
        return searchRoots.count != oldCount
    }

    public mutating func addBinding(
        modelID: String,
        url: URL,
        usageTermsAcknowledged: Bool
    ) {
        let normalizedID = modelID.lowercased()
        let path = Self.pathKey(url)
        bindings.removeAll {
            $0.modelID.lowercased() == normalizedID
                && Self.pathKey(URL(fileURLWithPath: $0.path)) == path
        }
        bindings.append(
            Binding(
                modelID: normalizedID,
                path: path,
                usageTermsAcknowledged: usageTermsAcknowledged
            )
        )
    }

    @discardableResult
    public mutating func removeBindings(modelID: String, url: URL? = nil) -> Int {
        let normalizedID = modelID.lowercased()
        let path = url.map(Self.pathKey)
        let oldCount = bindings.count
        bindings.removeAll { binding in
            guard binding.modelID.lowercased() == normalizedID else { return false }
            guard let path else { return true }
            return Self.pathKey(URL(fileURLWithPath: binding.path)) == path
        }
        return oldCount - bindings.count
    }

    public func normalized() -> Self {
        var rootPaths: [String] = []
        var seenRoots: Set<String> = []
        for rawPath in searchRoots {
            let path = Self.pathKey(URL(fileURLWithPath: rawPath))
            if seenRoots.insert(path).inserted {
                rootPaths.append(path)
            }
        }

        var normalizedBindings: [Binding] = []
        var seenBindings: Set<String> = []
        for binding in bindings {
            let id = binding.modelID.lowercased()
            let path = Self.pathKey(URL(fileURLWithPath: binding.path))
            if seenBindings.insert("\(id)\u{0}\(path)").inserted {
                normalizedBindings.append(
                    Binding(
                        modelID: id,
                        path: path,
                        usageTermsAcknowledged: binding.usageTermsAcknowledged
                    )
                )
            }
        }

        return Self(searchRoots: rootPaths, bindings: normalizedBindings)
    }

    private static func pathKey(_ url: URL) -> String {
        let expanded = (url.path as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }
}

public enum ModelLocationKind: String, Codable, Hashable, Sendable {
    case primaryStore = "primary_store"
    case registeredBinding = "registered_binding"
    case registeredSearchRoot = "registered_search_root"

    public var isExternallyManaged: Bool {
        self != .primaryStore
    }
}

public struct ModelLocationCandidate: Hashable, Sendable {
    public let modelID: String
    public let rootURL: URL
    public let kind: ModelLocationKind
    public let catalogRootURL: URL?
    public let usageTermsAcknowledged: Bool

    public init(
        modelID: String,
        rootURL: URL,
        kind: ModelLocationKind,
        catalogRootURL: URL? = nil,
        usageTermsAcknowledged: Bool = false
    ) {
        self.modelID = modelID
        self.rootURL = rootURL.standardizedFileURL
        self.kind = kind
        self.catalogRootURL = catalogRootURL?.standardizedFileURL
        self.usageTermsAcknowledged = usageTermsAcknowledged
    }
}

public struct ModelLocationSnapshot: Equatable, Sendable {
    public let primaryRoot: URL
    public let searchRoots: [URL]
    public let bindings: [ModelLocationRegistry.Binding]

    public init(
        primaryRoot: URL,
        searchRoots: [URL] = [],
        bindings: [ModelLocationRegistry.Binding] = []
    ) {
        self.primaryRoot = primaryRoot.standardizedFileURL
        self.searchRoots = searchRoots.map(\.standardizedFileURL)
        self.bindings = bindings
    }

    public func candidates(for modelID: String) -> [ModelLocationCandidate] {
        let normalizedID = modelID.lowercased()
        var candidates = [
            ModelLocationCandidate(
                modelID: normalizedID,
                rootURL: primaryRoot.appendingPathComponent(modelID, isDirectory: true),
                kind: .primaryStore,
                catalogRootURL: primaryRoot
            ),
        ]

        candidates += bindings
            .filter { $0.modelID.lowercased() == normalizedID }
            .map {
                ModelLocationCandidate(
                    modelID: normalizedID,
                    rootURL: URL(fileURLWithPath: $0.path, isDirectory: true),
                    kind: .registeredBinding,
                    usageTermsAcknowledged: $0.usageTermsAcknowledged
                )
            }

        candidates += searchRoots.map { root in
            ModelLocationCandidate(
                modelID: normalizedID,
                rootURL: root.appendingPathComponent(modelID, isDirectory: true),
                kind: .registeredSearchRoot,
                catalogRootURL: root
            )
        }

        var seen: Set<String> = []
        return candidates.filter { seen.insert($0.rootURL.standardizedFileURL.path).inserted }
    }
}

public enum MereRunModelLocations {
    public static func snapshot(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults = .standard,
        registryURL: URL = ModelLocationRegistry.fileURL
    ) -> ModelLocationSnapshot {
        let store = MereRunModelPaths.modelStoreResolution(
            fileManager: fileManager,
            environment: environment,
            defaults: defaults
        )
        guard MereRunModelPaths.includesRegisteredModelLocations(environment: environment) else {
            return ModelLocationSnapshot(primaryRoot: store.activeModelsDir)
        }
        let registry = (try? ModelLocationRegistry.load(from: registryURL, fileManager: fileManager)) ?? .init()
        return ModelLocationSnapshot(
            primaryRoot: store.activeModelsDir,
            searchRoots: registry.searchRoots.map { URL(fileURLWithPath: $0, isDirectory: true) },
            bindings: registry.bindings
        )
    }
}
