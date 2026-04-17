import Foundation

/// Resolves manifest `components` references into concrete local directories.
///
/// This is used to support hybrid/assembled models without requiring symlinks
/// (e.g. `zeta-base` using `zeta-max` tokenizer/text encoder/VAE).
public struct ModelComponentResolver {
    public enum Component: String, CaseIterable, Hashable, Sendable {
        case tokenizer
        case textEncoder
        case transformer
        case vae
        case scheduler

        public var manifestKey: String {
            switch self {
            case .tokenizer: return "tokenizer"
            case .textEncoder: return "text_encoder"
            case .transformer: return "transformer"
            case .vae: return "vae"
            case .scheduler: return "scheduler"
            }
        }
    }

    public struct ResolvedComponent: Hashable, Sendable {
        public let component: Component
        public let directoryURL: URL
        public let sourceModelRootURL: URL
        public let sourceManifest: MereRunModelManifest?

        public init(
            component: Component,
            directoryURL: URL,
            sourceModelRootURL: URL,
            sourceManifest: MereRunModelManifest?
        ) {
            self.component = component
            self.directoryURL = directoryURL
            self.sourceModelRootURL = sourceModelRootURL
            self.sourceManifest = sourceManifest
        }
    }

    public enum ResolutionError: LocalizedError, Sendable {
        case unknownModelID(component: Component, modelID: String)
        case modelNotFound(component: Component, modelID: String)
        case missingDirectory(component: Component, url: URL)
        case unsupportedRemote(component: Component, id: String)
        case anyOfFailed(component: Component, reasons: [String])

        public var errorDescription: String? {
            switch self {
            case .unknownModelID(let component, let modelID):
                return "Component \(component.manifestKey) references unknown model id: \(modelID)"
            case .modelNotFound(let component, let modelID):
                return "Component \(component.manifestKey) references missing model: \(modelID)"
            case .missingDirectory(let component, let url):
                return "Missing component directory for \(component.manifestKey): \(url.path)"
            case .unsupportedRemote(let component, let id):
                return "Component \(component.manifestKey) references remote content (\(id)), which is not yet supported."
            case .anyOfFailed(let component, let reasons):
                var lines: [String] = []
                lines.append("Component \(component.manifestKey) could not be resolved from anyOf candidates.")
                lines.append(contentsOf: reasons.map { "  - \($0)" })
                return lines.joined(separator: "\n")
            }
        }
    }

    private let modelRootURL: URL
    private let manifest: MereRunModelManifest?
    private let modelResolver: ModelResolver
    private let fileManager: FileManager

    public init(
        modelRootURL: URL,
        manifest: MereRunModelManifest?,
        modelResolver: ModelResolver = ModelResolver(),
        fileManager: FileManager = .default
    ) {
        self.modelRootURL = modelRootURL
        self.manifest = manifest
        self.modelResolver = modelResolver
        self.fileManager = fileManager
    }

    public func resolveDirectory(
        for component: Component,
        fallbackLocalPath: String
    ) throws -> ResolvedComponent {
        let fallbackRef = MereRunModelManifest.ComponentRef.local(path: fallbackLocalPath)
        guard let ref = componentRef(for: component) else {
            return try resolve(fallbackRef, component: component)
        }

        do {
            return try resolve(ref, component: component)
        } catch {
            // Compatibility: if a manifest references another model's component but that
            // reference is stale, prefer the local fallback path when it exists.
            if let local = localFallback(component: component, fallbackLocalPath: fallbackLocalPath) {
                return local
            }
            throw error
        }
    }

    private func componentRef(for component: Component) -> MereRunModelManifest.ComponentRef? {
        guard let components = manifest?.components else { return nil }
        switch component {
        case .tokenizer: return components.tokenizer
        case .textEncoder: return components.textEncoder
        case .transformer: return components.transformer
        case .vae: return components.vae
        case .scheduler: return components.scheduler
        }
    }

    private func localFallback(component: Component, fallbackLocalPath: String) -> ResolvedComponent? {
        let dir = modelRootURL
            .appendingPathComponent(fallbackLocalPath, isDirectory: true)
            .resolvingSymlinksInPath()
        guard existsDirectory(dir) else {
            return nil
        }
        return ResolvedComponent(
            component: component,
            directoryURL: dir,
            sourceModelRootURL: modelRootURL,
            sourceManifest: manifest
        )
    }

    private func existsDirectory(_ url: URL) -> Bool {
        var isDir: ObjCBool = false
        return fileManager.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
    }

    private func findNearestModelRoot(for directoryURL: URL, maxParentLevels: Int = 4) -> URL? {
        var current = directoryURL.standardizedFileURL
        for _ in 0...maxParentLevels {
            let manifestURL = MereRunModelManifest.url(in: current)
            if fileManager.fileExists(atPath: manifestURL.path) {
                return current
            }

            let parent = current.deletingLastPathComponent()
            if parent.path == current.path {
                break
            }
            current = parent
        }
        return nil
    }

    private func resolve(_ ref: MereRunModelManifest.ComponentRef, component: Component) throws -> ResolvedComponent {
        func resolvedLocalDirectory(relative path: String) throws -> ResolvedComponent {
            let dir = modelRootURL.appendingPathComponent(path, isDirectory: true).resolvingSymlinksInPath()
            guard existsDirectory(dir) else {
                throw ResolutionError.missingDirectory(component: component, url: dir)
            }
            return ResolvedComponent(
                component: component,
                directoryURL: dir,
                sourceModelRootURL: modelRootURL,
                sourceManifest: manifest
            )
        }

        func resolvedAbsoluteDirectory(path: String) throws -> ResolvedComponent {
            let dir = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
            guard existsDirectory(dir) else {
                throw ResolutionError.missingDirectory(component: component, url: dir)
            }

            let sourceRoot = findNearestModelRoot(for: dir) ?? dir
            let sourceManifest: MereRunModelManifest?
            if sourceRoot.path == dir.path {
                sourceManifest = nil
            } else {
                sourceManifest = try MereRunModelManifest.loadRequired(from: sourceRoot, fileManager: fileManager)
            }
            return ResolvedComponent(
                component: component,
                directoryURL: dir,
                sourceModelRootURL: sourceRoot,
                sourceManifest: sourceManifest
            )
        }

        func resolvedModelDirectory(modelID: String, path: String) throws -> ResolvedComponent {
            guard let id = ModelResolver.ModelID(rawValue: modelID) else {
                throw ResolutionError.unknownModelID(component: component, modelID: modelID)
            }

            let resolved: ModelResolver.Resolution
            do {
                resolved = try modelResolver.resolve(id)
            } catch {
                throw ResolutionError.modelNotFound(component: component, modelID: modelID)
            }

            let root = resolved.rootURL
            let dir = root.appendingPathComponent(path, isDirectory: true).resolvingSymlinksInPath()
            guard existsDirectory(dir) else {
                throw ResolutionError.missingDirectory(component: component, url: dir)
            }

            let sourceManifest = try MereRunModelManifest.loadRequired(from: root, fileManager: fileManager)
            return ResolvedComponent(
                component: component,
                directoryURL: dir,
                sourceModelRootURL: root,
                sourceManifest: sourceManifest
            )
        }

        switch ref {
        case .local(let path):
            return try resolvedLocalDirectory(relative: path)
        case .absolute(let path):
            return try resolvedAbsoluteDirectory(path: path)
        case .model(let modelID, let path):
            return try resolvedModelDirectory(modelID: modelID, path: path)
        case .remote(let id, _, _):
            throw ResolutionError.unsupportedRemote(component: component, id: id)
        case .anyOf(let candidates):
            var reasons: [String] = []
            for candidate in candidates {
                do {
                    return try resolve(candidate, component: component)
                } catch {
                    reasons.append(error.localizedDescription)
                }
            }
            throw ResolutionError.anyOfFailed(component: component, reasons: reasons)
        }
    }
}
