import ArgumentParser
import Foundation
import MereRunCore
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

struct PluginBundleStore {
    let root: URL
    var trustedKeys = PluginBundleEnvelope.trustedKeys
    var verifyPlatform: (URL, PluginBundleManifest) throws -> Void = { try PluginBundleIO.verifyMacOS($0, manifest: $1) }
    var smoke: (URL, PluginBundleManifest) throws -> Void = Self.checkEntrypoints

    static var standard: Self {
        Self(root: MereRunModelPaths.applicationSupportBase.appendingPathComponent("plugins"))
    }

    struct State: Codable, Equatable {
        let current: String
        let previous: String?
        let highestSequence: Int
    }

    func install(
        envelopeData: Data, archive: URL, package: String, pluginID: String,
        stage: (URL, PluginBundleManifest, URL) throws -> Void = PluginBundleIO.stageDMG
    ) throws -> PluginBundleManifest {
        let envelope = try JSONDecoder().decode(PluginBundleEnvelope.self, from: envelopeData)
        let manifest = try envelope.verified(trustedKeys: trustedKeys)
        guard manifest.package == package, manifest.entrypoints[pluginID] == pluginID else {
            throw ValidationError("Signed bundle does not contain the requested plugin/package.")
        }
        let size = try archive.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size.map(Int64.init) == manifest.artifact.size,
              try PluginBundleIO.hash(archive) == manifest.artifact.sha256 else {
            throw ValidationError("Plugin bundle size or SHA-256 does not match the signed release. Installation stopped.")
        }
        return try locked {
            let old = try state(package)
            if let old {
                guard manifest.sequence >= old.highestSequence,
                      manifest.sequence != old.highestSequence || old.current == manifest.artifact.sha256
                        || old.previous == manifest.artifact.sha256 else {
                    throw ValidationError("Refusing a replayed or conflicting bundle release. Use plugin rollback for a retained version.")
                }
            }
            let version = versionURL(package, manifest.artifact.sha256)
            let app = version.appendingPathComponent(manifest.appBundle)
            if FileManager.default.fileExists(atPath: version.path) {
                guard try receipt(package, manifest.artifact.sha256) == manifest else {
                    throw ValidationError("The retained bundle receipt conflicts with this release. Installation stopped.")
                }
            } else {
                let staging = root.appendingPathComponent("staging-\(UUID().uuidString)")
                try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
                defer { try? FileManager.default.removeItem(at: staging) }
                let stagedApp = staging.appendingPathComponent(manifest.appBundle)
                try stage(archive, manifest, stagedApp)
                try PluginBundleIO.validateTree(stagedApp)
                try verifyPlatform(stagedApp, manifest)
                try smoke(stagedApp, manifest)
                try envelopeData.write(to: staging.appendingPathComponent("release.json"), options: .atomic)
                try FileManager.default.createDirectory(at: version.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.moveItem(at: staging, to: version)
            }
            // Relocation must work too. Never switch the active version before this passes.
            try PluginBundleIO.validateTree(app)
            try verifyPlatform(app, manifest)
            try smoke(app, manifest)
            let next = State(current: manifest.artifact.sha256,
                             previous: old?.current == manifest.artifact.sha256 ? old?.previous : old?.current,
                             highestSequence: max(old?.highestSequence ?? 0, manifest.sequence))
            try save(next, package: package)
            return manifest
        }
    }

    func rollback(package: String) throws -> PluginBundleManifest {
        try locked {
            guard let old = try state(package), let previous = old.previous else {
                throw ValidationError("No retained plugin bundle is available for rollback.")
            }
            let manifest = try receipt(package, previous)
            let app = versionURL(package, previous).appendingPathComponent(manifest.appBundle)
            try PluginBundleIO.validateTree(app)
            try verifyPlatform(app, manifest)
            try smoke(app, manifest)
            try save(State(current: previous, previous: old.current, highestSequence: old.highestSequence), package: package)
            return manifest
        }
    }

    func resolve(_ entrypoint: String) throws -> URL? {
        for (package, state) in try activePackages() {
            let manifest = try receipt(package, state.current)
            if let executable = manifest.entrypoints[entrypoint] {
                return versionURL(package, state.current).appendingPathComponent(manifest.appBundle)
                    .appendingPathComponent("Contents/MacOS/\(executable)")
            }
        }
        return nil
    }

    func entrypoints() throws -> [String] {
        try activePackages().flatMap { package, state in
            try receipt(package, state.current).entrypoints.keys
        }.sorted()
    }

    func package(containing entrypoint: String) throws -> String? {
        for (package, state) in try activePackages() {
            if try receipt(package, state.current).entrypoints[entrypoint] != nil { return package }
        }
        return nil
    }

    func state(_ package: String) throws -> State? {
        guard PluginBundleManifest.matches(package, "^mere-[a-z0-9-]+$") else {
            throw ValidationError("Invalid plugin bundle package identifier.")
        }
        let url = packageURL(package).appendingPathComponent("active.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let value = try JSONDecoder().decode(State.self, from: Data(contentsOf: url))
        guard PluginBundleManifest.matches(value.current, "^[0-9a-f]{64}$"),
              value.previous.map({ PluginBundleManifest.matches($0, "^[0-9a-f]{64}$") }) ?? true,
              value.highestSequence > 0 else { throw ValidationError("Invalid plugin bundle installation state.") }
        return value
    }

    private func receipt(_ package: String, _ digest: String) throws -> PluginBundleManifest {
        let data = try Data(contentsOf: versionURL(package, digest).appendingPathComponent("release.json"))
        let envelope = try JSONDecoder().decode(PluginBundleEnvelope.self, from: data)
        // Expiry prevents new installs, not offline use of an already verified installation.
        let manifest = try envelope.verified(trustedKeys: trustedKeys, now: nil)
        guard manifest.package == package, manifest.artifact.sha256 == digest else {
            throw ValidationError("Plugin bundle receipt does not match its installation.")
        }
        return manifest
    }

    private func activePackages() throws -> [(String, State)] {
        let packages = root.appendingPathComponent("packages")
        guard FileManager.default.fileExists(atPath: packages.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(atPath: packages.path).sorted().compactMap { name in
            try state(name).map { (name, $0) }
        }
    }

    private func packageURL(_ package: String) -> URL { root.appendingPathComponent("packages/\(package)") }
    private func versionURL(_ package: String, _ digest: String) -> URL {
        packageURL(package).appendingPathComponent("versions/\(digest)")
    }
    private func save(_ value: State, package: String) throws {
        try JSONEncoder().encode(value).write(to: packageURL(package).appendingPathComponent("active.json"), options: .atomic)
    }

    private func locked<T>(_ body: () throws -> T) throws -> T {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let descriptor = open(root.appendingPathComponent("install.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, 0o600)
        guard descriptor >= 0 else { throw ValidationError("Cannot open plugin installation lock.") }
        defer { close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            throw ValidationError("Another plugin installation is running. Retry after it finishes.")
        }
        defer { flock(descriptor, LOCK_UN) }
        return try body()
    }

    static func checkEntrypoints(_ app: URL, _ manifest: PluginBundleManifest) throws {
        for (id, command) in manifest.entrypoints.sorted(by: { $0.key < $1.key }) {
            let executable = app.appendingPathComponent("Contents/MacOS/\(command)")
            let data = try PluginBundleIO.capture(executable.path, ["manifest", "--json"])
            let actual = try JSONDecoder().decode(PluginManifest.self, from: data)
            guard actual.contractVersion == "mere.run/plugin.v1", actual.name == id, actual.version == manifest.version else {
                throw ValidationError("Bundled plugin identity/version does not match the signed release: \(id)")
            }
            if actual.graphProvider != nil {
                _ = try WorkflowGraphProviderRegistry.loadProvider(entrypoint: executable.path)
            }
        }
    }
}
