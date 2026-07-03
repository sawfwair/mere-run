import ArgumentParser
#if canImport(Darwin)
import Darwin
#endif
import Foundation
import MereRunCore

enum MLXBundleSupport {
#if os(Linux)
    static func ensureAvailable(quiet _: Bool) throws {}
#else
    private static let bundleName = "mlx-swift_Cmlx.bundle"
    private static let stampName = "default.metallib.version"

    /// Relationship between a bundle's metallib version stamp
    /// (`default.metallib.version`, written by scripts/build_mlx_metallib.sh)
    /// and the mlx core version this binary was compiled against. `swift build`
    /// never regenerates the metallib, so a leftover library from an older
    /// mlx-swift silently corrupts inference — the stamp is the only guard.
    enum MetallibStamp {
        case matched
        case unstamped
        case mismatched(stamped: String, expected: String)
        /// The runtime mlx version is unavailable (e.g. prebuilt MLX); no
        /// judgement is possible.
        case unvalidatable
    }

    static func ensureAvailable(quiet: Bool) throws {
        let fm = FileManager.default
        let execDir = executableDirectory()
        let destBundleURL = execDir.appendingPathComponent(bundleName, isDirectory: true)

        if hasMetallib(destBundleURL) {
            try validateAndFinish(
                bundleURL: destBundleURL,
                executableDir: execDir,
                quiet: quiet,
                justInstalled: false
            )
            return
        }

        let flatResourcesURL = execDir.appendingPathComponent("Resources", isDirectory: true)
        if hasMetallib(resourcesURL: flatResourcesURL) {
            try validateFlatResources(
                resourcesURL: flatResourcesURL,
                executableDir: execDir,
                quiet: quiet
            )
            return
        }

        // No usable bundle next to the executable (missing entirely, or a
        // husk without a metallib inside): copy one in, preferring candidates
        // whose stamp matches this binary.
        if let source = locateCandidateBundle(executableDir: execDir) {
            try? fm.removeItem(at: destBundleURL)
            try fm.copyItem(at: source, to: destBundleURL)
            if !quiet {
                CLIStderr.write("[mererun] Installed \(bundleName) for mlx-swift Metal shaders (from \(source.path)).\n")
            }
            try validateAndFinish(
                bundleURL: destBundleURL,
                executableDir: execDir,
                quiet: quiet,
                justInstalled: true
            )
            return
        }

        if ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] != nil {
            if !quiet {
                CLIStderr.write("[mererun] warning: \(bundleName) not found near executable; relying on DYLD_FRAMEWORK_PATH.\n")
            }
            return
        }

        throw ValidationError(
            """
            Missing mlx-swift Metal shaders (\(bundleName) or Resources/default.metallib).

            `swift build` does not generate the Metal kernel library. Build a
            stamped one from the current mlx-swift checkout, then rerun:

              scripts/build_mlx_metallib.sh

            (Run `swift package resolve` first if .build/checkouts/mlx-swift
            is missing.)
            """
        )
    }

    private static func validateFlatResources(
        resourcesURL: URL,
        executableDir: URL,
        quiet _: Bool
    ) throws {
        switch stampStatus(resourcesURL: resourcesURL) {
        case .matched, .unvalidatable:
            return

        case .unstamped:
            CLIStderr.write(
                """
                [mererun] WARNING: the MLX Metal shader library has no version stamp:
                [mererun]   \(resourcesURL.path)
                [mererun] Its provenance is unknown, so it may not match this binary\(expectedVersionSuffix()).
                [mererun] A stale metallib silently corrupts output (gibberish, nondeterministic
                [mererun] generation past ~1024 tokens of context). Rebuild and stamp it with:
                [mererun]   scripts/build_mlx_metallib.sh

                """
            )

        case .mismatched(let stamped, let expected):
            if let replacement = locateCandidateBundle(executableDir: executableDir, matchedOnly: true) {
                try installCompatibilityMetallibs(bundleURL: replacement, executableDir: executableDir)
                CLIStderr.write("[mererun] Replaced stale MLX metallib (built for mlx \(stamped), this binary needs \(expected)) with matching copy from \(replacement.path).\n")
                return
            }

            if mismatchOverrideEnabled {
                CLIStderr.write("[mererun] WARNING: running with a mismatched MLX metallib (built for mlx \(stamped), this binary needs \(expected)) because MERERUN_ALLOW_METALLIB_MISMATCH=1. Expect corrupted output.\n")
                return
            }

            throw ValidationError(
                """
                Stale MLX Metal shader library.

                  metallib built for mlx core: \(stamped)
                  this binary requires:        \(expected)
                  resources: \(resourcesURL.path)

                A mismatched metallib produces silently corrupted inference
                (gibberish, nondeterministic generation past ~1024 tokens of
                context). Rebuild it from the current checkout:

                  scripts/build_mlx_metallib.sh

                Emergency override (unsafe): MERERUN_ALLOW_METALLIB_MISMATCH=1
                """
            )
        }
    }

    private static func validateAndFinish(
        bundleURL: URL,
        executableDir: URL,
        quiet: Bool,
        justInstalled: Bool
    ) throws {
        switch stampStatus(bundleURL: bundleURL) {
        case .matched, .unvalidatable:
            try installCompatibilityMetallibs(bundleURL: bundleURL, executableDir: executableDir)

        case .unstamped:
            CLIStderr.write(
                """
                [mererun] WARNING: the MLX Metal shader library has no version stamp:
                [mererun]   \(bundleURL.path)
                [mererun] Its provenance is unknown, so it may not match this binary\(expectedVersionSuffix()).
                [mererun] A stale metallib silently corrupts output (gibberish, nondeterministic
                [mererun] generation past ~1024 tokens of context). Rebuild and stamp it with:
                [mererun]   scripts/build_mlx_metallib.sh

                """
            )
            try installCompatibilityMetallibs(bundleURL: bundleURL, executableDir: executableDir)

        case .mismatched(let stamped, let expected):
            if !justInstalled,
               let replacement = locateCandidateBundle(executableDir: executableDir, matchedOnly: true) {
                let fm = FileManager.default
                try? fm.removeItem(at: bundleURL)
                try fm.copyItem(at: replacement, to: bundleURL)
                CLIStderr.write("[mererun] Replaced stale MLX metallib (built for mlx \(stamped), this binary needs \(expected)) with matching copy from \(replacement.path).\n")
                try installCompatibilityMetallibs(bundleURL: bundleURL, executableDir: executableDir)
                return
            }

            if mismatchOverrideEnabled {
                CLIStderr.write("[mererun] WARNING: running with a mismatched MLX metallib (built for mlx \(stamped), this binary needs \(expected)) because MERERUN_ALLOW_METALLIB_MISMATCH=1. Expect corrupted output.\n")
                try installCompatibilityMetallibs(bundleURL: bundleURL, executableDir: executableDir)
                return
            }

            throw ValidationError(
                """
                Stale MLX Metal shader library.

                  metallib built for mlx core: \(stamped)
                  this binary requires:        \(expected)
                  bundle: \(bundleURL.path)

                A mismatched metallib produces silently corrupted inference
                (gibberish, nondeterministic generation past ~1024 tokens of
                context). Rebuild it from the current checkout:

                  scripts/build_mlx_metallib.sh

                Emergency override (unsafe): MERERUN_ALLOW_METALLIB_MISMATCH=1
                """
            )
        }
    }

    private static var mismatchOverrideEnabled: Bool {
        let raw = (ProcessInfo.processInfo.environment["MERERUN_ALLOW_METALLIB_MISMATCH"] ?? "").lowercased()
        return raw == "1" || raw == "true" || raw == "yes"
    }

    private static func expectedVersionSuffix() -> String {
        guard let expected = MLXRuntimeVersion.coreVersion.flatMap(normalizedVersion) else {
            return ""
        }
        return " (mlx core \(expected))"
    }

    static func stampStatus(bundleURL: URL) -> MetallibStamp {
        let resourcesURL = bundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Resources", isDirectory: true)
        return stampStatus(resourcesURL: resourcesURL)
    }

    private static func stampStatus(resourcesURL: URL) -> MetallibStamp {
        guard let expected = MLXRuntimeVersion.coreVersion.flatMap(normalizedVersion) else {
            return .unvalidatable
        }
        let stampURL = resourcesURL.appendingPathComponent(stampName, isDirectory: false)
        guard let contents = try? String(contentsOf: stampURL, encoding: .utf8),
              let stamped = stampField("mlx-core-version", in: contents).flatMap(normalizedVersion) else {
            return .unstamped
        }
        return stamped == expected ? .matched : .mismatched(stamped: stamped, expected: expected)
    }

    private static func stampField(_ key: String, in contents: String) -> String? {
        for line in contents.split(separator: "\n") {
            let parts = line.split(separator: ":", maxSplits: 1)
            guard parts.count == 2,
                  parts[0].trimmingCharacters(in: .whitespaces) == key else {
                continue
            }
            return parts[1].trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    /// Reduces a version string to its MAJOR.MINOR.PATCH numeric prefix, so
    /// dev-build suffixes ("0.31.1.dev20260115+abc") still compare equal.
    private static func normalizedVersion(_ raw: String) -> String? {
        let numeric = raw.trimmingCharacters(in: .whitespaces).prefix { $0.isNumber || $0 == "." }
        let parts = numeric.split(separator: ".").prefix(3)
        guard !parts.isEmpty else {
            return nil
        }
        return parts.joined(separator: ".")
    }

    private static func hasMetallib(_ bundleURL: URL) -> Bool {
        let resourcesURL = bundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Resources", isDirectory: true)
        return hasMetallib(resourcesURL: resourcesURL)
    }

    private static func hasMetallib(resourcesURL: URL) -> Bool {
        let metallibURL = resourcesURL.appendingPathComponent("default.metallib", isDirectory: false)
        return FileManager.default.fileExists(atPath: metallibURL.path)
    }

    /// Finds a bundle to copy next to the executable. Preference order:
    /// stamp-matched candidates, then unstamped ones, then mismatched ones
    /// (`matchedOnly` restricts to the first tier, for self-healing).
    private static func locateCandidateBundle(executableDir: URL, matchedOnly: Bool = false) -> URL? {
        var firstUnstamped: URL?
        var firstMismatched: URL?

        for candidate in candidateBundleURLs(executableDir: executableDir) {
            guard hasMetallib(candidate) else {
                continue
            }
            switch stampStatus(bundleURL: candidate) {
            case .matched:
                return candidate
            case .unvalidatable:
                // No runtime version to compare against; without a better
                // signal the first available bundle wins (legacy behaviour).
                return matchedOnly ? nil : candidate
            case .unstamped:
                if firstUnstamped == nil { firstUnstamped = candidate }
            case .mismatched:
                if firstMismatched == nil { firstMismatched = candidate }
            }
        }

        return matchedOnly ? nil : (firstUnstamped ?? firstMismatched)
    }

    private static func candidateBundleURLs(executableDir: URL) -> [URL] {
        var urls: [URL] = []
        for candidateDir in bundleInstallCandidates(executableDir: executableDir) {
            urls.append(candidateDir.appendingPathComponent(bundleName, isDirectory: true))
        }
        for root in workspaceRootCandidates(executableDir: executableDir) {
            urls.append(contentsOf: findMlxSwiftBundles(workspaceRoot: root))
        }
        var seen = Set<String>()
        return urls.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private static func installCompatibilityMetallibs(bundleURL: URL, executableDir: URL) throws {
        let fm = FileManager.default
        let bundleResources = bundleURL
            .appendingPathComponent("Contents", isDirectory: true)
            .appendingPathComponent("Resources", isDirectory: true)
        let sourceMetallib = bundleResources.appendingPathComponent("default.metallib", isDirectory: false)

        guard fm.fileExists(atPath: sourceMetallib.path) else {
            return
        }

        let resourcesDir = executableDir.appendingPathComponent("Resources", isDirectory: true)
        try fm.createDirectory(at: resourcesDir, withIntermediateDirectories: true)

        let destinations: [URL] = [
            executableDir.appendingPathComponent("mlx.metallib", isDirectory: false),
            resourcesDir.appendingPathComponent("mlx.metallib", isDirectory: false),
            resourcesDir.appendingPathComponent("default.metallib", isDirectory: false),
        ]
        for destination in destinations {
            try refreshCopy(from: sourceMetallib, to: destination)
        }

        let sourceStamp = bundleResources.appendingPathComponent(stampName, isDirectory: false)
        if fm.fileExists(atPath: sourceStamp.path) {
            try? refreshCopy(from: sourceStamp, to: resourcesDir.appendingPathComponent(stampName, isDirectory: false))
        }
    }

    /// Copies `source` over `destination` unless the destination already has
    /// the same size. (The previous skip-if-exists behaviour let stale flat
    /// copies outlive bundle updates forever.) The copy goes through a temp
    /// file + rename so concurrent CLI startups never observe a torn file.
    private static func refreshCopy(from source: URL, to destination: URL) throws {
        let fm = FileManager.default
        if let sourceSize = fileSize(source),
           let destinationSize = fileSize(destination),
           sourceSize == destinationSize {
            return
        }

        let tmp = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).tmp-\(getpid())", isDirectory: false)
        try? fm.removeItem(at: tmp)
        do {
            try fm.copyItem(at: source, to: tmp)
            if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(destination, withItemAt: tmp)
            } else {
                try fm.moveItem(at: tmp, to: destination)
            }
        } catch {
            try? fm.removeItem(at: tmp)
            // Tolerate races with a concurrent startup that produced the file.
            if fm.fileExists(atPath: destination.path) {
                return
            }
            throw error
        }
    }

    private static func fileSize(_ url: URL) -> Int64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? Int64 else {
            return nil
        }
        return size
    }

    private static func executableDirectory() -> URL {
        // `Bundle.main.executableURL` / `argv[0]` can reflect the *invocation* path (e.g. a symlink in
        // `/usr/local/bin`). `proc_pidpath` gives us the actual on-disk executable path.
        if let execURL = processExecutableURL() {
            return execURL.deletingLastPathComponent()
        }
        if let execURL = Bundle.main.executableURL {
            return execURL.resolvingSymlinksInPath().deletingLastPathComponent()
        }
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
        let execURL = URL(fileURLWithPath: CommandLine.arguments[0], relativeTo: cwd)
            .resolvingSymlinksInPath()
        return execURL.deletingLastPathComponent()
    }

#if canImport(Darwin)
    private static func processExecutableURL() -> URL? {
        // `proc_pidpath` needs a buffer; 4096 is comfortably above typical macOS PATH_MAX.
        var buffer = [CChar](repeating: 0, count: 4096)
        let result = proc_pidpath(getpid(), &buffer, UInt32(buffer.count))
        guard result > 0 else {
            return nil
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        let path = String(decoding: bytes, as: UTF8.self)
        return URL(fileURLWithPath: path).resolvingSymlinksInPath()
    }
#else
    private static func processExecutableURL() -> URL? { nil }
#endif

    private static func workspaceRootCandidates(executableDir: URL) -> [URL] {
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true).standardizedFileURL
        let inferred = inferWorkspaceRoot(fromExecutableDir: executableDir)

        var roots: [URL] = ancestry(of: cwd)
        if let inferred {
            roots.append(contentsOf: ancestry(of: inferred))
        }

        var seen = Set<String>()
        return roots.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private static func bundleInstallCandidates(executableDir: URL) -> [URL] {
        var candidates: [URL] = []

        if let invocationPath = apparentInvocationPath() {
            candidates.append(invocationPath.deletingLastPathComponent())
        }

        candidates.append(URL(fileURLWithPath: "/usr/local/bin", isDirectory: true))
        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin", isDirectory: true))

        var seen = Set<String>([executableDir.standardizedFileURL.path])
        return candidates.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private static func apparentInvocationPath() -> URL? {
        let argument = CommandLine.arguments[0]
        guard !argument.isEmpty else {
            return nil
        }

        let fm = FileManager.default
        if argument.contains("/") {
            let cwd = URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true)
            return URL(fileURLWithPath: argument, relativeTo: cwd).standardizedFileURL
        }

        let pathEntries = ProcessInfo.processInfo.environment["PATH"]?.split(separator: ":") ?? []
        for entry in pathEntries {
            let candidate = URL(fileURLWithPath: String(entry), isDirectory: true)
                .appendingPathComponent(argument, isDirectory: false)
            if fm.isExecutableFile(atPath: candidate.path) {
                return candidate.standardizedFileURL
            }
        }

        return nil
    }

    private static func inferWorkspaceRoot(fromExecutableDir executableDir: URL) -> URL? {
        let components = executableDir.standardizedFileURL.pathComponents
        guard let buildIndex = components.lastIndex(of: ".build"), buildIndex > 0 else {
            return nil
        }
        let rootPath = NSString.path(withComponents: Array(components.prefix(buildIndex)))
        return URL(fileURLWithPath: rootPath, isDirectory: true)
    }

    private static func ancestry(of url: URL) -> [URL] {
        var candidates: [URL] = []
        var current = url.standardizedFileURL

        while true {
            candidates.append(current)
            let parent = current.deletingLastPathComponent()
            if parent.path == current.path {
                break
            }
            current = parent
        }

        return candidates
    }

    private static func findMlxSwiftBundles(workspaceRoot: URL) -> [URL] {
        let fm = FileManager.default
        var found: [URL] = []

        let vendoredBundle = workspaceRoot.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle", isDirectory: true)
        if fm.fileExists(atPath: vendoredBundle.path) {
            found.append(vendoredBundle)
        }

        let explicitCandidates: [String] = [
            ".build/xcode/Build/Products/Debug/mlx-swift_Cmlx.bundle",
            ".build/xcode/Build/Products/Release/mlx-swift_Cmlx.bundle",
            ".build/Build/Products/Debug/mlx-swift_Cmlx.bundle",
            ".build/Build/Products/Release/mlx-swift_Cmlx.bundle",
            ".build/DerivedData/Build/Products/Debug/mlx-swift_Cmlx.bundle",
            ".build/DerivedData/Build/Products/Release/mlx-swift_Cmlx.bundle",
        ]

        for relative in explicitCandidates {
            let url = workspaceRoot.appendingPathComponent(relative, isDirectory: true)
            if fm.fileExists(atPath: url.path) {
                found.append(url)
            }
        }

        let buildRoot = workspaceRoot.appendingPathComponent(".build", isDirectory: true)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: buildRoot.path, isDirectory: &isDir), isDir.boolValue else {
            return found
        }

        let derivedDataDirs: [URL]
        do {
            derivedDataDirs = try fm.contentsOfDirectory(
                at: buildRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.lastPathComponent.hasPrefix("DerivedData") }
        } catch {
            derivedDataDirs = []
        }

        for dir in derivedDataDirs {
            let candidates: [URL] = [
                dir.appendingPathComponent("Build/Products/Debug/mlx-swift_Cmlx.bundle", isDirectory: true),
                dir.appendingPathComponent("Build/Products/Release/mlx-swift_Cmlx.bundle", isDirectory: true),
            ]
            for candidate in candidates where fm.fileExists(atPath: candidate.path) {
                found.append(candidate)
            }
        }

        let skipped: Set<String> = [
            "checkouts",
            "SourcePackages",
            "clang-module-cache",
            "ModuleCache.noindex",
            "artifacts",
        ]

        guard let enumerator = fm.enumerator(
            at: buildRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return found
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "mlx-swift_Cmlx.bundle" {
                found.append(url)
            }
            if skipped.contains(url.lastPathComponent),
               (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                enumerator.skipDescendants()
            }
        }

        return found
    }
#endif
}
