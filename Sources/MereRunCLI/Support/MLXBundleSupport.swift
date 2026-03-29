import ArgumentParser
#if canImport(Darwin)
import Darwin
#endif
import Foundation

enum MLXBundleSupport {
    static func ensureAvailable(quiet: Bool) throws {
        let bundleName = "mlx-swift_Cmlx.bundle"
        let fm = FileManager.default
        let execDir = executableDirectory()
        let destBundleURL = execDir.appendingPathComponent(bundleName, isDirectory: true)

        if fm.fileExists(atPath: destBundleURL.path) {
            return
        }

        let rootCandidates = workspaceRootCandidates(executableDir: execDir)
        for root in rootCandidates {
            if let sourceBundleURL = findMlxSwiftBundle(workspaceRoot: root) {
                try? fm.removeItem(at: destBundleURL)
                try fm.copyItem(at: sourceBundleURL, to: destBundleURL)
                if !quiet {
                    CLIStderr.write("[mererun] Installed \(bundleName) for mlx-swift Metal shaders.\n")
                }
                return
            }
        }

        if ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] != nil {
            if !quiet {
                CLIStderr.write("[mererun] warning: \(bundleName) not found near executable; relying on DYLD_FRAMEWORK_PATH.\n")
            }
            return
        }

        throw ValidationError(
            """
            Missing mlx-swift Metal shaders (\(bundleName)).

            Build the package once to generate the metallib bundle, then rerun:
              swift build
            """
        )
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

    private static func findMlxSwiftBundle(workspaceRoot: URL) -> URL? {
        let fm = FileManager.default

        let vendoredBundle = workspaceRoot.appendingPathComponent("vendor/mlx-swift_Cmlx.bundle", isDirectory: true)
        if fm.fileExists(atPath: vendoredBundle.path) {
            return vendoredBundle
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
                return url
            }
        }

        let buildRoot = workspaceRoot.appendingPathComponent(".build", isDirectory: true)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: buildRoot.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
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
                return candidate
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
            return nil
        }

        for case let url as URL in enumerator {
            if url.lastPathComponent == "mlx-swift_Cmlx.bundle" {
                return url
            }
            if skipped.contains(url.lastPathComponent),
               (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                enumerator.skipDescendants()
            }
        }

        return nil
    }
}
