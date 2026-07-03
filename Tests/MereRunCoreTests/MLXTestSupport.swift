import Foundation

enum MLXTestSupport {
    /// Ensure MLX can locate its Metal library when running under `swift test`.
    ///
    /// In SwiftPM builds the metallib is emitted under:
    ///   `.build/.../mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib`
    ///
    /// But the MLX runtime first looks for a colocated `mlx.metallib` next to the
    /// running test binary. We satisfy that by symlinking (or copying) it there.
    static func ensureMetalLibraryAvailable() {
        let fm = FileManager.default
        let debugEnabled: Bool = {
            let raw = (ProcessInfo.processInfo.environment["MERERUN_TEST_DEBUG_MLX"] ?? "").lowercased()
            return raw == "1" || raw == "true" || raw == "yes"
        }()
        let arguments = CommandLine.arguments
        guard let executablePath = arguments.first, !executablePath.isEmpty else { return }
        let executableURL = URL(fileURLWithPath: executablePath).standardizedFileURL

        if debugEnabled {
            let xctest = resolveXCTestBundleURL(executableURL: executableURL, arguments: arguments)
            print("[MLXTestSupport] exe=\(executableURL.path)")
            print("[MLXTestSupport] xctest=\(xctest?.path ?? "<none>")")
        }

        let sourceBundle: URL?
        do {
            sourceBundle = try findCmlxBundle(executableURL: executableURL, arguments: arguments, fileManager: fm)
        } catch {
            if debugEnabled {
                print("[MLXTestSupport] Failed to resolve Cmlx bundle: \(error)")
            }
            sourceBundle = nil
        }

        guard let sourceBundle, fm.fileExists(atPath: sourceBundle.path) else {
            if debugEnabled {
                print("[MLXTestSupport] Cmlx bundle not found. exe=\(executableURL.path)")
            }
            return
        }

        if debugEnabled {
            print("[MLXTestSupport] cmlxBundle=\(sourceBundle.path)")
        }

        let sourceMetallib = sourceBundle.appendingPathComponent("Contents/Resources/default.metallib")
        if fm.fileExists(atPath: sourceMetallib.path) {
            installColocatedMetallib(
                executableURL: executableURL,
                sourceMetallib: sourceMetallib,
                fileManager: fm,
                debugEnabled: debugEnabled
            )
        } else if debugEnabled {
            print("[MLXTestSupport] default.metallib missing in bundle: \(sourceMetallib.path)")
        }

        installSwiftPMBundleIfPossible(
            sourceBundle: sourceBundle,
            executableURL: executableURL,
            arguments: arguments,
            fileManager: fm,
            debugEnabled: debugEnabled
        )
    }

    private static func installColocatedMetallib(
        executableURL: URL,
        sourceMetallib: URL,
        fileManager: FileManager,
        debugEnabled: Bool
    ) {
        let binaryDir = executableURL.deletingLastPathComponent()
        guard fileManager.isWritableFile(atPath: binaryDir.path) else { return }
        let destination = binaryDir.appendingPathComponent("mlx.metallib")
        // Replace rather than skip: a leftover link/copy from an earlier run
        // can point at a stale metallib, which silently corrupts kernels
        // (see scripts/build_mlx_metallib.sh). A symlink already pointing at
        // the current source is left alone so steady-state runs are no-ops.
        if symlinkTarget(of: destination, fileManager: fileManager) == sourceMetallib.path {
            return
        }
        try? fileManager.removeItem(at: destination)

        // Best-effort + race-safe: if concurrent tests call this at the same time,
        // one may succeed and the other will see "file exists" errors, which we ignore.
        do {
            try fileManager.createSymbolicLink(at: destination, withDestinationURL: sourceMetallib)
            if debugEnabled {
                print("[MLXTestSupport] Symlinked \(destination.path) -> \(sourceMetallib.path)")
            }
        } catch {
            do {
                try fileManager.copyItem(at: sourceMetallib, to: destination)
                if debugEnabled {
                    print("[MLXTestSupport] Copied \(sourceMetallib.path) -> \(destination.path)")
                }
            } catch {
                if debugEnabled {
                    print("[MLXTestSupport] Failed to install colocated metallib: \(error)")
                }
            }
        }
    }

    private static func installSwiftPMBundleIfPossible(
        sourceBundle: URL,
        executableURL: URL,
        arguments: [String],
        fileManager: FileManager,
        debugEnabled: Bool
    ) {
        guard let testBundleURL = resolveXCTestBundleURL(executableURL: executableURL, arguments: arguments) else {
            if debugEnabled {
                print("[MLXTestSupport] No .xctest bundle found in args; skipping SwiftPM bundle install. exe=\(executableURL.path)")
            }
            return
        }

        let resourcesDir = testBundleURL.appendingPathComponent("Contents/Resources", isDirectory: true)
        do {
            try fileManager.createDirectory(at: resourcesDir, withIntermediateDirectories: true)
            if debugEnabled {
                print("[MLXTestSupport] xctestResources=\(resourcesDir.path)")
            }
        } catch {
            if debugEnabled {
                print("[MLXTestSupport] Failed to create xctest Resources dir: \(error)")
            }
            return
        }

        let destinationBundle = resourcesDir.appendingPathComponent(sourceBundle.lastPathComponent, isDirectory: true)
        // Replace rather than skip: a leftover symlink/copy from an earlier run
        // can point at a stale bundle (this exact mechanism pinned tests to a
        // pre-0.30 metallib and produced NaN attention past 1024 keys). A
        // symlink already pointing at the current source is left alone so
        // steady-state runs are no-ops.
        if symlinkTarget(of: destinationBundle, fileManager: fileManager) == sourceBundle.path {
            return
        }
        try? fileManager.removeItem(at: destinationBundle)

        // MLX's SwiftPM lookup expects `<someBundle>/Contents/Resources/<SWIFTPM_BUNDLE>.bundle/...`.
        // Symlink (or copy) the entire Cmlx bundle into the test bundle's Resources directory.
        do {
            try fileManager.createSymbolicLink(at: destinationBundle, withDestinationURL: sourceBundle)
            if debugEnabled {
                print("[MLXTestSupport] Symlinked bundle \(destinationBundle.path) -> \(sourceBundle.path)")
            }
        } catch {
            do {
                try fileManager.copyItem(at: sourceBundle, to: destinationBundle)
                if debugEnabled {
                    print("[MLXTestSupport] Copied bundle \(sourceBundle.path) -> \(destinationBundle.path)")
                }
            } catch {
                if debugEnabled {
                    print("[MLXTestSupport] Failed to install SwiftPM bundle: \(error)")
                }
            }
        }
    }

    private static func symlinkTarget(of url: URL, fileManager: FileManager) -> String? {
        guard let target = try? fileManager.destinationOfSymbolicLink(atPath: url.path) else {
            return nil
        }
        if target.hasPrefix("/") {
            return target
        }
        return url.deletingLastPathComponent().appendingPathComponent(target).standardizedFileURL.path
    }

    private static func resolveXCTestBundleURL(executableURL: URL, arguments: [String]) -> URL? {
        // When invoked directly from the test bundle executable:
        //   .../<name>.xctest/Contents/MacOS/<exe>
        if executableURL.path.contains(".xctest/Contents/MacOS/") {
            return executableURL
                .deletingLastPathComponent() // .../MacOS
                .deletingLastPathComponent() // .../Contents
                .deletingLastPathComponent() // .../<name>.xctest
        }

        // When invoked via the `xctest` tool, the bundle path is usually an argument.
        for arg in arguments where arg.hasSuffix(".xctest") {
            return URL(fileURLWithPath: arg).standardizedFileURL
        }

        return nil
    }

    private static func findCmlxBundle(
        executableURL: URL,
        arguments: [String],
        fileManager: FileManager
    ) throws -> URL? {
        let workspaceRoots = candidateWorkspaceRoots(
            executableURL: executableURL,
            arguments: arguments,
            fileManager: fileManager
        )

        for root in workspaceRoots {
            let vendoredBundle = root
                .appendingPathComponent("vendor", isDirectory: true)
                .appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
            if fileManager.fileExists(atPath: vendoredBundle.path) {
                return vendoredBundle
            }
        }

        // Fast path: for SwiftPM test runs, the products directory is usually the
        // parent of the `.xctest` bundle and contains `mlx-swift_Cmlx.bundle`.
        if let testBundleURL = resolveXCTestBundleURL(executableURL: executableURL, arguments: arguments) {
            let productsDir = testBundleURL.deletingLastPathComponent()
            let direct = productsDir.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
            if fileManager.fileExists(atPath: direct.path) {
                return direct
            }
        }

        for root in workspaceRoots {
            let xcodeProductsDir = root
                .appendingPathComponent(".xcodebuild", isDirectory: true)
                .appendingPathComponent("Build", isDirectory: true)
                .appendingPathComponent("Products", isDirectory: true)
            if fileManager.fileExists(atPath: xcodeProductsDir.path),
               let bundle = findBundle(named: "mlx-swift_Cmlx.bundle", under: xcodeProductsDir, fileManager: fileManager) {
                return bundle
            }
        }

        // Final fallback: scan `.build/**/mlx-swift_Cmlx.bundle`.
        for root in workspaceRoots {
            let buildDir = root.appendingPathComponent(".build", isDirectory: true)
            if fileManager.fileExists(atPath: buildDir.path),
               let bundle = findBundle(named: "mlx-swift_Cmlx.bundle", under: buildDir, fileManager: fileManager) {
                return bundle
            }
        }

        return nil
    }

    private static func candidateWorkspaceRoots(
        executableURL: URL,
        arguments: [String],
        fileManager: FileManager
    ) -> [URL] {
        var candidates: [URL] = ancestry(
            of: URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
                .standardizedFileURL
        )

        if let testBundleURL = resolveXCTestBundleURL(executableURL: executableURL, arguments: arguments) {
            candidates.append(contentsOf: ancestry(of: testBundleURL))
        }

        candidates.append(contentsOf: ancestry(of: executableURL.standardizedFileURL))

        var seen = Set<String>()
        return candidates.filter { seen.insert($0.path).inserted }
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

    private static func findBundle(
        named bundleName: String,
        under root: URL,
        fileManager: FileManager
    ) -> URL? {
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return nil }

        for case let url as URL in enumerator {
            guard url.lastPathComponent == bundleName else { continue }
            enumerator.skipDescendants()
            return url
        }

        return nil
    }
}
