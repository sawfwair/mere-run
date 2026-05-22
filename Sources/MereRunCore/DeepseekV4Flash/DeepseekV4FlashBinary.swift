import Foundation

/// Locates the vendored `ds4-server` (and friends) on disk.
///
/// Search order:
/// 1. `$MERERUN_DS4_BIN_DIR/ds4-server` (explicit override, useful for tests).
/// 2. `$MERERUN_DS4_BIN_DIR/<platform>/ds4-server` when an override points at
///    a platform-rooted DS4 directory.
/// 3. `vendor/ds4/ds4-server` alongside the running executable on macOS, or
///    `vendor/ds4/<platform>/ds4-server` first on Linux.
///    (installed layout produced by `scripts/install.sh`).
/// 4. `vendor/ds4/ds4-server` under the SwiftPM build, walking upward from
///    the executable path until we find a directory containing `Package.swift`.
/// 5. `ds4-server` on `$PATH`.
public enum DeepseekV4FlashBinary {
    public enum Kind: String, Sendable {
        case server = "ds4-server"
        case cli = "ds4"
        case bench = "ds4-bench"
    }

    public static func locate(
        _ kind: Kind,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) throws -> URL {
        var searched: [URL] = []
        let platformDirectory = Self.platformDirectoryName(environment: environment)
        let preferPlatformDirectory = Self.prefersPlatformDirectory()

        if let override = environment["MERERUN_DS4_BIN_DIR"], !override.isEmpty {
            let overrideRoot = URL(fileURLWithPath: override)
            for candidate in ds4CandidateURLs(
                root: overrideRoot,
                kind: kind,
                platformDirectory: platformDirectory,
                preferPlatformDirectory: false
            ) {
                searched.append(candidate)
                if fileManager.isExecutableFile(atPath: candidate.path) {
                    return candidate
                }
            }
        }

        let executableURL = URL(fileURLWithPath: CommandLine.arguments.first ?? "")
            .resolvingSymlinksInPath()
        let executableDir = executableURL.deletingLastPathComponent()

        // Installed layout: vendor/ds4/<kind> next to mere.run.
        let installedRoot = executableDir
            .appendingPathComponent("vendor")
            .appendingPathComponent("ds4")
        for candidate in ds4CandidateURLs(
            root: installedRoot,
            kind: kind,
            platformDirectory: platformDirectory,
            preferPlatformDirectory: preferPlatformDirectory
        ) {
            searched.append(candidate)
            if fileManager.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }

        // Also try a flat layout (DMG / install.sh style): <kind> next to mere.run.
        let flatCandidate = executableDir.appendingPathComponent(kind.rawValue)
        searched.append(flatCandidate)
        if fileManager.isExecutableFile(atPath: flatCandidate.path) {
            return flatCandidate
        }

        // Dev mode: walk upward from the executable looking for Package.swift,
        // then check vendor/ds4/<kind> relative to that root.
        var dir = executableDir
        for _ in 0..<8 {
            let packageURL = dir.appendingPathComponent("Package.swift")
            if fileManager.fileExists(atPath: packageURL.path) {
                let devRoot = dir
                    .appendingPathComponent("vendor")
                    .appendingPathComponent("ds4")
                for candidate in ds4CandidateURLs(
                    root: devRoot,
                    kind: kind,
                    platformDirectory: platformDirectory,
                    preferPlatformDirectory: preferPlatformDirectory
                ) {
                    searched.append(candidate)
                    if fileManager.isExecutableFile(atPath: candidate.path) {
                        return candidate
                    }
                }
                break
            }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }

        // Last resort: PATH.
        if let path = environment["PATH"] {
            for component in path.split(separator: ":") {
                let candidate = URL(fileURLWithPath: String(component))
                    .appendingPathComponent(kind.rawValue)
                searched.append(candidate)
                if fileManager.isExecutableFile(atPath: candidate.path) {
                    return candidate
                }
            }
        }

        throw DeepseekV4FlashError.binaryNotFound(searched: searched)
    }

    private static func ds4CandidateURLs(
        root: URL,
        kind: Kind,
        platformDirectory: String?,
        preferPlatformDirectory: Bool
    ) -> [URL] {
        let rootCandidate = root.appendingPathComponent(kind.rawValue)
        guard let platformDirectory, !platformDirectory.isEmpty else {
            return [rootCandidate]
        }

        let platformCandidate = root
            .appendingPathComponent(platformDirectory, isDirectory: true)
            .appendingPathComponent(kind.rawValue)

        if preferPlatformDirectory {
            return [platformCandidate, rootCandidate]
        } else {
            return [rootCandidate, platformCandidate]
        }
    }

    private static func platformDirectoryName(environment: [String: String]) -> String? {
        if let override = environment["MERERUN_DS4_PLATFORM_DIR"], !override.isEmpty {
            return override
        }

        let osName: String
        #if os(macOS)
        osName = "macos"
        #elseif os(Linux)
        osName = "linux"
        #else
        return nil
        #endif

        let archName: String
        #if arch(arm64)
        archName = "arm64"
        #elseif arch(x86_64)
        archName = "x86_64"
        #else
        return nil
        #endif

        return "\(osName)-\(archName)"
    }

    private static func prefersPlatformDirectory() -> Bool {
        #if os(Linux)
        return true
        #else
        return false
        #endif
    }
}
