import Foundation

/// Locates the vendored `ds4-server` (and friends) on disk.
///
/// Search order:
/// 1. `$MERERUN_DS4_BIN_DIR/ds4-server` (explicit override, useful for tests).
/// 2. `vendor/ds4/ds4-server` alongside the running executable
///    (installed layout produced by `scripts/install.sh`).
/// 3. `vendor/ds4/ds4-server` under the SwiftPM build, walking upward from
///    the executable path until we find a directory containing `Package.swift`.
/// 4. `ds4-server` on `$PATH`.
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

        if let override = environment["MERERUN_DS4_BIN_DIR"], !override.isEmpty {
            let candidate = URL(fileURLWithPath: override).appendingPathComponent(kind.rawValue)
            searched.append(candidate)
            if fileManager.isExecutableFile(atPath: candidate.path) {
                return candidate
            }
        }

        let executableURL = URL(fileURLWithPath: CommandLine.arguments.first ?? "")
            .resolvingSymlinksInPath()
        let executableDir = executableURL.deletingLastPathComponent()

        // Installed layout: vendor/ds4/<kind> next to mere.run.
        let installedCandidate = executableDir
            .appendingPathComponent("vendor")
            .appendingPathComponent("ds4")
            .appendingPathComponent(kind.rawValue)
        searched.append(installedCandidate)
        if fileManager.isExecutableFile(atPath: installedCandidate.path) {
            return installedCandidate
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
                let devCandidate = dir
                    .appendingPathComponent("vendor")
                    .appendingPathComponent("ds4")
                    .appendingPathComponent(kind.rawValue)
                searched.append(devCandidate)
                if fileManager.isExecutableFile(atPath: devCandidate.path) {
                    return devCandidate
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
}
