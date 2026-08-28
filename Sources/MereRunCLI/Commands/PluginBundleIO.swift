import ArgumentParser
import Crypto
import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

enum PluginBundleIO {
    static let maximumArchiveSize: Int64 = 1_073_741_824

    static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty { digest.update(data: bytes) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func fetch(_ source: String, to destination: URL, limit: Int64, local: Bool = false) throws {
        if let url = URL(string: source), url.scheme == "https", url.host != nil,
           url.user == nil, url.password == nil {
            _ = try capture("/usr/bin/curl", [
                "--disable", "--globoff", "--fail", "--silent", "--show-error", "--location", "--proto", "=https",
                "--proto-redir", "=https", "--max-time", "180", "--max-filesize", String(limit),
                "--output", destination.path, "--", source,
            ], timeout: 190)
        } else if local, !source.contains("://") {
            try FileManager.default.copyItem(
                at: URL(fileURLWithPath: NSString(string: source).expandingTildeInPath), to: destination
            )
        } else {
            throw ValidationError("Plugin bundle downloads require HTTPS; local files must be supplied explicitly.")
        }
        let size = try destination.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
        guard size > 0, size <= limit else { throw ValidationError("Plugin bundle download has an invalid size.") }
    }

    /// Files avoid pipe deadlocks when signing tools emit more than a pipe buffer.
    static func capture(_ executable: String, _ arguments: [String], timeout: TimeInterval = 60) throws -> Data {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false,
                                                attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: temporary) }
        let output = temporary.appendingPathComponent("stdout")
        let errors = temporary.appendingPathComponent("stderr")
        FileManager.default.createFile(atPath: output.path, contents: nil)
        FileManager.default.createFile(atPath: errors.path, contents: nil)
        let outputHandle = try FileHandle(forWritingTo: output)
        let errorHandle = try FileHandle(forWritingTo: errors)
        defer { try? outputHandle.close(); try? errorHandle.close() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = outputHandle
        process.standardError = errorHandle
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline { Thread.sleep(forTimeInterval: 0.05) }
        if process.isRunning {
            process.terminate()
            let grace = Date().addingTimeInterval(1)
            while process.isRunning, Date() < grace { Thread.sleep(forTimeInterval: 0.05) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            throw ValidationError("Plugin bundle operation timed out: \(URL(fileURLWithPath: executable).lastPathComponent)")
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let detail = String(decoding: try Data(contentsOf: errors).prefix(2_000), as: UTF8.self)
            throw ValidationError("Plugin bundle operation failed (\(process.terminationStatus)): \(detail)")
        }
        return try Data(contentsOf: output)
    }

    static func validateTree(_ root: URL) throws {
        let manager = FileManager.default
        guard try manager.attributesOfItem(atPath: root.path)[.type] as? FileAttributeType == .typeDirectory,
              let entries = manager.enumerator(at: root, includingPropertiesForKeys: nil) else {
            throw ValidationError("Plugin bundle must be a directory.")
        }
        var total: Int64 = 0
        for case let url as URL in entries {
            let attributes = try manager.attributesOfItem(atPath: url.path)
            let type = attributes[.type] as? FileAttributeType
            if type == .typeSymbolicLink {
                let target = try manager.destinationOfSymbolicLink(atPath: url.path)
                let resolved = url.resolvingSymlinksInPath().standardizedFileURL
                let prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
                guard !target.hasPrefix("/"), resolved.path.hasPrefix(prefix), manager.fileExists(atPath: resolved.path) else {
                    throw ValidationError("Plugin bundle link escapes the bundle or has no target: \(url.lastPathComponent)")
                }
                continue
            }
            guard type == .typeDirectory || type == .typeRegular else {
                throw ValidationError("Plugin bundles cannot contain special files: \(url.lastPathComponent)")
            }
            total += (attributes[.size] as? NSNumber)?.int64Value ?? 0
            guard total <= maximumArchiveSize * 3 else { throw ValidationError("Expanded plugin bundle is too large.") }
        }
    }

    static func verifyMacOS(
        _ app: URL, manifest: PluginBundleManifest,
        run: (String, [String]) throws -> Data = { try capture($0, $1) }
    ) throws {
        #if os(macOS)
        let requirement = "anchor apple generic and certificate leaf[subject.OU] = \"\(PluginBundleEnvelope.developerTeam)\""
            + " and identifier \"\(manifest.bundleIdentifier)\""
        _ = try run("/usr/bin/codesign", ["--verify", "--deep", "--strict", "-R", "=" + requirement, app.path])
        // Built into macOS 14+. Unlike xcrun stapler, this does not require developer tools.
        _ = try run("/usr/bin/syspolicy_check", ["distribution", app.path])
        _ = try run("/usr/sbin/spctl", ["--assess", "--type", "execute", app.path])
        #else
        throw ValidationError("Signed plugin bundles require macOS on Apple Silicon.")
        #endif
    }

    static func stageDMG(_ archive: URL, manifest: PluginBundleManifest, destination: URL) throws {
        let mount = destination.deletingLastPathComponent().appendingPathComponent("mount-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: mount, withIntermediateDirectories: false)
        var needsDetach = true
        defer {
            // Also attempt detachment if hdiutil times out after mounting the image.
            if needsDetach { _ = try? capture("/usr/bin/hdiutil", ["detach", mount.path]) }
            try? FileManager.default.removeItem(at: mount)
        }
        _ = try capture("/usr/bin/hdiutil", ["attach", archive.path, "-readonly", "-nobrowse", "-noautoopen",
                                            "-mountpoint", mount.path, "-plist"])
        let app = mount.appendingPathComponent(manifest.appBundle)
        try validateTree(app)
        try verifyMacOS(app, manifest: manifest)
        try FileManager.default.copyItem(at: app, to: destination)
        _ = try capture("/usr/bin/hdiutil", ["detach", mount.path])
        needsDetach = false
    }
}
