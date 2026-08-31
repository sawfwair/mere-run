import Darwin
import Foundation

enum CLIBootstrapInstallError: LocalizedError {
    case missingBinary(String)
    case invalidPayload(String)
    case unwritableDestination(String)
    case validationFailed(String)
    case atomicActivationFailed(String)

    var errorDescription: String? {
        switch self {
        case .missingBinary(let path):
            return "Bundled CLI binary not found at \(path)."
        case .invalidPayload(let message):
            return "Bundled CLI payload is invalid: \(message)"
        case .unwritableDestination(let path):
            return "The CLI destination is not writable: \(path). Choose a writable bin directory or update its permissions."
        case .validationFailed(let message):
            return "The staged CLI did not pass validation: \(message)"
        case .atomicActivationFailed(let message):
            return "The CLI could not be activated atomically: \(message)"
        }
    }
}

struct CLIManagedInstallation: Equatable, Sendable {
    let destinationURL: URL
    let payloadURL: URL
    let receipt: CLIInstallationReceipt
    let installedVersion: String
}

enum CLIBootstrapInstaller {
    static func installManagedPayload(
        from payloadURL: URL,
        to destinationURL: URL,
        paths: CLIInstallationPaths,
        studioBuild: CLIStudioBuild,
        installationDate: Date = Date(),
        fileManager fm: FileManager = .default
    ) throws -> CLIManagedInstallation {
        let bundledBinaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
        guard fm.isExecutableFile(atPath: bundledBinaryURL.path) else {
            throw CLIBootstrapInstallError.missingBinary(bundledBinaryURL.path)
        }

        let manifest = try CLIPayloadManifestBuilder.manifest(at: payloadURL, fileManager: fm)
        guard manifest.assetNames.contains("mere.run") else {
            throw CLIBootstrapInstallError.invalidPayload("the top-level executable is absent")
        }
        guard destinationDirectoryIsWritable(destinationURL, fileManager: fm) else {
            throw CLIBootstrapInstallError.unwritableDestination(destinationURL.deletingLastPathComponent().path)
        }

        let oldReceipt = try? CLIInstallationReceipt.read(from: paths.receiptURL)
        try fm.createDirectory(at: paths.payloadRoot, withIntermediateDirectories: true)
        let targetURL = try stagedPayloadURL(
            for: payloadURL,
            manifest: manifest,
            paths: paths,
            studioBuild: studioBuild,
            fileManager: fm
        )
        let installedBinaryURL = targetURL.appendingPathComponent("mere.run", isDirectory: false)
        let installedVersion = try validatedVersion(executableURL: installedBinaryURL)

        let receipt = CLIInstallationReceipt(
            destinationPath: destinationURL.standardizedFileURL.path,
            payloadPath: targetURL.standardizedFileURL.path,
            studioVersion: studioBuild.version,
            studioBuild: studioBuild.build,
            payloadFingerprint: manifest.fingerprint,
            installedAssetNames: manifest.assetNames,
            installationTimestamp: installationDate
        )

        let backup = try activateSymlink(
            destinationURL: destinationURL,
            executableURL: installedBinaryURL,
            fileManager: fm
        )
        do {
            try receipt.write(to: paths.receiptURL, fileManager: fm)
        } catch {
            try? restoreDestination(backup, destinationURL: destinationURL, fileManager: fm)
            throw error
        }
        discardDestinationBackup(backup, fileManager: fm)

        if let oldReceipt {
            removePreviouslyOwnedPayloadIfSafe(
                oldReceipt,
                replacingWith: receipt,
                paths: paths,
                fileManager: fm
            )
        }

        return CLIManagedInstallation(
            destinationURL: destinationURL,
            payloadURL: targetURL,
            receipt: receipt,
            installedVersion: installedVersion
        )
    }

    static func bundledPayloadURL(
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> URL? {
        let candidates = [
            bundle.bundleURL.appendingPathComponent("Contents/Helpers", isDirectory: true),
            bundle.resourceURL?.appendingPathComponent("mere.run", isDirectory: true),
        ].compactMap { $0 }

        for payloadURL in candidates {
            let binaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
            if fm.isExecutableFile(atPath: binaryURL.path) {
                return payloadURL
            }
        }
        return nil
    }

    static func preferredAutomaticInstallURL(fileManager fm: FileManager = .default) -> URL {
        preferredAutomaticInstallURL(
            fileManager: fm,
            installDirectoryCandidates: [
                URL(fileURLWithPath: "/usr/local/bin", isDirectory: true),
                URL(fileURLWithPath: "/opt/homebrew/bin", isDirectory: true),
            ]
        )
    }

    static func preferredAutomaticInstallURL(
        fileManager fm: FileManager = .default,
        installDirectoryCandidates: [URL]
    ) -> URL {
        for candidate in installDirectoryCandidates where directoryIsWritable(candidate, fileManager: fm) {
            return candidate.appendingPathComponent("mere.run", isDirectory: false)
        }
        return fm.homeDirectoryForCurrentUser
            .appendingPathComponent(".local", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: false)
    }

    static func existingInstallationCandidate(
        paths: CLIInstallationPaths,
        fileManager fm: FileManager = .default
    ) -> URL? {
        paths.destinationCandidates.first { itemExistsIncludingSymbolicLink($0, fileManager: fm) }
    }

    static func itemExistsIncludingSymbolicLink(_ url: URL, fileManager fm: FileManager = .default) -> Bool {
        fm.fileExists(atPath: url.path) || (try? fm.destinationOfSymbolicLink(atPath: url.path)) != nil
    }

    static func resolvedSymbolicLink(at url: URL, fileManager fm: FileManager = .default) -> URL? {
        guard let destination = try? fm.destinationOfSymbolicLink(atPath: url.path) else { return nil }
        let destinationURL: URL
        if destination.hasPrefix("/") {
            destinationURL = URL(fileURLWithPath: destination)
        } else {
            destinationURL = url.deletingLastPathComponent().appendingPathComponent(destination)
        }
        return destinationURL.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func validatedVersion(executableURL: URL) throws -> String {
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            throw CLIBootstrapInstallError.validationFailed("\(executableURL.path) is not executable")
        }

        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.executableURL = executableURL
        process.arguments = ["--version"]
        process.currentDirectoryURL = executableURL.resolvingSymlinksInPath().deletingLastPathComponent()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        do {
            try process.run()
        } catch {
            throw CLIBootstrapInstallError.validationFailed(error.localizedDescription)
        }
        process.waitUntilExit()
        let stdout = String(decoding: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let stderr = String(decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard process.terminationStatus == 0, !stdout.isEmpty else {
            let message = stderr.isEmpty ? "`--version` exited with status \(process.terminationStatus)" : stderr
            throw CLIBootstrapInstallError.validationFailed(message)
        }
        return stdout
    }

    static func payloadMatches(
        receipt: CLIInstallationReceipt,
        paths: CLIInstallationPaths,
        fileManager fm: FileManager = .default
    ) -> Bool {
        let payloadURL = URL(fileURLWithPath: receipt.payloadPath, isDirectory: true)
        guard payloadURL != paths.payloadRoot,
              payloadURL.isContained(in: paths.payloadRoot),
              fm.isExecutableFile(atPath: payloadURL.appendingPathComponent("mere.run").path),
              let manifest = try? CLIPayloadManifestBuilder.manifest(at: payloadURL, fileManager: fm) else {
            return false
        }
        return manifest.fingerprint == receipt.payloadFingerprint
            && manifest.assetNames == receipt.installedAssetNames
    }

    static func destinationIsWritable(_ destinationURL: URL, fileManager fm: FileManager = .default) -> Bool {
        destinationDirectoryIsWritable(destinationURL, fileManager: fm)
    }

    private static func stagedPayloadURL(
        for payloadURL: URL,
        manifest: CLIPayloadManifest,
        paths: CLIInstallationPaths,
        studioBuild: CLIStudioBuild,
        fileManager fm: FileManager
    ) throws -> URL {
        let baseName = "\(safePathComponent(studioBuild.build))-\(manifest.fingerprint.prefix(12))"
        var targetURL = paths.payloadRoot.appendingPathComponent(baseName, isDirectory: true)
        if fm.fileExists(atPath: targetURL.path) {
            if let existingManifest = try? CLIPayloadManifestBuilder.manifest(at: targetURL, fileManager: fm),
               existingManifest == manifest,
               (try? validatedVersion(executableURL: targetURL.appendingPathComponent("mere.run"))) != nil {
                return targetURL
            }
            targetURL = paths.payloadRoot.appendingPathComponent(
                "\(baseName)-repair-\(UUID().uuidString)",
                isDirectory: true
            )
        }

        let stagingURL = paths.payloadRoot.appendingPathComponent(
            ".stage-\(UUID().uuidString)",
            isDirectory: true
        )
        do {
            try fm.createDirectory(at: stagingURL, withIntermediateDirectories: false)
            for assetURL in try fm.contentsOfDirectory(at: payloadURL, includingPropertiesForKeys: nil) {
                try fm.copyItem(
                    at: assetURL,
                    to: stagingURL.appendingPathComponent(assetURL.lastPathComponent)
                )
            }
            let stagedManifest = try CLIPayloadManifestBuilder.manifest(at: stagingURL, fileManager: fm)
            guard stagedManifest == manifest else {
                throw CLIBootstrapInstallError.invalidPayload("the staged payload fingerprint changed during copy")
            }
            _ = try validatedVersion(executableURL: stagingURL.appendingPathComponent("mere.run"))
            try fm.moveItem(at: stagingURL, to: targetURL)
            return targetURL
        } catch {
            try? fm.removeItem(at: stagingURL)
            throw error
        }
    }

    private enum DestinationBackup {
        case missing
        case symbolicLink(String)
        case copied(URL)
    }

    private static func activateSymlink(
        destinationURL: URL,
        executableURL: URL,
        fileManager fm: FileManager
    ) throws -> DestinationBackup {
        let destinationDirectoryURL = destinationURL.deletingLastPathComponent()
        try fm.createDirectory(at: destinationDirectoryURL, withIntermediateDirectories: true)
        guard destinationDirectoryIsWritable(destinationURL, fileManager: fm) else {
            throw CLIBootstrapInstallError.unwritableDestination(destinationDirectoryURL.path)
        }

        let backup: DestinationBackup
        if let symbolicDestination = try? fm.destinationOfSymbolicLink(atPath: destinationURL.path) {
            backup = .symbolicLink(symbolicDestination)
        } else if fm.fileExists(atPath: destinationURL.path) {
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: destinationURL.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw CLIBootstrapInstallError.atomicActivationFailed("\(destinationURL.path) is a directory")
            }
            let backupURL = destinationDirectoryURL.appendingPathComponent(
                ".mere.run-backup-\(UUID().uuidString)",
                isDirectory: false
            )
            try fm.copyItem(at: destinationURL, to: backupURL)
            backup = .copied(backupURL)
        } else {
            backup = .missing
        }

        let temporaryLinkURL = destinationDirectoryURL.appendingPathComponent(
            ".mere.run-link-\(UUID().uuidString)",
            isDirectory: false
        )
        do {
            try fm.createSymbolicLink(at: temporaryLinkURL, withDestinationURL: executableURL)
            try renameAtomically(from: temporaryLinkURL, to: destinationURL)
            return backup
        } catch {
            try? fm.removeItem(at: temporaryLinkURL)
            discardDestinationBackup(backup, fileManager: fm)
            throw error
        }
    }

    private static func restoreDestination(
        _ backup: DestinationBackup,
        destinationURL: URL,
        fileManager fm: FileManager
    ) throws {
        switch backup {
        case .missing:
            try? fm.removeItem(at: destinationURL)
        case .symbolicLink(let target):
            let restoreURL = destinationURL.deletingLastPathComponent().appendingPathComponent(
                ".mere.run-restore-\(UUID().uuidString)"
            )
            try fm.createSymbolicLink(atPath: restoreURL.path, withDestinationPath: target)
            try renameAtomically(from: restoreURL, to: destinationURL)
        case .copied(let backupURL):
            try renameAtomically(from: backupURL, to: destinationURL)
        }
    }

    private static func discardDestinationBackup(_ backup: DestinationBackup, fileManager fm: FileManager) {
        if case .copied(let backupURL) = backup {
            try? fm.removeItem(at: backupURL)
        }
    }

    private static func renameAtomically(from sourceURL: URL, to destinationURL: URL) throws {
        let result = sourceURL.path.withCString { sourcePath in
            destinationURL.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard result == 0 else {
            throw CLIBootstrapInstallError.atomicActivationFailed(String(cString: strerror(errno)))
        }
    }

    private static func removePreviouslyOwnedPayloadIfSafe(
        _ oldReceipt: CLIInstallationReceipt,
        replacingWith newReceipt: CLIInstallationReceipt,
        paths: CLIInstallationPaths,
        fileManager fm: FileManager
    ) {
        guard oldReceipt.payloadPath != newReceipt.payloadPath,
              oldReceipt.destinationPath == newReceipt.destinationPath,
              payloadMatches(receipt: oldReceipt, paths: paths, fileManager: fm) else {
            return
        }
        try? fm.removeItem(at: URL(fileURLWithPath: oldReceipt.payloadPath, isDirectory: true))
    }

    private static func destinationDirectoryIsWritable(_ destinationURL: URL, fileManager fm: FileManager) -> Bool {
        directoryIsWritable(destinationURL.deletingLastPathComponent(), fileManager: fm)
    }

    private static func directoryIsWritable(_ directoryURL: URL, fileManager fm: FileManager) -> Bool {
        var candidateURL = directoryURL
        while !fm.fileExists(atPath: candidateURL.path) {
            let parentURL = candidateURL.deletingLastPathComponent()
            guard parentURL.path != candidateURL.path else { return false }
            candidateURL = parentURL
        }
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: candidateURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              let attributes = try? fm.attributesOfItem(atPath: candidateURL.path),
              let permissions = attributes[.posixPermissions] as? NSNumber,
              permissions.intValue & 0o222 != 0 else {
            return false
        }
        return fm.isWritableFile(atPath: candidateURL.path)
    }

    private static func safePathComponent(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-._"))
        let scalars = value.unicodeScalars.map { allowed.contains($0) ? Character(String($0)) : "-" }
        let component = String(scalars)
        return component.isEmpty ? "dev" : component
    }
}
