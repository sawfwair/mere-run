import Foundation

enum CLIInstallationKind: Equatable, Sendable {
    case none
    case studioManaged
    case bundledSymlink
    case unowned
    case externallyManaged
    case custom
}

enum CLIInstallationPhase: Equatable, Sendable {
    case install
    case upToDate
    case update
    case repair
    case externallyManaged
    case synchronizing
}

struct CLIInstallationStatus: Equatable, Sendable {
    let phase: CLIInstallationPhase
    let kind: CLIInstallationKind
    let resolvedPath: String?
    let installedVersion: String?
    let bundledVersion: String?
    let detail: String
    let lastSynchronizationError: String?
    let allowsManualAction: Bool

    static let checking = Self(
        phase: .synchronizing,
        kind: .none,
        resolvedPath: nil,
        installedVersion: nil,
        bundledVersion: nil,
        detail: "Checking the Terminal CLI installation…",
        lastSynchronizationError: nil,
        allowsManualAction: false
    )

    var title: String {
        switch phase {
        case .install: "Install CLI"
        case .upToDate: "CLI is up to date"
        case .update: "Update CLI"
        case .repair: "Repair CLI"
        case .externallyManaged: "CLI managed externally"
        case .synchronizing: "Synchronizing CLI…"
        }
    }

    var actionTitle: String? {
        guard allowsManualAction else { return nil }
        return switch phase {
        case .install: "Install CLI"
        case .update: "Update CLI"
        case .repair, .upToDate: "Repair CLI"
        case .externallyManaged, .synchronizing: nil
        }
    }
}

struct CLIInstallationContext: Equatable, Sendable {
    let bundledPayloadURL: URL?
    let paths: CLIInstallationPaths
    let studioBuild: CLIStudioBuild
    let customCLIPath: String

    static func current(
        customCLIPath: String,
        fileManager fm: FileManager = .default,
        bundle: Bundle = .main
    ) -> Self {
        Self(
            bundledPayloadURL: CLIBootstrapInstaller.bundledPayloadURL(fileManager: fm, bundle: bundle),
            paths: .current(fileManager: fm),
            studioBuild: .current(bundle: bundle),
            customCLIPath: customCLIPath
        )
    }
}

private struct CLIInstallationInspection {
    let status: CLIInstallationStatus
    let destinationURL: URL?
    let automaticUpdateAllowed: Bool
}

enum CLIInstallationSynchronizer {
    static func inspect(
        context: CLIInstallationContext,
        fileManager fm: FileManager = .default
    ) -> CLIInstallationStatus {
        inspection(context: context, fileManager: fm).status
    }

    static func synchronizeAfterLaunch(
        context: CLIInstallationContext,
        fileManager fm: FileManager = .default
    ) -> CLIInstallationStatus {
        let current = inspection(context: context, fileManager: fm)
        guard current.automaticUpdateAllowed,
              let payloadURL = context.bundledPayloadURL,
              let destinationURL = current.destinationURL else {
            return current.status
        }

        do {
            _ = try CLIBootstrapInstaller.installManagedPayload(
                from: payloadURL,
                to: destinationURL,
                paths: context.paths,
                studioBuild: context.studioBuild,
                fileManager: fm
            )
            return inspection(context: context, fileManager: fm).status
        } catch {
            return failedStatus(from: current.status, error: error)
        }
    }

    static func performManualAction(
        context: CLIInstallationContext,
        fileManager fm: FileManager = .default
    ) -> CLIInstallationStatus {
        let current = inspection(context: context, fileManager: fm)
        guard current.status.allowsManualAction else { return current.status }
        guard let payloadURL = context.bundledPayloadURL else {
            return failedStatus(
                from: current.status,
                error: CLIBootstrapInstallError.invalidPayload("the app does not contain a bundled CLI")
            )
        }

        let destinationURL = current.destinationURL ?? preferredDestination(
            paths: context.paths,
            fileManager: fm
        )
        do {
            _ = try CLIBootstrapInstaller.installManagedPayload(
                from: payloadURL,
                to: destinationURL,
                paths: context.paths,
                studioBuild: context.studioBuild,
                fileManager: fm
            )
            return inspection(context: context, fileManager: fm).status
        } catch {
            return failedStatus(from: current.status, error: error)
        }
    }

    private static func inspection(
        context: CLIInstallationContext,
        fileManager fm: FileManager
    ) -> CLIInstallationInspection {
        let bundled = bundledInformation(context: context, fileManager: fm)
        let customPath = NSString(string: context.customCLIPath).expandingTildeInPath
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !customPath.isEmpty {
            let customURL = URL(fileURLWithPath: customPath, isDirectory: false)
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .externallyManaged,
                    kind: .custom,
                    resolvedPath: customURL.path,
                    installedVersion: try? CLIBootstrapInstaller.validatedVersion(executableURL: customURL),
                    bundledVersion: bundled.version,
                    detail: "Studio will not replace the explicitly configured CLI path.",
                    lastSynchronizationError: nil,
                    allowsManualAction: false
                ),
                destinationURL: customURL,
                automaticUpdateAllowed: false
            )
        }

        var receipt: CLIInstallationReceipt?
        var receiptError: String?
        if CLIBootstrapInstaller.itemExistsIncludingSymbolicLink(context.paths.receiptURL, fileManager: fm) {
            do {
                receipt = try CLIInstallationReceipt.read(from: context.paths.receiptURL)
            } catch {
                receiptError = error.localizedDescription
            }
        }

        let receiptDestinationURL = receipt.map {
            URL(fileURLWithPath: $0.destinationPath, isDirectory: false)
        }
        let destinationURL = receiptDestinationURL
            ?? CLIBootstrapInstaller.existingInstallationCandidate(paths: context.paths, fileManager: fm)

        guard let destinationURL else {
            let bundledMissing = context.bundledPayloadURL == nil
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .install,
                    kind: .none,
                    resolvedPath: nil,
                    installedVersion: nil,
                    bundledVersion: bundled.version,
                    detail: bundledMissing
                        ? "This build does not contain a bundled CLI payload."
                        : "No Terminal CLI is installed in a supported command location.",
                    lastSynchronizationError: bundled.error,
                    allowsManualAction: !bundledMissing
                ),
                destinationURL: nil,
                automaticUpdateAllowed: false
            )
        }

        if receipt != nil, !isSupportedDestination(destinationURL, paths: context.paths) {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .externallyManaged,
                    kind: .custom,
                    resolvedPath: destinationURL.path,
                    installedVersion: try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL),
                    bundledVersion: bundled.version,
                    detail: "The receipt names a destination outside Studio's supported command locations.",
                    lastSynchronizationError: "Studio won't replace this custom CLI destination.",
                    allowsManualAction: false
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        let resolvedLinkURL = CLIBootstrapInstaller.resolvedSymbolicLink(at: destinationURL, fileManager: fm)
        if let bundledBinaryURL = context.bundledPayloadURL?.appendingPathComponent("mere.run"),
           resolvedLinkURL == bundledBinaryURL.standardizedFileURL.resolvingSymlinksInPath() {
            let installedVersion = try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL)
            let isValid = installedVersion != nil
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: isValid ? .upToDate : .repair,
                    kind: .bundledSymlink,
                    resolvedPath: destinationURL.path,
                    installedVersion: installedVersion,
                    bundledVersion: bundled.version,
                    detail: isValid
                        ? "The command resolves directly into this Studio app bundle; no copy is required."
                        : "The command points into this Studio app bundle but does not pass `--version`.",
                    lastSynchronizationError: isValid ? nil : "The bundled CLI symlink could not be validated.",
                    allowsManualAction: false
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        if let receipt {
            return inspectOwnedInstallation(
                receipt: receipt,
                destinationURL: destinationURL,
                resolvedLinkURL: resolvedLinkURL,
                bundled: bundled,
                context: context,
                fileManager: fm
            )
        }

        if let receiptError {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .repair,
                    kind: .unowned,
                    resolvedPath: destinationURL.path,
                    installedVersion: try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL),
                    bundledVersion: bundled.version,
                    detail: "Studio cannot prove ownership because the installation receipt is invalid. Explicit repair is required.",
                    lastSynchronizationError: receiptError,
                    allowsManualAction: context.bundledPayloadURL != nil
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        if let resolvedLinkURL, !resolvedLinkURL.isContained(in: context.paths.payloadRoot) {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .externallyManaged,
                    kind: .externallyManaged,
                    resolvedPath: destinationURL.path,
                    installedVersion: try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL),
                    bundledVersion: bundled.version,
                    detail: "This command is a symlink managed outside Studio. Update it with its package manager or owner.",
                    lastSynchronizationError: nil,
                    allowsManualAction: false
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        let installedVersion = try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL)
        let matchesBundledBinary = context.bundledPayloadURL.map {
            filesMatch(
                destinationURL,
                $0.appendingPathComponent("mere.run", isDirectory: false)
            )
        } ?? false
        let detail = if matchesBundledBinary && installedVersion != nil {
            "The CLI matches this Studio build but has no ownership receipt. Update once to adopt the managed layout."
        } else {
            "This unmarked CLI will not be changed automatically. Update explicitly to migrate it into Studio management."
        }
        return CLIInstallationInspection(
            status: CLIInstallationStatus(
                phase: .update,
                kind: .unowned,
                resolvedPath: destinationURL.path,
                installedVersion: installedVersion,
                bundledVersion: bundled.version,
                detail: detail,
                lastSynchronizationError: nil,
                allowsManualAction: context.bundledPayloadURL != nil
            ),
            destinationURL: destinationURL,
            automaticUpdateAllowed: false
        )
    }

    private static func inspectOwnedInstallation(
        receipt: CLIInstallationReceipt,
        destinationURL: URL,
        resolvedLinkURL: URL?,
        bundled: (manifest: CLIPayloadManifest?, version: String?, error: String?),
        context: CLIInstallationContext,
        fileManager fm: FileManager
    ) -> CLIInstallationInspection {
        let receiptDestinationPath = URL(fileURLWithPath: receipt.destinationPath).standardizedFileURL.path
        let expectedExecutableURL = URL(fileURLWithPath: receipt.payloadPath, isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: false)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let ownershipMatches = destinationURL.standardizedFileURL.path == receiptDestinationPath
            && resolvedLinkURL == expectedExecutableURL
            && CLIBootstrapInstaller.payloadMatches(receipt: receipt, paths: context.paths, fileManager: fm)
        let installedVersion = ownershipMatches
            ? try? CLIBootstrapInstaller.validatedVersion(executableURL: destinationURL)
            : nil

        guard ownershipMatches, installedVersion != nil else {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .repair,
                    kind: .studioManaged,
                    resolvedPath: destinationURL.path,
                    installedVersion: installedVersion,
                    bundledVersion: bundled.version,
                    detail: "The installed destination or payload no longer matches Studio's ownership receipt.",
                    lastSynchronizationError: "Automatic replacement was blocked because the installed CLI changed after Studio installed it.",
                    allowsManualAction: context.bundledPayloadURL != nil
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        guard let bundledManifest = bundled.manifest else {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .repair,
                    kind: .studioManaged,
                    resolvedPath: destinationURL.path,
                    installedVersion: installedVersion,
                    bundledVersion: bundled.version,
                    detail: "The managed CLI is intact, but this app does not contain a valid replacement payload.",
                    lastSynchronizationError: bundled.error,
                    allowsManualAction: false
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        if receipt.payloadFingerprint == bundledManifest.fingerprint {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .upToDate,
                    kind: .studioManaged,
                    resolvedPath: destinationURL.path,
                    installedVersion: installedVersion,
                    bundledVersion: bundled.version,
                    detail: "Studio owns this versioned CLI payload and its runtime assets match the app bundle.",
                    lastSynchronizationError: nil,
                    allowsManualAction: true
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        guard CLIBootstrapInstaller.destinationIsWritable(destinationURL, fileManager: fm) else {
            return CLIInstallationInspection(
                status: CLIInstallationStatus(
                    phase: .repair,
                    kind: .studioManaged,
                    resolvedPath: destinationURL.path,
                    installedVersion: installedVersion,
                    bundledVersion: bundled.version,
                    detail: "A newer bundled CLI is available, but Studio cannot write the command destination.",
                    lastSynchronizationError: "Make \(destinationURL.deletingLastPathComponent().path) writable, then choose Repair CLI.",
                    allowsManualAction: true
                ),
                destinationURL: destinationURL,
                automaticUpdateAllowed: false
            )
        }

        return CLIInstallationInspection(
            status: CLIInstallationStatus(
                phase: .update,
                kind: .studioManaged,
                resolvedPath: destinationURL.path,
                installedVersion: installedVersion,
                bundledVersion: bundled.version,
                detail: "Studio will synchronize its managed CLI payload with this app build.",
                lastSynchronizationError: nil,
                allowsManualAction: true
            ),
            destinationURL: destinationURL,
            automaticUpdateAllowed: true
        )
    }

    private static func bundledInformation(
        context: CLIInstallationContext,
        fileManager fm: FileManager
    ) -> (manifest: CLIPayloadManifest?, version: String?, error: String?) {
        guard let payloadURL = context.bundledPayloadURL else {
            return (nil, nil, "Bundled CLI payload was not found.")
        }
        do {
            let manifest = try CLIPayloadManifestBuilder.manifest(at: payloadURL, fileManager: fm)
            let version = try CLIBootstrapInstaller.validatedVersion(
                executableURL: payloadURL.appendingPathComponent("mere.run", isDirectory: false)
            )
            return (manifest, version, nil)
        } catch {
            return (nil, nil, error.localizedDescription)
        }
    }

    private static func preferredDestination(paths: CLIInstallationPaths, fileManager fm: FileManager) -> URL {
        for candidate in paths.destinationCandidates
        where CLIBootstrapInstaller.destinationIsWritable(candidate, fileManager: fm) {
            return candidate
        }
        return paths.destinationCandidates.last
            ?? fm.homeDirectoryForCurrentUser
                .appendingPathComponent(".local/bin/mere.run", isDirectory: false)
    }

    private static func isSupportedDestination(_ destinationURL: URL, paths: CLIInstallationPaths) -> Bool {
        let path = destinationURL.standardizedFileURL.path
        return paths.destinationCandidates.contains { $0.standardizedFileURL.path == path }
    }

    private static func filesMatch(_ firstURL: URL, _ secondURL: URL) -> Bool {
        guard let first = try? Data(contentsOf: firstURL, options: .mappedIfSafe),
              let second = try? Data(contentsOf: secondURL, options: .mappedIfSafe) else {
            return false
        }
        return first == second
    }

    private static func failedStatus(from status: CLIInstallationStatus, error: Error) -> CLIInstallationStatus {
        CLIInstallationStatus(
            phase: status.phase == .install ? .install : .repair,
            kind: status.kind,
            resolvedPath: status.resolvedPath,
            installedVersion: status.installedVersion,
            bundledVersion: status.bundledVersion,
            detail: status.detail,
            lastSynchronizationError: error.localizedDescription,
            allowsManualAction: status.kind != .externallyManaged && status.kind != .custom
        )
    }
}
