@testable import StudioKit
import XCTest

final class CLIBootstrapInstallerTests: XCTestCase {
    func testPreferredAutomaticInstallURLUsesFirstWritableCandidate() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        let secondCandidateURL = layout.rootURL.appendingPathComponent("second-bin", isDirectory: true)
        try FileManager.default.createDirectory(at: secondCandidateURL, withIntermediateDirectories: true)

        let installURL = CLIBootstrapInstaller.preferredAutomaticInstallURL(
            installDirectoryCandidates: [layout.binURL, secondCandidateURL]
        )

        XCTAssertEqual(installURL, layout.destinationURL)
    }

    func testReceiptRoundTripsAndRejectsUnknownSchema() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let receipt = CLIInstallationReceipt(
            destinationPath: layout.destinationURL.path,
            payloadPath: layout.paths.payloadRoot.appendingPathComponent("42-deadbeef").path,
            studioVersion: "1.2.3",
            studioBuild: "42",
            payloadFingerprint: "deadbeef",
            installedAssetNames: ["Resources", "mere.run"],
            installationTimestamp: timestamp
        )

        try receipt.write(to: layout.paths.receiptURL)
        XCTAssertEqual(try CLIInstallationReceipt.read(from: layout.paths.receiptURL), receipt)

        let unsupported = CLIInstallationReceipt(
            schemaVersion: 99,
            destinationPath: receipt.destinationPath,
            payloadPath: receipt.payloadPath,
            studioVersion: receipt.studioVersion,
            studioBuild: receipt.studioBuild,
            payloadFingerprint: receipt.payloadFingerprint,
            installedAssetNames: receipt.installedAssetNames,
            installationTimestamp: timestamp
        )
        try unsupported.write(to: layout.paths.receiptURL)
        XCTAssertThrowsError(try CLIInstallationReceipt.read(from: layout.paths.receiptURL)) { error in
            XCTAssertEqual(error as? CLIInstallationReceiptError, .unsupportedSchemaVersion(99))
        }
    }

    func testManagedInstallStagesCompletePayloadAndWritesReceipt() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "1.0.0")
        try "leave me".write(
            to: layout.binURL.appendingPathComponent("unrelated-tool"),
            atomically: true,
            encoding: .utf8
        )

        let installation = try CLIBootstrapInstaller.installManagedPayload(
            from: layout.payloadURL,
            to: layout.destinationURL,
            paths: layout.paths,
            studioBuild: CLIStudioBuild(version: "1.0.0", build: "100")
        )

        XCTAssertEqual(installation.installedVersion, "mere.run 1.0.0")
        XCTAssertTrue(installation.payloadURL.isContained(in: layout.paths.payloadRoot))
        XCTAssertEqual(
            CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL),
            installation.payloadURL.appendingPathComponent("mere.run").resolvingSymlinksInPath()
        )
        XCTAssertEqual(try CLIBootstrapInstaller.validatedVersion(executableURL: layout.destinationURL), "mere.run 1.0.0")
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.payloadURL.appendingPathComponent("llama.framework").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.payloadURL.appendingPathComponent("mlx-swift_Cmlx.bundle").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.payloadURL.appendingPathComponent("libonnxruntime.dylib").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.payloadURL.appendingPathComponent("Resources/default.metallib").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: installation.payloadURL.appendingPathComponent("vendor/ds4/helper").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: layout.binURL.appendingPathComponent("unrelated-tool").path))

        let receipt = try CLIInstallationReceipt.read(from: layout.paths.receiptURL)
        XCTAssertEqual(receipt.destinationPath, layout.destinationURL.path)
        XCTAssertEqual(receipt.payloadPath, installation.payloadURL.path)
        XCTAssertEqual(receipt.studioVersion, "1.0.0")
        XCTAssertEqual(receipt.studioBuild, "100")
        XCTAssertEqual(receipt.installedAssetNames, [
            "Resources",
            "libonnxruntime.dylib",
            "llama.framework",
            "mere.run",
            "mlx-swift_Cmlx.bundle",
            "vendor",
        ])
    }

    func testUnownedCopiedCLIRequiresExplicitMigration() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "2.0.0")
        try makeLegacyCopiedInstallation(from: layout.payloadURL, in: layout.binURL)
        let context = makeContext(layout: layout, version: "2.0.0", build: "200")

        let before = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: context)
        XCTAssertEqual(before.phase, .update)
        XCTAssertEqual(before.kind, .unowned)
        XCTAssertFalse(FileManager.default.fileExists(atPath: layout.paths.receiptURL.path))
        XCTAssertNil(CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL))

        let after = CLIInstallationSynchronizer.performManualAction(context: context)
        XCTAssertEqual(after.phase, .upToDate)
        XCTAssertEqual(after.kind, .studioManaged)
        XCTAssertTrue(FileManager.default.fileExists(atPath: layout.paths.receiptURL.path))
        XCTAssertNotNil(CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL))
        XCTAssertTrue(FileManager.default.fileExists(atPath: layout.binURL.appendingPathComponent("Resources/default.metallib").path))
    }

    func testMatchingManagedBuildIsANoOp() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "3.0.0")
        let context = makeContext(layout: layout, version: "3.0.0", build: "300")
        XCTAssertEqual(CLIInstallationSynchronizer.performManualAction(context: context).phase, .upToDate)
        let receiptData = try Data(contentsOf: layout.paths.receiptURL)
        let payloads = try FileManager.default.contentsOfDirectory(atPath: layout.paths.payloadRoot.path)

        let status = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: context)

        XCTAssertEqual(status.phase, .upToDate)
        XCTAssertEqual(try Data(contentsOf: layout.paths.receiptURL), receiptData)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: layout.paths.payloadRoot.path), payloads)
    }

    func testStartupSynchronizerUpdatesOwnedPayloadAndPreservesUnrelatedFiles() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "4.0.0")
        let firstContext = makeContext(layout: layout, version: "4.0.0", build: "400")
        XCTAssertEqual(CLIInstallationSynchronizer.performManualAction(context: firstContext).phase, .upToDate)
        let oldReceipt = try CLIInstallationReceipt.read(from: layout.paths.receiptURL)
        let oldPayloadURL = URL(fileURLWithPath: oldReceipt.payloadPath, isDirectory: true)
        let unmanagedPayloadURL = layout.paths.payloadRoot.appendingPathComponent("do-not-delete", isDirectory: true)
        try FileManager.default.createDirectory(at: unmanagedPayloadURL, withIntermediateDirectories: true)
        let unrelatedURL = layout.binURL.appendingPathComponent("unrelated-tool")
        try "unrelated".write(to: unrelatedURL, atomically: true, encoding: .utf8)

        try FileManager.default.removeItem(at: layout.payloadURL)
        try makePayload(at: layout.payloadURL, version: "4.1.0")
        let updateContext = makeContext(layout: layout, version: "4.1.0", build: "401")
        let status = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: updateContext)

        XCTAssertEqual(status.phase, .upToDate)
        XCTAssertEqual(status.kind, .studioManaged)
        XCTAssertEqual(try CLIBootstrapInstaller.validatedVersion(executableURL: layout.destinationURL), "mere.run 4.1.0")
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldPayloadURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: unmanagedPayloadURL.path))
        XCTAssertEqual(try String(contentsOf: unrelatedURL, encoding: .utf8), "unrelated")
    }

    func testDamagedOwnedPayloadRequiresExplicitRepair() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "5.0.0")
        let context = makeContext(layout: layout, version: "5.0.0", build: "500")
        XCTAssertEqual(CLIInstallationSynchronizer.performManualAction(context: context).phase, .upToDate)
        let receipt = try CLIInstallationReceipt.read(from: layout.paths.receiptURL)
        let damagedAssetURL = URL(fileURLWithPath: receipt.payloadPath, isDirectory: true)
            .appendingPathComponent("Resources/default.metallib")
        try FileManager.default.removeItem(at: damagedAssetURL)

        let automatic = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: context)
        XCTAssertEqual(automatic.phase, .repair)
        XCTAssertFalse(FileManager.default.fileExists(atPath: damagedAssetURL.path))

        let repaired = CLIInstallationSynchronizer.performManualAction(context: context)
        XCTAssertEqual(repaired.phase, .upToDate)
        let repairedReceipt = try CLIInstallationReceipt.read(from: layout.paths.receiptURL)
        XCTAssertNotEqual(repairedReceipt.payloadPath, receipt.payloadPath)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: repairedReceipt.payloadPath)
                    .appendingPathComponent("Resources/default.metallib")
                    .path
            )
        )
    }

    func testFailedStagedVersionValidationPreservesWorkingInstallation() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "6.0.0")
        _ = try CLIBootstrapInstaller.installManagedPayload(
            from: layout.payloadURL,
            to: layout.destinationURL,
            paths: layout.paths,
            studioBuild: CLIStudioBuild(version: "6.0.0", build: "600")
        )
        let originalTarget = CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL)

        try FileManager.default.removeItem(at: layout.payloadURL)
        try makePayload(at: layout.payloadURL, version: "6.1.0", exitsSuccessfully: false)
        XCTAssertThrowsError(
            try CLIBootstrapInstaller.installManagedPayload(
                from: layout.payloadURL,
                to: layout.destinationURL,
                paths: layout.paths,
                studioBuild: CLIStudioBuild(version: "6.1.0", build: "601")
            )
        )
        XCTAssertEqual(CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL), originalTarget)
        XCTAssertEqual(try CLIBootstrapInstaller.validatedVersion(executableURL: layout.destinationURL), "mere.run 6.0.0")
    }

    func testUnwritableDestinationReportsRecoveryWithoutReplacement() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: layout.binURL.path) }
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "7.0.0")
        try makeLegacyCopiedInstallation(from: layout.payloadURL, in: layout.binURL)
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: layout.binURL.path)

        let status = CLIInstallationSynchronizer.performManualAction(
            context: makeContext(layout: layout, version: "7.0.0", build: "700")
        )

        XCTAssertEqual(status.phase, .repair)
        XCTAssertTrue(status.lastSynchronizationError?.contains("not writable") == true)
        XCTAssertNil(CLIBootstrapInstaller.resolvedSymbolicLink(at: layout.destinationURL))
    }

    func testBrokenExternalSymlinkAndCustomPathRemainExternal() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "8.0.0")
        let externalTargetURL = layout.rootURL.appendingPathComponent("external/mere.run")
        try FileManager.default.createSymbolicLink(at: layout.destinationURL, withDestinationURL: externalTargetURL)
        let regularContext = makeContext(layout: layout, version: "8.0.0", build: "800")

        let external = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: regularContext)
        XCTAssertEqual(external.phase, .externallyManaged)
        XCTAssertEqual(external.kind, .externallyManaged)
        XCTAssertFalse(external.allowsManualAction)

        let customContext = CLIInstallationContext(
            bundledPayloadURL: layout.payloadURL,
            paths: layout.paths,
            studioBuild: CLIStudioBuild(version: "8.0.0", build: "800"),
            customCLIPath: externalTargetURL.path
        )
        let custom = CLIInstallationSynchronizer.synchronizeAfterLaunch(context: customContext)
        XCTAssertEqual(custom.phase, .externallyManaged)
        XCTAssertEqual(custom.kind, .custom)
        XCTAssertFalse(custom.allowsManualAction)
    }

    func testReceiptCannotAuthorizeAPathOutsideSupportedDestinations() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "8.1.0")
        let unsupportedDestinationURL = layout.rootURL.appendingPathComponent("custom/mere.run")
        try FileManager.default.createDirectory(
            at: unsupportedDestinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.createSymbolicLink(
            at: unsupportedDestinationURL,
            withDestinationURL: layout.payloadURL.appendingPathComponent("mere.run")
        )
        let manifest = try CLIPayloadManifestBuilder.manifest(at: layout.payloadURL)
        let receipt = CLIInstallationReceipt(
            destinationPath: unsupportedDestinationURL.path,
            payloadPath: layout.payloadURL.path,
            studioVersion: "8.1.0",
            studioBuild: "810",
            payloadFingerprint: manifest.fingerprint,
            installedAssetNames: manifest.assetNames,
            installationTimestamp: Date(timeIntervalSince1970: 1_700_000_000)
        )
        try receipt.write(to: layout.paths.receiptURL)

        let status = CLIInstallationSynchronizer.synchronizeAfterLaunch(
            context: makeContext(layout: layout, version: "8.1.0", build: "810")
        )

        XCTAssertEqual(status.phase, .externallyManaged)
        XCTAssertEqual(status.kind, .custom)
        XCTAssertFalse(status.allowsManualAction)
        XCTAssertEqual(
            CLIBootstrapInstaller.resolvedSymbolicLink(at: unsupportedDestinationURL),
            layout.payloadURL.appendingPathComponent("mere.run").resolvingSymlinksInPath()
        )
    }

    func testSymlinkIntoBundledPayloadNeedsNoReceiptOrCopy() throws {
        let layout = try makeLayout()
        defer { try? FileManager.default.removeItem(at: layout.rootURL) }
        try makePayload(at: layout.payloadURL, version: "9.0.0")
        try FileManager.default.createSymbolicLink(
            at: layout.destinationURL,
            withDestinationURL: layout.payloadURL.appendingPathComponent("mere.run")
        )

        let status = CLIInstallationSynchronizer.synchronizeAfterLaunch(
            context: makeContext(layout: layout, version: "9.0.0", build: "900")
        )

        XCTAssertEqual(status.phase, .upToDate)
        XCTAssertEqual(status.kind, .bundledSymlink)
        XCTAssertFalse(status.allowsManualAction)
        XCTAssertFalse(FileManager.default.fileExists(atPath: layout.paths.receiptURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: layout.paths.payloadRoot.path))
    }

    private struct Layout {
        let rootURL: URL
        let payloadURL: URL
        let binURL: URL
        let destinationURL: URL
        let paths: CLIInstallationPaths
    }

    private func makeLayout() throws -> Layout {
        let rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("CLIBootstrapInstallerTests-\(UUID().uuidString)", isDirectory: true)
        let payloadURL = rootURL.appendingPathComponent("bundled-payload", isDirectory: true)
        let binURL = rootURL.appendingPathComponent("bin", isDirectory: true)
        let destinationURL = binURL.appendingPathComponent("mere.run", isDirectory: false)
        try FileManager.default.createDirectory(at: binURL, withIntermediateDirectories: true)
        let paths = CLIInstallationPaths(
            applicationSupportRoot: rootURL.appendingPathComponent("Application Support/mere.run", isDirectory: true),
            destinationCandidates: [destinationURL]
        )
        return Layout(
            rootURL: rootURL,
            payloadURL: payloadURL,
            binURL: binURL,
            destinationURL: destinationURL,
            paths: paths
        )
    }

    private func makeContext(layout: Layout, version: String, build: String) -> CLIInstallationContext {
        CLIInstallationContext(
            bundledPayloadURL: layout.payloadURL,
            paths: layout.paths,
            studioBuild: CLIStudioBuild(version: version, build: build),
            customCLIPath: ""
        )
    }

    private func makePayload(
        at payloadURL: URL,
        version: String,
        exitsSuccessfully: Bool = true
    ) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: payloadURL, withIntermediateDirectories: true)
        try fm.createDirectory(at: payloadURL.appendingPathComponent("llama.framework"), withIntermediateDirectories: true)
        try fm.createDirectory(at: payloadURL.appendingPathComponent("mlx-swift_Cmlx.bundle"), withIntermediateDirectories: true)
        try fm.createDirectory(at: payloadURL.appendingPathComponent("Resources"), withIntermediateDirectories: true)
        try fm.createDirectory(at: payloadURL.appendingPathComponent("vendor/ds4"), withIntermediateDirectories: true)
        try "framework".write(
            to: payloadURL.appendingPathComponent("llama.framework/runtime"),
            atomically: true,
            encoding: .utf8
        )
        try "bundle".write(
            to: payloadURL.appendingPathComponent("mlx-swift_Cmlx.bundle/resource"),
            atomically: true,
            encoding: .utf8
        )
        try "dylib".write(
            to: payloadURL.appendingPathComponent("libonnxruntime.dylib"),
            atomically: true,
            encoding: .utf8
        )
        try "metal".write(
            to: payloadURL.appendingPathComponent("Resources/default.metallib"),
            atomically: true,
            encoding: .utf8
        )
        try "helper".write(
            to: payloadURL.appendingPathComponent("vendor/ds4/helper"),
            atomically: true,
            encoding: .utf8
        )

        let script = exitsSuccessfully
            ? "#!/bin/sh\n"
                + "[ -f Resources/default.metallib ] || exit 70\n"
                + "[ -d llama.framework ] || exit 70\n"
                + "[ -d mlx-swift_Cmlx.bundle ] || exit 70\n"
                + "[ -f libonnxruntime.dylib ] || exit 70\n"
                + "[ -f vendor/ds4/helper ] || exit 70\n"
                + "printf 'mere.run \(version)\\n'\n"
            : "#!/bin/sh\necho 'fixture validation failed' >&2\nexit 71\n"
        let binaryURL = payloadURL.appendingPathComponent("mere.run", isDirectory: false)
        try script.write(to: binaryURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: binaryURL.path)
    }

    private func makeLegacyCopiedInstallation(from payloadURL: URL, in binURL: URL) throws {
        let fm = FileManager.default
        for assetURL in try fm.contentsOfDirectory(at: payloadURL, includingPropertiesForKeys: nil) {
            try fm.copyItem(at: assetURL, to: binURL.appendingPathComponent(assetURL.lastPathComponent))
        }
    }
}
