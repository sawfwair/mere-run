import XCTest
@testable import MereRunCore

final class ManagedModelResolverTests: XCTestCase {
    func testHiddenMuseAssistantResolvesFromManagedInstallRoot() throws {
        let modelsRoot = try makeTemporaryDirectory()
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let assistantRoot = modelsRoot.appendingPathComponent(
            MuseGlimmerResources.assistantModelId,
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: assistantRoot, withIntermediateDirectories: true)
        try Data(#"{"model_type":"muse_glimmer_assistant"}"#.utf8).write(
            to: assistantRoot.appendingPathComponent("config.json")
        )
        try Data([0]).write(to: assistantRoot.appendingPathComponent("model.safetensors"))

        XCTAssertNil(ModelResolver.ModelID(rawValue: MuseGlimmerResources.assistantModelId))
        XCTAssertEqual(
            ManagedModelResolver.resolveInstalledModel(id: MuseGlimmerResources.assistantModelId),
            assistantRoot.standardizedFileURL
        )
    }

    func testRestrictedInstallRequiresCoreAcknowledgementBeforeDownload() async throws {
        let modelsRoot = try makeTemporaryDirectory()
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        do {
            _ = try await ManagedModelResolver.installManagedModel(id: FaceAnalysisResources.modelID)
            XCTFail("Restricted install should fail before contacting its download source.")
        } catch let error as ManagedModelResolver.ResolverError {
            guard case .usageTermsNotAcknowledged(let modelID) = error else {
                return XCTFail("Unexpected resolver error: \(error)")
            }
            XCTAssertEqual(modelID, FaceAnalysisResources.modelID)
        }

        XCTAssertFalse(FileManager.default.fileExists(
            atPath: modelsRoot.appendingPathComponent(FaceAnalysisResources.modelID).path
        ))
    }

    func testMaterializedGeometryInstallsExactBundledLicenseEvidence() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        for pin in GeometryModelPins.all where pin.licenseEvidence != nil {
            let snapshot = root.appendingPathComponent("snapshot-\(pin.modelID)", isDirectory: true)
            let install = root.appendingPathComponent("install-\(pin.modelID)", isDirectory: true)
            try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: pin.modelID))
            _ = try ManagedModelResolver.materializeManagedInstallRoot(
                for: spec,
                snapshotURL: snapshot,
                modelDir: install,
                fileManager: .default
            )
            let evidence = try XCTUnwrap(pin.licenseEvidence)
            _ = try evidence.installedPin.verify(in: install)
            try Data("tampered".utf8).write(
                to: install.appendingPathComponent(evidence.installedPin.filename)
            )
            XCTAssertTrue(
                spec.missingPaths(in: install).contains {
                    $0.lastPathComponent == evidence.installedPin.filename
                },
                pin.modelID
            )
        }
    }

    func testPinnedGeometryModelsWithManifestAndPlaceholderBytesAreNotRunnable() throws {
        let modelsRoot = try makeTemporaryDirectory()
        defer {
            MereRunModelPaths.setProcessModelsDirOverride(nil)
            try? FileManager.default.removeItem(at: modelsRoot)
        }
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        for pin in GeometryModelPins.all {
            let modelID = try XCTUnwrap(ModelResolver.ModelID(rawValue: pin.modelID))
            let install = modelsRoot.appendingPathComponent(pin.modelID, isDirectory: true)
            try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
            for artifact in pin.artifacts {
                try Data([0]).write(to: install.appendingPathComponent(artifact.filename))
            }
            try MereRunModelManifest.template(for: modelID, createdAt: Date(timeIntervalSince1970: 0))
                .write(to: install)

            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: pin.modelID))
            XCTAssertNil(ModelResolver().resolveIfPresent(modelID), pin.modelID)
            XCTAssertNil(spec.managedRuntimeURL(), pin.modelID)
            XCTAssertFalse(
                ManagedModelResolver.isManagedInstallComplete(spec: spec, at: install),
                pin.modelID
            )
        }
    }

    func testMaterializedInstallRootsKeepSharedHubAliasesManifestIsolated() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let snapshot = root.appendingPathComponent("hub/models/google/gemma-4-31B-it", isDirectory: true)
        try writeMinimalGemma4Snapshot(at: snapshot)
        try MereRunModelManifest.template(for: .gemma4, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: snapshot)

        let defaultSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4"))
        let maxSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4-max"))
        let defaultInstall = root.appendingPathComponent("models/text-chat-gemma4", isDirectory: true)
        let maxInstall = root.appendingPathComponent("models/text-chat-gemma4-max", isDirectory: true)

        let defaultManifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: defaultSpec,
            snapshotURL: snapshot,
            modelDir: defaultInstall,
            fileManager: .default
        )
        let maxManifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: maxSpec,
            snapshotURL: snapshot,
            modelDir: maxInstall,
            fileManager: .default
        )

        XCTAssertEqual(defaultManifest?.id, "text-chat-gemma4")
        XCTAssertEqual(maxManifest?.id, "text-chat-gemma4-max")
        XCTAssertEqual(try MereRunModelManifest.loadRequired(from: snapshot).id, "text-chat-gemma4")
        XCTAssertTrue(ManagedModelResolver.isManagedInstallComplete(spec: defaultSpec, at: defaultInstall))
        XCTAssertTrue(ManagedModelResolver.isManagedInstallComplete(spec: maxSpec, at: maxInstall))

        let defaultTokenizer = defaultInstall.appendingPathComponent("tokenizer.json")
        let maxTokenizer = maxInstall.appendingPathComponent("tokenizer.json")
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: defaultTokenizer.path))
                .standardizedFileURL.path,
            snapshot.appendingPathComponent("tokenizer.json").standardizedFileURL.path
        )
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: maxTokenizer.path))
                .standardizedFileURL.path,
            snapshot.appendingPathComponent("tokenizer.json").standardizedFileURL.path
        )
    }

    func testManagedInstallCompletenessRejectsMismatchedManifest() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        try writeMinimalGemma4Snapshot(at: root)
        try MereRunModelManifest.template(for: .gemma4, createdAt: Date(timeIntervalSince1970: 0))
            .write(to: root)

        let maxSpec = try XCTUnwrap(ManagedModelCatalog.spec(for: "text-chat-gemma4-max"))
        XCTAssertFalse(ManagedModelResolver.isManagedInstallComplete(spec: maxSpec, at: root))
    }

    func testMaterializedInstallRootMountsAdditionalHubSnapshots() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let baseSnapshot = root.appendingPathComponent("hub/base", isDirectory: true)
        try FileManager.default.createDirectory(
            at: baseSnapshot.appendingPathComponent("vae", isDirectory: true),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: baseSnapshot.appendingPathComponent("Qwen3-Embedding-0.6B", isDirectory: true),
            withIntermediateDirectories: true
        )

        let mountedSnapshot = root.appendingPathComponent("hub/xl", isDirectory: true)
        try FileManager.default.createDirectory(at: mountedSnapshot, withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(
            atPath: mountedSnapshot.appendingPathComponent("config.json").path,
            contents: Data("{}".utf8)
        ))

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: ModelResolver.ModelID.aceStepXLTurbo.rawValue))
        let install = root.appendingPathComponent("models/music-acestep-xl-turbo", isDirectory: true)
        let mounted = try XCTUnwrap(spec.mountedHubFallbacks.first)

        let manifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: spec,
            snapshotURL: baseSnapshot,
            mountedSnapshots: [(mounted, mountedSnapshot)],
            modelDir: install,
            fileManager: .default
        )

        XCTAssertEqual(manifest?.id, ModelResolver.ModelID.aceStepXLTurbo.rawValue)
        let mountedConfig = install.appendingPathComponent("acestep-v15-xl-turbo/config.json")
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: mountedConfig.path))
                .standardizedFileURL.path,
            mountedSnapshot.appendingPathComponent("config.json").standardizedFileURL.path
        )
        XCTAssertTrue(spec.isManagedRootComplete(install, fileManager: .default))
    }

    func testMaterializedInstallRootMountsSnapshotSubdirectoriesIntoComponentDirectories() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let baseSnapshot = root.appendingPathComponent("hub/base-9b", isDirectory: true)
        try makeFile(baseSnapshot.appendingPathComponent("model_index.json"))
        try makeFile(baseSnapshot.appendingPathComponent("transformer/config.json"))
        try makeFile(baseSnapshot.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors.index.json"))

        let sharedSnapshot = root.appendingPathComponent("hub/shared-9b", isDirectory: true)
        try makeFile(sharedSnapshot.appendingPathComponent("tokenizer/tokenizer_config.json"))
        try makeFile(sharedSnapshot.appendingPathComponent("text_encoder/config.json"))
        try makeFile(sharedSnapshot.appendingPathComponent("text_encoder/model.safetensors.index.json"))
        try makeFile(sharedSnapshot.appendingPathComponent("vae/config.json"))
        try makeFile(sharedSnapshot.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        try makeFile(sharedSnapshot.appendingPathComponent("scheduler/scheduler_config.json"))

        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: "image-klein-base-9b"))
        let install = root.appendingPathComponent("models/image-klein-base-9b", isDirectory: true)
        let mountedSnapshots = spec.mountedHubFallbacks.map { ($0, sharedSnapshot) }

        let manifest = try ManagedModelResolver.materializeManagedInstallRoot(
            for: spec,
            snapshotURL: baseSnapshot,
            mountedSnapshots: mountedSnapshots,
            modelDir: install,
            usageTermsAcknowledged: true,
            fileManager: .default
        )

        XCTAssertEqual(manifest?.id, "image-klein-base-9b")
        XCTAssertEqual(manifest?.usageTermsAcknowledged, true)
        let mountedTextConfig = install.appendingPathComponent("text_encoder/config.json")
        XCTAssertEqual(
            URL(fileURLWithPath: try FileManager.default.destinationOfSymbolicLink(atPath: mountedTextConfig.path))
                .standardizedFileURL.path,
            sharedSnapshot.appendingPathComponent("text_encoder/config.json").standardizedFileURL.path
        )
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: install.appendingPathComponent("text_encoder/text_encoder/config.json").path
        ))
        XCTAssertTrue(spec.isManagedRootComplete(install, fileManager: .default))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-managed-model-resolver-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func makeFile(_ url: URL, contents: String = "{}") throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: Data(contents.utf8)))
    }

    private func writeMinimalGemma4Snapshot(at root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for file in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            try makeFile(root.appendingPathComponent(file))
        }
    }
}
