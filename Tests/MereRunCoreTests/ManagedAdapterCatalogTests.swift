import Foundation
import Testing
@testable import MereRunCore

@Suite("Managed adapter catalog")
struct ManagedAdapterCatalogTests {
    @Test("Mere platform assistant release is pinned")
    func releaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.merePlatformAssistantID)
        )
        #expect(spec.version == "22")
        #expect(spec.baseModelID == Gemma4Resources.twelveB4BitModelId)
        #expect(spec.format == TextLoRATrainingManifest.format)
        #expect(spec.artifact.filename == "mere-platform-assistant-v22.safetensors")
        #expect(spec.artifact.byteCount == 128_131_022)
        #expect(spec.artifact.sha256 == "c4fec5979631b4031196c1e21c0b990437a26c5ebc52aec32f89338d64063290")
        #expect(spec.downloadURL.scheme == "https")
        #expect(spec.downloadURL.host == "releases.merekit.com")
    }

    @Test("Catalog ids resolve case-insensitively")
    func normalizedLookup() {
        #expect(ManagedAdapterCatalog.spec(for: " MERE-PLATFORM-ASSISTANT ")?.version == "22")
    }

    @Test("Resolver enforces the base model before checking disk")
    func rejectsWrongBaseModel() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        #expect(throws: ManagedAdapterResolutionError.self) {
            _ = try ManagedAdapterCatalog.resolveInstalledReference(
                ManagedAdapterCatalog.merePlatformAssistantID,
                baseModelID: Gemma4Resources.nanoModelId,
                adaptersRoot: root
            )
        }
    }

    @Test("Resolver reports the exact missing install path")
    func reportsMissingInstall() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            _ = try ManagedAdapterCatalog.resolveInstalledReference(
                ManagedAdapterCatalog.merePlatformAssistantID,
                baseModelID: Gemma4Resources.twelveB4BitModelId,
                adaptersRoot: root
            )
            Issue.record("missing adapter should fail")
        } catch let error as ManagedAdapterResolutionError {
            #expect(error.localizedDescription.contains("mere.run adapter pull mere-platform-assistant"))
            #expect(error.localizedDescription.contains("mere-platform-assistant-v22.safetensors"))
        }
    }
}
