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

    @Test("SCAIL-2 LightX2V release is an immutable remote-only pin")
    func scail2LightX2VReleaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.scail2LightX2VFourStepID)
        )
        #expect(spec.version == "27ae38da9101")
        #expect(spec.baseModelID == SCAIL2Resources.modelID)
        #expect(spec.format == SCAIL2DistilledAdapter.format)
        #expect(spec.license == "Apache-2.0")
        #expect(spec.upstreamRevision == "27ae38da91014b947dd39cc3fa78b97cd7b386dd")
        #expect(spec.artifact.filename == "wan2.1_i2v_lora_rank64_lightx2v_4step.safetensors")
        #expect(spec.artifact.byteCount == 739_472_104)
        #expect(spec.artifact.sha256 == "8833bd4fd7c8eabebf0bc8ee5cfaf47f4f310ce116928a02c1adf8941dd4b0f1")
        #expect(spec.downloadURL.scheme == "https")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

    @Test("MiniMax-H3 Turbo release is an immutable BF16 runtime pin")
    func miniMaxH3TurboReleaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3TurboFourStepID)
        )
        #expect(spec.version == "b604dd5fe25c")
        #expect(spec.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(spec.format == MiniMaxH3TurboAdapter.format)
        #expect(spec.upstreamRevision == "b604dd5fe25c4c747699f698a1e63f6c46d4a066")
        #expect(spec.artifact.filename == "minimax_h3_turbo_4step_ema_ckpt850.safetensors")
        #expect(spec.artifact.byteCount == 779_849_816)
        #expect(spec.artifact.sha256 == "5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
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
