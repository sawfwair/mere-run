import Foundation
import Testing
@testable import MereRunCore

@Suite("Managed adapter catalog")
struct ManagedAdapterCatalogTests {
    @Test("FLUX.2-dev Turbo is an immutable gated eight-step pin")
    func flux2DevTurboIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.flux2DevTurboEightStepID)
        )
        #expect(spec.version == "9ee51cd87578")
        #expect(spec.baseModelID == ModelResolver.ModelID.flux2Dev.rawValue)
        #expect(spec.format == ManagedAdapterCatalog.flux2DevLoRAFormat)
        #expect(spec.upstreamRevision == ManagedAdapterCatalog.flux2DevTurboEightStepRevision)
        #expect(spec.usageRestriction != nil)
        #expect(spec.artifact.filename == "flux.2-turbo-lora.safetensors")
        #expect(spec.artifact.byteCount == 2_760_818_216)
        #expect(spec.artifact.sha256 == "f76cf9c2cc546ddca878799136434a1098477af3f4b0adff2cfd79f2ebe4aa01")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

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
        #expect(spec.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(!spec.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAMLX.rawValue))
        #expect(spec.format == MiniMaxH3TurboAdapter.format)
        #expect(spec.upstreamRevision == "b604dd5fe25c4c747699f698a1e63f6c46d4a066")
        #expect(spec.artifact.filename == "minimax_h3_turbo_4step_ema_ckpt850.safetensors")
        #expect(spec.artifact.byteCount == 779_849_816)
        #expect(spec.artifact.sha256 == "5a6eeba171cf183020a4ad48774bb2968f29f8168afd6ec17a04987f3528b4ea")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

    @Test("MiniMax-H3 LightX2V release is an immutable BF16 runtime pin")
    func miniMaxH3LightX2VReleaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3LightX2VFourStepID)
        )
        #expect(spec.version == "b65e359c0d12")
        #expect(spec.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(spec.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(spec.format == MiniMaxH3TurboAdapter.lightX2VFormat)
        #expect(spec.upstreamRevision == "b65e359c0d128b3c5e08e0f5bf2791b794378588")
        #expect(spec.artifact.filename == "minimax_h3_fl2v_turbo_4step_v0.1.safetensors")
        #expect(spec.artifact.byteCount == 1_383_677_888)
        #expect(spec.artifact.sha256 == "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

    @Test("MiniMax-H3 LightX2V v1.0 releases are immutable BF16 runtime pins")
    func miniMaxH3LightX2VV1ReleasesArePinned() throws {
        let eightStep = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3LightX2VEightStepV1ID)
        )
        #expect(eightStep.version == "e6346777701a")
        #expect(eightStep.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(eightStep.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(eightStep.format == MiniMaxH3TurboAdapter.lightX2VFormat)
        #expect(eightStep.upstreamRevision == ManagedAdapterCatalog.miniMaxH3LightX2VV1Revision)
        #expect(eightStep.artifact.byteCount == 1_383_677_768)
        #expect(eightStep.artifact.sha256 == "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f")

        let fourStep = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3LightX2VFourStepV1_768pID)
        )
        #expect(fourStep.version == "e6346777701a")
        #expect(fourStep.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(fourStep.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(fourStep.format == MiniMaxH3TurboAdapter.lightX2VFormat)
        #expect(fourStep.upstreamRevision == ManagedAdapterCatalog.miniMaxH3LightX2VV1Revision)
        #expect(fourStep.artifact.byteCount == 1_383_677_808)
        #expect(fourStep.artifact.sha256 == "1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c")

        let eightStep768p = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3LightX2VEightStepV1_768pID)
        )
        #expect(eightStep768p.version == "05ef678438e8")
        #expect(eightStep768p.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(eightStep768p.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(eightStep768p.format == MiniMaxH3TurboAdapter.lightX2VFormat)
        #expect(
            eightStep768p.upstreamRevision
                == ManagedAdapterCatalog.miniMaxH3LightX2VEightStepV1_768pRevision
        )
        #expect(eightStep768p.artifact.filename == "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors")
        #expect(eightStep768p.artifact.byteCount == 1_383_677_808)
        #expect(eightStep768p.artifact.sha256 == "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4")
        #expect(eightStep768p.downloadURL.absoluteString.contains(eightStep768p.upstreamRevision!))
    }

    @Test("MiniMax-H3 LightX2V Ref2VA release is an immutable runtime pin")
    func miniMaxH3LightX2VRef2VAReleaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3LightX2VRef2VFourStepV01ID)
        )
        #expect(spec.version == "5d1d4829fe61")
        #expect(spec.baseModelID == ModelResolver.ModelID.miniMaxH3Ref2VAMLX.rawValue)
        #expect(spec.format == MiniMaxH3TurboAdapter.lightX2VFormat)
        #expect(spec.upstreamRevision == ManagedAdapterCatalog.miniMaxH3LightX2VRef2VFourStepV01Revision)
        #expect(spec.artifact.filename == "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
        #expect(spec.artifact.byteCount == 1_383_677_768)
        #expect(spec.artifact.sha256 == "9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

    @Test("FastH3 VSA DataFree release is an immutable restricted BF16 pin")
    func miniMaxH3FastH3VSADataFreeReleaseIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.miniMaxH3FastH3VSADataFreeID)
        )
        #expect(spec.version == "bcf40ca6f457")
        #expect(spec.baseModelID == ModelResolver.ModelID.miniMaxH3FL2VABF16MLX.rawValue)
        #expect(spec.supports(
            baseModelID: ModelResolver.ModelID.miniMaxH3FastH3VSADataFreeMLX.rawValue
        ))
        #expect(!spec.supports(baseModelID: ModelResolver.ModelID.miniMaxH3FL2VAQ8MLX.rawValue))
        #expect(spec.format == MiniMaxH3TurboAdapter.fastVideoFormat)
        #expect(spec.usageRestriction != nil)
        #expect(spec.upstreamRevision == ManagedAdapterCatalog.miniMaxH3FastH3VSADataFreeRevision)
        #expect(spec.artifact.filename == MiniMaxH3TurboAdapter.fastH3VSADataFreeFilename)
        #expect(spec.artifact.byteCount == 5_339_117_712)
        #expect(spec.artifact.sha256 == "42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
    }

    @Test("Catalog ids resolve case-insensitively")
    func normalizedLookup() {
        #expect(ManagedAdapterCatalog.spec(for: " MERE-PLATFORM-ASSISTANT ")?.version == "22")
    }

    @Test("Official LTX-2.5 DFR detailer is an immutable gated pin")
    func ltx25PixelSpatialUpscalerIsPinned() throws {
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.ltx25PixelSpatialUpscalerID)
        )
        #expect(spec.version == "74c4e68ee7dd")
        #expect(spec.baseModelID == ModelResolver.ModelID.ltxVideo25FullBF16.rawValue)
        #expect(spec.format == "ltx-2.5-ic-lora")
        #expect(spec.upstreamRevision == ManagedAdapterCatalog.ltx25PixelSpatialUpscalerRevision)
        #expect(spec.usageRestriction != nil)
        #expect(spec.artifact.byteCount == 327_322_640)
        #expect(spec.artifact.sha256 == "984851b769ea2bcb4c9e0a239a7676239e42c6a6001ddc69943b41ff0b283c1d")
        #expect(spec.downloadURL.host == "huggingface.co")
        #expect(spec.downloadURL.absoluteString.contains(spec.upstreamRevision!))
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
