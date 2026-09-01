import Foundation
import XCTest
@testable import MereRunCore

final class MereRunModelManifestTests: MereRunCoreTestCase {

    func testLagunaTemplatesPinTheValidatedTargetAndDFlashRevisions() {
        let createdAt = Date(timeIntervalSince1970: 0)
        let target = MereRunModelManifest.template(for: .lagunaS21, createdAt: createdAt)
        let xs = MereRunModelManifest.template(for: .lagunaXS21, createdAt: createdAt)
        let dflash = MereRunModelManifest.template(for: .lagunaS21DFlash, createdAt: createdAt)

        XCTAssertEqual(target.id, LagunaResources.modelID)
        XCTAssertEqual(target.engine, .laguna)
        XCTAssertEqual(target.family, .laguna)
        XCTAssertEqual(target.precision, .int4)
        XCTAssertEqual(target.quantization?.bits, 4)
        XCTAssertEqual(target.quantization?.groupSize, 16)
        XCTAssertEqual(target.quantization?.scheme, "mlx-nvfp4")
        XCTAssertEqual(target.supports, [.chat, .codeGeneration])
        XCTAssertEqual(
            target.upstreamRepoId,
            "\(LagunaResources.upstreamModelID)@\(LagunaResources.upstreamRevision)"
        )

        XCTAssertEqual(xs.id, LagunaResources.xsModelID)
        XCTAssertEqual(xs.engine, .laguna)
        XCTAssertEqual(xs.family, .laguna)
        XCTAssertEqual(xs.tier, .small)
        XCTAssertEqual(xs.precision, .int4)
        XCTAssertEqual(xs.quantization?.bits, 4)
        XCTAssertEqual(xs.quantization?.groupSize, 16)
        XCTAssertEqual(xs.quantization?.scheme, "mlx-nvfp4")
        XCTAssertEqual(xs.supports, [.chat, .codeGeneration])
        XCTAssertEqual(
            xs.upstreamRepoId,
            "\(LagunaResources.xsUpstreamModelID)@\(LagunaResources.xsUpstreamRevision)"
        )

        XCTAssertEqual(dflash.id, LagunaResources.dflashModelID)
        XCTAssertEqual(dflash.engine, .laguna)
        XCTAssertEqual(dflash.family, .laguna)
        XCTAssertEqual(dflash.precision, .bf16)
        XCTAssertEqual(
            dflash.upstreamRepoId,
            "\(LagunaResources.dflashUpstreamModelID)@\(LagunaResources.dflashUpstreamRevision)"
        )
    }

    func testGemma4TwelveB4BitTemplatePinsSawfwairConversion() throws {
        let manifest = MereRunModelManifest.template(
            for: .gemma4TwelveB4Bit,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            Gemma4Resources.twelveB4BitUpstreamModelId
                + "@\(Gemma4Resources.twelveB4BitUpstreamRevision)"
        )
    }

    func testInklingSmallTemplatePinsTheNativeMLXConversion() {
        let manifest = MereRunModelManifest.template(
            for: .inklingSmall,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, InklingResources.modelID)
        XCTAssertEqual(manifest.engine, .inkling)
        XCTAssertEqual(manifest.family, .inkling)
        XCTAssertEqual(manifest.tier, .small)
        XCTAssertEqual(manifest.precision, .int2)
        XCTAssertEqual(manifest.quantization?.bits, 2)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.quantization?.scheme, "affine-routed-experts")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(InklingResources.artifactRepoID)@\(InklingResources.artifactRevision)"
        )
    }

    func testMuseGlimmerTemplatePinsSawfwairSelectiveQ4Conversion() {
        let manifest = MereRunModelManifest.template(
            for: .museGlimmer30B,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, MuseGlimmerResources.modelId)
        XCTAssertEqual(manifest.engine, .museGlimmer)
        XCTAssertEqual(manifest.family, .muse)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(MuseGlimmerResources.artifactRepoId)@\(MuseGlimmerResources.artifactRevision)"
        )
    }

    func testTemplateRoundTrip() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let manifest = MereRunModelManifest.template(for: .kleinNano, createdAt: Date(timeIntervalSince1970: 0))
        try manifest.write(to: temp)

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-klein-nano")
        XCTAssertEqual(loaded.engine, .flux2Klein)
        XCTAssertEqual(loaded.family, .klein)
        XCTAssertEqual(loaded.tier, .nano)
        XCTAssertEqual(loaded.variant, .distilled)
        XCTAssertEqual(loaded.precision, .int4)
        XCTAssertEqual(loaded.quantization?.bits, 4)
        XCTAssertEqual(loaded.quantization?.groupSize, 64)
        XCTAssertEqual(loaded.defaults?.steps, 4)
        XCTAssertEqual(loaded.supports?.contains(.referenceEdit), true)
    }

    func testWriteTemplateIfKnownOverwritesInvalidJSON() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        // Write a broken manifest file.
        let url = MereRunModelManifest.url(in: temp)
        try TestFileSystem.writeFile(url, contents: Data("{not-json".utf8))

        let written = try MereRunModelManifest.writeTemplateIfKnown(modelId: "image-zimage-max", to: temp, createdAt: Date(timeIntervalSince1970: 0))
        XCTAssertNotNil(written)
        guard let written else { return }
        XCTAssertEqual(written.id, "image-zimage-max")

        let loaded = try MereRunModelManifest.loadIfPresent(from: temp)
        XCTAssertNotNil(loaded)
        guard let loaded else { return }
        XCTAssertEqual(loaded.id, "image-zimage-max")
        XCTAssertEqual(loaded.engine, .zimageTurbo)
    }

    func testZImageNanoTemplateUsesMFluxUpstream() throws {
        let manifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-zimage-nano")
        XCTAssertEqual(manifest.engine, .zimageTurbo)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "filipstrand/Z-Image-Turbo-mflux-4bit@b3a8f31115a11f2f9e2fa0bfbc8d78dcc3e6568b"
        )
    }

    func testBonsaiTernaryTemplateHasExpectedNativeLayout() throws {
        let manifest = MereRunModelManifest.template(for: .bonsaiTernary, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-bonsai-ternary")
        XCTAssertEqual(manifest.engine, .flux2Klein)
        XCTAssertEqual(manifest.family, .klein)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .int2)
        XCTAssertEqual(manifest.quantization?.bits, 2)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3.0)
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder-mlx-4bit"))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer-packed-mflux"))
        XCTAssertEqual(manifest.upstreamRepoId, "prism-ml/bonsai-image-ternary-4B-mlx-2bit@main")
    }

    func testBonsaiBinaryTemplateHasExpectedNativeLayout() throws {
        let manifest = MereRunModelManifest.template(for: .bonsaiBinary, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-bonsai-binary")
        XCTAssertEqual(manifest.engine, .flux2Klein)
        XCTAssertEqual(manifest.family, .klein)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .int1)
        XCTAssertEqual(manifest.quantization?.bits, 1)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.defaults?.steps, 4)
        XCTAssertEqual(manifest.defaults?.cfg, 1.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, 3.0)
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder-mlx-4bit"))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer-packed-mflux"))
        XCTAssertEqual(manifest.upstreamRepoId, "prism-ml/bonsai-image-binary-4B-mlx-1bit@main")
    }

    func testTernaryBonsai27BTemplateHasExpectedNativeLayout() throws {
        let manifest = MereRunModelManifest.template(
            for: .bonsai27B2Bit,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.bonsai27B2BitModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .qwen)
        XCTAssertEqual(manifest.precision, .int2)
        XCTAssertEqual(manifest.quantization?.bits, 2)
        XCTAssertEqual(manifest.quantization?.groupSize, 128)
        XCTAssertEqual(manifest.quantization?.scheme, "prism-packed-affine-ternary")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.bonsai27B2BitUpstreamRepoId)@\(Q35Resources.bonsai27B2BitUpstreamRevision)"
        )
    }

    func testQ38TwentySevenBTemplatePinsOfficialBF16Checkpoint() throws {
        let manifest = MereRunModelManifest.template(
            for: .q38TwentySevenB,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.q38TwentySevenBModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .qwen)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.q38TwentySevenBUpstreamRepoId)@\(Q35Resources.q38TwentySevenBUpstreamRevision)"
        )
    }

    func testQ38TwentySevenB4BitTemplateRecordsTargetMTPAndVisionSources() throws {
        let manifest = MereRunModelManifest.template(
            for: .q38TwentySevenB4Bit,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.q38TwentySevenB4BitModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .qwen)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.q38TwentySevenB4BitUpstreamRepoId)"
                + "@\(Q35Resources.q38TwentySevenB4BitUpstreamRevision)"
        )
        XCTAssertEqual(manifest.sources?.count, 4)
        XCTAssertEqual(manifest.sources?.first?.role, "primary")
        XCTAssertEqual(manifest.sources?.first?.repository, Q35Resources.q38TwentySevenB4BitUpstreamRepoId)
        XCTAssertEqual(manifest.sources?[1].role, "component")
        XCTAssertEqual(manifest.sources?[1].repository, Q35Resources.q38MTP4BitUpstreamRepoId)
        XCTAssertEqual(manifest.sources?[1].destinationPath, Q35Resources.q38MTPComponentPath)
        XCTAssertEqual(manifest.sources?[2].role, "component")
        XCTAssertEqual(manifest.sources?[2].repository, Q35Resources.q38TwentySevenBUpstreamRepoId)
        XCTAssertEqual(manifest.sources?[2].destinationPath, Q35Resources.q38VisionComponentPath)
        XCTAssertEqual(manifest.sources?.last?.role, "component")
        XCTAssertEqual(manifest.sources?.last?.repository, Q35Resources.q38TwentySevenBUpstreamRepoId)
        XCTAssertEqual(manifest.sources?.last?.destinationPath, Q35Resources.q38LicenseComponentPath)
    }

    func testQ38FlashNext3BitTemplatePinsActivationWeightedArtifact() {
        let manifest = MereRunModelManifest.template(
            for: .q38FlashNext3Bit,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.q38FlashNext3BitModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .qwen)
        XCTAssertEqual(manifest.precision, .int3)
        XCTAssertEqual(manifest.quantization?.bits, 3)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(
            manifest.quantization?.scheme,
            "mlx-mixed-q3-q4-activation-refit-affine"
        )
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.q38FlashNext3BitUpstreamRepoId)"
                + "@\(Q35Resources.q38FlashNext3BitUpstreamRevision)"
        )
    }

    func testQ38FlashNext3BitNativePLETemplatePinsCompletePack() {
        let manifest = MereRunModelManifest.template(
            for: .q38FlashNext3BitNativePLE,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.q38FlashNext3BitNativePLEModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .qwen)
        XCTAssertEqual(manifest.precision, .int3)
        XCTAssertEqual(manifest.quantization?.bits, 3)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(
            manifest.quantization?.scheme,
            "mlx-mixed-q3-q4-activation-refit-affine-native-ple"
        )
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.q38FlashNext3BitNativePLEUpstreamRepoId)"
                + "@\(Q35Resources.q38FlashNext3BitNativePLEUpstreamRevision)"
        )
    }

    func testFalconPerceptionTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .visionGroundFalconPerception, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "vision-ground-falcon-perception")
        XCTAssertEqual(manifest.engine, .falconPerception)
        XCTAssertEqual(manifest.family, .falcon)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .unknown)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.visionGrounding, .visionDetection, .visionSegmentation]))
        XCTAssertEqual(manifest.upstreamRepoId, "tiiuae/Falcon-Perception")
    }

    func testTerraMindFloodTemplateHasExpectedNativeMetadata() {
        let manifest = MereRunModelManifest.template(
            for: .visionFloodTerraMindBase,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, TerraMindFloodResources.defaultModelID)
        XCTAssertEqual(manifest.engine, .terramindFlood)
        XCTAssertEqual(manifest.family, .terramind)
        XCTAssertEqual(manifest.tier, .base)
        XCTAssertEqual(manifest.precision, .fp32)
        XCTAssertEqual(manifest.supports, [.floodSegmentation])
        XCTAssertNil(manifest.components)
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(TerraMindFloodResources.sourceRepository)@\(TerraMindFloodResources.sourceRevision)"
        )
    }

    func testTerraMindFireTemplateHasExpectedNativeMetadata() {
        let manifest = MereRunModelManifest.template(
            for: .visionFireTerraMindBase,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.engine, .terramindFire)
        XCTAssertEqual(manifest.family, .terramind)
        XCTAssertEqual(manifest.tier, .base)
        XCTAssertEqual(manifest.precision, .fp32)
        XCTAssertEqual(manifest.supports, [.fireSegmentation])
        XCTAssertNil(manifest.components)
    }

    func testTESSERAAndOlmoEarthTemplatesPreserveHardwareTiers() {
        let tessera = MereRunModelManifest.template(
            for: .visionEmbedTESSERAV2Teacher,
            createdAt: Date(timeIntervalSince1970: 0)
        )
        let olmoEarth = MereRunModelManifest.template(
            for: .visionEmbedOlmoEarthV12Base,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(tessera.engine, .tessera)
        XCTAssertEqual(tessera.family, .tessera)
        XCTAssertEqual(tessera.tier, .max)
        XCTAssertEqual(tessera.supports, [.earthObservationEmbedding])
        XCTAssertEqual(olmoEarth.engine, .olmoEarth)
        XCTAssertEqual(olmoEarth.family, .olmoEarth)
        XCTAssertEqual(olmoEarth.tier, .base)
        XCTAssertEqual(olmoEarth.supports, [.earthObservationEmbedding])
    }

    func testInfinityParser2ProTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .infinityParser2Pro, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Q35Resources.infinityParser2ProModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .ocr)
        XCTAssertEqual(manifest.tier, .max)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .visionChat, .visionOCR]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.infinityParser2ProUpstreamRepoId)@\(Q35Resources.infinityParser2ProUpstreamRevision)"
        )
    }

    func testInfinityParser2ProInt8TemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .infinityParser2ProInt8, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Q35Resources.infinityParser2ProInt8ModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .ocr)
        XCTAssertEqual(manifest.tier, .max)
        XCTAssertEqual(manifest.precision, .int8)
        XCTAssertEqual(manifest.quantization?.bits, 8)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-quantized-linear")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .visionChat, .visionOCR]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.infinityParser2ProInt8UpstreamRepoId)@\(Q35Resources.infinityParser2ProInt8UpstreamRevision)"
        )
    }

    func testNorthMiniCodeTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .northMiniCode, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, NorthMiniCodeResources.modelId)
        XCTAssertEqual(manifest.engine, .northMiniCode)
        XCTAssertEqual(manifest.family, .code)
        XCTAssertEqual(manifest.tier, .small)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(NorthMiniCodeResources.upstreamRepoId)@\(NorthMiniCodeResources.upstreamRevision)"
        )
    }

    func testOrnith9BTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .ornith9B, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Q35Resources.ornith9BModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .code)
        XCTAssertEqual(manifest.tier, .nano)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-optiq-mixed-affine")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.ornith9BUpstreamRepoId)@\(Q35Resources.ornith9BUpstreamRevision)"
        )
    }

    func testOrnith35BMLXTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .ornith35BMLX, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Q35Resources.ornith35BMLXModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .code)
        XCTAssertEqual(manifest.tier, .large)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.ornith35BMLXUpstreamRepoId)@\(Q35Resources.ornith35BMLXUpstreamRevision)"
        )
    }

    func testOrnith35BVisionTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(
            for: .ornith35BVision,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.ornith35BVisionModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.family, .code)
        XCTAssertEqual(manifest.tier, .large)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertNil(manifest.quantization)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Q35Resources.ornith35BVisionUpstreamRepoId)"
                + "@\(Q35Resources.ornith35BVisionUpstreamRevision)"
        )
    }

    func testOrnith35BMLX4BitTemplateRecordsTargetAndVisionSources() throws {
        let manifest = MereRunModelManifest.template(
            for: .ornith35BMLX4Bit,
            createdAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(manifest.id, Q35Resources.ornith35BMLX4BitModelId)
        XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration, .visionChat]))
        XCTAssertEqual(manifest.sources?.count, 2)
        XCTAssertEqual(manifest.sources?.first?.role, "primary")
        XCTAssertEqual(
            manifest.sources?.first?.repository,
            Q35Resources.ornith35BMLX4BitUpstreamRepoId
        )
        XCTAssertEqual(manifest.sources?.last?.role, "component")
        XCTAssertEqual(
            manifest.sources?.last?.repository,
            Q35Resources.ornith35BVisionUpstreamRepoId
        )
        XCTAssertEqual(
            manifest.sources?.last?.destinationPath,
            Q35Resources.ornith35BVisionComponentPath
        )
    }

    func testOrnith35BMLXQuantizedTemplatesPinOfficialConversions() throws {
        let createdAt = Date(timeIntervalSince1970: 0)
        let cases: [(
            ModelResolver.ModelID,
            String,
            MereRunModelManifest.Precision,
            Int,
            String,
            String
        )] = [
            (
                .ornith35BMLX4Bit,
                Q35Resources.ornith35BMLX4BitModelId,
                .int4,
                4,
                Q35Resources.ornith35BMLX4BitUpstreamRepoId,
                Q35Resources.ornith35BMLX4BitUpstreamRevision
            ),
            (
                .ornith35BMLX6Bit,
                Q35Resources.ornith35BMLX6BitModelId,
                .int6,
                6,
                Q35Resources.ornith35BMLX6BitUpstreamRepoId,
                Q35Resources.ornith35BMLX6BitUpstreamRevision
            ),
            (
                .ornith35BMLX8Bit,
                Q35Resources.ornith35BMLX8BitModelId,
                .int8,
                8,
                Q35Resources.ornith35BMLX8BitUpstreamRepoId,
                Q35Resources.ornith35BMLX8BitUpstreamRevision
            ),
        ]

        for (modelID, id, precision, bits, repoID, revision) in cases {
            let manifest = MereRunModelManifest.template(for: modelID, createdAt: createdAt)
            XCTAssertEqual(manifest.id, id)
            XCTAssertEqual(manifest.engine, .qwen35HybridMoE)
            XCTAssertEqual(manifest.family, .code)
            XCTAssertEqual(manifest.tier, .large)
            XCTAssertEqual(manifest.precision, precision)
            XCTAssertEqual(manifest.quantization?.bits, bits)
            XCTAssertEqual(manifest.quantization?.groupSize, 64)
            XCTAssertEqual(manifest.quantization?.scheme, "mlx-affine")
            let expectedSupports: Set<MereRunModelManifest.Capability> = bits == 4
                ? [.chat, .codeGeneration, .visionChat]
                : [.chat, .codeGeneration]
            XCTAssertEqual(Set(manifest.supports ?? []), expectedSupports)
            XCTAssertEqual(manifest.upstreamRepoId, "\(repoID)@\(revision)")
        }
    }

    func testOrnith35BTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .ornith35B, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Ornith35BCodeResources.modelId)
        XCTAssertEqual(manifest.engine, .qwen3Coder)
        XCTAssertEqual(manifest.family, .code)
        XCTAssertEqual(manifest.tier, .small)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "gguf-q4-k-m")
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.chat, .codeGeneration]))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "\(Ornith35BCodeResources.upstreamRepoId)@\(Ornith35BCodeResources.upstreamRevision)"
        )
    }

    func testHiDreamO1DevTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .hidreamO1Dev, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-hidream-o1-dev")
        XCTAssertEqual(manifest.engine, .hidreamO1)
        XCTAssertEqual(manifest.family, .hidream)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 28)
        XCTAssertEqual(manifest.defaults?.cfg, 0.0)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .referenceEdit, .subjectPersonalization]))
        XCTAssertEqual(manifest.upstreamRepoId, "HiDream-ai/HiDream-O1-Image-Dev")
    }

    func testHiDreamO1FullTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .hidreamO1, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "image-hidream-o1")
        XCTAssertEqual(manifest.engine, .hidreamO1)
        XCTAssertEqual(manifest.family, .hidream)
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 50)
        XCTAssertEqual(manifest.defaults?.cfg, 5.0)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .referenceEdit, .subjectPersonalization]))
        XCTAssertEqual(manifest.upstreamRepoId, "HiDream-ai/HiDream-O1-Image")
    }

    func testKrea2TurboTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .krea2Turbo, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Krea2Resources.modelId)
        XCTAssertEqual(manifest.engine, .krea2)
        XCTAssertEqual(manifest.family, .krea)
        XCTAssertEqual(manifest.tier, .turbo)
        XCTAssertEqual(manifest.variant, .distilled)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 8)
        XCTAssertEqual(manifest.defaults?.cfg, 0.0)
        XCTAssertEqual(manifest.defaults?.sigmaShift, Double(Krea2SampleBuilder.defaultMu))
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .loraInference]))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer"))
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder"))
        XCTAssertEqual(manifest.components?.vae, .local(path: "vae"))
        XCTAssertEqual(manifest.upstreamRepoId, "\(Krea2Resources.upstreamRepoId)@\(Krea2Resources.upstreamRevision)")
    }

    func testKrea2RawTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .krea2Raw, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Krea2RawResources.modelId)
        XCTAssertEqual(manifest.engine, .krea2)
        XCTAssertEqual(manifest.family, .krea)
        XCTAssertEqual(manifest.tier, .max)
        XCTAssertEqual(manifest.variant, .base)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(manifest.defaults?.steps, 52)
        XCTAssertEqual(manifest.defaults?.cfg, 3.5)
        XCTAssertEqual(manifest.defaults?.sigmaShift, Double(Krea2SampleBuilder.defaultMu))
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img, .loraTraining]))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer"))
        XCTAssertEqual(manifest.components?.textEncoder, .local(path: "text_encoder"))
        XCTAssertEqual(manifest.components?.vae, .local(path: "vae"))
        XCTAssertEqual(manifest.upstreamRepoId, "\(Krea2RawResources.upstreamRepoId)@\(Krea2RawResources.upstreamRevision)")
    }

    func testIdeogram4TemplateHasExpectedSDNQMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .ideogram4SDNQUInt4, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, Ideogram4Resources.modelId)
        XCTAssertEqual(manifest.engine, .ideogram4)
        XCTAssertEqual(manifest.family, .ideogram)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .int4)
        XCTAssertEqual(manifest.quantization?.bits, 4)
        XCTAssertEqual(manifest.quantization?.groupSize, 64)
        XCTAssertEqual(manifest.quantization?.scheme, "sdnq-uint4")
        XCTAssertEqual(manifest.defaults?.steps, 20)
        XCTAssertEqual(manifest.defaults?.cfg, 7.0)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.txt2img]))
        XCTAssertEqual(manifest.components?.transformer, .local(path: "transformer"))
        XCTAssertEqual(manifest.components?.unconditionalTransformer, .local(path: "unconditional_transformer"))
        XCTAssertEqual(
            manifest.upstreamRepoId,
            "WaveCut/ideogram-4-sdnq-uint4@\(Ideogram4Resources.upstreamRevision)"
        )
    }

    func testPrivacyFilterTemplateHasExpectedMetadata() throws {
        let manifest = MereRunModelManifest.template(for: .privacyFilter, createdAt: Date(timeIntervalSince1970: 0))

        XCTAssertEqual(manifest.id, "text-anonymize-privacy-filter")
        XCTAssertEqual(manifest.engine, .openAIPrivacyFilter)
        XCTAssertEqual(manifest.family, .privacy)
        XCTAssertEqual(manifest.variant, .standard)
        XCTAssertEqual(manifest.precision, .bf16)
        XCTAssertEqual(Set(manifest.supports ?? []), Set([.textAnonymization]))
        XCTAssertEqual(manifest.upstreamRepoId, "openai/privacy-filter")
    }
}
