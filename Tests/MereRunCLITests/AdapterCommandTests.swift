import ArgumentParser
import Foundation
import Testing
@testable import MereRunCLI
@testable import MereRunCore

@Suite("Adapter commands")
struct AdapterCommandTests {
    @Test("Adapter pull parses the canonical id")
    func parsesPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.merePlatformAssistantID,
            "--force",
            "--quiet",
        ])
        #expect(command.target == ManagedAdapterCatalog.merePlatformAssistantID)
        #expect(command.force)
        #expect(command.quiet)
    }

    @Test("FLUX.2-dev Turbo pull parses explicit license acceptance")
    func parsesFlux2DevTurboPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.flux2DevTurboEightStepID,
            "--accept-license",
        ])
        #expect(command.target == ManagedAdapterCatalog.flux2DevTurboEightStepID)
        #expect(command.acceptLicense)
    }

    @Test("SCAIL-2 distilled adapter pull parses its canonical id")
    func parsesSCAIL2Pull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.scail2LightX2VFourStepID,
        ])
        #expect(command.target == ManagedAdapterCatalog.scail2LightX2VFourStepID)
    }

    @Test("MiniMax-H3 Turbo adapter pull parses its canonical id")
    func parsesMiniMaxH3TurboPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.miniMaxH3TurboFourStepID,
        ])
        #expect(command.target == ManagedAdapterCatalog.miniMaxH3TurboFourStepID)
    }

    @Test("MiniMax-H3 LightX2V adapter pull parses its canonical id")
    func parsesMiniMaxH3LightX2VPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.miniMaxH3LightX2VFourStepID,
        ])
        #expect(command.target == ManagedAdapterCatalog.miniMaxH3LightX2VFourStepID)
    }

    @Test("MiniMax-H3 LightX2V Ref2VA adapter pull parses its canonical id")
    func parsesMiniMaxH3LightX2VRef2VAPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.miniMaxH3LightX2VRef2VFourStepV01ID,
        ])
        #expect(command.target == ManagedAdapterCatalog.miniMaxH3LightX2VRef2VFourStepV01ID)
    }

    @Test("FastH3 VSA adapter pull parses its canonical id and license acknowledgement")
    func parsesMiniMaxH3FastH3VSADataFreePull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.miniMaxH3FastH3VSADataFreeID,
            "--accept-license",
        ])
        #expect(command.target == ManagedAdapterCatalog.miniMaxH3FastH3VSADataFreeID)
        #expect(command.acceptLicense)
    }

    @Test("Gated LTX-2.5 DFR adapter pull requires explicit terms acceptance")
    func parsesLTX25DFRAdapterPull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.ltx25PixelSpatialUpscalerID,
            "--accept-license",
        ])
        #expect(command.target == ManagedAdapterCatalog.ltx25PixelSpatialUpscalerID)
        #expect(command.acceptLicense)
    }

    @Test("Local LoRA paths remain supported")
    func resolvesLocalPath() throws {
        let relative = "fixtures/local-adapter.safetensors"
        let resolved = try ManagedAdapterArgumentResolver.resolve(
            relative,
            baseModelID: Gemma4Resources.twelveB4BitModelId
        )
        #expect(resolved == URL(fileURLWithPath: relative).standardizedFileURL.path)
    }

    @Test("Catalog id requires a verified pull")
    func catalogIDRequiresInstall() {
        let emptyRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-adapter-test-\(UUID().uuidString)", isDirectory: true)
        #expect(throws: ManagedAdapterResolutionError.self) {
            _ = try ManagedAdapterArgumentResolver.resolve(
                ManagedAdapterCatalog.merePlatformAssistantID,
                baseModelID: Gemma4Resources.twelveB4BitModelId,
                adaptersRoot: emptyRoot
            )
        }
    }

    @Test("Preflight resolves an uninstalled catalog id to its pinned destination")
    func preflightResolvesUninstalledCatalogDestination() throws {
        let emptyRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-adapter-preflight-test-\(UUID().uuidString)", isDirectory: true)
        let resolved = try ManagedAdapterArgumentResolver.resolve(
            ManagedAdapterCatalog.scail2LightX2VFourStepID,
            baseModelID: SCAIL2Resources.modelID,
            adaptersRoot: emptyRoot,
            requireInstalled: false
        )
        let spec = try #require(
            ManagedAdapterCatalog.spec(for: ManagedAdapterCatalog.scail2LightX2VFourStepID)
        )
        #expect(resolved == spec.installedFileURL(adaptersRoot: emptyRoot).path)
    }
}
