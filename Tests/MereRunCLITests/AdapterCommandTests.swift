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

    @Test("SCAIL-2 distilled adapter pull parses its canonical id")
    func parsesSCAIL2Pull() throws {
        let command = try AdapterPull.parse([
            ManagedAdapterCatalog.scail2LightX2VFourStepID,
        ])
        #expect(command.target == ManagedAdapterCatalog.scail2LightX2VFourStepID)
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
}
