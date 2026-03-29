import Foundation
import XCTest
@testable import MereRunCore

final class QwenImageEditRepositoryTests: MereRunCoreTestCase {
    func testResolveInstalledModelRootFindsDirectRoot() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let modelRoot = modelsRoot.appendingPathComponent(QwenImageEditRepository.modelId, isDirectory: true)
        try writeMinimalQwenImageEditModel(at: modelRoot)

        let resolved = QwenImageEditRepository.resolveInstalledModelRoot()
        XCTAssertEqual(resolved?.standardizedFileURL, modelRoot.standardizedFileURL)
    }

    func testResolveInstalledModelRootFindsSingleNestedRoot() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        let parentRoot = modelsRoot.appendingPathComponent(QwenImageEditRepository.modelId, isDirectory: true)
        let nestedRoot = parentRoot.appendingPathComponent("Qwen-Image-Edit", isDirectory: true)
        try writeMinimalQwenImageEditModel(at: nestedRoot)

        let resolved = QwenImageEditRepository.resolveInstalledModelRoot()
        XCTAssertEqual(resolved?.standardizedFileURL, nestedRoot.standardizedFileURL)
    }

    func testResolveInstalledModelRootReturnsNilWhenMissing() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }
        defer { MereRunModelPaths.setProcessModelsDirOverride(nil) }

        let modelsRoot = temp.appendingPathComponent("models", isDirectory: true)
        MereRunModelPaths.setProcessModelsDirOverride(modelsRoot)

        XCTAssertNil(QwenImageEditRepository.resolveInstalledModelRoot())
    }

    private func writeMinimalQwenImageEditModel(at root: URL) throws {
        let files: [String] = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer_config.json",
            "tokenizer/tokenizer.json",
        ]

        for relativePath in files {
            let url = root.appendingPathComponent(relativePath, isDirectory: false)
            try TestFileSystem.writeFile(url)
        }
    }
}
