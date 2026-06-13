import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ImageValidateCommandTests: XCTestCase {
    func testRuntimeErrorDoesNotEmitUsageText() {
        let error = ImageValidateRuntimeError(
            "Model directory is incomplete. Repair the manifest or pull the model again."
        )

        XCTAssertTrue(error.localizedDescription.contains("Model directory is incomplete."))
        XCTAssertFalse(error.localizedDescription.contains("Usage:"))
    }

    func testResolvedZImageResourcesUsesManifestComponentDirectories() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        var manifest = MereRunModelManifest.template(for: .zetaNano, createdAt: Date(timeIntervalSince1970: 0))
        manifest.components = .init(
            tokenizer: .local(path: "tok"),
            textEncoder: .local(path: "enc-mlx"),
            transformer: .local(path: "trans-mflux"),
            vae: .local(path: "vae-mflux"),
            scheduler: .local(path: "sched")
        )
        try manifest.write(to: root)

        let tokenizer = try createDirectory(named: "tok", under: root)
        let textEncoder = try createDirectory(named: "enc-mlx", under: root)
        let transformer = try createDirectory(named: "trans-mflux", under: root)
        let vae = try createDirectory(named: "vae-mflux", under: root)
        let scheduler = try createDirectory(named: "sched", under: root)

        let (resources, loadedManifest) = try ImageValidate().resolvedZImageResources(modelURL: root)

        XCTAssertEqual(loadedManifest?.id, "image-zimage-nano")
        XCTAssertEqual(resources.tokenizerDirURL.standardizedFileURL, tokenizer.standardizedFileURL)
        XCTAssertEqual(resources.textEncoderDirURL.standardizedFileURL, textEncoder.standardizedFileURL)
        XCTAssertEqual(resources.transformerDirURL.standardizedFileURL, transformer.standardizedFileURL)
        XCTAssertEqual(resources.vaeDirURL.standardizedFileURL, vae.standardizedFileURL)
        XCTAssertEqual(resources.schedulerDirURL.standardizedFileURL, scheduler.standardizedFileURL)
        XCTAssertTrue(resources.transformerMFluxWeightsIndexURL.path.hasSuffix("trans-mflux/model.safetensors.index.json"))
        XCTAssertTrue(resources.vaeWeightsIndexURL.path.hasSuffix("vae-mflux/model.safetensors.index.json"))
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func createDirectory(named name: String, under root: URL) throws -> URL {
        let url = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url.resolvingSymlinksInPath()
    }
}
