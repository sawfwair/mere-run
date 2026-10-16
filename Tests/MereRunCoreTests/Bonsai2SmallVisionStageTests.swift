import Foundation
import MereRunAdmission
import MLX
import MLXNN
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel
@testable import MereRunTextEncoder

final class Bonsai2SmallVisionStageTests: MereRunCoreTestCase {
    func testInstalledVisionStages() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let path = env["MERERUN_TEST_BONSAI_PARITY_ROOT"],
              let size = env["MERERUN_TEST_BONSAI_PARITY_SIZE"],
              let mode = env["MERERUN_TEST_BONSAI_PARITY_MODE"],
              ["reference-stages", "native-stages"].contains(mode) else {
            throw XCTSkip("Set the Bonsai parity root, size, stage mode, and GPU opt-in.")
        }
        let root = URL(fileURLWithPath: path)
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "Bonsai \(size)B vision stage comparison", resourceClass: .standard))
        defer { lease.release() }
        if mode == "reference-stages" {
            let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let process = Process()
            process.executableURL = root.appending(path: "venv/bin/python")
            process.arguments = [repo.appending(path: "scripts/reference-parity/export_bonsai2_vision_stages.py").path,
                                 "--root", path, "--size", size]
            var environment = env
            environment["HF_HUB_OFFLINE"] = "1"
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = environment
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0)
            return
        }
        let generator = Q35Generator(modelId: "text-chat-bonsai-2-\(size)b-2bit",
                                     prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        try await generator.exportVisionStages(root: root, size: size)
    }
}

extension Q35Generator {
    fileprivate func exportVisionStages(root: URL, size: String) async throws {
        try await withRequestStream {
            let checkpoint = root.deletingLastPathComponent().appending(path: "models/text-chat-bonsai-2-\(size)b-2bit")
            try await ensureLoaded(rootURL: checkpoint, progressHandler: nil)
            try ensureVisionWeightsLoaded(progressHandler: nil)
            let wrapper = try XCTUnwrap(visionTower)
            let children = Dictionary(uniqueKeysWithValues: wrapper.children().flattened())
            let tower = try XCTUnwrap(children["visionTower"] as? QwenVisionTower)
            let parts = Dictionary(uniqueKeysWithValues: tower.children().flattened())
            let patch = try XCTUnwrap(parts["patch_embed"] as? QwenVisionPatchEmbed)
            let position = try XCTUnwrap(parts["pos_embed"] as? Embedding)
            let merger = try XCTUnwrap(parts["patch_merger"] as? QwenVisionPatchMerger)
            let source = try MLX.loadArrays(url: root.appending(path: "reference-\(size)/vision-chart.safetensors"))
            let patches = try XCTUnwrap(source["patches"])
            let gridArray = try XCTUnwrap(source["grid"]).asArray(Int32.self).map(Int.init)
            let geometry = [(t: gridArray[0], h: gridArray[1], w: gridArray[2])]
            let grid = [QwenVisionGrid(temporal: gridArray[0], height: gridArray[1], width: gridArray[2])]
            let production = try tower(patchInputs: patches.expandedDimensions(axis: 0), grid: grid)
            let rotary = QwenVisionTower.prepareQwen3Rotary(rotaryPosEmb: production.rotaryCos)
            var hidden = patch(patches)
            let pos = tower.fastPosEmbedInterpolate(gridThw: geometry, posEmbed: position)
            var arrays = ["patch_embed": hidden, "pos_embed": pos, "rotary_freqs": production.rotaryCos]
            hidden = hidden + pos.asType(hidden.dtype)
            arrays["input"] = hidden
            hidden = hidden.expandedDimensions(axis: 0)
            let cu = MLXArray([Int32(0), Int32(patches.dim(0))])
            for index in 0..<tower.configuration.depth {
                let block = try XCTUnwrap(parts["blocks.\(index)"] as? QwenVisionBlock)
                hidden = block(hidden, rotaryEmbedding: rotary, cuSeqlens: cu)
                MLX.eval(hidden)
                arrays["block_\(index)"] = hidden.squeezed(axis: 0)
            }
            arrays["merged"] = try merger(hidden.squeezed(axis: 0))
            arrays["production_merged"] = production.hiddenStates
            try MLX.save(arrays: arrays, url: root.appending(path: "native-vision-stages-\(size).safetensors"))
            await unload()
        }
    }
}
