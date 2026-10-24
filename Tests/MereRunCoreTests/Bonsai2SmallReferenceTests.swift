import Foundation
import MediaIO
import MereRunAdmission
import MLX
import MLXNN
@testable import MereRunTextEncoder
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

/// Opt-in exports for independent comparison; an export alone is not a parity pass.
final class Bonsai2SmallReferenceTests: MereRunCoreTestCase {
    func testExportedPackedComponentsOnCPU() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_BONSAI_PARITY_MODE"] == "components",
              let root = env["MERERUN_TEST_BONSAI_PARITY_ROOT"] else {
            throw XCTSkip("Set the parity evidence root and components mode for bounded real-row checks.")
        }
        struct Record: Decodable { let size: String; let blocks: [String: Int] }
        struct Result: Encodable {
            let size: String
            let embeddingMaxDelta: Float
            let headMaxDelta: Float
            let patchMaxDelta: Float
            let flattenedPatchMaxDelta: Float
        }
        let destination = URL(fileURLWithPath: root).appending(path: "cpu-components")
        let records = try JSONDecoder().decode([Record].self,
            from: Data(contentsOf: destination.appending(path: "reference.json")))
        var results: [Result] = []
        for record in records {
            let arrays = try MLX.loadArrays(url: destination.appending(path: "\(record.size).safetensors"))
            func array(_ key: String) throws -> MLXArray { try XCTUnwrap(arrays[key]) }
            let embedding = Q35PrismEmbedding(weight: try array("embedding_weight"),
                scales: try array("embedding_scales"), biases: try array("embedding_biases"),
                signs: try array("embedding_signs"), block: try XCTUnwrap(record.blocks["embedding"]))
            let rows = embedding(MLXArray([Int32(0), 1, 2, 3]).reshaped(1, 4))
            let hidden = try array("hidden")
            let head: MLXArray
            if record.size == "9" {
                let projection = Q35PrismLinear(weight: try array("head_weight"), scales: try array("head_scales"),
                    biases: try array("head_biases"), signs: try array("head_signs"),
                    block: try XCTUnwrap(record.blocks["head"]))
                head = projection(hidden)
            } else {
                head = embedding.asLinear(hidden)
            }
            let embeddingDelta = MLX.max(MLX.abs(rows - (try array("expected_embedding")))).item(Float.self)
            let headDelta = MLX.max(MLX.abs(head - (try array("expected_head")))).item(Float.self)
            XCTAssertLessThanOrEqual(embeddingDelta, 0.00001, "\(record.size)B embedding")
            XCTAssertLessThanOrEqual(headDelta, 0.00001, "\(record.size)B head")
            let patchArrays = try MLX.loadArrays(url: URL(fileURLWithPath: root)
                .appending(path: "cpu-patch-projection/\(record.size).safetensors"))
            let patchWeight = try XCTUnwrap(patchArrays["weight"])
            let patch = QwenVisionPatchEmbed(patchSize: 16, temporalPatchSize: 2,
                inChannels: 3, embedDim: patchWeight.dim(0), bias: true)
            try patch.update(parameters: ModuleParameters.unflattened([
                ("proj.weight", patchWeight), ("proj.bias", try XCTUnwrap(patchArrays["bias"]))
            ]), verify: [.all])
            let patches = try XCTUnwrap(patchArrays["patches"])
            let expectedPatch = try XCTUnwrap(patchArrays["expected"])
            let flattened = patch(patches)
            patch.useConvolution = true
            let convolved = patch(patches)
            let patchDelta = MLX.max(MLX.abs(convolved - expectedPatch)).item(Float.self)
            let flattenedDelta = MLX.max(MLX.abs(flattened - expectedPatch)).item(Float.self)
            XCTAssertLessThanOrEqual(patchDelta, 0.00001, "\(record.size)B patch projection")
            results.append(Result(size: record.size, embeddingMaxDelta: embeddingDelta, headMaxDelta: headDelta,
                patchMaxDelta: patchDelta, flattenedPatchMaxDelta: flattenedDelta))
            try MLX.save(arrays: ["embedding": rows, "head": head],
                url: destination.appending(path: "\(record.size)-native.safetensors"))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(results).write(to: destination.appending(path: "comparison.json"))
    }

    func testInstalledCheckpointExport() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let rootPath = env["MERERUN_TEST_BONSAI_PARITY_ROOT"],
              let size = env["MERERUN_TEST_BONSAI_PARITY_SIZE"],
              let mode = env["MERERUN_TEST_BONSAI_PARITY_MODE"] else {
            throw XCTSkip("Set the Bonsai parity root, size, mode, and GPU opt-in variables.")
        }
        XCTAssertTrue(["2", "4", "9"].contains(size))
        let root = URL(fileURLWithPath: rootPath)
        let checkpoint = root.deletingLastPathComponent().appending(path: "models/text-chat-bonsai-2-\(size)b-2bit")
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(label: "Bonsai \(size)B \(mode) parity", resourceClass: .standard))
        defer { lease.release() }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(coordinator.snapshot()).write(to: root.appending(path: "admission-\(mode)-\(size).json"))
        if mode == "reference" {
            let process = Process()
            process.executableURL = root.appending(path: "venv/bin/python")
            let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            process.arguments = [repo.appending(path: "scripts/reference-parity/export_bonsai2_small.py").path,
                                 "--root", root.path, "--model", checkpoint.path, "--size", size]
            if let version = env["MERERUN_TEST_BONSAI_REFERENCE_MLX_VERSION"] {
                process.arguments?.append(contentsOf: ["--mlx-version", version])
            }
            var environment = env
            environment["HF_HUB_OFFLINE"] = "1"
            environment["HF_HOME"] = root.appending(path: "hub-cache").path
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            process.environment = environment
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0)
            return
        }
        XCTAssertEqual(mode, "native")
        let plan = try JSONDecoder().decode(BonsaiParityPlan.self, from: Data(contentsOf: root.appending(path: "cases.json")))
        let reference = try JSONDecoder().decode(BonsaiParityReference.self,
            from: Data(contentsOf: root.appending(path: "reference-\(size)/results.json")))
        let destination = root.appending(path: "native-\(size)")
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        guard !FileManager.default.fileExists(atPath: destination.appending(path: "results.json").path) else {
            XCTFail("Preserve the completed native export before running another comparison.")
            return
        }
        let generator = Q35Generator(modelId: "text-chat-bonsai-2-\(size)b-2bit",
                                     prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        let results = try await generator.runBonsaiParity(
            checkpoint: checkpoint, plan: plan, reference: reference, destination: destination
        )
        try encoder.encode(results).write(to: destination.appending(path: "results.json"))
    }
}

private struct BonsaiParityPlan: Decodable, Sendable {
    let system: String
    let cases: [Probe]
    struct Probe: Decodable, Sendable {
        let id: String
        let prompt: String
        let image: String?
        let thinking: Bool
        let steps: Int
    }
}

private struct BonsaiParityReference: Decodable, Sendable {
    let cases: [Probe]
    struct Probe: Decodable, Sendable {
        let id: String
        let continuation: [Int]
    }
}

private struct BonsaiParityOutput: Encodable, Sendable {
    let id: String
    let prompt_ids: [Int]
    let continuation: [Int]
    let native_argmax: [Int]
    let grid: [Int]
    let rope_delta: Int
}

extension Q35Generator {
    fileprivate func runBonsaiParity(
        checkpoint: URL, plan: BonsaiParityPlan, reference: BonsaiParityReference, destination: URL
    ) async throws -> [BonsaiParityOutput] {
        try await withRequestStream {
            try await ensureLoaded(rootURL: checkpoint, progressHandler: nil)
            let results = try await exportBonsaiParity(plan: plan, reference: reference, destination: destination)
            await unload()
            return results
        }
    }

    fileprivate func exportBonsaiParity(
        plan: BonsaiParityPlan, reference: BonsaiParityReference, destination: URL
    ) async throws -> [BonsaiParityOutput] {
        let model = try XCTUnwrap(model)
        let template = try XCTUnwrap(tokenizerAndTemplate)
        let config = try XCTUnwrap(loadedConfig)
        var results: [BonsaiParityOutput] = []
        for probe in plan.cases {
            let target = try XCTUnwrap(reference.cases.first { $0.id == probe.id })
            var arrays: [String: MLXArray] = [:]
            var replacements: [Q35VisionReplacement] = []
            var gridValues: [Int] = []
            let imagePath = probe.image.map {
                destination.deletingLastPathComponent().deletingLastPathComponent().appending(path: "fixtures/\($0)").path
            }
            if let imagePath {
                try ensureVisionWeightsLoaded(progressHandler: nil)
                let tower = try XCTUnwrap(visionTower)
                let prepared = try loadImageTensor(from: imagePath, patchSize: tower.patchSize,
                    spatialMergeSize: tower.spatialMergeSize, maximumVisionTokens: 8_000)
                let embeddings = try tower.encodeImage(pixelValues: prepared.tensor, gridTHW: prepared.gridTHW)
                replacements = [Q35VisionReplacement(embeddings: embeddings, gridTHW: prepared.gridTHW)]
                gridValues = [prepared.gridTHW.0, prepared.gridTHW.1, prepared.gridTHW.2]
                arrays["pixels_chw"] = prepared.tensor
                arrays["patches"] = Q35VisionTower.preparePatchInputs(pixelValues: prepared.tensor,
                    patchSize: tower.patchSize, temporalPatchSize: tower.temporalPatchSize,
                    mergeSize: tower.spatialMergeSize).reshaped(-1, 3 * tower.temporalPatchSize * tower.patchSize * tower.patchSize)
                arrays["image_embeddings"] = embeddings
            }
            let messages = [ChatMessage(role: .system, content: plan.system),
                            ChatMessage(role: .user, content: probe.prompt, imageUrl: imagePath)]
            let ids = try template.encodeForGeneration(messages: messages, includeThinking: probe.thinking,
                maxLength: 16_384, imageTokenCounts: replacements.map { $0.embeddings.dim(0) })
            let input = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            var embeddings = model.embeddings(for: input)
            var positionData: Q35MRoPEPositionData?
            if !replacements.isEmpty {
                let imageID = try XCTUnwrap(config.imageTokenId)
                embeddings = insertVisionEmbeddings(hiddenStates: embeddings, inputIds: input,
                                                     imageTokenId: imageID, replacements: replacements)
                positionData = try buildMRoPEPositionData(inputIds: input, imageTokenId: imageID,
                    replacements: replacements, spatialMergeSize: try XCTUnwrap(visionTower).spatialMergeSize)
                arrays["positions"] = positionData?.positionIds
            }
            arrays["input_embeddings"] = embeddings
            arrays["prompt_ids"] = input
            let caches = makeLayerCaches(config: config)
            var output = try await chunkedPrefillEmbeddings(model: model, inputIds: input, inputEmbeddings: embeddings,
                cache: caches, positionIds: positionData?.positionIds, retainHidden: false, progressHandler: nil)
            var logits: [MLXArray] = []
            var greedy: [Int] = []
            for (index, token) in target.continuation.enumerated() {
                let row = output.logits[0, -1, 0...].asType(.float32)
                MLX.eval(row)
                logits.append(row)
                greedy.append(Int(MLX.argMax(row).item(Int32.self)))
                if index + 1 < target.continuation.count {
                    let next = MLXArray([Int32(token)]).reshaped(1, 1)
                    let positions = positionData.map {
                        MLXArray.full([3, 1, 1], values: MLXArray(Int32(ids.count + index + $0.ropeDelta)))
                    }
                    let decoded = model.forward(next, cache: caches, positionIds: positions)
                    MLX.eval(decoded.logits)
                    output = Q35PrefillOutput(logits: decoded.logits, hidden: nil, mtpHistoryHidden: nil, mtpSession: nil)
                }
            }
            arrays["logits"] = MLX.stacked(logits)
            try MLX.save(arrays: arrays, url: destination.appending(path: "\(probe.id).safetensors"))
            results.append(BonsaiParityOutput(id: probe.id, prompt_ids: ids, continuation: target.continuation,
                native_argmax: greedy, grid: gridValues, rope_delta: positionData?.ropeDelta ?? 0))
            print("[bonsai-parity] \(probe.id) exported \(logits.count) positions")
            Memory.clearCache()
        }
        return results
    }
}
