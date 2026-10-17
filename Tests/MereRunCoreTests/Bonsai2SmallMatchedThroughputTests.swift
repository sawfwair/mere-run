import Foundation
import MediaIO
import MereRunAdmission
import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

/// A fixed-continuation model-loop comparison, separate from product API timings.
final class Bonsai2SmallMatchedThroughputTests: MereRunCoreTestCase {
    func testInstalledMatchedModelLoop() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_TEST_MLX_DEVICE"] == "gpu",
              let path = env["MERERUN_BONSAI_MATCHED_ROOT"],
              let size = env["MERERUN_BONSAI_MATCHED_SIZE"],
              let stack = env["MERERUN_BONSAI_MATCHED_STACK"],
              let outputPath = env["MERERUN_BONSAI_MATCHED_OUTPUT"] else {
            throw XCTSkip("Set the matched Bonsai root, size, stack, output, and GPU opt-in variables.")
        }
        XCTAssertTrue(["2", "4", "9"].contains(size))
        XCTAssertTrue(["native", "reference"].contains(stack))
        let root = URL(fileURLWithPath: path)
        let output = URL(fileURLWithPath: outputPath)
        guard !FileManager.default.fileExists(atPath: output.path) else {
            XCTFail("Preserve the previous matched benchmark before collecting another cohort.")
            return
        }
        let coordinator = MachineInferenceCoordinator(stateDirectory:
            MereRunModelPaths.applicationSupportBase.appendingPathComponent("admission", isDirectory: true))
        let lease = try await coordinator.acquire(.init(
            label: "Bonsai \(size)B matched \(stack) model loop", resourceClass: .standard))
        defer { lease.release() }
        if stack == "reference" {
            let python = try XCTUnwrap(env["MERERUN_BONSAI_REFERENCE_PYTHON"])
            let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
            let process = Process()
            process.executableURL = URL(fileURLWithPath: python)
            process.arguments = ["-B", repo.appending(path:
                "scripts/reference-parity/benchmark_bonsai2_small_matched.py").path,
                "--root", root.path, "--size", size, "--output", output.path]
            process.environment = env
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0)
            return
        }
        let inputs = try JSONDecoder().decode(MatchedBonsaiInputs.self,
            from: Data(contentsOf: root.appending(path: "matched-inputs-\(size).json")))
        let generator = Q35Generator(modelId: "text-chat-bonsai-2-\(size)b-2bit",
                                     prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        let checkpoint = root.appending(path: "models/text-chat-bonsai-2-\(size)b-2bit")
        var records: [MatchedBonsaiRecord] = []
        for probe in inputs.probes {
            for repetition in 0..<4 {
                let count = repetition == 0 ? 128 : probe.continuation.count
                records.append(try await generator.measureMatchedBonsai(
                    checkpoint: checkpoint, root: root, probe: probe,
                    repetition: repetition, count: count))
            }
        }
        await generator.unload()
        let report = MatchedBonsaiReport(stack: stack, size: size, records: records)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output)
        XCTAssertEqual(records.count, 8)
        XCTAssertTrue(records.allSatisfy { $0.seconds > 0 && $0.argmax.count == $0.tokenCount })
    }
}

private struct MatchedBonsaiInputs: Decodable, Sendable {
    let probes: [Probe]
    struct Probe: Decodable, Sendable {
        let id: String
        let prompt: String
        let prompt_ids: [Int]
        let continuation: [Int]
        let image: String?
        let image_features: String?
    }
}

private struct MatchedBonsaiRecord: Encodable, Sendable {
    let modality: String
    let repetition: Int
    let tokenCount: Int
    let promptIDs: [Int]
    let continuation: [Int]
    let argmax: [Int]
    let seconds: Double
    let thermalBefore: Int
    let thermalAfter: Int
}

private struct MatchedBonsaiReport: Encodable {
    let stack: String
    let size: String
    let records: [MatchedBonsaiRecord]
}

extension Q35Generator {
    fileprivate func measureMatchedBonsai(
        checkpoint: URL, root: URL, probe: MatchedBonsaiInputs.Probe,
        repetition: Int, count: Int
    ) async throws -> MatchedBonsaiRecord {
        try await withRequestStream {
            try await ensureLoaded(rootURL: checkpoint, progressHandler: nil)
            let model = try XCTUnwrap(model)
            let template = try XCTUnwrap(tokenizerAndTemplate)
            let config = try XCTUnwrap(loadedConfig)
            var replacements: [Q35VisionReplacement] = []
            let imagePath = probe.image.map { root.appending(path: "fixtures/\($0)").path }
            if let imagePath {
                try ensureVisionWeightsLoaded(progressHandler: nil)
                let tower = try XCTUnwrap(visionTower)
                let prepared = try loadImageTensor(from: imagePath, patchSize: tower.patchSize,
                    spatialMergeSize: tower.spatialMergeSize, maximumVisionTokens: 8_000)
                let featurePath = try XCTUnwrap(probe.image_features)
                let features = try MLX.loadArrays(url: root.appending(path: featurePath))
                let grid = try XCTUnwrap(features["grid"]).asArray(Int32.self).map(Int.init)
                XCTAssertEqual(grid, [prepared.gridTHW.0, prepared.gridTHW.1, prepared.gridTHW.2])
                let embeddings = try XCTUnwrap(features["image_embeddings"])
                replacements = [Q35VisionReplacement(embeddings: embeddings, gridTHW: prepared.gridTHW)]
            }
            let ids = try template.encodeForGeneration(
                messages: [ChatMessage(role: .user, content: probe.prompt, imageUrl: imagePath)],
                includeThinking: true, maxLength: 8192,
                imageTokenCounts: replacements.map { $0.embeddings.dim(0) })
            XCTAssertEqual(ids, probe.prompt_ids)
            let input = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            var embeddings = model.embeddings(for: input)
            var positions: Q35MRoPEPositionData?
            if !replacements.isEmpty {
                let imageID = try XCTUnwrap(config.imageTokenId)
                embeddings = insertVisionEmbeddings(hiddenStates: embeddings, inputIds: input,
                    imageTokenId: imageID, replacements: replacements)
                positions = try buildMRoPEPositionData(inputIds: input, imageTokenId: imageID,
                    replacements: replacements, spatialMergeSize: try XCTUnwrap(visionTower).spatialMergeSize)
            }
            let caches = makeLayerCaches(config: config)
            let prefill = try await chunkedPrefillEmbeddings(model: model, inputIds: input,
                inputEmbeddings: embeddings, cache: caches, positionIds: positions?.positionIds,
                retainHidden: false, progressHandler: nil)
            MLX.eval(prefill.logits)
            StreamOrDevice.default.stream.synchronize()
            var greedy: [Int] = []
            greedy.reserveCapacity(count)
            let tokens = Array(probe.continuation.prefix(count))
            let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
            let start = DispatchTime.now().uptimeNanoseconds
            for (index, token) in tokens.enumerated() {
                let next = MLXArray([Int32(token)]).reshaped(1, 1)
                let nextPositions = positions.map {
                    MLXArray.full([3, 1, 1], values: MLXArray(Int32(ids.count + index + $0.ropeDelta)))
                }
                let decoded = model.forward(next, cache: caches, positionIds: nextPositions)
                greedy.append(Int(MLX.argMax(decoded.logits[0, -1, 0...].asType(.float32)).item(Int32.self)))
            }
            StreamOrDevice.default.stream.synchronize()
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
            let record = MatchedBonsaiRecord(modality: probe.id, repetition: repetition, tokenCount: count,
                promptIDs: ids, continuation: tokens, argmax: greedy, seconds: seconds,
                thermalBefore: thermalBefore, thermalAfter: ProcessInfo.processInfo.thermalState.rawValue)
            print("BONSAI_MATCHED \(probe.id) repetition=\(repetition) tokens=\(count) seconds=\(seconds)")
            return record
        }
    }
}
