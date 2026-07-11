import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class QwenImageEditRepositoryTests: MereRunCoreTestCase {
    func testCFGExecutionModeParsing() {
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse(nil), .automatic)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("auto"), .automatic)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("on"), .batched)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("batched"), .batched)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("off"), .serial)
        XCTAssertEqual(QwenImageEditCFGExecutionMode.parse("serial"), .serial)
    }

    func testCFGAutoBatchingRequiresUnifiedMemoryHeadroom() {
        let gibibyte = UInt64(1_073_741_824)
        XCTAssertTrue(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 64 * gibibyte,
            activeMemoryBytes: 20 * Int(gibibyte),
            cacheMemoryBytes: 2 * Int(gibibyte),
            isUnifiedMemory: true
        ))
        XCTAssertFalse(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 24 * gibibyte,
            activeMemoryBytes: 20 * Int(gibibyte),
            cacheMemoryBytes: 1 * Int(gibibyte),
            isUnifiedMemory: true
        ))
        XCTAssertFalse(QwenImageEditCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 64 * gibibyte,
            activeMemoryBytes: 0,
            cacheMemoryBytes: 0,
            isUnifiedMemory: false
        ))
        XCTAssertTrue(QwenImageEditCFGExecution.shouldBatch(
            mode: .batched,
            width: 4_096,
            height: 4_096,
            physicalMemoryBytes: gibibyte,
            activeMemoryBytes: Int(gibibyte),
            cacheMemoryBytes: 0,
            isUnifiedMemory: false
        ))
    }

    func testBatchedCFGCombinationPreservesSerialFormula() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 1, 1, 2])
        let combined = QwenImageEditCFGExecution.combinePredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 1, 1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [10, 20])
    }

    func testBatchedCFGCombinationSupportsNonImageRanks() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 2])
        let combined = DiffusionCFGExecution.combinePredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [10, 20])
    }

    func testPositiveAnchoredCFGCombinationPreservesZImageFormula() {
        let predictions = MLXArray([Float(1), 2, 4, 8], [2, 2])
        let combined = DiffusionCFGExecution.combinePositiveAnchoredPredictions(
            predictions,
            guidanceScale: 3
        )
        MLX.eval(combined)

        XCTAssertEqual(combined.shape, [1, 2])
        XCTAssertEqual(combined.asArray(Float.self), [13, 26])
    }

    func testCFGBatchingPairsOnlyShapeCompatibleRows() {
        let unconditional = MLXArray([Float(1), 2, 3, 4], [1, 2, 2])
        let conditional = MLXArray([Float(5), 6, 7, 8], [1, 2, 2])
        let incompatible = MLXArray([Float(1), 2, 3], [1, 3, 1])

        XCTAssertTrue(DiffusionCFGExecution.canPair(unconditional, conditional))
        XCTAssertFalse(DiffusionCFGExecution.canPair(unconditional, incompatible))

        let paired = DiffusionCFGExecution.paired(unconditional, conditional)
        MLX.eval(paired)
        XCTAssertEqual(paired.shape, [2, 2, 2])
        XCTAssertEqual(paired.asArray(Float.self), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    func testModelSpecificHeadroomReserveCanKeepCFGSerial() {
        let gibibyte = DiffusionCFGExecution.gibibyte
        XCTAssertFalse(DiffusionCFGExecution.shouldBatch(
            mode: .automatic,
            width: 1_024,
            height: 1_024,
            physicalMemoryBytes: 32 * gibibyte,
            activeMemoryBytes: 19 * Int(gibibyte),
            cacheMemoryBytes: 2 * Int(gibibyte),
            isUnifiedMemory: true,
            baseReserveBytes: 8 * gibibyte,
            activationBytesPerPixel: 4_096
        ))
    }

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
