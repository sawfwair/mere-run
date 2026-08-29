import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class LTX25NativeModelPackTests: MereRunCoreTestCase {
    func testRealNativePackCoversV2TransformerParameters() throws {
        guard let rootPath = ProcessInfo.processInfo.environment["MERERUN_LTX25_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_LTX25_MODEL_ROOT for real-checkpoint coverage.")
        }
        let resources = LTX25Resources(rootURL: URL(fileURLWithPath: rootPath))
        let packURL = try XCTUnwrap(
            LTX25NativeModelPack.optimizedURLIfValid(resources: resources, kind: .dev)
        )
        let packed = try SafetensorsStreamingLoader.metadata(url: packURL)
        let expected = ltx25UnifiedTransformerParameterShapes()

        XCTAssertEqual(Set(packed.keys).subtracting(expected.keys), [])
        XCTAssertEqual(Set(expected.keys).subtracting(packed.keys), [])
        for (key, shape) in expected {
            XCTAssertEqual(packed[key]?.shape, shape, "Shape mismatch for \(key)")
        }
    }

    func testRealNativePackMLXValuesMatchOfficialCheckpoint() throws {
        guard let rootPath = ProcessInfo.processInfo.environment["MERERUN_LTX25_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_LTX25_MODEL_ROOT for real-checkpoint value parity.")
        }
        let resources = LTX25Resources(rootURL: URL(fileURLWithPath: rootPath))
        let packURL = try XCTUnwrap(
            LTX25NativeModelPack.optimizedURLIfValid(resources: resources, kind: .dev)
        )
        let officialMetadata = try SafetensorsStreamingLoader.metadata(
            url: resources.devTransformerURL
        )
        let mappedOfficial = officialMetadata.compactMap { sourceKey, metadata -> (String, String, Int)? in
            guard let nativeKey = mapUnifiedTransformerKey(sourceKey) else { return nil }
            return (sourceKey, nativeKey, metadata.endOffset - metadata.startOffset)
        }.sorted { lhs, rhs in
            officialMetadata[lhs.0]!.startOffset < officialMetadata[rhs.0]!.startOffset
        }
        let largest = try XCTUnwrap(mappedOfficial.max { $0.2 < $1.2 })
        let samples = [
            try XCTUnwrap(mappedOfficial.first),
            mappedOfficial[mappedOfficial.count / 2],
            largest,
            try XCTUnwrap(mappedOfficial.last),
        ]
        let nativeArrays = try MLX.loadArrays(url: packURL)

        for (sourceKey, nativeKey, _) in samples {
            let sourceArrays = try SafetensorsStreamingLoader.loadArrays(
                url: resources.devTransformerURL,
                where: { $0 == sourceKey }
            )
            let source = try XCTUnwrap(sourceArrays[sourceKey])
            let native = try XCTUnwrap(nativeArrays[nativeKey])
            XCTAssertEqual(native.shape, source.shape, nativeKey)
            XCTAssertEqual(native.dtype, source.dtype, nativeKey)
            XCTAssertTrue(
                MLX.arrayEqual(native, source).item(Bool.self),
                "Value mismatch for \(nativeKey)"
            )
            Memory.clearCache()
        }
    }

    func testRealAppliedTransformerParametersMatchCheckpoint() throws {
        guard ProcessInfo.processInfo.environment["MERERUN_LTX25_APPLIED_WEIGHT_PARITY"] == "1" else {
            throw XCTSkip("Set MERERUN_LTX25_APPLIED_WEIGHT_PARITY=1 for the full applied-weight audit.")
        }
        guard let rootPath = ProcessInfo.processInfo.environment["MERERUN_LTX25_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_LTX25_MODEL_ROOT for the full applied-weight audit.")
        }
        let resources = LTX25Resources(rootURL: URL(fileURLWithPath: rootPath))
        let packURL = try XCTUnwrap(
            LTX25NativeModelPack.optimizedURLIfValid(resources: resources, kind: .dev)
        )

        for url in [packURL, resources.devTransformerURL] {
            let mismatches = try appliedTransformerParameterMismatches(url: url)
            XCTAssertEqual(mismatches, [], "Applied parameter mismatch from \(url.path)")
            Memory.clearCache()
        }
    }

    func testOptimizerStreamsNativeTransformerAndConnectorPacks() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = LTX25Resources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.distilledTransformerURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let expected = MLXArray([Float(1), 2, 3, 4]).reshaped(2, 2).asType(.bfloat16)
        try MLX.save(
            arrays: [
                "model.diffusion_model.patchify_proj.weight": expected,
                "model.diffusion_model.video_embeddings_connector.learnable_registers": MLX.ones(
                    [4, 4],
                    dtype: .bfloat16
                ),
                "unrelated.weight": MLX.ones([8, 8], dtype: .bfloat16),
            ],
            metadata: ["source": "fixture"],
            url: resources.distilledTransformerURL
        )

        let result = try LTX25NativeModelPack.optimize(
            resources: resources,
            kind: .distilled
        )
        XCTAssertEqual(result.tensorCount, 1)
        XCTAssertLessThan(result.packedBytes, result.sourceBytes)
        XCTAssertEqual(
            LTX25NativeModelPack.optimizedURLIfValid(
                resources: resources,
                kind: .distilled
            ),
            result.outputURL
        )
        let packed = try SafetensorsStreamingLoader.loadArrays(url: result.outputURL)
        XCTAssertEqual(Set(packed.keys), Set(["patchify_proj.weight"]))
        MLX.eval(expected, packed["patchify_proj.weight"]!)
        let error = MLX.max(
            MLX.abs(expected.asType(.float32) - packed["patchify_proj.weight"]!.asType(.float32))
        ).item(Float.self)
        XCTAssertEqual(error, 0)

        let connectorResult = try LTX25NativeModelPack.optimize(
            resources: resources,
            kind: .connector
        )
        XCTAssertEqual(connectorResult.tensorCount, 1)
        let connectorPacked = try SafetensorsStreamingLoader.loadArrays(
            url: connectorResult.outputURL
        )
        XCTAssertEqual(
            Set(connectorPacked.keys),
            Set(["model.diffusion_model.video_embeddings_connector.learnable_registers"])
        )
    }

    func testOptimizerRequiresExplicitReplacement() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString,
            isDirectory: true
        )
        defer { try? FileManager.default.removeItem(at: root) }
        let resources = LTX25Resources(rootURL: root)
        try FileManager.default.createDirectory(
            at: resources.distilledTransformerURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try MLX.save(
            arrays: [
                "model.diffusion_model.patchify_proj.bias": MLX.ones(
                    [2],
                    dtype: .bfloat16
                ),
            ],
            url: resources.distilledTransformerURL
        )
        _ = try LTX25NativeModelPack.optimize(resources: resources, kind: .distilled)
        XCTAssertThrowsError(
            try LTX25NativeModelPack.optimize(resources: resources, kind: .distilled)
        ) { error in
            guard case LTX25NativeModelPackError.outputExists = error else {
                return XCTFail("Expected outputExists, got \(error)")
            }
        }
        _ = try LTX25NativeModelPack.optimize(
            resources: resources,
            kind: .distilled,
            replacing: true
        )
    }

    private func appliedTransformerParameterMismatches(url: URL) throws -> [String] {
        let isNative = LTX25NativeModelPack.isNativePack(url)
        var expected = try expectedTransformerArrays(url: url, isNative: isNative)
        let parameters: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: try loadLTX25UnifiedTransformerParametersForValidation(
                url: url,
                dtype: .bfloat16
            )
        )
        var mismatches: [String] = []

        for key in parameters.keys.sorted() {
            guard let parameter = parameters[key],
                  let expectedValue = expected.removeValue(forKey: key) else {
                mismatches.append(key)
                continue
            }
            if parameter.shape != expectedValue.shape
                || parameter.dtype != expectedValue.dtype
                || !MLX.arrayEqual(parameter, expectedValue).item(Bool.self)
            {
                mismatches.append(key)
            }
            Memory.clearCache()
        }
        mismatches.append(contentsOf: expected.keys.sorted())
        return mismatches
    }

    private func expectedTransformerArrays(
        url: URL,
        isNative: Bool
    ) throws -> [String: MLXArray] {
        let arrays = try MLX.loadArrays(url: url)
        if isNative {
            return Dictionary(
                uniqueKeysWithValues: arrays.compactMap { key, value in
                    guard !isLTX25ConnectorTensorKey(key) else { return nil }
                    let casted = value.dtype.isFloatingPoint && value.dtype != .bfloat16
                        ? value.asType(.bfloat16)
                        : value
                    return (key, casted)
                }
            )
        }
        return Dictionary(
            uniqueKeysWithValues: arrays.flatMap { key, value in
                mapUnifiedTransformerWeight(key: key, value: value, dtype: .bfloat16)
            }
        )
    }
}
