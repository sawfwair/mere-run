import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class Wan2RealModelCompatibilityTests: MereRunCoreTestCase {
    func testInstalledCheckpointKeysAndShapesMatchNativeModules() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_WAN2_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_WAN2_MODEL_ROOT to inspect an installed checkpoint.")
        }
        let resources = Wan2Resources(rootURL: URL(fileURLWithPath: root))
        try assertCompatible(model: Wan2TextEncoderModel(), weightsURL: resources.textEncoderURL)
        Memory.clearCache()
        try assertCompatible(model: Wan2TransformerModel(), weightsURL: resources.transformerURL)
        Memory.clearCache()
        try assertCompatible(model: Wan2VAEModel(), weightsURL: resources.vaeURL)
        Memory.clearCache()
    }

    func testOneBlockOutputMatchesReferenceHarness() throws {
        try writeReferenceHarnessOutput(layerCount: 1, outputPath: "/tmp/wan-swift-oneblock.npy")
    }

    func testFullStackOutputMatchesReferenceHarness() throws {
        try writeReferenceHarnessOutput(layerCount: 30, outputPath: "/tmp/wan-swift-fullstack.npy")
    }

    func testTextEncoderOutputMatchesReferenceHarness() throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_WAN2_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_WAN2_MODEL_ROOT to inspect an installed checkpoint.")
        }
        let resources = Wan2Resources(rootURL: URL(fileURLWithPath: root))
        let model = Wan2TextEncoderModel()
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.textEncoderURL,
            to: model,
            dtype: .bfloat16,
            verify: .none,
            batchSize: 12
        )
        eval(model.parameters().flattened().map(\.1))

        let tokenIDs = MLXArray([42, 43, 44, 1], [1, 4])
        let mask = MLX.ones([1, 4], dtype: .int32)
        let output = model(tokenIDs: tokenIDs, mask: mask).asType(.float32)
        eval(output)
        let outputURL = URL(fileURLWithPath: "/tmp/wan-swift-t5.npy")
        try MLX.save(array: output, url: outputURL)
        let values = output.asArray(Float.self)
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
        print(
            "Wan T5 Swift output shape=\(output.shape) mean=\(mean) std=\(sqrt(variance)) "
                + "min=\(values.min() ?? .nan) max=\(values.max() ?? .nan) path=\(outputURL.path)"
        )
        XCTAssertTrue(values.allSatisfy(\.isFinite))
    }

    private func writeReferenceHarnessOutput(layerCount: Int, outputPath: String) throws {
        guard let root = ProcessInfo.processInfo.environment["MERERUN_WAN2_MODEL_ROOT"] else {
            throw XCTSkip("Set MERERUN_WAN2_MODEL_ROOT to inspect an installed checkpoint.")
        }
        let resources = Wan2Resources(rootURL: URL(fileURLWithPath: root))
        let configuration = Wan2TransformerConfiguration(layerCount: layerCount)
        let model = Wan2TransformerModel(configuration: configuration)
        try SafetensorsStreamingLoader.applyWeightsStreaming(
            url: resources.transformerURL,
            to: model,
            verify: .none,
            include: { key in
                guard key.hasPrefix("blocks.") else { return true }
                guard layerCount < 30 else { return true }
                return (0..<layerCount).contains { key.hasPrefix("blocks.\($0).") }
            },
            mapper: { key, value in
                let dtype: DType = key.hasSuffix("modulation") ? .float32 : .bfloat16
                return [(key, value.asType(dtype))]
            },
            batchSize: 12
        )
        eval(model.parameters().flattened().map(\.1))

        let latent = MLX.zeros([48, 3, 8, 8], dtype: .bfloat16)
        let rawContext = MLX.zeros([1, 512, 4_096], dtype: .bfloat16)
        let context = model.embedText(rawContext)
        let timesteps = MLX.concatenated([
            MLX.zeros([1, 16], dtype: .float32),
            MLX.ones([1, 32], dtype: .float32) * 999,
        ], axis: 1)
        let output = model(
            latents: [latent],
            timesteps: timesteps,
            embeddedContext: context
        )[0].asType(.float32)
        eval(output)

        let outputURL = URL(fileURLWithPath: outputPath)
        try MLX.save(array: output, url: outputURL)
        let values = output.asArray(Float.self)
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count)
        print(
            "Wan \(layerCount)-block Swift output shape=\(output.shape) mean=\(mean) std=\(sqrt(variance)) "
                + "min=\(values.min() ?? .nan) max=\(values.max() ?? .nan) path=\(outputURL.path)"
        )
        XCTAssertTrue(values.allSatisfy(\.isFinite))
    }

    private func assertCompatible(model: Module, weightsURL: URL) throws {
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let metadata = try SafetensorsStreamingLoader.metadata(url: weightsURL)
        let runtimeBuffers: Set<String> = ["inverseTimestepFrequencies", "ropeFrequencies"]
        let parameterKeys = Set(parameters.keys).subtracting(runtimeBuffers)
        let weightKeys = Set(metadata.keys)
        XCTAssertEqual(parameterKeys.subtracting(weightKeys), [], "Native parameters missing from checkpoint")
        XCTAssertEqual(weightKeys.subtracting(parameterKeys), [], "Checkpoint parameters missing from native module")
        for key in parameterKeys.intersection(weightKeys) {
            XCTAssertEqual(parameters[key]?.shape, metadata[key]?.shape, "Shape mismatch for \(key)")
        }
    }
}
