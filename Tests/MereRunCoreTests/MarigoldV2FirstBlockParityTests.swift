import Foundation
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MarigoldV2FirstBlockParityTests: MereRunCoreTestCase {
    func testInstalledFirstBlockWhenRequested() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_MARIGOLD_ROOT"],
              let fixture = environment["MERERUN_TEST_MARIGOLD_BLOCK_FIXTURE"],
              let output = environment["MERERUN_TEST_MARIGOLD_BLOCK_OUTPUT"] else {
            throw XCTSkip("Set the Marigold root, first-block reference fixture, and output variables.")
        }
        let resources = MarigoldV2Resources(rootURL: URL(fileURLWithPath: root))
        let config = try JSONDecoder().decode(QwenImageEditTransformerConfig.self, from: Data("""
        {"num_attention_heads":24,"attention_head_dim":128,"num_layers":1,
         "joint_attention_dim":3584,"in_channels":64,"out_channels":16,"patch_size":2,
         "axes_dims_rope":[16,56,56],"guidance_embeds":false}
        """.utf8))
        let model = MMDiT(config: config)
        let keys = Set(model.parameters().flattened().map(\.0))
        let index = try JSONDecoder().decode(
            HFSafetensorsIndex.self, from: Data(contentsOf: resources.transformerWeightsIndexURL)
        )
        let mapper = QwenImageEditGenerator.transformerWeightKey
        var weights: [String: MLXArray] = [:]
        for file in Set(index.weightMap.values) {
            try SafetensorsStreamingLoader.forEachTensor(
                url: resources.transformerWeightsIndexURL.deletingLastPathComponent().appendingPathComponent(file),
                where: { keys.contains(mapper($0)) }, dtype: .float32
            ) { key, value in weights[mapper(key)] = value }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        weights.removeAll()
        let leaves = model.leafModules().flattened()
        let modules = Dictionary(uniqueKeysWithValues: leaves)
        var replacements: [String: Module] = [:]
        try SafetensorsStreamingLoader.forEachTensorPair(
            url: resources.trainablesURL,
            firstSuffix: MarigoldV2LoRAAdapter.downSuffix, secondSuffix: MarigoldV2LoRAAdapter.upSuffix
        ) { sourcePath, down, up in
            let path = MarigoldV2LoRAAdapter.mappedTargetPath(sourcePath)
            guard let base = modules[path] as? Linear else { return }
            let layer = LoRALinear(base: base, rank: 128, alpha: 128, zeroInitUp: true)
            layer.loraDown = down.asType(.float32)
            layer.loraUp = up.asType(.float32)
            replacements[path] = layer
        }
        XCTAssertEqual(replacements.count, 15)
        Krea2LoRAInjector.applyModuleReplacements(replacements, leafModules: leaves, to: model)
        MLX.eval(model)
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let packed = try XCTUnwrap(arrays["packed"])
        let prompt = try MarigoldV2PromptEmbeddingLoader.load(
            embedsURL: resources.promptEmbedsURL, maskURL: resources.promptMaskURL, dtype: .float32
        )
        let imageStream = model.xEmbedder(packed)
        let contextEmbedder = try XCTUnwrap(model.contextEmbedder)
        let textStream = contextEmbedder(model.textNorm(prompt.embeddings))
        let timestep = model.tEmbedder(MLXArray([Float(0.5)]))
        let rope = model.rope.frequencies(imageShapes: [(temporal: 1, height: 11, width: 16)], textSequenceLength: prompt.embeddings.dim(1))
        let block = model.transformerBlocks[0].forwardJoint(
            x: imageStream, context: textStream, imageConditioning: timestep, textConditioning: timestep,
            xFreqsCis: rope.image, contextFreqsCis: rope.text, attnMask: prompt.mask
        )
        let captured = [
            "packed": packed, "prompt": prompt.embeddings,
            "image_embedded": imageStream, "text_embedded": textStream, "timestep": timestep,
            "rope_image": rope.image, "rope_text": rope.text,
            "block_image": block.image, "block_text": block.context
        ]
        try MLX.save(arrays: captured, url: URL(fileURLWithPath: output))
        for key in captured.keys.sorted() {
            let actual = try XCTUnwrap(captured[key])
            let expected = try XCTUnwrap(arrays[key])
            XCTAssertEqual(actual.shape, expected.shape, key)
            let error = MLX.mean(MLX.abs(actual - expected)).item(Float.self)
            let maximum = MLX.max(MLX.abs(actual - expected)).item(Float.self)
            FileHandle.standardError.write(Data("marigold_block \(key) mae=\(error) max=\(maximum)\n".utf8))
            let errorRMS = MLX.sqrt(MLX.mean((actual - expected).square()))
            let referenceRMS = MLX.sqrt(MLX.mean(expected.square()))
            XCTAssertLessThan((errorRMS / referenceRMS).item(Float.self), 0.001, key)
        }
    }
}
