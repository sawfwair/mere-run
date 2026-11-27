import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class NemotronImageTests: MereRunCoreTestCase {
    func testManagedQ8PinsTheCommunityPackAndRecognizesBothLayouts() throws {
        let id = ManagedModelID.nemotronImageQ8.rawValue
        let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: id))
        XCTAssertEqual(spec.hubFallback?.revision, NemotronImageCommunityPack.revision)
        XCTAssertNotNil(spec.usageRestriction)
        XCTAssertFalse(spec.runtimeAutoDownloadAllowed)
        let template = MereRunModelManifest.template(for: .nemotronImageQ8)
        XCTAssertEqual(try NemotronImageQuantization.expertBits(template), 8)
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for file in NemotronImageCommunityPack.files {
            let path = root.appending(path: file)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1]).write(to: path)
        }
        XCTAssertTrue(spec.missingPaths(in: root).isEmpty)
        try FileManager.default.removeItem(at: root.appending(path: "layers-51.safetensors"))
        XCTAssertEqual(spec.missingPaths(in: root).map(\.lastPathComponent), ["layers-51.safetensors"])
        try FileManager.default.removeItem(at: root.appending(path: "embed.npy"))
        try Data([1]).write(to: root.appending(path: "backbone.embeddings.safetensors"))
        XCTAssertTrue(spec.missingPaths(in: root).contains(root.appending(path: "backbone.layers.51.safetensors")))
    }

    func testLoadsSplitExpertsInNumericOrderAndTransposesDepthwiseConvolution() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = NemotronHConfig(modelType: "nemotron_h", vocabSize: 32, hiddenSize: 8,
            numHiddenLayers: 3, layersBlockType: ["mamba", "moe", "attention"],
            numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 4, maxPositionEmbeddings: 32,
            normEps: 1e-5, mambaHeadDim: 4, mambaNumHeads: 2, ssmStateSize: 4, nGroups: 1,
            convKernel: 3, timeStepMin: 0, timeStepMax: 0.1, nRoutedExperts: 16, nSharedExperts: 1,
            numExpertsPerToken: 2, moeIntermediateSize: 4, sharedExpertIntermediateSize: 8,
            routedScalingFactor: 2.5, normTopKProbability: true, nGroup: 1, topKGroup: 1,
            eosTokenIDs: [2], quantization: .init(bits: 4, groupSize: 16, mode: "nvfp4", globalScale: true))
        let template = NemotronOmniCausalLM(config: config)
        var tensors: [(String, MLXArray)] = []
        for (key, parameter) in template.parameters().flattened() {
            if key.contains(".mixer.experts.") {
                for expert in 0..<config.nRoutedExperts {
                    let name = key.replacingOccurrences(of: ".experts.", with: ".experts.\(expert).")
                    let value = MLXArray.full(Array(parameter.shape.dropFirst()), values: MLXArray(Float(expert)))
                    tensors.append((name, value))
                }
            } else if key.hasSuffix(".conv1d.weight") {
                let shape = parameter.shape
                tensors.append((key, MLXArray(0..<parameter.size).asType(.float32).reshaped(shape[0], shape[2], shape[1])))
            } else {
                tensors.append((key, MLXArray.ones(parameter.shape)))
            }
        }
        var map: [String: String] = [:]
        for (offset, name) in NemotronImageResources.shards.enumerated() {
            let chunk = stride(from: offset, to: tensors.count, by: 32).map { tensors[$0] }
            for (key, _) in chunk { map[key] = name }
            try MLX.save(arrays: Dictionary(uniqueKeysWithValues: chunk), url: root.appending(path: name))
        }
        let index = HFSafetensorsIndex(metadata: nil, weightMap: map)
        try JSONEncoder().encode(index).write(to: root.appending(path: "model.safetensors.index.json"))
        let model = try NemotronImageLoader.generator(at: root, config: config)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let experts = try XCTUnwrap(weights["backbone.layers.1.mixer.experts.up_proj.weight"])
        XCTAssertEqual(experts.dtype, .bfloat16)
        XCTAssertEqual(experts[0..., 0, 0].asType(.float32).asArray(Float.self), (0..<16).map(Float.init))
        let convolution = try XCTUnwrap(weights["backbone.layers.0.mixer.conv1d.weight"])
        XCTAssertEqual(convolution.shape, [16, 3, 1])
        XCTAssertEqual(convolution[1].asType(.float32).asArray(Float.self), [3, 4, 5])
    }

    func testPackedExpertsMatchDensePrefillAndDecode() throws {
        let config = NemotronHConfig(modelType: "nemotron_h", vocabSize: 32, hiddenSize: 64,
            numHiddenLayers: 1, layersBlockType: ["moe"], numAttentionHeads: 2, numKeyValueHeads: 1,
            headDim: 32, maxPositionEmbeddings: 32, normEps: 1e-5, mambaHeadDim: 4, mambaNumHeads: 2,
            ssmStateSize: 4, nGroups: 1, convKernel: 3, timeStepMin: 0, timeStepMax: 0.1,
            nRoutedExperts: 4, nSharedExperts: 1, numExpertsPerToken: 2, moeIntermediateSize: 64,
            sharedExpertIntermediateSize: 64, routedScalingFactor: 2.5, normTopKProbability: true,
            nGroup: 1, topKGroup: 1, eosTokenIDs: [2],
            quantization: .init(bits: 4, groupSize: 16, mode: "nvfp4", globalScale: true))
        let dense = NemotronOmniLanguageBlock(config: config, blockType: "moe")
        let parameters = dense.parameters().flattened().map { key, parameter in
            // Integer levels 0...15 are exactly representable by both affine formats.
            let values = MLXArray((0..<parameter.size).map { Float($0 % 16) / 128 })
            return (key, key == "norm.weight" ? MLXArray.ones(parameter.shape, dtype: .bfloat16)
                    : values.reshaped(parameter.shape).asType(.bfloat16))
        }
        try dense.update(parameters: .unflattened(parameters), verify: .all)
        for bits in [4, 8] {
            let packed = NemotronOmniLanguageBlock(config: config, blockType: "moe", expertBits: bits)
            let weights = NemotronImageQuantization.pack(parameters, bits: bits)
            try NemotronImageQuantization.validate(weights, for: packed)
            try packed.update(parameters: .unflattened(Array(weights)), verify: .all)
            XCTAssertEqual(weights["mixer.experts.up_proj.weight"]?.dtype, .uint32)
            for tokens in [1, 3] {
                let input = MLXArray.ones([1, tokens, 64], dtype: .bfloat16)
                let expected = dense(input, cache: nil)
                let actual = packed(input, cache: nil)
                XCTAssertTrue(allClose(actual, expected, rtol: 0.02).item(Bool.self),
                    "Q\(bits), \(tokens) tokens: actual \(actual.flattened()[0].item(Float.self)), expected \(expected.flattened()[0].item(Float.self))")
            }
        }
    }

    func testUnitronBlockChecksSizeRowsAndVocabularyBeforeDecode() throws {
        let data = Data(#"""
        {"unitron_token_ranges":{"framing":{"<|img_start|>":131072,"<|img_end|>":131073,"<|vis_row|>":131074},
        "vision":{"start_id":131075,"end_id":327635}}}
        """#.utf8)
        let format = try JSONDecoder().decode(NemotronImageTokens.self, from: data)
        try format.validate()
        let start = 131_072, end = 131_073, row = 131_074
        let first = 131_075, last = 327_634
        func parse(_ tokens: [Int], header: String = "height=32 width=16") throws -> [Int32] {
            try format.codes(from: tokens, width: 16, height: 32) { ids in
                XCTAssertEqual(ids, [3, 4])
                return header
            }
        }
        let valid = [7, start, 3, 4, row, first, row, last, end]
        XCTAssertEqual(try parse(valid), [0, 196_559])
        XCTAssertThrowsError(try parse(valid, header: "height=16 width=32"))
        for malformed in [Array(valid.dropLast()), [start, first, row, first, row, last, end],
                          [start, 3, 4, row, first, last, end], [start, 3, 4, row, first, first, last, end],
                          [start, 3, 4, row, first, row, last + 1, end]] {
            XCTAssertThrowsError(try parse(malformed))
        }
    }

    func testManagedPlanAcceptsRecommendedGridsAndRejectsOversizedOrDiffusionRequests() throws {
        let manifest = MereRunModelManifest.template(for: .nemotronImage)
        XCTAssertEqual(try ImageGenerationBackend(manifest: manifest), .nemotronImage)
        var options = ImageGenerationOptions(prompt: "A lighthouse", outputURL: URL(fileURLWithPath: "/tmp/test.png"))
        for (width, height) in [(1_024, 1_024), (1_184, 880), (1_248, 832), (880, 1_184)] {
            options.width = width
            options.height = height
            XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).isEmpty)
        }
        for width in [0, 15, 8_192, Int.max - 15] {
            options.width = width
            XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "dimensions_invalid" })
        }
        options.width = 1_024
        options.steps = 4
        XCTAssertTrue(ImageGenerationPlan.issues(options, manifest: manifest).contains { $0.code == "sampling_unsupported" })
    }
}
