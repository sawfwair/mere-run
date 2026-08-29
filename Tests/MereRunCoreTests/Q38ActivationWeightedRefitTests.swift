import Foundation
import MLX
import MLXRandom
import XCTest
@testable import MereRunCore

final class Q38ActivationWeightedRefitTests: MereRunCoreTestCase {
    func testProfilerInstallsAcrossArrayBackedLayersWithoutReplacingWeights() throws {
        try XCTSkipUnless(!Q35FusedSwitchGLUPolicy.enabled, "Profiler installation requires fusion disabled.")
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen4_exp","quantization":{"bits":4,"group_size":64},"text_config":{
          "model_type":"qwen4_exp_text","hidden_size":128,"intermediate_size":64,
          "num_hidden_layers":2,"layer_types":["linear_attention","linear_attention"],
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "num_experts":4,"num_experts_per_tok":2,"norm_topk_prob":true,
          "moe_intermediate_size":64,"shared_expert_intermediate_size":64,
          "linear_num_key_heads":2,"linear_num_value_heads":2,
          "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
          "max_position_embeddings":1024,"vocab_size":32,"rms_norm_eps":0.000001,
          "attention_bias":false,"attention_dropout":0,
          "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}}}
        """#.utf8))
        let model = Q35Model(config: config)
        let original = Dictionary(uniqueKeysWithValues: model.leafModules().flattened().compactMap { path, module in
            (module as? Q35SwitchLinear).map { (path, ObjectIdentifier($0.weight)) }
        })
        Q38ExpertActivationProfile().install(on: model)
        let observed = model.leafModules().flattened().compactMap { path, module in
            (module as? Q38ExpertActivationProfile.ObservedExpert).map { (path, ObjectIdentifier($0.weight)) }
        }
        XCTAssertEqual(observed.count, 6)
        XCTAssertEqual(Dictionary(uniqueKeysWithValues: observed), original)
    }

    func testMomentsAccumulateRepeatedRoutes() {
        var moments = Q38ExpertActivationProfile.Moments(experts: 3, input: 2)
        moments.observe(MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped([3, 2]),
                        indices: MLXArray([Int32(0), 1, 0]))
        moments.observe(MLXArray([Float(2), 3]).reshaped([1, 2]), indices: MLXArray([Int32(1)]))
        XCTAssertEqual(moments.count.asArray(Float.self), [2, 2, 0])
        XCTAssertEqual(moments.squaredSum.asArray(Float.self), [26, 40, 13, 25, 0, 0])
    }

    func testImportanceBalancesModalitiesAndLeavesUnobservedExpertsZero() throws {
        let result = try Q38ExpertActivationProfile.importance([
            "projection.image.squared_sum": MLXArray([Float(40), 80, 0, 0]).reshaped([2, 2]),
            "projection.image.count": MLXArray([Float(10), 0]),
            "projection.text.squared_sum": MLXArray([Float(2), 5, 0, 0]).reshaped([2, 2]),
            "projection.text.count": MLXArray([Float(1), 0]),
        ], path: "projection")
        XCTAssertEqual(result.asArray(Float.self), [6, 13, 0, 0])
    }

    func testRefitKeepsPackedCodesAndNeverIncreasesWeightedGroupError() throws {
        MLXRandom.seed(741)
        let dense = MLXRandom.normal([4, 64, 128]).asType(.bfloat16)
        let q4 = MLX.quantized(dense, groupSize: 64, bits: 4)
        let source = Q35SwitchLinear(weight: q4.0, scales: q4.1, biases: q4.2,
                                    bias: nil, groupSize: 64, bits: 4)
        let q3 = try Q38ExpertRequantization.q3Group64.arrays(from: source)
        let importance = MLX.concatenated([
            MLXArray.zeros([1, 128]), MLX.exp(MLXRandom.normal([3, 128]) * 2),
        ], axis: 0)
        let sourceCodes = source.weight.asArray(UInt32.self)
        let result = try Q38ActivationWeightedRefit.arrays(source: source, candidate: q3, importance: importance)
        XCTAssertEqual(result.0.asArray(UInt32.self), q3.0.asArray(UInt32.self))
        XCTAssertEqual(source.weight.asArray(UInt32.self), sourceCodes)
        XCTAssertEqual(result.1.dtype, .bfloat16)
        XCTAssertEqual(result.2.dtype, .bfloat16)
        XCTAssertEqual(result.1[0].asArray(Float.self), q3.1[0].asArray(Float.self))
        XCTAssertEqual(result.2[0].asArray(Float.self), q3.2[0].asArray(Float.self))
        let teacher = MLX.dequantized(q4.0, scales: q4.1, biases: q4.2, groupSize: 64, bits: 4,
                                     dtype: .bfloat16).asType(.float32)
        func error(_ arrays: (MLXArray, MLXArray, MLXArray)) -> MLXArray {
            let difference = teacher - MLX.dequantized(arrays.0, scales: arrays.1, biases: arrays.2,
                                                       groupSize: 64, bits: 3, dtype: .float32)
            return (difference * difference * importance.expandedDimensions(axis: 1))
                .reshaped([4, 64, 2, 64]).sum(axis: -1)
        }
        XCTAssertLessThanOrEqual((error(result) - error(q3)).max().item(Float.self), 0.0001)
        XCTAssertLessThan(error(result).sum().item(Float.self), error(q3).sum().item(Float.self))
    }

    func testDegenerateGroupsKeepOriginalParameters() {
        let scales = MLXArray([Float(0.5), 0]).asType(.bfloat16)
        let biases = MLXArray([Float(1), 2]).asType(.bfloat16)
        let result = Q38ActivationWeightedRefit.fit(
            teacher: MLXArray.ones([2, 64]), codes: MLXArray.ones([2, 64]),
            importance: MLXArray.ones([2, 64]), scales: scales, biases: biases
        )
        XCTAssertEqual(result.scales.asArray(Float.self), scales.asArray(Float.self))
        XCTAssertEqual(result.biases.asArray(Float.self), biases.asArray(Float.self))
    }

    func testObservationPreservesTeacherOutputAndCountsEachDownRouteOnce() throws {
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Routed observation requires the GPU.")
        MLXRandom.seed(742)
        let q4 = MLX.quantized(MLXRandom.normal([4, 64, 128]).asType(.bfloat16), groupSize: 64, bits: 4)
        let source = Q35SwitchLinear(weight: q4.0, scales: q4.1, biases: q4.2,
                                    bias: nil, groupSize: 64, bits: 4)
        let profile = Q38ExpertActivationProfile()
        let observed = Q38ExpertActivationProfile.ObservedExpert(source: source, path: "projection", profile: profile)
        let indices = MLXArray([Int32(0), 1, 0]).reshaped([1, 1, 3])
        for shape in [[1, 1, 128], [1, 1, 3, 128]] {
            let input = MLXRandom.normal(shape).asType(.bfloat16)
            let actual = observed(input, indices: indices)
            let expected = source(input, indices: indices)
            XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("moments.safetensors")
        try profile.save(to: url, metadata: [:])
        let saved = try MLX.loadArrays(url: url)
        XCTAssertEqual(saved["projection.text.count"]?.asArray(Float.self), [4, 2, 0, 0])
        XCTAssertNil(saved["projection.image.count"])
    }
}
