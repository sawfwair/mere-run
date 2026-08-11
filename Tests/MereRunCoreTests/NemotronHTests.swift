import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class NemotronHTests: MereRunCoreTestCase {
    func testReleasedTargetConfigDecodesExactHybridContract() throws {
        let config = try JSONDecoder().decode(
            NemotronHConfig.self,
            from: Data(Self.targetConfigJSON.utf8)
        )

        XCTAssertEqual(config.modelType, "nemotron_h")
        XCTAssertEqual(config.layersBlockType.count, 52)
        XCTAssertEqual(config.layersBlockType.filter { $0 == "mamba" }.count, 23)
        XCTAssertEqual(config.layersBlockType.filter { $0 == "attention" }.count, 6)
        XCTAssertEqual(config.layersBlockType.filter { $0 == "moe" }.count, 23)
        XCTAssertEqual(config.mambaNumHeads * config.mambaHeadDim, 4_096)
        XCTAssertEqual(config.nRoutedExperts, 128)
        XCTAssertEqual(config.numExpertsPerToken, 6)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertEqual(config.quantization.mode, "nvfp4")
        XCTAssertTrue(config.quantization.globalScale)
    }

    func testReleasedDSparkConfigDecodesExactSpeculationContract() throws {
        let config = try JSONDecoder().decode(
            NemotronHDSparkConfig.self,
            from: Data(Self.dsparkConfigJSON.utf8)
        )

        XCTAssertEqual(config.architectures, ["Qwen3DSparkModel"])
        XCTAssertEqual(config.numHiddenLayers, 6)
        XCTAssertEqual(config.blockSize, 8)
        XCTAssertEqual(config.markovHeadDim, 512)
        XCTAssertTrue(config.bonusAnchor)
        XCTAssertEqual(config.speculation.maskTokenID, 990)
        XCTAssertEqual(config.speculation.slidingWindow, 1_024)
        XCTAssertEqual(config.speculation.targetLayerIDs, [1, 5, 19, 29, 41, 51])
        XCTAssertEqual(config.eagleAuxHiddenStateLayerIDs, [2, 6, 20, 30, 42, 52])
    }

    func testNVFP4LinearRetainsModelOptGlobalScale() throws {
        MLXRandom.seed(35)
        let dense = MLXRandom.uniform(low: -0.5, high: 0.5, [4, 16])
        let quantized = MLX.quantized(
            dense,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        )
        let layer = NemotronHNVFP4Linear(inputDimensions: 16, outputDimensions: 4)
        let globalScale = MLXArray(0.375, dtype: .float32)
        try layer.update(
            parameters: ModuleParameters.unflattened([
                ("weight", quantized.wq),
                ("scales", quantized.scales),
                ("global_scale", globalScale),
            ]),
            verify: .all
        )
        let input = MLXRandom.uniform(low: -1, high: 1, [1, 1, 16])
        let expected = MLX.quantizedMM(
            input,
            quantized.wq,
            scales: quantized.scales,
            biases: nil,
            transpose: true,
            groupSize: 16,
            bits: 4,
            mode: .nvfp4
        ).asType(.float32) * globalScale
        let actual = layer(input)
        MLX.eval(expected, actual)

        XCTAssertLessThan(
            MLX.max(MLX.abs(expected - actual.asType(.float32))).item(Float.self),
            0.0001
        )
    }

    func testMetalSSMScanMatchesPortableRecurrenceAndChunking() throws {
        #if os(macOS)
        Device.withDefaultDevice(.gpu) {
            MLXRandom.seed(3_505)
            let x = MLXRandom.uniform(low: -0.4, high: 0.4, [1, 5, 2, 4])
            let b = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 5, 1, 32])
            let c = MLXRandom.uniform(low: -0.3, high: 0.3, [1, 5, 1, 32])
            let dt = MLXRandom.uniform(low: -0.5, high: 0.5, [1, 5, 2])
            let aLog = MLXArray([0.0, 0.2] as [Float])
            let d = MLXArray([0.75, 1.25] as [Float])
            let dtBias = MLXArray([-0.1, 0.1] as [Float])

            let portable = nemotronHSSMUpdate(
                x: x,
                b: b,
                c: c,
                dt: dt,
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: nil,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude,
                useMetalKernel: false
            )
            let first = nemotronHSSMUpdate(
                x: x[0..., ..<3, 0..., 0...],
                b: b[0..., ..<3, 0..., 0...],
                c: c[0..., ..<3, 0..., 0...],
                dt: dt[0..., ..<3, 0...],
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: nil,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude
            )
            let second = nemotronHSSMUpdate(
                x: x[0..., 3..., 0..., 0...],
                b: b[0..., 3..., 0..., 0...],
                c: c[0..., 3..., 0..., 0...],
                dt: dt[0..., 3..., 0...],
                aLog: aLog,
                d: d,
                dtBias: dtBias,
                state: first.1,
                minimumTimeStep: 0.001,
                maximumTimeStep: .greatestFiniteMagnitude
            )
            let chunked = MLX.concatenated([first.0, second.0], axis: 1)
            MLX.eval(portable.0, portable.1, chunked, second.1)

            XCTAssertLessThan(
                MLX.max(MLX.abs(portable.0.asType(.float32) - chunked.asType(.float32)))
                    .item(Float.self),
                0.002
            )
            XCTAssertLessThan(
                MLX.max(MLX.abs(portable.1 - second.1)).item(Float.self),
                0.002
            )
        }
        #endif
    }

    func testRejectionDistributionUsesPositiveResidual() {
        let target = MLXArray([0.6, 0.3, 0.1] as [Float])
        let draft = MLXArray([0.2, 0.5, 0.3] as [Float])
        let residual = NemotronHDSparkDecoder.rejectionDistribution(
            target: target,
            draft: draft
        )
        MLX.eval(residual)

        XCTAssertEqual(residual[0].item(Float.self), 1, accuracy: 0.0001)
        XCTAssertEqual(residual[1].item(Float.self), 0, accuracy: 0.0001)
        XCTAssertEqual(residual[2].item(Float.self), 0, accuracy: 0.0001)
        XCTAssertEqual(residual.sum().item(Float.self), 1, accuracy: 0.0001)
    }

    func testDSparkPolicyDefaultsAndEnvironmentOverrides() {
        XCTAssertTrue(NemotronHDSparkPolicy.enabled(environment: [:]))
        XCTAssertFalse(NemotronHDSparkPolicy.enabled(environment: [
            "MERERUN_NEMOTRON35_DSPARK": "off",
        ]))
        XCTAssertEqual(
            NemotronHDSparkPolicy.minimumOutputTokens(environment: [
                "MERERUN_NEMOTRON35_DSPARK_MIN_OUTPUT": "24",
            ]),
            24
        )
        XCTAssertEqual(
            NemotronHDSparkPolicy.minimumAcceptanceRate(environment: [
                "MERERUN_NEMOTRON35_DSPARK_MIN_ACCEPTANCE": "0.72",
            ]),
            0.72,
            accuracy: 0.0001
        )
    }

    func testCatalogConnectsTargetToManagedDSparkCompanion() throws {
        let target = try XCTUnwrap(
            ManagedModelCatalog.spec(for: NemotronHResources.modelID)
        )
        let dspark = try XCTUnwrap(
            ManagedModelCatalog.spec(for: NemotronHResources.dsparkModelID)
        )

        XCTAssertEqual(target.validationKind, .nemotronH)
        XCTAssertEqual(target.defaultRuntimeServingEngine, .textChatNemotronH)
        XCTAssertEqual(target.companionModelIDs, [NemotronHResources.dsparkModelID])
        XCTAssertEqual(target.hubFallback?.repoId, NemotronHResources.artifactRepoID)
        XCTAssertEqual(target.hubFallback?.revision, NemotronHResources.artifactRevision)
        XCTAssertEqual(target.estimatedDownloadBytes, NemotronHResources.estimatedDownloadBytes)
        XCTAssertFalse(target.runtimeAutoDownloadAllowed)
        XCTAssertFalse(ManagedModelCatalog.allSpecs.contains { $0.id == dspark.id })
        XCTAssertEqual(dspark.validationKind, .nemotronHDSpark)
        XCTAssertEqual(dspark.hubFallback?.repoId, NemotronHResources.dsparkArtifactRepoID)
        XCTAssertEqual(
            dspark.hubFallback?.revision,
            NemotronHResources.dsparkArtifactRevision
        )
        XCTAssertEqual(
            dspark.estimatedDownloadBytes,
            NemotronHResources.dsparkEstimatedDownloadBytes
        )
        XCTAssertFalse(dspark.runtimeAutoDownloadAllowed)
    }

    func testManifestTemplatesRetainPinnedNVIDIAProvenance() {
        let target = MereRunModelManifest.template(for: .nemotron35Lightning)
        let dspark = MereRunModelManifest.template(for: .nemotron35LightningDSpark)

        XCTAssertEqual(target.engine, .nemotronH)
        XCTAssertEqual(target.family, .nemotron)
        XCTAssertEqual(
            target.upstreamRepoId,
            "\(NemotronHResources.upstreamRepoID)@\(NemotronHResources.upstreamRevision)"
        )
        XCTAssertEqual(dspark.engine, .nemotronH)
        XCTAssertEqual(
            dspark.upstreamRepoId,
            "\(NemotronHResources.dsparkUpstreamRepoID)@\(NemotronHResources.dsparkUpstreamRevision)"
        )
    }

    private static let targetConfigJSON: String = {
        let types = [
            "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe",
            "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe",
            "mamba", "attention", "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention",
            "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba",
            "moe", "mamba", "moe", "mamba", "moe", "mamba", "attention", "moe", "mamba",
            "moe", "mamba", "moe", "mamba", "moe", "mamba", "moe",
        ].map { "\"\($0)\"" }.joined(separator: ",")
        return """
        {
          "model_type":"nemotron_h","vocab_size":131072,"hidden_size":2688,
          "num_hidden_layers":52,"layers_block_type":[\(types)],
          "num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,
          "max_position_embeddings":1048576,"norm_eps":0.00001,
          "mamba_head_dim":64,"mamba_num_heads":64,"ssm_state_size":128,
          "n_groups":8,"conv_kernel":4,"time_step_min":0.001,"time_step_max":0.1,
          "n_routed_experts":128,"n_shared_experts":1,"num_experts_per_tok":6,
          "moe_intermediate_size":1856,"moe_shared_expert_intermediate_size":3712,
          "routed_scaling_factor":2.5,"norm_topk_prob":true,"n_group":1,"topk_group":1,
          "eos_token_id":2,
          "quantization":{"bits":4,"group_size":16,"mode":"nvfp4","global_scale":true}
        }
        """
    }()

    private static let dsparkConfigJSON = """
    {
      "model_type":"qwen3","architectures":["Qwen3DSparkModel"],
      "vocab_size":131072,"hidden_size":2688,"intermediate_size":6144,
      "num_hidden_layers":6,"num_attention_heads":32,"num_key_value_heads":2,
      "head_dim":128,"max_position_embeddings":1048576,"rms_norm_eps":0.000001,
      "rope_theta":10000,"eagle_aux_hidden_state_layer_ids":[2,6,20,30,42,52],
      "block_size":8,"markov_rank":512,"dspark_bonus_anchor":true,
      "dflash_config":{"attention_sink_bias":true,"causal":true,"mask_token_id":990,
        "swa_window_size":1024,"target_layer_ids":[1,5,19,29,41,51],
        "use_swa":true,"sample_from_anchor":false}
    }
    """
}
