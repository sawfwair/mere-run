import Foundation
import MLX
import MLXNN
import MLXRandom

final class HiDreamO1Model: Module {
    @ModuleInfo(key: "language_model") var languageModel: QwenEncoder
    @ModuleInfo(key: "vision_tower") var visionTower: QwenVisionTower
    @ModuleInfo(key: "pixel_head") var pixelHead: HiDreamO1PixelHead

    init(config: HiDreamO1Config) {
        let visionActivation: QwenVisionConfiguration.Activation = switch config.visionConfig.hiddenAct.lowercased() {
        case "gelu_pytorch_tanh", "gelu_tanh", "gelu":
            .geluApproximate
        default:
            .silu
        }
        self._languageModel.wrappedValue = QwenEncoder(configuration: .init(
            vocabSize: config.textConfig.vocabSize,
            hiddenSize: config.textConfig.hiddenSize,
            numHiddenLayers: config.textConfig.numHiddenLayers,
            numAttentionHeads: config.textConfig.numAttentionHeads,
            numKeyValueHeads: config.textConfig.numKeyValueHeads,
            intermediateSize: config.textConfig.intermediateSize,
            ropeTheta: config.textConfig.ropeTheta,
            maxPositionEmbeddings: config.textConfig.maxPositionEmbeddings,
            rmsNormEps: config.textConfig.rmsNormEps,
            headDim: config.textConfig.headDim,
            mropeSection: config.textConfig.ropeScaling.mropeSection,
            mropeInterleaved: config.textConfig.ropeScaling.mropeInterleaved
        ))
        self._visionTower.wrappedValue = QwenVisionTower(configuration: .init(
            depth: config.visionConfig.depth,
            embedDim: config.visionConfig.hiddenSize,
            mlpHiddenDim: config.visionConfig.intermediateSize,
            hiddenAct: visionActivation,
            numHeads: config.visionConfig.numHeads,
            eps: 1e-6,
            patchSize: config.visionConfig.patchSize,
            temporalPatchSize: config.visionConfig.temporalPatchSize,
            spatialMergeSize: config.visionConfig.spatialMergeSize,
            inChannels: config.visionConfig.inChannels,
            outHiddenDim: config.visionConfig.outHiddenSize,
            windowSize: 112,
            fullAttentionBlockIndices: [],
            patchEmbedBias: true,
            numPositionEmbeddings: config.visionConfig.numPositionEmbeddings,
            useLearnedPosEmbed: true,
            deepstackVisualIndexes: config.visionConfig.deepstackVisualIndexes
        ))
        self._pixelHead.wrappedValue = HiDreamO1PixelHead(config: config)
        super.init()
    }

    func forward(
        inputIds: MLXArray,
        positionIds: MLXArray,
        vinputs: MLXArray,
        timestep: MLXArray,
        tokenTypes: MLXArray,
        visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor] = []
    ) throws -> MLXArray {
        let textEmbeddings = try preparedTextEmbeddings(
            inputIds: inputIds,
            timestep: timestep,
            visionConditions: visionConditions
        )
        let visualEmbeddings = pixelHead.patchEmbeddings(vinputs[0]).expandedDimensions(axis: 0)
        let embeddings = MLX.concatenated([textEmbeddings, visualEmbeddings], axis: 1)
        let hiddenStates = languageModel.forward(
            embeddings: embeddings,
            attentionMask: nil,
            positionIds: positionIds,
            tokenTypes: tokenTypes,
            outputHiddenStates: false
        ).lastHiddenState
        return pixelHead.pixelPredictions(hiddenStates)
    }

    private func preparedTextEmbeddings(
        inputIds: MLXArray,
        timestep: MLXArray,
        visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor]
    ) throws -> MLXArray {
        var embeddings = languageModel.embed(inputIds: inputIds)
        if !visionConditions.isEmpty {
            embeddings = try imagePadReplacedTextEmbeddings(
                embeddings,
                inputIds: inputIds,
                visionConditions: visionConditions
            )
        }
        let timestepEmbedding = pixelHead.timestepEmbeddings(timestep).asType(embeddings.dtype)
        let expandedTimestep = MLX.broadcast(
            timestepEmbedding.expandedDimensions(axis: 1),
            to: embeddings.shape
        )
        let mask = MLX.broadcast(
            (inputIds .== MLXArray(Int32(HiDreamO1TokenizerAndTemplate.tmsTokenId)))
                .expandedDimensions(axis: 2),
            to: embeddings.shape
        )
        return MLX.where(mask, expandedTimestep, embeddings)
    }

    private func imagePadReplacedTextEmbeddings(
        _ embeddings: MLXArray,
        inputIds: MLXArray,
        visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor]
    ) throws -> MLXArray {
        let replacementList = try visionConditions.map { condition in
            try visionTower(patchInputs: condition.pixelValues, grid: [condition.grid]).hiddenStates
        }
        let replacement = replacementList.count == 1
            ? replacementList[0]
            : MLX.concatenated(replacementList, axis: 0)
        return replaceImagePadTokens(
            embeddings,
            inputIds: inputIds,
            replacement: replacement.asType(embeddings.dtype)
        )
    }

    private func replaceImagePadTokens(
        _ embeddings: MLXArray,
        inputIds: MLXArray,
        replacement: MLXArray
    ) -> MLXArray {
        let batch = embeddings.dim(0)
        let seqLen = embeddings.dim(1)
        let replacementCount = replacement.dim(0)

        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        let result = embeddings
        for row in 0..<batch {
            let base = row * seqLen
            var positions: [Int] = []
            positions.reserveCapacity(replacementCount)
            for position in 0..<seqLen
                where tokenValues[base + position] == Int32(HiDreamO1TokenizerAndTemplate.imagePadTokenId) {
                positions.append(position)
            }

            precondition(
                positions.count == replacementCount,
                "[HiDreamO1] image token mismatch in row \(row): found \(positions.count), expected \(replacementCount)"
            )

            guard let start = positions.first else { continue }
            let end = start + replacementCount
            let isContiguous = positions.enumerated().allSatisfy { offset, value in
                value == start + offset
            }

            if isContiguous {
                result[row, start..<end, 0...] = replacement
            } else {
                let idx = MLXArray(positions.map(Int32.init))
                result[row, idx, 0...] = replacement
            }
        }
        return result
    }
}

enum HiDreamO1ModelLoader {
    static func load(
        resources: HiDreamO1Resources,
        config: HiDreamO1Config,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> HiDreamO1Model {
        let model = HiDreamO1Model(config: config)
        try HFSafetensorsWeightsLoader.applyShardedWeights(
            indexURL: resources.weightsIndexURL,
            to: model,
            dtype: .bfloat16,
            verify: .shapeMismatch,
            mapper: mapKey,
            progressHandler: { shard in
                progressHandler?(GenerationProgress(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex + 1,
                    totalSteps: shard.shardCount
                ))
            }
        )
        return model
    }

    private static func mapKey(_ key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("model.language_model.") {
            return [("language_model." + key.dropPrefix("model.language_model."), value)]
        }
        if key.hasPrefix("model.visual.") {
            let mappedKey = mapVisionKey("vision_tower." + key.dropPrefix("model.visual."))
            if key == "model.visual.patch_embed.proj.weight" {
                return [(mappedKey, value.transposed(0, 2, 3, 4, 1))]
            }
            return [(mappedKey, value)]
        }
        if key.hasPrefix("model.t_embedder1.") {
            return [("pixel_head.t_embedder1." + key.dropPrefix("model.t_embedder1."), value)]
        }
        if key.hasPrefix("model.x_embedder.") {
            return [("pixel_head.x_embedder." + key.dropPrefix("model.x_embedder."), value)]
        }
        if key.hasPrefix("model.final_layer2.") {
            return [("pixel_head.final_layer2." + key.dropPrefix("model.final_layer2."), value)]
        }
        return []
    }

    private static func mapVisionKey(_ rawKey: String) -> String {
        var key = rawKey
        key = key.replacingOccurrences(of: ".merger.", with: ".patch_merger.")
        key = key.replacingOccurrences(of: ".patch_merger.norm.", with: ".patch_merger.ln_q.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc1.", with: ".patch_merger.mlp_0.")
        key = key.replacingOccurrences(of: ".patch_merger.linear_fc2.", with: ".patch_merger.mlp_2.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc1.", with: ".mlp.fc1.")
        key = key.replacingOccurrences(of: ".mlp.linear_fc2.", with: ".mlp.fc2.")
        if key.contains(".deepstack_merger_list.") {
            key = key.replacingOccurrences(of: ".norm.", with: ".ln_q.")
            key = key.replacingOccurrences(of: ".linear_fc1.", with: ".mlp_0.")
            key = key.replacingOccurrences(of: ".linear_fc2.", with: ".mlp_2.")
        }
        return key
    }
}

enum HiDreamO1Denoiser {
    static let noiseScale: Float = 8.0
    static let timestepEpsilon: Float = 0.001

    struct Conditioning {
        var sample: HiDreamO1SampleBuilder.Sample
        var visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor]
    }

    private struct PreparedSample {
        let sample: HiDreamO1SampleBuilder.Sample
        let inputIds: MLXArray
        let positionIds: MLXArray
        let tokenTypes: MLXArray
        let selection: MLXArray
        let visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor]

        init(conditioning: Conditioning) {
            self.sample = conditioning.sample
            self.inputIds = MLXArray(conditioning.sample.inputIds.map(Int32.init), [1, conditioning.sample.inputIds.count])
            self.positionIds = HiDreamO1Denoiser.positionIdsArray(conditioning.sample.positionIds)
            self.tokenTypes = MLXArray(conditioning.sample.tokenTypes.map(Int32.init), [1, conditioning.sample.tokenTypes.count])
            let selectedPositions = conditioning.sample.vinputMask.enumerated().compactMap { index, keep in
                keep ? Int32(index) : nil
            }
            self.selection = MLXArray(selectedPositions)
            self.visionConditions = conditioning.visionConditions
        }
    }

    static func initialPatches(
        height: Int,
        width: Int,
        seed: UInt64
    ) -> MLXArray {
        let image = MLXRandom.normal(
            [3, height, width],
            key: MLXRandom.key(seed &+ 1)
        ).asType(.bfloat16) * MLXArray(noiseScale).asType(.bfloat16)
        return HiDreamO1SampleBuilder.patchifyCHW(image).expandedDimensions(axis: 0)
    }

    static func runTextOnly(
        model: HiDreamO1Model,
        sample: HiDreamO1SampleBuilder.Sample,
        scheduler: HiDreamO1Scheduler,
        height: Int,
        width: Int,
        seed: UInt64,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        try run(
            model: model,
            conditioning: Conditioning(sample: sample, visionConditions: []),
            unconditionalConditioning: nil,
            scheduler: scheduler,
            height: height,
            width: width,
            seed: seed,
            referencePatches: nil,
            guidanceScale: 0,
            progressHandler: progressHandler
        )
    }

    static func runWithReferences(
        model: HiDreamO1Model,
        sample: HiDreamO1SampleBuilder.Sample,
        scheduler: HiDreamO1Scheduler,
        height: Int,
        width: Int,
        seed: UInt64,
        referenceTensors: [HiDreamO1ImagePreprocessor.PatchTensor],
        visionConditions: [HiDreamO1ImagePreprocessor.VisionConditionTensor],
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        try runWithReferences(
            model: model,
            conditioning: Conditioning(sample: sample, visionConditions: visionConditions),
            unconditionalConditioning: nil,
            scheduler: scheduler,
            height: height,
            width: width,
            seed: seed,
            referenceTensors: referenceTensors,
            guidanceScale: 0,
            progressHandler: progressHandler
        )
    }

    static func runTextOnly(
        model: HiDreamO1Model,
        conditioning: Conditioning,
        unconditionalConditioning: Conditioning?,
        scheduler: HiDreamO1Scheduler,
        height: Int,
        width: Int,
        seed: UInt64,
        guidanceScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        try run(
            model: model,
            conditioning: conditioning,
            unconditionalConditioning: unconditionalConditioning,
            scheduler: scheduler,
            height: height,
            width: width,
            seed: seed,
            referencePatches: nil,
            guidanceScale: guidanceScale,
            progressHandler: progressHandler
        )
    }

    static func runWithReferences(
        model: HiDreamO1Model,
        conditioning: Conditioning,
        unconditionalConditioning: Conditioning?,
        scheduler: HiDreamO1Scheduler,
        height: Int,
        width: Int,
        seed: UInt64,
        referenceTensors: [HiDreamO1ImagePreprocessor.PatchTensor],
        guidanceScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        let referencePatches: MLXArray?
        if referenceTensors.isEmpty {
            referencePatches = nil
        } else {
            let patches = referenceTensors.map { $0.patches.asType(.bfloat16) }
            referencePatches = (patches.count == 1 ? patches[0] : MLX.concatenated(patches, axis: 0))
                .expandedDimensions(axis: 0)
        }
        return try run(
            model: model,
            conditioning: conditioning,
            unconditionalConditioning: unconditionalConditioning,
            scheduler: scheduler,
            height: height,
            width: width,
            seed: seed,
            referencePatches: referencePatches,
            guidanceScale: guidanceScale,
            progressHandler: progressHandler
        )
    }

    private static func run(
        model: HiDreamO1Model,
        conditioning: Conditioning,
        unconditionalConditioning: Conditioning?,
        scheduler: HiDreamO1Scheduler,
        height: Int,
        width: Int,
        seed: UInt64,
        referencePatches: MLXArray?,
        guidanceScale: Float,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) throws -> MLXArray {
        let steps = scheduler.timesteps.dim(0)
        let preparedConditioning = PreparedSample(conditioning: conditioning)
        let preparedUnconditional = unconditionalConditioning.map(PreparedSample.init(conditioning:))
        let usesCFG = guidanceScale > 1.0 && preparedUnconditional != nil
        var uniPCState = scheduler.usesFlashStep ? nil : HiDreamO1UniPCState(scheduler: scheduler)
        var z = initialPatches(height: height, width: width, seed: seed)

        progressHandler?(GenerationProgress(stage: .denoising, stepIndex: 0, totalSteps: steps))
        for stepIndex in 0..<steps {
            let timestep = scheduler.timestep(at: stepIndex)
            let sigma = MLX.maximum(timestep / MLXArray(1_000.0), MLXArray(timestepEpsilon))
            let tPixelDiT = MLXArray(1.0) - timestep / MLXArray(1_000.0)
            let vinputs = referencePatches.map { MLX.concatenated([z, $0], axis: 1) } ?? z
            let conditionalVelocity = try predictedVelocity(
                model: model,
                prepared: preparedConditioning,
                vinputs: vinputs,
                timestep: tPixelDiT,
                sigma: sigma,
                z: z
            )
            let guidedVelocity: MLXArray
            if usesCFG, let preparedUnconditional {
                let unconditionalVelocity = try predictedVelocity(
                    model: model,
                    prepared: preparedUnconditional,
                    vinputs: vinputs,
                    timestep: tPixelDiT,
                    sigma: sigma,
                    z: z
                )
                guidedVelocity = unconditionalVelocity
                    + (conditionalVelocity - unconditionalVelocity) * MLXArray(guidanceScale)
            } else {
                guidedVelocity = conditionalVelocity
            }

            let modelOutput = -guidedVelocity
            if scheduler.usesFlashStep {
                z = flashStep(
                    modelOutput: modelOutput,
                    sample: z,
                    sigma: scheduler.sigma(at: stepIndex),
                    sigmaNext: scheduler.sigma(at: stepIndex + 1),
                    noiseScale: scheduler.noiseScales[stepIndex],
                    seed: seed &+ UInt64(stepIndex + 10_000)
                ).asType(.bfloat16)
            } else {
                z = uniPCState!.step(
                    modelOutput: modelOutput.asType(.float32),
                    sample: z.asType(.float32),
                    stepIndex: stepIndex
                ).asType(.bfloat16)
            }
            MLX.eval(z)
            progressHandler?(GenerationProgress(stage: .denoising, stepIndex: stepIndex + 1, totalSteps: steps))
        }
        return z[0]
    }

    private static func predictedVelocity(
        model: HiDreamO1Model,
        prepared: PreparedSample,
        vinputs: MLXArray,
        timestep: MLXArray,
        sigma: MLXArray,
        z: MLXArray
    ) throws -> MLXArray {
        let xPred = try model.forward(
            inputIds: prepared.inputIds,
            positionIds: prepared.positionIds,
            vinputs: vinputs,
            timestep: timestep.reshaped(1),
            tokenTypes: prepared.tokenTypes,
            visionConditions: prepared.visionConditions
        )
        let selected = xPred[0, prepared.selection, 0...]
        let predicted = selected[0..<prepared.sample.targetImageLength, 0...].expandedDimensions(axis: 0)
        return (predicted.asType(.float32) - z.asType(.float32)) / sigma.asType(.float32)
    }

    private static func flashStep(
        modelOutput: MLXArray,
        sample: MLXArray,
        sigma: MLXArray,
        sigmaNext: MLXArray,
        noiseScale: Float,
        seed: UInt64
    ) -> MLXArray {
        let sampleF32 = sample.asType(.float32)
        let denoised = sampleF32 - modelOutput.asType(.float32) * sigma.asType(.float32)
        guard sigmaNext.item(Float.self) > 0 else {
            return denoised
        }
        let noise = MLXRandom.normal(sample.shape, key: MLXRandom.key(seed)).asType(.float32)
        return sigmaNext.asType(.float32) * noise * MLXArray(noiseScale) + (1.0 - sigmaNext.asType(.float32)) * denoised
    }

    private static func positionIdsArray(_ positionIds: [[Int]]) -> MLXArray {
        let axes = positionIds.count
        let length = positionIds.first?.count ?? 0
        let values = positionIds.flatMap { $0.map(Int32.init) }
        return MLXArray(values, [axes, 1, length])
    }
}

private extension String {
    func dropPrefix(_ prefix: String) -> String {
        String(dropFirst(prefix.count))
    }
}
