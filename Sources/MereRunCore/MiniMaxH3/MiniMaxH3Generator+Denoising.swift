import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3Generator {
    func denoise(
        transformer: MiniMaxH3Transformer,
        videoRows initialVideoRows: MLXArray,
        audioRows initialAudioRows: MLXArray,
        conditionVideoRows: MLXArray?,
        conditionAudioRows: MLXArray?,
        promptStates: MLXArray,
        layout: MiniMaxH3PackedLayout,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        adaLNCache: MiniMaxH3AdaLNCache?,
        accelerationMode: MiniMaxH3AccelerationMode,
        permitsCacheReuse: Bool,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> (videoRows: MLXArray, audioRows: MLXArray) {
        precondition(videoSchedule.timesteps.count == audioSchedule.timesteps.count)
        let blockProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_BLOCKS"],
            prefix: "[minimax-h3-profile]"
        )
        let fastVSAProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_FASTVSA"],
            prefix: "[minimax-h3-fastvsa-profile]"
        )
        let stepProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_STEPS"],
            prefix: "[minimax-h3-step-profile]"
        )
        blockProfileLogger?("rows=\(layout.sequenceLength) blocks=\(transformer.configuration.layerCount)")
        fastVSAProfileLogger?(
            "rows=\(layout.sequenceLength) blocks=\(transformer.configuration.layerCount) "
                + "kernel=\(MiniMaxH3FastVSAKernelMode.runtimeDefault.rawValue)"
        )
        stepProfileLogger?(
            "rows=\(layout.sequenceLength) steps=\(videoSchedule.timesteps.count) "
                + "resident_bf16=\(transformer.usesResidentBF16)"
        )
        transformer.blockTimingHandler = blockProfileLogger.map { logger in
            { index, elapsed, memory in
                logger(String(
                    format: "block=%02d seconds=%.3f active_gib=%.2f cache_gib=%.2f peak_gib=%.2f",
                    index,
                    elapsed,
                    Double(memory.activeMemory) / 1_073_741_824,
                    Double(memory.cacheMemory) / 1_073_741_824,
                    Double(memory.peakMemory) / 1_073_741_824
                ))
            }
        }
        transformer.fastH3BlockPhaseTimingHandler = fastVSAProfileLogger.map { logger in
            { index, projection, attention, postAttention in
                logger(String(
                    format: "block=%02d projection=%.3f attention=%.3f post_attention=%.3f",
                    index,
                    projection,
                    attention,
                    postAttention
                ))
            }
        }
        let context = transformer.prepare(textStates: promptStates, layout: layout)
        var videoRows = initialVideoRows
        var audioRows = initialAudioRows
        MLX.eval(videoRows, audioRows)
        if let conditionVideoRows { MLX.eval(conditionVideoRows) }
        if let conditionAudioRows { MLX.eval(conditionAudioRows) }

        // The practical H3 tier uses independent compiled block boundaries to
        // bound the graph, reuse intermediate buffers, and hold steadier clocks
        // during a sustained denoise pass. Smaller graphs can still amortize a
        // compiled whole-step transform; very large graphs also stay blockwise
        // to avoid the macOS watchdog.
        let environment = ProcessInfo.processInfo.environment
        let exactKernelMode = try MiniMaxH3ExactKernelMode.resolveForRuntime(
            environmentValue: environment["MERERUN_H3_EXACT_KERNELS"],
            usesFastH3VSA: transformer.usesFastH3VSA,
            supportsAffineQ8ExactKernels: transformer.supportsAffineQ8ExactKernels,
            usesResidentBF16: transformer.usesResidentBF16
        )
        if exactKernelMode != .disabled {
            guard accelerationMode == .quality else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "exact H3 kernels must be measured with h3 acceleration quality"
                )
            }
        }
        if exactKernelMode.usesAffineQ8FeedForward {
            guard transformer.supportsAffineQ8ExactKernels,
                  !transformer.usesResidentBF16 else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "affine Q8 exact kernels require the managed Q8/group-64 transformer"
                )
            }
        }
        transformer.exactKernelMode = exactKernelMode
        let usesScheduledTailCache = environment["MERERUN_H3_CACHE_STRATEGY"] == "scheduled-tail"
        let requestedReuseStreak = Int(environment["MERERUN_H3_REUSE_STREAK"] ?? "")
        let adaptiveCachePolicy = !permitsCacheReuse || usesScheduledTailCache
            ? nil
            : accelerationMode.adaptiveFirstBlockCachePolicy.map { policy in
                let globalThreshold = Float(environment["MERERUN_H3_FIRST_BLOCK_THRESHOLD"] ?? "")
                    .flatMap { $0 > 0 ? $0 : nil }
                    ?? policy.globalThreshold
                let temporalThreshold = Float(
                    environment["MERERUN_H3_FIRST_BLOCK_TEMPORAL_THRESHOLD"] ?? ""
                ).flatMap { $0 > 0 ? $0 : nil }
                    ?? policy.temporalThreshold
                let maximumConsecutiveCachedSteps = requestedReuseStreak.flatMap {
                    $0 >= 0 ? $0 : nil
                } ?? policy.maximumConsecutiveCachedSteps
                return MiniMaxH3AdaptiveFirstBlockCachePolicy(
                    globalThreshold: globalThreshold,
                    temporalThreshold: temporalThreshold,
                    window: policy.window,
                    maximumConsecutiveCachedSteps: maximumConsecutiveCachedSteps,
                    minimumFullSteps: policy.minimumFullSteps,
                    requiredFinalFullSteps: policy.requiredFinalFullSteps
                )
            }
        let blockReusePolicy = permitsCacheReuse && usesScheduledTailCache
            ? accelerationMode.blockReusePolicy.map { policy in
                let requestedCacheDepth = Double(environment["MERERUN_H3_REUSE_DEPTH"] ?? "")
                let cacheDepth = requestedCacheDepth.flatMap {
                    (0..<1).contains($0) ? $0 : nil
                } ?? policy.cacheDepth
                let maximumConsecutiveCachedSteps = requestedReuseStreak.flatMap {
                    $0 >= 0 ? $0 : nil
                } ?? policy.maximumConsecutiveCachedSteps
                return MiniMaxH3BlockReusePolicy(
                    cacheDepth: cacheDepth,
                    window: policy.window,
                    maximumConsecutiveCachedSteps: maximumConsecutiveCachedSteps
                )
            } : nil
        let configuredDynamicSparseAttentionPolicy = environment["MERERUN_H3_DYNAMIC_SPARSE"] == "0"
            ? nil
            : accelerationMode.dynamicSparseAttentionPolicy.map { policy in
                DynamicSparseAttentionPolicy(
                    thresholdStandardDeviations: Float(
                        environment["MERERUN_H3_DYNAMIC_SPARSE_TAU"] ?? ""
                    ).flatMap { $0 >= 0 ? $0 : nil } ?? policy.thresholdStandardDeviations,
                    minimumSequenceLength: policy.minimumSequenceLength,
                    denseLeadingStepFraction: policy.denseLeadingStepFraction,
                    denseTrailingStepCount: policy.denseTrailingStepCount,
                    denseLeadingLayerCount: policy.denseLeadingLayerCount
                )
            }
        let dynamicSparseAttentionPolicy = configuredDynamicSparseAttentionPolicy.flatMap { policy in
            layout.sequenceLength >= policy.minimumSequenceLength ? policy : nil
        }
        let velocityReusePolicy = accelerationMode.velocityReusePolicy
        let layerThinningPolicy = accelerationMode.layerThinningPolicy
        let tokenReductionPolicy = accelerationMode.tokenReductionPolicy
        if let layerThinningPolicy {
            guard let adaLNCache else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "layer thinning requires a compatible precomputed AdaLN table"
                )
            }
            transformer.activeBlockIndices = Set(layerThinningPolicy.activeBlockIndices(
                blockModulations: adaLNCache.blockModulations
            ))
        } else {
            transformer.activeBlockIndices = nil
        }
        let tokenReduction = tokenReductionPolicy.map { _ in
            transformer.prepareTokenReduction(context: context)
        }
        let executionMode: MiniMaxH3DenoiseExecutionMode
        if transformer.usesFastH3VSA {
            executionMode = environment["MERERUN_H3_EXECUTION_MODE"]?.lowercased() == "eager"
                ? .eagerStep
                : .blockwiseCompiled
        } else if exactKernelMode.requiresEagerExecution(sequenceLength: layout.sequenceLength) {
            executionMode = .eagerStep
        } else if layerThinningPolicy == nil,
                  velocityReusePolicy == nil,
                  tokenReductionPolicy == nil,
                  adaptiveCachePolicy == nil,
                  blockReusePolicy == nil,
                  dynamicSparseAttentionPolicy == nil {
            executionMode = MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: transformer.usesResidentBF16,
                sequenceLength: layout.sequenceLength,
                usesBlockProfiling: blockProfileLogger != nil,
                denoiseStepCount: videoSchedule.timesteps.count,
                profilingOverride: ProcessInfo.processInfo.environment["MERERUN_H3_EXECUTION_MODE"]
            )
        } else {
            executionMode = .blockwiseCompiled
        }
        let usesCompiledStep = executionMode == .compiledStep
        transformer.usesBlockwiseCompilation = executionMode == .blockwiseCompiled
        let attentionKernelSchedule = MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(
            sequenceLength: layout.sequenceLength
        )
        transformer.maximumAttentionQueryTokensPerKernel = max(
            1,
            Int(environment["MERERUN_H3_ATTENTION_QUERY_TOKENS"] ?? "")
                ?? attentionKernelSchedule.maximumQueryTokens
        )
        transformer.maximumAttentionHeadsPerKernel = Int(
            environment["MERERUN_H3_ATTENTION_HEADS_PER_KERNEL"] ?? ""
        ).map { max(1, $0) } ?? attentionKernelSchedule.maximumHeadsPerKernel
        transformer.maximumAttentionKernelsPerEvaluation = max(
            1,
            Int(environment["MERERUN_H3_ATTENTION_EVALUATION_BATCH"] ?? "")
                ?? attentionKernelSchedule.maximumKernelsPerEvaluation
        )
        transformer.dynamicSparseAttentionPolicy = dynamicSparseAttentionPolicy
        transformer.dynamicSparseAttentionStepCount = videoSchedule.timesteps.count
        transformer.dynamicSparseAttentionLogHandler = stepProfileLogger
        transformer.usesFusedPostAttention = ProcessInfo.processInfo
            .environment["MERERUN_H3_FUSED_POST_ATTENTION"] == "1"
        transformer.usesLayerwiseEvaluation = executionMode.usesLayerwiseEvaluation
        transformer.clearsCacheAfterLayerwiseEvaluation = executionMode == .eagerStep
        let attentionHeadsPerKernel = transformer.maximumAttentionHeadsPerKernel
            ?? transformer.configuration.attentionHeadCount
        stepProfileLogger?(
            "execution_mode=\(executionMode) acceleration=\(accelerationMode.rawValue) "
                + "exact_kernels=\(exactKernelMode.rawValue) "
                + "fastvsa_kernel=\(MiniMaxH3FastVSAKernelMode.runtimeDefault.rawValue) "
                + "fused_post_attention=\(transformer.usesFusedPostAttention) "
                + "attention_query_tokens=\(transformer.maximumAttentionQueryTokensPerKernel) "
                + "attention_heads_per_kernel=\(attentionHeadsPerKernel) "
                + "attention_evaluation_batch="
                + "\(transformer.maximumAttentionKernelsPerEvaluation) "
                + "dynamic_sparse=\(dynamicSparseAttentionPolicy != nil) "
                + "velocity_reuse_interval=\(velocityReusePolicy?.interval ?? 0) "
                + "token_reduction=\(tokenReductionPolicy != nil) "
                + "active_blocks=\(transformer.activeBlockCount)"
        )
        if let tokenReductionPolicy, let tokenReduction {
            stepProfileLogger?(
                "token_reduction_begin=\(tokenReductionPolicy.beginBlock) "
                    + "token_reduction_end=\(tokenReductionPolicy.endBlock) "
                    + "token_reduction_early_steps=\(tokenReductionPolicy.earlyStepCount) "
                    + "token_reduction_early_end=\(tokenReductionPolicy.earlyEndBlock) "
                    + "full_rows=\(layout.sequenceLength) "
                    + "reduced_rows=\(tokenReduction.reducedContext.layout.sequenceLength)"
            )
        }
        if let dynamicSparseAttentionPolicy {
            stepProfileLogger?(
                "dynamic_sparse_tau=\(dynamicSparseAttentionPolicy.thresholdStandardDeviations) "
                    + "dense_step_fraction=\(dynamicSparseAttentionPolicy.denseLeadingStepFraction) "
                    + "dense_trailing_steps=\(dynamicSparseAttentionPolicy.denseTrailingStepCount) "
                    + "dense_leading_layers=\(dynamicSparseAttentionPolicy.denseLeadingLayerCount) "
                    + "prefix_sink=\(layout.targetVideoRows.lowerBound)"
            )
        }
        if let blockReusePolicy {
            stepProfileLogger?(
                "cache_strategy=scheduled-tail "
                    + "reuse_depth=\(blockReusePolicy.cacheDepth) "
                    + "reuse_streak=\(blockReusePolicy.maximumConsecutiveCachedSteps)"
            )
        }
        if let adaptiveCachePolicy {
            stepProfileLogger?(
                "cache_strategy=adaptive-first-block "
                    + "global_threshold=\(adaptiveCachePolicy.globalThreshold) "
                    + "temporal_threshold=\(adaptiveCachePolicy.temporalThreshold) "
                    + "reuse_streak=\(adaptiveCachePolicy.maximumConsecutiveCachedSteps) "
                    + "final_full_steps=\(adaptiveCachePolicy.requiredFinalFullSteps)"
            )
        }

        let compiledStep = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            let videoSample = inputs[0]
            let audioSample = inputs[1]
            let timestepValues = inputs[2]
            let videoCoefficients = inputs[3]
            let audioCoefficients = inputs[4]
            let videoInput = conditionVideoRows.map {
                MLX.concatenated([$0, videoSample], axis: 1)
            } ?? videoSample
            let audioInput = conditionAudioRows.map {
                MLX.concatenated([$0, audioSample], axis: 1)
            } ?? audioSample
            let predicted = transformer(
                videoRows: videoInput,
                audioRows: audioInput,
                context: context,
                timesteps: timestepValues,
                cachedAdaLN: adaLNCache.map { _ in
                    let blockStart = 6
                    let blockEnd = blockStart + transformer.configuration.layerCount
                    return MiniMaxH3AdaLNStep(
                        timeEmbedding: inputs[5],
                        blockModulations: Array(inputs[blockStart..<blockEnd]),
                        finalModulation: inputs[blockEnd]
                    )
                }
            )
            return [
                Self.advance(
                    sample: videoSample,
                    velocity: predicted.videoVelocityRows,
                    coefficients: videoCoefficients
                ),
                Self.advance(
                    sample: audioSample,
                    velocity: predicted.audioVelocityRows,
                    coefficients: audioCoefficients
                ),
            ]
        }

        let totalBlockCount = transformer.activeBlockCount
        let warmBlockCount = blockReusePolicy?.warmBlockCount(totalBlockCount: totalBlockCount)
        var cachedTailResidual: MLXArray?
        var previousFirstResidual: MLXArray?
        var cachedTargetTailResidual: MLXArray?
        var consecutiveCachedSteps = 0
        var cachedStepCount = 0
        var fullStepCount = 0
        var executedBlockCount = 0
        var previousVideoVelocity: MLXArray?
        var previousAudioVelocity: MLXArray?
        var cachedVideoVelocity: MLXArray?
        var cachedAudioVelocity: MLXArray?
        var previousVelocityStepIndex: Int?
        var cachedVelocityStepIndex: Int?

        for index in videoSchedule.timesteps.indices {
            transformer.dynamicSparseAttentionStepIndex = index
            let stepStarted = stepProfileLogger.map { _ in CFAbsoluteTimeGetCurrent() }
            progressHandler?(.init(
                stage: .denoising,
                stepIndex: index,
                totalSteps: videoSchedule.timesteps.count
            ))
            let videoTimestep = videoSchedule.timesteps[index]
            let audioTimestep = audioSchedule.timesteps[index]
            let timestepValues = MLXArray([
                videoTimestep,
                audioTimestep,
                max(videoTimestep, 0.999),
            ])
            let videoCoefficients = Self.scheduleCoefficients(videoSchedule, index: index)
            let audioCoefficients = Self.scheduleCoefficients(audioSchedule, index: index)
            let reusesTail = blockReusePolicy?.shouldReuseTail(
                stepIndex: index,
                stepCount: videoSchedule.timesteps.count,
                videoSigmas: videoSchedule.sigmas,
                audioSigmas: audioSchedule.sigmas,
                hasCachedResidual: cachedTailResidual != nil,
                consecutiveCachedSteps: consecutiveCachedSteps
            ) ?? false
            var cacheHitThisStep = false
            var executedBlocksThisStep = totalBlockCount
            var firstBlockChange: MiniMaxH3FirstBlockChange?
            let reusesVelocity = velocityReusePolicy?.shouldReuse(
                stepIndex: index,
                stepCount: videoSchedule.timesteps.count,
                hasCachedVelocity: cachedVideoVelocity != nil && cachedAudioVelocity != nil
            ) ?? false
            if reusesVelocity,
               let cachedVideoVelocity,
               let cachedAudioVelocity,
               let cachedVelocityStepIndex {
                let previousVideoSigma = previousVelocityStepIndex.map {
                    videoSchedule.sigmas[$0]
                }
                let previousAudioSigma = previousVelocityStepIndex.map {
                    audioSchedule.sigmas[$0]
                }
                let videoRatio = MiniMaxH3ServingContract.extrapolationRatio(
                    currentSigma: videoSchedule.sigmas[index],
                    lastSigma: videoSchedule.sigmas[cachedVelocityStepIndex],
                    previousSigma: previousVideoSigma
                )
                let audioRatio = MiniMaxH3ServingContract.extrapolationRatio(
                    currentSigma: audioSchedule.sigmas[index],
                    lastSigma: audioSchedule.sigmas[cachedVelocityStepIndex],
                    previousSigma: previousAudioSigma
                )
                let videoVelocity = previousVideoVelocity.map {
                    cachedVideoVelocity + videoRatio * (cachedVideoVelocity - $0)
                } ?? cachedVideoVelocity
                let audioVelocity = previousAudioVelocity.map {
                    cachedAudioVelocity + audioRatio * (cachedAudioVelocity - $0)
                } ?? cachedAudioVelocity
                videoRows = Self.advance(
                    sample: videoRows,
                    velocity: videoVelocity,
                    coefficients: videoCoefficients
                )
                audioRows = Self.advance(
                    sample: audioRows,
                    velocity: audioVelocity,
                    coefficients: audioCoefficients
                )
                cacheHitThisStep = true
                executedBlocksThisStep = 0
                cachedStepCount += 1
            } else if usesCompiledStep {
                var inputs = [
                    videoRows,
                    audioRows,
                    timestepValues,
                    videoCoefficients,
                    audioCoefficients,
                ]
                if let cacheStep = adaLNCache?.step(at: index) {
                    inputs.append(cacheStep.timeEmbedding)
                    inputs.append(contentsOf: cacheStep.blockModulations)
                    inputs.append(cacheStep.finalModulation)
                }
                let outputs = compiledStep(inputs)
                videoRows = outputs[0]
                audioRows = outputs[1]
                fullStepCount += 1
            } else {
                let videoInput = conditionVideoRows.map {
                    MLX.concatenated([$0, videoRows], axis: 1)
                } ?? videoRows
                let audioInput = conditionAudioRows.map {
                    MLX.concatenated([$0, audioRows], axis: 1)
                } ?? audioRows
                let predicted: MiniMaxH3TransformerOutput
                if let tokenReductionPolicy, let tokenReduction {
                    predicted = transformer.callWithTokenReduction(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        reduction: tokenReduction,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        policy: tokenReductionPolicy,
                        stepIndex: index
                    )
                    fullStepCount += 1
                    stepProfileLogger?(
                        "step_plan=\(index + 1)/\(videoSchedule.timesteps.count) "
                            + "token_reduction_blocks=\(tokenReductionPolicy.beginBlock)..<"
                            + "\(tokenReductionPolicy.restoreBeforeBlock(stepIndex: index)) "
                            + "rows=\(tokenReduction.reducedContext.layout.sequenceLength)"
                    )
                } else if let adaptiveCachePolicy {
                    let canConsiderReuse = adaptiveCachePolicy.canConsiderReuse(
                        stepIndex: index,
                        stepCount: videoSchedule.timesteps.count,
                        fullStepCount: fullStepCount,
                        consecutiveCachedSteps: consecutiveCachedSteps,
                        hasCachedState: previousFirstResidual != nil && cachedTargetTailResidual != nil
                    )
                    let result = transformer.callWithAdaptiveFirstBlockReuse(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        policy: adaptiveCachePolicy,
                        canConsiderReuse: canConsiderReuse,
                        previousFirstResidual: previousFirstResidual,
                        cachedTargetTailResidual: cachedTargetTailResidual
                    )
                    predicted = result.output
                    firstBlockChange = result.change
                    if result.reusedTail {
                        cacheHitThisStep = true
                        executedBlocksThisStep = 1
                        consecutiveCachedSteps += 1
                        cachedStepCount += 1
                    } else if let refreshedFirstResidual = result.refreshedFirstResidual,
                              let refreshedTargetTailResidual = result.refreshedTargetTailResidual {
                        previousFirstResidual = refreshedFirstResidual
                        cachedTargetTailResidual = refreshedTargetTailResidual
                        consecutiveCachedSteps = 0
                        fullStepCount += 1
                    } else {
                        preconditionFailure("adaptive MiniMax-H3 cache did not return refreshed state")
                    }
                } else if let warmBlockCount {
                    let result = transformer.callWithBlockResidualReuse(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        warmBlockCount: warmBlockCount,
                        cachedTailResidual: reusesTail ? cachedTailResidual : nil
                    )
                    predicted = result.output
                    if let refreshedTailResidual = result.refreshedTailResidual {
                        cachedTailResidual = refreshedTailResidual
                        consecutiveCachedSteps = 0
                        fullStepCount += 1
                    } else {
                        cacheHitThisStep = true
                        executedBlocksThisStep = warmBlockCount
                        consecutiveCachedSteps += 1
                        cachedStepCount += 1
                    }
                } else {
                    predicted = transformer(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index)
                    )
                    fullStepCount += 1
                }
                if velocityReusePolicy != nil {
                    previousVideoVelocity = cachedVideoVelocity
                    previousAudioVelocity = cachedAudioVelocity
                    previousVelocityStepIndex = cachedVelocityStepIndex
                    cachedVideoVelocity = predicted.videoVelocityRows
                    cachedAudioVelocity = predicted.audioVelocityRows
                    cachedVelocityStepIndex = index
                }
                videoRows = Self.advance(
                    sample: videoRows,
                    velocity: predicted.videoVelocityRows,
                    coefficients: videoCoefficients
                )
                audioRows = Self.advance(
                    sample: audioRows,
                    velocity: predicted.audioVelocityRows,
                    coefficients: audioCoefficients
                )
            }
            executedBlockCount += executedBlocksThisStep
            if adaptiveCachePolicy != nil || blockReusePolicy != nil || velocityReusePolicy != nil {
                var plan = "step_plan=\(index + 1)/\(videoSchedule.timesteps.count) "
                    + "cache_hit=\(cacheHitThisStep) blocks=\(executedBlocksThisStep)"
                if let firstBlockChange {
                    plan += String(
                        format: " video_global=%.5f audio_global=%.5f "
                            + "video_temporal=%.5f audio_temporal=%.5f",
                        firstBlockChange.videoGlobal,
                        firstBlockChange.audioGlobal,
                        firstBlockChange.videoTemporalMaximum,
                        firstBlockChange.audioTemporalMaximum
                    )
                }
                stepProfileLogger?(plan)
            }
            MLX.eval(videoRows, audioRows)
            if let stepStarted, let stepProfileLogger {
                let memory = Memory.snapshot()
                stepProfileLogger(String(
                    format: "step=%02d/%02d seconds=%.3f active_gib=%.2f cache_gib=%.2f peak_gib=%.2f",
                    index + 1,
                    videoSchedule.timesteps.count,
                    CFAbsoluteTimeGetCurrent() - stepStarted,
                    Double(memory.activeMemory) / 1_073_741_824,
                    Double(memory.cacheMemory) / 1_073_741_824,
                    Double(memory.peakMemory) / 1_073_741_824
                ))
            }
        }
        if adaptiveCachePolicy != nil || blockReusePolicy != nil || velocityReusePolicy != nil {
            stepProfileLogger?(
                "cached_steps=\(cachedStepCount)/\(videoSchedule.timesteps.count) "
                    + "full_steps=\(fullStepCount) executed_blocks=\(executedBlockCount) "
                    + "baseline_blocks=\(videoSchedule.timesteps.count * totalBlockCount)"
            )
        }
        return (videoRows, audioRows)
    }

}
