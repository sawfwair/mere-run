import CoreFoundation
import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

extension MiniMaxH3Transformer {
    func prepareBlockInput(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> (hidden: MLXArray, timeEmbedding: MLXArray) {
        let layout = context.layout
        precondition(videoRows.dim(1) == layout.conditionVideoRowCount + layout.targetVideoRows.count)
        precondition(audioRows.dim(1) == layout.conditionAudioRowCount + layout.targetAudioRows.count)
        precondition(timesteps.shape == [3])
        let video = miniMaxH3Linear(videoInput, videoRows).asType(context.text.dtype)
        let audio = miniMaxH3Linear(audioInput, audioRows).asType(context.text.dtype)
        let targetVideo = video[0..., layout.conditionVideoRowCount..., 0...]
        let targetAudio = audio[0..., layout.conditionAudioRowCount..., 0...]
        var packed: [MLXArray] = [context.text]
        for segment in layout.conditionSegments {
            switch segment.modality {
            case .video:
                packed.append(video[0..., segment.sourceRows, 0...])
            case .audio:
                packed.append(audio[0..., segment.sourceRows, 0...])
            case .text:
                preconditionFailure("condition segments cannot contain text")
            }
        }
        packed.append(targetAudio)
        packed.append(targetVideo)
        let hidden = MLX.concatenated(packed, axis: 1)

        let timeEmbedding = cachedAdaLN?.timeEmbedding ?? embedTimesteps(timesteps)
        if let cachedAdaLN {
            precondition(cachedAdaLN.blockModulations.count == blocks.count)
        }
        return (hidden, timeEmbedding)
    }

    func runBlocks(
        _ initialHidden: MLXArray,
        range: Range<Int>,
        context: MiniMaxH3TransformerPreparedContext,
        timeEmbedding: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MLXArray {
        precondition(range.lowerBound >= 0 && range.upperBound <= blocks.count)
        var hidden = initialHidden
        for index in range {
            if let activeBlockIndices, !activeBlockIndices.contains(index) { continue }
            let block = blocks[index]
            let blockStarted = blockTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
            if let compressionGate = fastH3CompressionGates[index] {
                let phaseTimingHandler = fastH3BlockPhaseTimingHandler
                let projectionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                let compiled = usesBlockwiseCompilation
                    ? compiledBlockForward(for: block, compressionGate: compressionGate)
                    : nil
                let projectedAttention: [MLXArray]
                if let compiled, let fastH3Projection = compiled.fastH3AttentionProjection {
                    var projectionInputs = [
                        hidden,
                        timeEmbedding,
                        context.adaLNIndices,
                        context.rope.cosine,
                        context.rope.sine,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        projectionInputs.append(modulation)
                    }
                    projectionInputs.append(contentsOf: compressionGate.parameters)
                    projectedAttention = fastH3Projection(projectionInputs)
                } else {
                    projectedAttention = block.fastH3AttentionProjection(
                        hidden,
                        timeEmbedding: timeEmbedding,
                        adaLNIndices: context.adaLNIndices,
                        rope: context.rope,
                        cachedModulation: cachedAdaLN?.blockModulations[index],
                        compressionGate: compressionGate
                    )
                }
                MLX.eval(projectedAttention)
                let projectionSeconds = projectionStarted.map {
                    CFAbsoluteTimeGetCurrent() - $0
                }
                guard let fastVSA = context.fastVSA else {
                    preconditionFailure("FastH3 VSA prepared context is missing")
                }
                let attentionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                guard let attended = MiniMaxH3FastVSA.call(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    compressionGate: projectedAttention[4],
                    prepared: fastVSA,
                    kernelMode: .runtimeDefault
                ) else {
                    preconditionFailure("FastH3 VSA requires batch-one BF16 attention on Metal")
                }
                if phaseTimingHandler != nil { MLX.eval(attended) }
                let attentionSeconds = attentionStarted.map {
                    CFAbsoluteTimeGetCurrent() - $0
                }
                let postAttentionStarted = phaseTimingHandler.map { _ in CFAbsoluteTimeGetCurrent() }
                if let compiled {
                    var postAttentionInputs = [
                        hidden,
                        attended,
                        projectedAttention[3],
                        timeEmbedding,
                        context.adaLNIndices,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        postAttentionInputs.append(modulation)
                    }
                    if exactKernelMode.usesTiledAffineQ8FeedForward,
                       Self.requiresTiledFeedForwardEvaluationBoundary(
                        rowCount: hidden.dim(1),
                        feedForwardSize: configuration.feedForwardSize,
                        itemSize: hidden.itemSize
                    ) {
                        // MLX buffers use 32-bit byte offsets. Materialize the
                        // compact SwiGLU result before compiling the FC2 stage
                        // when that single activation exceeds 4 GiB.
                        let feedForwardParts = compiled.postAttentionProjection(
                            postAttentionInputs
                        )
                        MLX.eval(feedForwardParts)
                        hidden = compiled.feedForwardOutput(feedForwardParts)[0]
                    } else {
                        hidden = compiled.postAttention(postAttentionInputs)[0]
                    }
                    MLX.eval(hidden)
                } else {
                    hidden = block.postAttention(
                        hidden,
                        attended: attended,
                        gate: projectedAttention[3],
                        timeEmbedding: timeEmbedding,
                        adaLNIndices: context.adaLNIndices,
                        cachedModulation: cachedAdaLN?.blockModulations[index]
                    )
                }
                if let phaseTimingHandler,
                   let projectionSeconds,
                   let attentionSeconds,
                   let postAttentionStarted {
                    MLX.eval(hidden)
                    phaseTimingHandler(
                        index,
                        projectionSeconds,
                        attentionSeconds,
                        CFAbsoluteTimeGetCurrent() - postAttentionStarted
                    )
                }
            } else if usesBlockwiseCompilation {
                var attentionInputs = [
                    hidden,
                    timeEmbedding,
                    context.adaLNIndices,
                    context.rope.cosine,
                    context.rope.sine,
                ]
                if let modulation = cachedAdaLN?.blockModulations[index] {
                    attentionInputs.append(modulation)
                }
                let compiled = compiledBlockForward(for: block)
                let projectedAttention = compiled.attentionProjection(attentionInputs)
                MLX.eval(projectedAttention)
                let dynamicSparseRequest = qualifiedDynamicSparseAttentionRequest(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    layerIndex: index,
                    layout: context.layout
                )
                let attended = block.scaledDotProductAttention(
                    queries: projectedAttention[0],
                    keys: projectedAttention[1],
                    values: projectedAttention[2],
                    maximumQueryTokens: maximumAttentionQueryTokensPerKernel,
                    maximumHeadsPerKernel: maximumAttentionHeadsPerKernel,
                    maximumKernelsPerEvaluation: maximumAttentionKernelsPerEvaluation,
                    dynamicSparseRequest: dynamicSparseRequest
                )
                if usesFusedPostAttention {
                    var postAttentionInputs = [
                        hidden,
                        attended,
                        projectedAttention[3],
                        timeEmbedding,
                        context.adaLNIndices,
                    ]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        postAttentionInputs.append(modulation)
                    }
                    hidden = compiled.postAttention(postAttentionInputs)[0]
                    MLX.eval(hidden)
                } else {
                    hidden = compiled.attentionOutput([hidden, attended, projectedAttention[3]])[0]
                    MLX.eval(hidden)
                    var feedForwardInputs = [hidden, timeEmbedding, context.adaLNIndices]
                    if let modulation = cachedAdaLN?.blockModulations[index] {
                        feedForwardInputs.append(modulation)
                    }
                    let projectedFeedForward = compiled.feedForwardProjection(feedForwardInputs)
                    MLX.eval(projectedFeedForward)
                    hidden = compiled.feedForwardOutput([
                        hidden,
                        projectedFeedForward[0],
                        projectedFeedForward[1],
                    ])[0]
                    MLX.eval(hidden)
                }
            } else {
                hidden = block(
                    hidden,
                    timeEmbedding: timeEmbedding,
                    adaLNIndices: context.adaLNIndices,
                    rope: context.rope,
                    cachedModulation: cachedAdaLN?.blockModulations[index]
                )
            }
            if usesLayerwiseEvaluation || blockTimingHandler != nil {
                MLX.eval(hidden)
                if clearsCacheAfterLayerwiseEvaluation {
                    MLX.Memory.clearCache()
                }
            }
            if let blockStarted, let blockTimingHandler {
                blockTimingHandler(
                    index,
                    CFAbsoluteTimeGetCurrent() - blockStarted,
                    Memory.snapshot()
                )
            }
        }
        return hidden
    }

    func qualifiedDynamicSparseAttentionRequest(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        layerIndex: Int,
        layout: MiniMaxH3PackedLayout
    ) -> DynamicSparseAttentionRequest? {
        guard let request = dynamicSparseAttentionPolicy?.request(
            stepIndex: dynamicSparseAttentionStepIndex,
            stepCount: dynamicSparseAttentionStepCount,
            layerIndex: layerIndex,
            sequenceLength: layout.sequenceLength,
            prefixTokenCount: layout.targetVideoRows.lowerBound
        ) else { return nil }

        let gateShape = "\(queries.dim(1))x\(queries.dim(2))x\(request.prefixTokenCount)"
        let gateKey = "\(gateShape):\(queries.dtype):\(keys.dtype):\(values.dtype)"
        if dynamicSparseAttentionGateResults[gateKey] == nil {
            let gate = DynamicSparseAttention.denseRouteGate(
                queries: queries,
                keys: keys,
                values: values,
                queryStart: request.prefixTokenCount,
                scale: 1 / sqrt(Float(configuration.attentionHeadDimension))
            )
            dynamicSparseAttentionGateResults[gateKey] = gate?.passed ?? false
            if let gate {
                dynamicSparseAttentionLogHandler?(String(
                    format: "dynamic_sparse_gate=%@ shape=%@ max_abs=%.6g mean_abs=%.6g "
                        + "max_rel=%.6g mean_rel=%.6g rel_l2=%.6g",
                    gate.passed ? "pass" : "fail",
                    gateShape,
                    gate.maximumAbsoluteError,
                    gate.meanAbsoluteError,
                    gate.maximumRelativeError,
                    gate.meanRelativeError,
                    gate.relativeL2Error
                ))
            } else {
                dynamicSparseAttentionLogHandler?(
                    "dynamic_sparse_gate=unavailable shape=\(gateShape) "
                        + "q_dtype=\(queries.dtype) k_dtype=\(keys.dtype) "
                        + "v_dtype=\(values.dtype) device="
                        + String(describing: Device.defaultDevice().deviceType)
                )
            }
        }
        return dynamicSparseAttentionGateResults[gateKey] == true ? request : nil
    }

    func finalize(
        _ hidden: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timeEmbedding: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MiniMaxH3TransformerOutput {
        finalLayer(
            hidden,
            timeEmbedding: timeEmbedding,
            videoRows: context.layout.targetVideoRows,
            videoTimeIndex: 0,
            audioRows: context.layout.targetAudioRows,
            audioTimeIndex: 1,
            cachedModulation: cachedAdaLN?.finalModulation
        )
    }

}
