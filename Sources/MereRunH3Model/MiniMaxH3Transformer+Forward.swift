import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor

extension MiniMaxH3Transformer {
    public func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        textStates: MLXArray,
        layout: MiniMaxH3PackedLayout,
        videoTimestep: Float,
        audioTimestep: Float,
        conditionVideoTimestep: Float = 0.999
    ) -> MiniMaxH3TransformerOutput {
        let context = prepare(textStates: textStates, layout: layout)
        return self(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: MLXArray([
                videoTimestep,
                audioTimestep,
                max(videoTimestep, conditionVideoTimestep),
            ]),
            cachedAdaLN: nil
        )
    }

    package func prepare(
        textStates: MLXArray,
        layout: MiniMaxH3PackedLayout
    ) -> MiniMaxH3TransformerPreparedContext {
        precondition(textStates.dim(1) == layout.textRows.count)
        let text = tokenRefiner(miniMaxH3Linear(textInput, textStates))
        var timeIndices = Array(repeating: 0, count: layout.sequenceLength)
        for row in layout.conditionRows { timeIndices[row] = 2 }
        for row in layout.targetAudioRows { timeIndices[row] = 1 }
        let adaLNIndices = MLXArray(zip(timeIndices, layout.tokenTags).map { timeIndex, tag in
            Int32(timeIndex * 3) + max(tag, 0)
        })
        let rope = rotaryEmbedding(positions: layout.positions)
        let fastVSA = usesFastH3VSA ? MiniMaxH3FastVSA.prepare(layout: layout) : nil
        MLX.eval(text, adaLNIndices, rope.cosine, rope.sine)
        return MiniMaxH3TransformerPreparedContext(
            text: text,
            adaLNIndices: adaLNIndices,
            rope: rope,
            layout: layout,
            fastVSA: fastVSA
        )
    }

    package func prepareTokenReduction(
        context: MiniMaxH3TransformerPreparedContext
    ) -> MiniMaxH3TokenReductionPreparedContext {
        let map = MiniMaxH3TokenReductionMap(layout: context.layout)
        let firstPositions = MLX.take(
            context.layout.positions[context.layout.targetVideoRows, 0...],
            map.firstVideoIndices,
            axis: 0
        ).asType(.float32)
        let secondPositions = MLX.take(
            context.layout.positions[context.layout.targetVideoRows, 0...],
            map.secondVideoIndices,
            axis: 0
        ).asType(.float32)
        let reducedVideoPositions = (firstPositions + secondPositions) * 0.5
        let reducedPositions = map.targetVideoStart == 0
            ? reducedVideoPositions
            : MLX.concatenated([
                context.layout.positions[0..<map.targetVideoStart, 0...],
                reducedVideoPositions,
            ], axis: 0)
        let reducedAdaLNIndices = MLX.take(
            context.adaLNIndices,
            map.absoluteSourceIndices,
            axis: 0
        )
        let absoluteSources = map.absoluteSourceIndices.asArray(Int32.self).map(Int.init)
        let reducedTags = absoluteSources.map { context.layout.tokenTags[$0] }
        let reducedWidth = ((context.layout.latentWidth / 2 + 1) / 2) * 2
        let reducedLayout = MiniMaxH3PackedLayout(
            positions: reducedPositions,
            tokenTags: reducedTags,
            textRows: context.layout.textRows,
            conditionRows: context.layout.conditionRows,
            conditionSegments: context.layout.conditionSegments,
            conditionVideoRowCount: context.layout.conditionVideoRowCount,
            conditionAudioRowCount: context.layout.conditionAudioRowCount,
            targetAudioRows: context.layout.targetAudioRows,
            targetVideoRows: map.targetVideoStart..<(map.targetVideoStart + map.reducedVideoRowCount),
            videoLatentFrames: context.layout.videoLatentFrames,
            latentHeight: context.layout.latentHeight,
            latentWidth: reducedWidth,
            audioLatentFrames: context.layout.audioLatentFrames
        )
        let reducedRope = rotaryEmbedding(positions: reducedPositions)
        MLX.eval(
            map.firstVideoIndices,
            map.secondVideoIndices,
            map.parentVideoIndices,
            map.absoluteSourceIndices,
            reducedAdaLNIndices,
            reducedRope.cosine,
            reducedRope.sine
        )
        return MiniMaxH3TokenReductionPreparedContext(
            map: map,
            reducedContext: MiniMaxH3TransformerPreparedContext(
                text: context.text,
                adaLNIndices: reducedAdaLNIndices,
                rope: reducedRope,
                layout: reducedLayout,
                fastVSA: nil
            )
        )
    }

    package func callAsFunction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?
    ) -> MiniMaxH3TransformerOutput {
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let hidden = runBlocks(
            prepared.hidden,
            range: blocks.indices,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        return finalize(
            hidden,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
    }

    package func callWithTokenReduction(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        reduction: MiniMaxH3TokenReductionPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        policy: MiniMaxH3TokenReductionPolicy,
        stepIndex: Int
    ) -> MiniMaxH3TransformerOutput {
        let restoreBeforeBlock = policy.restoreBeforeBlock(stepIndex: stepIndex)
        precondition(policy.beginBlock < blocks.count)
        precondition(restoreBeforeBlock <= blocks.count)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let fullBeforeReduction = runBlocks(
            prepared.hidden,
            range: 0..<policy.beginBlock,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let reductionState = reduction.map.pool(fullBeforeReduction)
        let reducedHidden = runBlocks(
            reductionState.reducedHidden,
            range: policy.beginBlock..<restoreBeforeBlock,
            context: reduction.reducedContext,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let restoredHidden = reduction.map.restore(
            reducedHidden,
            state: reductionState,
            updateScale: policy.updateScale
        )
        let finalHidden = runBlocks(
            restoredHidden,
            range: restoreBeforeBlock..<blocks.count,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        return finalize(
            finalHidden,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
    }

    package func callWithBlockResidualReuse(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        warmBlockCount: Int,
        cachedTailResidual: MLXArray?
    ) -> MiniMaxH3BlockReuseResult {
        precondition(warmBlockCount >= 0 && warmBlockCount < blocks.count)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let warmHidden = runBlocks(
            prepared.hidden,
            range: 0..<warmBlockCount,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )

        let hidden: MLXArray
        let refreshedTailResidual: MLXArray?
        if let cachedTailResidual {
            precondition(cachedTailResidual.shape == warmHidden.shape)
            hidden = warmHidden + cachedTailResidual
            refreshedTailResidual = nil
            MLX.eval(hidden)
        } else {
            hidden = runBlocks(
                warmHidden,
                range: warmBlockCount..<blocks.count,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            )
            let residual = hidden - warmHidden
            MLX.eval(residual)
            refreshedTailResidual = residual
        }

        return MiniMaxH3BlockReuseResult(
            output: finalize(
                hidden,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            ),
            refreshedTailResidual: refreshedTailResidual
        )
    }

    package func callWithAdaptiveFirstBlockReuse(
        videoRows: MLXArray,
        audioRows: MLXArray,
        context: MiniMaxH3TransformerPreparedContext,
        timesteps: MLXArray,
        cachedAdaLN: MiniMaxH3AdaLNStep?,
        policy: MiniMaxH3AdaptiveFirstBlockCachePolicy,
        canConsiderReuse: Bool,
        previousFirstResidual: MLXArray?,
        cachedTargetTailResidual: MLXArray?
    ) -> MiniMaxH3AdaptiveBlockReuseResult {
        precondition(blocks.count > 1)
        let prepared = prepareBlockInput(
            videoRows: videoRows,
            audioRows: audioRows,
            context: context,
            timesteps: timesteps,
            cachedAdaLN: cachedAdaLN
        )
        let firstHidden = runBlocks(
            prepared.hidden,
            range: 0..<1,
            context: context,
            timeEmbedding: prepared.timeEmbedding,
            cachedAdaLN: cachedAdaLN
        )
        let targetRows = context.layout.targetAudioRows.lowerBound..<context.layout.targetVideoRows.upperBound
        let initialTarget = prepared.hidden[0..., targetRows, 0...]
        let firstTarget = firstHidden[0..., targetRows, 0...]
        let firstResidual = firstTarget - initialTarget

        let change: MiniMaxH3FirstBlockChange?
        let reusesTail: Bool
        if canConsiderReuse,
           let previousFirstResidual,
           let cachedTargetTailResidual,
           previousFirstResidual.shape == firstResidual.shape,
           cachedTargetTailResidual.shape == firstTarget.shape {
            let measured = MiniMaxH3FirstBlockChange.measure(
                current: firstResidual,
                previous: previousFirstResidual,
                layout: context.layout
            )
            change = measured
            reusesTail = policy.shouldReuse(change: measured)
        } else {
            change = nil
            reusesTail = false
        }

        let hidden: MLXArray
        let refreshedFirstResidual: MLXArray?
        let refreshedTargetTailResidual: MLXArray?
        if reusesTail, let cachedTargetTailResidual {
            let completedTarget = firstTarget + cachedTargetTailResidual
            hidden = targetRows.lowerBound == 0
                ? completedTarget
                : MLX.concatenated([
                    firstHidden[0..., 0..<targetRows.lowerBound, 0...],
                    completedTarget,
                ], axis: 1)
            MLX.eval(hidden)
            refreshedFirstResidual = nil
            refreshedTargetTailResidual = nil
        } else {
            hidden = runBlocks(
                firstHidden,
                range: 1..<blocks.count,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            )
            let targetTailResidual = hidden[0..., targetRows, 0...] - firstTarget
            MLX.eval(firstResidual, targetTailResidual)
            refreshedFirstResidual = firstResidual
            refreshedTargetTailResidual = targetTailResidual
        }

        return MiniMaxH3AdaptiveBlockReuseResult(
            output: finalize(
                hidden,
                context: context,
                timeEmbedding: prepared.timeEmbedding,
                cachedAdaLN: cachedAdaLN
            ),
            refreshedFirstResidual: refreshedFirstResidual,
            refreshedTargetTailResidual: refreshedTargetTailResidual,
            reusedTail: reusesTail,
            change: change
        )
    }

}
