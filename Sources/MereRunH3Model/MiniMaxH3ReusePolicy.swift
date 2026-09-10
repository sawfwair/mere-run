import MLX

package struct MiniMaxH3TokenReductionPolicy: Sendable, Equatable {
    package let beginBlock: Int
    package let endBlock: Int
    package let earlyStepCount: Int
    package let earlyEndBlock: Int
    package let updateScale: Float

    package init(
        beginBlock: Int = 4,
        endBlock: Int = 30,
        earlyStepCount: Int = 10,
        earlyEndBlock: Int = 40,
        updateScale: Float = 1
    ) {
        precondition(beginBlock >= 0)
        precondition(beginBlock < endBlock)
        precondition(endBlock < earlyEndBlock)
        precondition(earlyStepCount > 0)
        precondition(updateScale >= 0 && updateScale <= 2)
        self.beginBlock = beginBlock
        self.endBlock = endBlock
        self.earlyStepCount = earlyStepCount
        self.earlyEndBlock = earlyEndBlock
        self.updateScale = updateScale
    }

    package func restoreBeforeBlock(stepIndex: Int) -> Int {
        stepIndex < earlyStepCount ? earlyEndBlock : endBlock
    }
}

package struct MiniMaxH3LayerThinningPolicy: Sendable, Equatable {
    package let activeBlockCount: Int

    package init(activeBlockCount: Int) {
        precondition(activeBlockCount >= 3)
        self.activeBlockCount = activeBlockCount
    }

    package func activeBlockIndices(blockModulations: [MLXArray]) -> [Int] {
        precondition(activeBlockCount <= blockModulations.count)
        guard activeBlockCount < blockModulations.count else {
            return Array(blockModulations.indices)
        }
        let finalIndex = blockModulations.index(before: blockModulations.endIndex)
        let candidates = blockModulations.indices.filter { index in
            index >= 2 && index < finalIndex
        }
        let scoreArrays = candidates.map { index in
            let parts = MLX.split(blockModulations[index], parts: 6, axis: -1)
            return MLX.mean(MLX.abs(MLX.concatenated([parts[2], parts[5]], axis: -1)).asType(.float32))
        }
        let scoreValues = MLX.stacked(scoreArrays).asArray(Float.self)
        let ranked = zip(candidates, scoreValues).sorted { lhs, rhs in
            lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
        }
        let skippedCount = blockModulations.count - activeBlockCount
        let skipped = Set(ranked.prefix(skippedCount).map(\.0))
        return blockModulations.indices.filter { !skipped.contains($0) }
    }
}

package struct MiniMaxH3VelocityReusePolicy: Sendable, Equatable {
    package let interval: Int
    package let requiredFinalFullSteps: Int

    package init(interval: Int, requiredFinalFullSteps: Int = 1) {
        precondition(interval >= 2)
        precondition(requiredFinalFullSteps >= 1)
        self.interval = interval
        self.requiredFinalFullSteps = requiredFinalFullSteps
    }

    package func shouldReuse(stepIndex: Int, stepCount: Int, hasCachedVelocity: Bool) -> Bool {
        guard stepIndex > 0,
              stepIndex < stepCount,
              hasCachedVelocity,
              stepIndex < stepCount - requiredFinalFullSteps else { return false }
        return !stepIndex.isMultiple(of: interval)
    }
}

package struct MiniMaxH3AdaptiveFirstBlockCachePolicy: Sendable, Equatable {
    package let globalThreshold: Float
    package let temporalThreshold: Float
    package let window: ClosedRange<Float>
    package let maximumConsecutiveCachedSteps: Int
    package let minimumFullSteps: Int
    package let requiredFinalFullSteps: Int

    package init(
        globalThreshold: Float,
        temporalThreshold: Float,
        window: ClosedRange<Float> = 0.1...0.9,
        maximumConsecutiveCachedSteps: Int = 2,
        minimumFullSteps: Int = 2,
        requiredFinalFullSteps: Int = 1
    ) {
        precondition(globalThreshold > 0)
        precondition(temporalThreshold > 0)
        precondition((0...1).contains(window.lowerBound))
        precondition((0...1).contains(window.upperBound))
        precondition(maximumConsecutiveCachedSteps >= 0)
        precondition(minimumFullSteps >= 1)
        precondition(requiredFinalFullSteps >= 1)
        self.globalThreshold = globalThreshold
        self.temporalThreshold = temporalThreshold
        self.window = window
        self.maximumConsecutiveCachedSteps = maximumConsecutiveCachedSteps
        self.minimumFullSteps = minimumFullSteps
        self.requiredFinalFullSteps = requiredFinalFullSteps
    }

    package func canConsiderReuse(
        stepIndex: Int,
        stepCount: Int,
        fullStepCount: Int,
        consecutiveCachedSteps: Int,
        hasCachedState: Bool
    ) -> Bool {
        guard stepIndex >= 0,
              stepIndex < stepCount,
              fullStepCount >= minimumFullSteps,
              hasCachedState,
              consecutiveCachedSteps < maximumConsecutiveCachedSteps,
              stepIndex < stepCount - requiredFinalFullSteps else { return false }
        return window.contains(Float(stepIndex) / Float(stepCount))
    }

    package func shouldReuse(change: MiniMaxH3FirstBlockChange) -> Bool {
        change.isFinite
            && change.videoGlobal <= globalThreshold
            && change.audioGlobal <= globalThreshold
            && change.videoTemporalMaximum <= temporalThreshold
            && change.audioTemporalMaximum <= temporalThreshold
    }
}
