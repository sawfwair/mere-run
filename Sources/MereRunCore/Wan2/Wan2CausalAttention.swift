import MLX
import MLXFast

public struct Wan2CausalTransformerStateSnapshot: Hashable, Sendable {
    public let cachedFrames: Int
    public let globalFrames: Int
    public let localAttentionFrames: Int
    public let sinkFrames: Int
}

public struct Wan2CausalTransformerCheckpoint: @unchecked Sendable {
    fileprivate let blocks: [Wan2CausalBlockCheckpoint]
    public let localAttentionFrames: Int
    public let sinkFrames: Int
    public let cachesProjectiveAttention: Bool
}

public final class Wan2CausalTransformerState: @unchecked Sendable {
    let blocks: [Wan2CausalBlockState]
    public let localAttentionFrames: Int
    public let sinkFrames: Int
    public let cachesProjectiveAttention: Bool

    public init(
        layerCount: Int = 30,
        localAttentionFrames: Int = 12,
        sinkFrames: Int = 3,
        cachesProjectiveAttention: Bool = false
    ) {
        precondition(layerCount > 0)
        precondition(localAttentionFrames > 0)
        precondition(sinkFrames >= 0 && sinkFrames < localAttentionFrames)
        self.localAttentionFrames = localAttentionFrames
        self.sinkFrames = sinkFrames
        self.cachesProjectiveAttention = cachesProjectiveAttention
        self.blocks = (0..<layerCount).map { _ in
            Wan2CausalBlockState(
                localAttentionFrames: localAttentionFrames,
                sinkFrames: sinkFrames,
                cachesProjectiveAttention: cachesProjectiveAttention
            )
        }
    }

    public func reset() {
        blocks.forEach { $0.reset() }
    }

    public func snapshot(spatialTokensPerFrame: Int) -> Wan2CausalTransformerStateSnapshot {
        precondition(spatialTokensPerFrame > 0)
        let cache = blocks[0].selfAttention
        return Wan2CausalTransformerStateSnapshot(
            cachedFrames: cache.cachedTokenCount / spatialTokensPerFrame,
            globalFrames: cache.globalEndToken / spatialTokensPerFrame,
            localAttentionFrames: localAttentionFrames,
            sinkFrames: sinkFrames
        )
    }

    public func checkpoint() -> Wan2CausalTransformerCheckpoint {
        Wan2CausalTransformerCheckpoint(
            blocks: blocks.map { $0.checkpoint() },
            localAttentionFrames: localAttentionFrames,
            sinkFrames: sinkFrames,
            cachesProjectiveAttention: cachesProjectiveAttention
        )
    }

    public func restore(_ checkpoint: Wan2CausalTransformerCheckpoint) {
        precondition(checkpoint.blocks.count == blocks.count)
        precondition(checkpoint.localAttentionFrames == localAttentionFrames)
        precondition(checkpoint.sinkFrames == sinkFrames)
        precondition(checkpoint.cachesProjectiveAttention == cachesProjectiveAttention)
        for (block, saved) in zip(blocks, checkpoint.blocks) {
            block.restore(saved)
        }
    }
}

private struct Wan2CausalBlockCheckpoint: @unchecked Sendable {
    let selfAttention: Wan2CausalKVCheckpoint
    let cameraAttention: Wan2CausalKVCheckpoint
}

final class Wan2CausalBlockState {
    let selfAttention: Wan2CausalKVCache
    let cameraAttention: Wan2CausalKVCache
    let cachesProjectiveAttention: Bool

    init(localAttentionFrames: Int, sinkFrames: Int, cachesProjectiveAttention: Bool) {
        self.cachesProjectiveAttention = cachesProjectiveAttention
        self.selfAttention = Wan2CausalKVCache(
            localAttentionFrames: localAttentionFrames,
            sinkFrames: sinkFrames
        )
        self.cameraAttention = Wan2CausalKVCache(
            localAttentionFrames: localAttentionFrames,
            sinkFrames: sinkFrames
        )
    }

    func reset() {
        selfAttention.reset()
        cameraAttention.reset()
    }


    fileprivate func checkpoint() -> Wan2CausalBlockCheckpoint {
        Wan2CausalBlockCheckpoint(
            selfAttention: selfAttention.checkpoint(),
            cameraAttention: cameraAttention.checkpoint()
        )
    }

    fileprivate func restore(_ checkpoint: Wan2CausalBlockCheckpoint) {
        selfAttention.restore(checkpoint.selfAttention)
        cameraAttention.restore(checkpoint.cameraAttention)
    }
}

private struct Wan2CausalKVCheckpoint: @unchecked Sendable {
    let key: MLXArray?
    let value: MLXArray?
    let globalEndToken: Int
}

struct Wan2CausalCacheWindow {
    let key: MLXArray
    let value: MLXArray
    let frameCount: Int
    let queryFrameIndices: [Int]
}

final class Wan2CausalKVCache {
    let localAttentionFrames: Int
    let sinkFrames: Int
    private var key: MLXArray?
    private var value: MLXArray?
    private(set) var globalEndToken = 0

    init(localAttentionFrames: Int, sinkFrames: Int) {
        self.localAttentionFrames = localAttentionFrames
        self.sinkFrames = sinkFrames
    }

    var cachedTokenCount: Int { key?.dim(1) ?? 0 }

    func reset() {
        key = nil
        value = nil
        globalEndToken = 0
    }

    fileprivate func checkpoint() -> Wan2CausalKVCheckpoint {
        Wan2CausalKVCheckpoint(key: key, value: value, globalEndToken: globalEndToken)
    }

    fileprivate func restore(_ checkpoint: Wan2CausalKVCheckpoint) {
        precondition((checkpoint.key == nil) == (checkpoint.value == nil))
        if let key = checkpoint.key, let value = checkpoint.value {
            precondition(key.shape == value.shape)
        }
        key = checkpoint.key
        value = checkpoint.value
        globalEndToken = checkpoint.globalEndToken
    }

    func update(
        key newKey: MLXArray,
        value newValue: MLXArray,
        currentStartToken: Int,
        spatialTokensPerFrame: Int
    ) -> Wan2CausalCacheWindow {
        precondition(newKey.ndim == 4 && newValue.ndim == 4)
        precondition(newKey.shape == newValue.shape)
        precondition(newKey.dim(1).isMultiple(of: spatialTokensPerFrame))
        precondition(currentStartToken.isMultiple(of: spatialTokensPerFrame))
        let newTokenCount = newKey.dim(1)
        let currentEnd = currentStartToken + newTokenCount
        let maxTokens = localAttentionFrames * spatialTokensPerFrame
        let sinkTokens = sinkFrames * spatialTokensPerFrame

        let mergedKey: MLXArray
        let mergedValue: MLXArray
        if let key, let value {
            if currentStartToken == globalEndToken {
                mergedKey = MLX.concatenated([key, newKey], axis: 1)
                mergedValue = MLX.concatenated([value, newValue], axis: 1)
                globalEndToken = currentEnd
            } else {
                precondition(
                    currentEnd == globalEndToken,
                    "Causal Wan cache only accepts sequential appends or latest-chunk recomputation."
                )
                precondition(newTokenCount <= key.dim(1))
                let start = key.dim(1) - newTokenCount
                let keyCopy = key + MLX.zeros(key.shape, dtype: key.dtype)
                let valueCopy = value + MLX.zeros(value.shape, dtype: value.dtype)
                keyCopy[0..., start..<key.dim(1), 0..., 0...] = newKey
                valueCopy[0..., start..<value.dim(1), 0..., 0...] = newValue
                mergedKey = keyCopy
                mergedValue = valueCopy
            }
        } else {
            precondition(currentStartToken == 0)
            mergedKey = newKey
            mergedValue = newValue
            globalEndToken = currentEnd
        }

        if mergedKey.dim(1) > maxTokens {
            let tailCount = maxTokens - sinkTokens
            self.key = MLX.concatenated([
                mergedKey[0..., 0..<sinkTokens, 0..., 0...],
                mergedKey[0..., (mergedKey.dim(1) - tailCount)..., 0..., 0...]
            ], axis: 1)
            self.value = MLX.concatenated([
                mergedValue[0..., 0..<sinkTokens, 0..., 0...],
                mergedValue[0..., (mergedValue.dim(1) - tailCount)..., 0..., 0...]
            ], axis: 1)
        } else {
            self.key = mergedKey
            self.value = mergedValue
        }

        let storedKey = self.key!
        let storedValue = self.value!
        let frameCount = storedKey.dim(1) / spatialTokensPerFrame
        let newFrameCount = newTokenCount / spatialTokensPerFrame
        return Wan2CausalCacheWindow(
            key: storedKey,
            value: storedValue,
            frameCount: frameCount,
            queryFrameIndices: Array((frameCount - newFrameCount)..<frameCount)
        )
    }
}

extension Wan2SelfAttention {
    func callCausal(
        _ input: MLXArray,
        grid: Wan2GridSize,
        frequencies: MLXArray,
        cache: Wan2CausalKVCache,
        currentStartToken: Int
    ) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let dtype = query.weight.dtype
        let typed = input.asType(dtype)
        let q = queryNorm(query(typed)).reshaped(batch, sequence, heads, headDimension)
        let k = keyNorm(key(typed)).reshaped(batch, sequence, heads, headDimension)
        let v = value(typed).reshaped(batch, sequence, heads, headDimension)
        let spatialTokens = grid.height * grid.width
        let window = cache.update(
            key: k,
            value: v,
            currentStartToken: currentStartToken,
            spatialTokensPerFrame: spatialTokens
        )
        let queryRoPE = Wan2RoPE.prepare(
            grid: grid,
            frequencies: frequencies,
            dtype: .float32,
            temporalFrameIndices: window.queryFrameIndices
        )
        let keyGrid = Wan2GridSize(frames: window.frameCount, height: grid.height, width: grid.width)
        let keyRoPE = Wan2RoPE.prepare(
            grid: keyGrid,
            frequencies: frequencies,
            dtype: .float32,
            temporalFrameIndices: Array(0..<window.frameCount)
        )
        let rotatedQuery = Wan2RoPE.apply(q.asType(.float32), grid: grid, cache: queryRoPE)
            .transposed(0, 2, 1, 3)
        let rotatedKey = Wan2RoPE.apply(window.key.asType(.float32), grid: keyGrid, cache: keyRoPE)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: rotatedQuery,
            keys: rotatedKey,
            values: window.value.transposed(0, 2, 1, 3),
            scale: scale,
            mask: .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDimension))
    }
}

extension Wan2ProjectiveSelfAttention {
    func callCausal(
        _ input: MLXArray,
        conditioning: Wan2ProjectiveCameraConditioning,
        grid: Wan2GridSize,
        cache: Wan2CausalKVCache,
        currentStartToken: Int
    ) -> MLXArray {
        precondition(conditioning.frameCount == grid.frames)
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let dtype = query.weight.dtype
        let typed = input.asType(dtype)
        var q = queryNorm(query(typed)).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        var k = keyNorm(key(typed)).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        var v = value(typed).reshaped(batch, sequence, heads, headDimension).transposed(0, 2, 1, 3)
        let transforms = Wan2ProjectivePositionEncoding.prepare(
            conditioning: conditioning,
            batchSize: batch,
            dtype: dtype
        )
        q = Wan2ProjectivePositionEncoding.apply(q, matrices: transforms.query, cameraFrames: grid.frames)
        k = Wan2ProjectivePositionEncoding.apply(k, matrices: transforms.keyValue, cameraFrames: grid.frames)
        v = Wan2ProjectivePositionEncoding.apply(v, matrices: transforms.keyValue, cameraFrames: grid.frames)
        let window = cache.update(
            key: k.transposed(0, 2, 1, 3),
            value: v.transposed(0, 2, 1, 3),
            currentStartToken: currentStartToken,
            spatialTokensPerFrame: grid.height * grid.width
        )
        var attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: window.key.transposed(0, 2, 1, 3),
            values: window.value.transposed(0, 2, 1, 3),
            scale: scale,
            mask: .none
        )
        attended = Wan2ProjectivePositionEncoding.apply(
            attended,
            matrices: transforms.output,
            cameraFrames: grid.frames
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDimension))
    }
}

extension Wan2TransformerBlock {
    func callCausal(
        _ input: MLXArray,
        modulationInput: MLXArray,
        context: MLXArray,
        grid: Wan2GridSize,
        frequencies: MLXArray,
        crossCache: Wan2AttentionKVCache?,
        cameraConditioning: Wan2ProjectiveCameraConditioning?,
        state: Wan2CausalBlockState,
        currentStartToken: Int
    ) -> MLXArray {
        let parts = MLX.split(modulation.expandedDimensions(axis: 1) + modulationInput, parts: 6, axis: 2)
            .map { $0.squeezed(axis: 2) }
        var hidden = input
        let hiddenType = hidden.dtype
        let selfInput = selfNorm(hidden.asType(.float32)) * (1 + parts[1]) + parts[0]
        var selfOutput = selfAttention.callCausal(
            selfInput.asType(hiddenType),
            grid: grid,
            frequencies: frequencies,
            cache: state.selfAttention,
            currentStartToken: currentStartToken
        )
        if let cameraAttention, let cameraConditioning {
            let cameraOutput = state.cachesProjectiveAttention
                ? cameraAttention.callCausal(
                    selfInput.asType(hiddenType),
                    conditioning: cameraConditioning,
                    grid: grid,
                    cache: state.cameraAttention,
                    currentStartToken: currentStartToken
                )
                : cameraAttention(
                    selfInput.asType(hiddenType),
                    conditioning: cameraConditioning
                )
            selfOutput = selfOutput + cameraOutput
        }
        hidden = (hidden.asType(.float32) + selfOutput * parts[2]).asType(hiddenType)
        let crossInput = crossNorm(hidden.asType(.float32)).asType(hiddenType)
        hidden = hidden + crossAttention(crossInput, context: context, cache: crossCache)
        let feedForwardInput = feedForwardNorm(hidden.asType(.float32)) * (1 + parts[4]) + parts[3]
        let feedForwardOutput = feedForward(feedForwardInput.asType(hiddenType))
        return (hidden.asType(.float32) + feedForwardOutput.asType(.float32) * parts[5])
            .asType(hiddenType)
    }
}
