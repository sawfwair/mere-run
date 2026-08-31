import Foundation
import MLX
import MLXNN

/// Flash-Next keeps its readout, router, and indexer in BF16. Small-batch GEMM
/// can round differently from one-token GEMV, including at three/four rows.
/// Keep those short batches on serial arithmetic; quantized projections
/// retain their own weight-reusing QMV path and prefill stays batched.
func q38SmallBatchProjection(_ layer: Linear, _ input: MLXArray) -> MLXArray {
    #if os(macOS)
    if Device.defaultDevice().deviceType == .gpu,
       input.ndim == 3, input.dim(0) == 1,
       (2...9).contains(input.dim(1)),
       input.dtype == .bfloat16, layer.weight.dtype == .bfloat16 {
        return MLX.concatenated((0..<input.dim(1)).map { row in
            layer(input[0..., row..<(row + 1), 0...])
        }, axis: 1)
    }
    #endif
    return layer(input)
}

final class Q38GatedResidual: Module {
    @ModuleInfo(key: "hc_norm") var norm: Q35RMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInjectWeight: Linear?

    private let streamCount: Int
    private let hiddenSize: Int

    init(config: Q35Config, combinesBlockOutput: Bool = true) {
        let text = config.textConfig
        let hyperDimensions = text.hyperConnectionCount * text.hiddenSize
        self.streamCount = text.hyperConnectionCount
        self.hiddenSize = text.hiddenSize
        self._norm.wrappedValue = Q35RMSNorm(
            dimensions: hyperDimensions,
            eps: text.rmsNormEps,
            groupSize: text.hiddenSize,
            zeroCenteredWeight: false
        )
        self._inputMixDown.wrappedValue = Linear(
            hyperDimensions,
            text.hyperConnectionLowRank,
            bias: false
        )
        self._inputMixUp.wrappedValue = Linear(
            text.hyperConnectionLowRank,
            hyperDimensions,
            bias: false
        )
        self._blockInjectWeight.wrappedValue = combinesBlockOutput
            ? Linear(hyperDimensions, text.hyperConnectionCount, bias: false)
            : nil
        super.init()
    }

    func mix(_ hyperInput: MLXArray) -> (
        mixed: MLXArray,
        residual: MLXArray,
        injectionWeights: MLXArray
    ) {
        let normalized = norm(hyperInput)
        var inputWeights = MLXNN.silu(
            q38SmallBatchProjection(inputMixDown, normalized)
                / MLXArray(Float(streamCount)).asType(normalized.dtype)
        )
        inputWeights = MLX.sigmoid(q38SmallBatchProjection(inputMixUp, inputWeights))
            .reshaped(Array(hyperInput.shape.dropLast()) + [streamCount, hiddenSize])
        let streams = normalized.reshaped(
            Array(hyperInput.shape.dropLast()) + [streamCount, hiddenSize]
        )
        let mixed = (inputWeights * streams).mean(axis: -2)
        let rawInjection = blockInjectWeight.map { q38SmallBatchProjection($0, normalized) }
            ?? MLXArray.zeros(
                Array(hyperInput.shape.dropLast()) + [streamCount],
                dtype: normalized.dtype
            )
        let injectionWeights = MLXArray(2.0).asType(normalized.dtype) * MLX.sigmoid(
            rawInjection / MLXArray(Float(streamCount)).asType(normalized.dtype)
        )
        return (mixed, hyperInput, injectionWeights)
    }

    func combine(_ hyperInput: MLXArray) -> MLXArray {
        let normalized = norm(hyperInput)
        var inputWeights = MLXNN.silu(
            q38SmallBatchProjection(inputMixDown, normalized)
                / MLXArray(Float(streamCount)).asType(normalized.dtype)
        )
        inputWeights = MLX.sigmoid(q38SmallBatchProjection(inputMixUp, inputWeights))
            .reshaped(Array(hyperInput.shape.dropLast()) + [streamCount, hiddenSize])
        let streams = normalized.reshaped(
            Array(hyperInput.shape.dropLast()) + [streamCount, hiddenSize]
        )
        return (inputWeights * streams).mean(axis: -2)
    }

    func inject(
        blockOutput: MLXArray,
        residual: MLXArray,
        injectionWeights: MLXArray
    ) -> MLXArray {
        let injection = MLX.expandedDimensions(blockOutput, axis: -2)
            * MLX.expandedDimensions(injectionWeights, axis: -1)
        return residual + injection.reshaped(residual.shape)
    }
}

final class Q38NGramEmbedding: Module {
    private struct Shard {
        let rowOffset: Int
        let embedding: PreQuantizedEmbedding
    }

    final class PreparedLookup {
        fileprivate let diskLookup: Q38DiskNGramTable.PreparedLookup
        fileprivate let nextTokenContext: [Int32]
        fileprivate let batch: Int
        fileprivate let sequence: Int

        fileprivate init(
            diskLookup: Q38DiskNGramTable.PreparedLookup,
            nextTokenContext: [Int32],
            batch: Int,
            sequence: Int
        ) {
            self.diskLookup = diskLookup
            self.nextTokenContext = nextTokenContext
            self.batch = batch
            self.sequence = sequence
        }
    }

    private struct HashedTokenIDs {
        let ids: [Int32]
        let nextTokenContext: [Int32]
    }

    private let ngramSize: Int
    private let headsPerNgram: Int
    private let contextLength: Int
    private let eosTokenId: Int32
    private let multipliers: [Int64]
    private let headVocabSizes: [Int64]
    private let headOffsets: [Int64]
    private let outputDimensions: Int
    private var shards: [Shard] = []
    private var diskTable: Q38DiskNGramTable?

    init(config: Q35Config, pleLayerIndex: Int) {
        let text = config.textConfig
        self.ngramSize = text.ngramSize
        self.headsPerNgram = text.headsPerNgram
        self.contextLength = max(0, text.ngramSize - 1)
        self.eosTokenId = Int32(text.eosTokenId ?? config.eosTokenIds.first ?? 0)
        self.outputDimensions = text.pleEmbeddingDimensions

        let headCount = (text.ngramSize - 1) * text.headsPerNgram
        var sizes: [Int64] = []
        var offsets: [Int64] = []
        var offset: Int64 = 0
        for headIndex in 0..<headCount {
            let globalHeadIndex = pleLayerIndex * headCount + headIndex
            let size = Self.nthPrime(after: text.ngramVocabSizeBase - 1, count: globalHeadIndex + 1)
            offsets.append(offset)
            sizes.append(Int64(size))
            offset += Int64(size)
        }
        self.headVocabSizes = sizes
        self.headOffsets = offsets
        self.multipliers = Self.layerMultipliers(
            vocabularySize: text.vocabSize,
            ngramSize: text.ngramSize,
            pleLayerIndex: pleLayerIndex,
            seed: text.ngramSeed
        )
        super.init()
    }

    var isLoaded: Bool { diskTable != nil || !shards.isEmpty }
    var minimumRowCount: Int { Int((headOffsets.last ?? 0) + (headVocabSizes.last ?? 0)) }

    func installDiskTable(_ table: Q38DiskNGramTable) {
        diskTable = table
        shards = []
    }

    func installShards(_ embeddings: [PreQuantizedEmbedding]) {
        diskTable = nil
        var offset = 0
        shards = embeddings.map { embedding in
            defer { offset += embedding.weight.dim(0) }
            return Shard(rowOffset: offset, embedding: embedding)
        }
    }

    func callAsFunction(_ inputIds: MLXArray, cache: Q35LinearCache?) -> MLXArray {
        precondition(isLoaded, "Qwen4Exp n-gram embedding shards are not loaded")
        let ids = hashedTokenIDs(inputIds, cache: cache)
        return lookup(ids).reshaped(inputIds.dim(0), inputIds.dim(1), outputDimensions)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: Q35LinearCache?,
        prepared: PreparedLookup
    ) -> MLXArray {
        precondition(prepared.batch == inputIds.dim(0) && prepared.sequence == inputIds.dim(1))
        cache?.pleTokenContext = MLXArray(prepared.nextTokenContext)
            .reshaped(prepared.batch, contextLength)
        return prepared.diskLookup.materialize()
            .reshaped(prepared.batch, prepared.sequence, outputDimensions)
    }

    func prefetch(_ inputIds: MLXArray, cache: Q35LinearCache?) -> PreparedLookup? {
        guard let diskTable else { return nil }
        let hashed = hash(inputIds, cache: cache)
        return PreparedLookup(
            diskLookup: diskTable.prefetch(hashed.ids),
            nextTokenContext: hashed.nextTokenContext,
            batch: inputIds.dim(0),
            sequence: inputIds.dim(1)
        )
    }

    func hashedTokenIDs(_ inputIds: MLXArray, cache: Q35LinearCache?) -> [Int32] {
        let hashed = hash(inputIds, cache: cache)
        cache?.pleTokenContext = MLXArray(hashed.nextTokenContext)
            .reshaped(inputIds.dim(0), contextLength)
        return hashed.ids
    }

    private func hash(_ inputIds: MLXArray, cache: Q35LinearCache?) -> HashedTokenIDs {
        let batch = inputIds.dim(0)
        let sequence = inputIds.dim(1)
        let current = inputIds.asType(.int32).asArray(Int32.self)
        let previous: [Int32]
        if let cached = cache?.pleTokenContext {
            previous = cached.asType(.int32).asArray(Int32.self)
        } else {
            previous = [Int32](repeating: eosTokenId, count: batch * contextLength)
        }

        var ids: [Int32] = []
        ids.reserveCapacity(batch * sequence * headVocabSizes.count)
        var nextContext: [Int32] = []
        nextContext.reserveCapacity(batch * contextLength)

        for batchIndex in 0..<batch {
            let previousStart = batchIndex * contextLength
            let currentStart = batchIndex * sequence
            let history = Array(previous[previousStart..<(previousStart + contextLength)])
                + Array(current[currentStart..<(currentStart + sequence)])
            for tokenIndex in 0..<sequence {
                let position = contextLength + tokenIndex
                let segmentStart = Self.segmentStart(in: history, before: position, eosTokenId: eosTokenId)
                for ngram in 2...ngramSize {
                    var mixed = Self.multiply(history[position], multipliers[0])
                    for shift in 1..<ngram {
                        let sourcePosition = position - shift
                        let token = sourcePosition >= segmentStart && sourcePosition >= 0
                            ? history[sourcePosition]
                            : eosTokenId
                        mixed ^= Self.multiply(token, multipliers[shift])
                    }
                    let startHead = (ngram - 2) * headsPerNgram
                    for head in startHead..<(startHead + headsPerNgram) {
                        let local = Self.positiveRemainder(mixed, modulus: headVocabSizes[head])
                        ids.append(Int32(local + headOffsets[head]))
                    }
                }
            }
            nextContext.append(contentsOf: history.suffix(contextLength))
        }
        return HashedTokenIDs(ids: ids, nextTokenContext: nextContext)
    }

    func verificationTokenHistory(_ inputIds: MLXArray, cache: Q35LinearCache?) -> MLXArray {
        let batch = inputIds.dim(0)
        let previous = cache?.pleTokenContext ?? MLXArray(
            [Int32](repeating: eosTokenId, count: batch * contextLength)
        ).reshaped(batch, contextLength)
        return MLX.concatenated([previous, inputIds.asType(.int32)], axis: 1)
    }

    private func lookup(_ ids: [Int32]) -> MLXArray {
        if let diskTable { return diskTable.lookup(ids) }
        var grouped: [Int: [(position: Int, localId: Int32)]] = [:]
        grouped.reserveCapacity(shards.count)
        for (position, rawId) in ids.enumerated() {
            let id = Int(rawId)
            let shardIndex = shardIndex(containing: id)
            grouped[shardIndex, default: []].append((position, Int32(id - shards[shardIndex].rowOffset)))
        }

        var pieces: [MLXArray] = []
        var groupedPositions: [Int] = []
        for shardIndex in grouped.keys.sorted() {
            guard let entries = grouped[shardIndex] else { continue }
            pieces.append(shards[shardIndex].embedding(MLXArray(entries.map(\.localId))))
            groupedPositions.append(contentsOf: entries.map(\.position))
        }
        let groupedOutput = pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 0)
        var inverse = [Int32](repeating: 0, count: groupedPositions.count)
        for (groupedIndex, originalIndex) in groupedPositions.enumerated() {
            inverse[originalIndex] = Int32(groupedIndex)
        }
        return groupedOutput.take(MLXArray(inverse), axis: 0)
    }

    private func shardIndex(containing id: Int) -> Int {
        var low = 0
        var high = shards.count
        while low + 1 < high {
            let middle = (low + high) / 2
            if shards[middle].rowOffset <= id {
                low = middle
            } else {
                high = middle
            }
        }
        return low
    }

    private static func segmentStart(in history: [Int32], before position: Int, eosTokenId: Int32) -> Int {
        guard position > 0 else { return 0 }
        for index in stride(from: position - 1, through: 0, by: -1) where history[index] == eosTokenId {
            return index + 1
        }
        return 0
    }

    private static func multiply(_ token: Int32, _ multiplier: Int64) -> UInt64 {
        UInt64(UInt32(bitPattern: token)) &* UInt64(bitPattern: multiplier)
    }

    private static func positiveRemainder(_ value: UInt64, modulus: Int64) -> Int64 {
        let signed = Int64(bitPattern: value)
        let remainder = signed % modulus
        return remainder >= 0 ? remainder : remainder + modulus
    }

    private static func layerMultipliers(
        vocabularySize: Int,
        ngramSize: Int,
        pleLayerIndex: Int,
        seed: Int
    ) -> [Int64] {
        let multiplierMax = Int64.max / Int64(max(vocabularySize, 1))
        let halfBound = UInt64(max(1, multiplierMax / 2))
        let baseSeed = UInt64(seed + 10_007 * pleLayerIndex)
        return (0..<ngramSize).map { index in
            let value = baseSeed &+ 0x9E37_79B9_7F4A_7C15 &* UInt64(index + 1)
            return Int64(2 * (splitMix64(value) % halfBound) + 1)
        }
    }

    private static func splitMix64(_ input: UInt64) -> UInt64 {
        var value = input &+ 0x9E37_79B9_7F4A_7C15
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }

    private static func nthPrime(after start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0..<count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    private static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value.isMultiple(of: 2) { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value.isMultiple(of: divisor) { return false }
            divisor += 2
        }
        return true
    }
}

/// Inputs retained only for a speculative block, so rejection can rewind both
/// PLE histories without recomputing embeddings or reading model weights.
struct Q38PLEVerificationReplay {
    let convolutionInput: MLXArray
    let tokenHistory: MLXArray
    let tokenCount: Int

    var convolutionStateLength: Int { convolutionInput.dim(1) - tokenCount }
    var tokenContextLength: Int { tokenHistory.dim(1) - tokenCount }
}

final class Q38PLELayer: Module {
    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Q38NGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProjection: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "norm_key") var keyNorm: Q35RMSNorm
    @ModuleInfo(key: "norm_query") var queryNorm: Q35RMSNorm
    @ModuleInfo(key: "norm_conv") var convolutionNorm: Q35RMSNorm
    @ModuleInfo(key: "conv1d") var convolution: Conv1d

    private let streamCount: Int
    private let hiddenSize: Int
    private let convolutionStateLength: Int

    struct PreparedInput {
        let lookup: Q38NGramEmbedding.PreparedLookup
        let verificationTokens: MLXArray?
    }

    init(config: Q35Config, pleLayerIndex: Int) {
        let text = config.textConfig
        let hyperDimensions = text.hyperConnectionCount * text.hiddenSize
        self.streamCount = text.hyperConnectionCount
        self.hiddenSize = text.hiddenSize
        self.convolutionStateLength = (text.pleConvKernelSize - 1) * text.ngramSize
        self._pleEmbedding.wrappedValue = Q38NGramEmbedding(
            config: config,
            pleLayerIndex: pleLayerIndex
        )
        self._keyProjection.wrappedValue = Linear(
            text.pleEmbeddingDimensions,
            hyperDimensions,
            bias: false
        )
        self._valueProjection.wrappedValue = Linear(
            text.pleEmbeddingDimensions,
            text.hiddenSize,
            bias: false
        )
        self._keyNorm.wrappedValue = Q35RMSNorm(
            dimensions: hyperDimensions,
            eps: text.rmsNormEps,
            groupSize: text.hiddenSize,
            zeroCenteredWeight: false
        )
        self._queryNorm.wrappedValue = Q35RMSNorm(
            dimensions: hyperDimensions,
            eps: text.rmsNormEps,
            groupSize: text.hiddenSize,
            zeroCenteredWeight: false
        )
        self._convolutionNorm.wrappedValue = Q35RMSNorm(
            dimensions: hyperDimensions,
            eps: text.rmsNormEps,
            groupSize: text.hiddenSize,
            zeroCenteredWeight: false
        )
        self._convolution.wrappedValue = Conv1d(
            inputChannels: hyperDimensions,
            outputChannels: hyperDimensions,
            kernelSize: text.pleConvKernelSize,
            stride: 1,
            padding: 0,
            dilation: text.ngramSize,
            groups: hyperDimensions,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        inputIds: MLXArray,
        cache: Q35LinearCache?,
        targetVerify: Bool = false,
        preparedInput: PreparedInput? = nil
    ) -> MLXArray {
        let verificationTokens: MLXArray?
        let embeddings: MLXArray
        if let preparedInput {
            verificationTokens = preparedInput.verificationTokens
            embeddings = pleEmbedding(inputIds, cache: cache, prepared: preparedInput.lookup)
        } else {
            verificationTokens = targetVerify
                ? pleEmbedding.verificationTokenHistory(inputIds, cache: cache)
                : nil
            embeddings = pleEmbedding(inputIds, cache: cache)
        }
        let shapePrefix = Array(hiddenStates.shape.dropLast())
        let keys = keyNorm(q38SmallBatchProjection(keyProjection, embeddings))
            .reshaped(shapePrefix + [streamCount, hiddenSize])
        let values = q38SmallBatchProjection(valueProjection, embeddings)
        let queries = queryNorm(hiddenStates)
            .reshaped(shapePrefix + [streamCount, hiddenSize])
        var gate = (keys * queries).sum(axis: -1, keepDims: true)
            / MLXArray(sqrt(Float(hiddenSize))).asType(hiddenStates.dtype)
        gate = MLX.sqrt(
            MLX.maximum(MLX.abs(gate), MLXArray(Float(1e-6)).asType(gate.dtype))
        ) * MLX.sign(gate)
        let gatedValues = MLX.sigmoid(gate) * MLX.expandedDimensions(values, axis: -2)
        let flattened = gatedValues.reshaped(shapePrefix + [streamCount * hiddenSize])
        let normalized = convolutionNorm(flattened)
        return flattened + shortConvolution(normalized, cache: cache, verificationTokens: verificationTokens)
    }

    func prefetch(
        inputIds: MLXArray,
        cache: Q35LinearCache?,
        targetVerify: Bool
    ) -> PreparedInput? {
        let verificationTokens = targetVerify
            ? pleEmbedding.verificationTokenHistory(inputIds, cache: cache)
            : nil
        guard let lookup = pleEmbedding.prefetch(inputIds, cache: cache) else { return nil }
        return PreparedInput(lookup: lookup, verificationTokens: verificationTokens)
    }

    private func shortConvolution(
        _ hiddenStates: MLXArray, cache: Q35LinearCache?, verificationTokens: MLXArray?
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let previous = cache?.pleConvState
            ?? MLXArray.zeros(
                [batch, convolutionStateLength, streamCount * hiddenSize],
                dtype: hiddenStates.dtype
            )
        let convolutionInput = MLX.concatenated([previous, hiddenStates], axis: 1)
        cache?.pleVerificationReplay = verificationTokens.map {
            Q38PLEVerificationReplay(
                convolutionInput: convolutionInput, tokenHistory: $0, tokenCount: hiddenStates.dim(1)
            )
        }
        cache?.pleConvState = convolutionInput[
            0...,
            (convolutionInput.dim(1) - convolutionStateLength)...,
            0...
        ]
        return MLXNN.silu(convolution(convolutionInput))
    }
}
