import Foundation
import MLX
import MLXFast

enum GLM47CompressedMLAPolicy {
    static let defaultMinimumPromptTokens = 2_048

    static func isEnabled(
        promptTokenCount: Int,
        config: GLM47FlashConfig,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        isEnabled(
            promptTokenCount: promptTokenCount,
            dimensions: (
                rank: config.kvLoraRank,
                nope: config.qkNopeHeadDim,
                rope: config.qkRopeHeadDim,
                value: config.vHeadDim
            ),
            environment: environment
        )
    }

    static func isEnabled(
        promptTokenCount: Int,
        dimensions: (rank: Int, nope: Int, rope: Int, value: Int),
        environment: [String: String]
    ) -> Bool {
        guard dimensions.rank > 0,
              dimensions.nope > 0,
              dimensions.rope > 0,
              dimensions.value > 0 else {
            return false
        }
        let raw = environment["MERERUN_PSI_COMPRESSED_MLA"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch raw {
        case "1", "true", "yes", "on":
            return true
        case "auto":
            let threshold = environment["MERERUN_PSI_COMPRESSED_MLA_MIN_PROMPT_TOKENS"]
                .flatMap(Int.init)
                .map { max(1, $0) }
                ?? defaultMinimumPromptTokens
            return promptTokenCount >= threshold
        default:
            // Algebraic weight absorption changes floating-point operation
            // order. Keep it quality-gated until a real checkpoint parity
            // corpus proves a safe automatic threshold.
            return false
        }
    }

    static func elementsPerToken(config: GLM47FlashConfig) -> (expanded: Int, compressed: Int) {
        elementsPerToken(
            heads: config.numAttentionHeads,
            rank: config.kvLoraRank,
            nope: config.qkNopeHeadDim,
            rope: config.qkRopeHeadDim,
            value: config.vHeadDim
        )
    }

    static func elementsPerToken(
        heads: Int,
        rank: Int,
        nope: Int,
        rope: Int,
        value: Int
    ) -> (expanded: Int, compressed: Int) {
        let expanded = heads * (nope + rope + value)
        // The compressed cache stores latent+RoPE as K and latent as V, each
        // once rather than once per attention head.
        let compressed = (2 * rank) + rope
        return (expanded, compressed)
    }
}

enum GLM47FusedMoEPolicy {
    static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let raw = environment["MERERUN_PSI_FUSED_MOE"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
    }
}

/// KVCache conformance for the MLA latent plus its decoupled RoPE key. The
/// wrapped cache retains all existing fork/batch behavior; only the stored
/// width and head count differ from expanded attention K/V.
final class GLM47CompressedMLACache: KVCache {
    private let storage: KVCache

    init(step: Int = 256) {
        self.storage = KVCacheSimple(step: step)
    }

    private init(storage: KVCache) {
        self.storage = storage
    }

    var offset: Int { storage.offset }
    var rowOffsets: [Int]? { storage.rowOffsets }
    var supportsVariablePositionBatching: Bool { storage.supportsVariablePositionBatching }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        storage.update(keys: keys, values: values)
    }

    func makeMask(n: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        storage.makeMask(n: n)
    }

    func fork() -> KVCache {
        GLM47CompressedMLACache(storage: storage.fork())
    }

    func batched(with caches: [KVCache]) -> KVCache? {
        guard let typed = caches as? [GLM47CompressedMLACache],
              let batched = storage.batched(with: typed.map(\.storage)) else {
            return nil
        }
        return GLM47CompressedMLACache(storage: batched)
    }

    func unbatchedRows(count: Int) -> [KVCache]? {
        storage.unbatchedRows(count: count)?.map(GLM47CompressedMLACache.init(storage:))
    }
}

struct GLM47AbsorbedMLAWeights {
    let key: MLXArray       // [heads, qk_nope, rank]
    let value: MLXArray     // [heads, value_dim, rank]

    var sourceDType: DType { key.dtype }

    /// Apply the exact MLA algebra with a different floating-point operation
    /// order: Wk is absorbed into Q before attention and Wv is applied after
    /// the latent-value reduction.
    func attend(
        qNope: MLXArray,
        qPe: MLXArray,
        latentKeys: MLXArray,
        ropeKeys: MLXArray,
        latentValues: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let transformedQueries = MLX.matmul(qNope.asType(sourceDType), key)
        let queries = concatenated([transformedQueries, qPe.asType(sourceDType)], axis: -1)
        let keys = concatenated([latentKeys, ropeKeys], axis: -1)

        let latentOutput = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: latentValues.asType(.float32),
            scale: scale,
            mask: mask
        )
        return MLX.matmul(
            latentOutput.asType(sourceDType),
            value.swappedAxes(-1, -2)
        )
    }
}
