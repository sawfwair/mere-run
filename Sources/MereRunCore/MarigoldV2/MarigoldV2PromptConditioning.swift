import Foundation
import MLX

/// The frozen text conditioning a Marigold V2 modality runs with.
///
/// Like the other conditioning carriers in this runtime it is not `Sendable`:
/// it owns MLX arrays, which stay on the runtime that produced them.
public struct MarigoldV2PromptConditioning {
    /// Prompt embeddings shaped `[1, tokens, width]`.
    public let embeddings: MLXArray
    /// Attention mask shaped `[1, tokens]`, one for a real token and zero for padding.
    public let mask: MLXArray
    /// Number of unpadded tokens, mirroring the reference `txt_seq_lens`.
    public let tokenCount: Int

    public init(embeddings: MLXArray, mask: MLXArray, tokenCount: Int) {
        self.embeddings = embeddings
        self.mask = mask
        self.tokenCount = tokenCount
    }
}

/// Loads the precomputed prompt embeddings that stand in for the base text encoder.
///
/// Marigold ships one `torch.save`d tensor per file: embeddings shaped
/// `[contexts, tokens, width]` and a mask shaped `[contexts, tokens]`. Inference
/// conditions on the first context row, which is what the reference pipeline
/// selects for a single-image batch.
public enum MarigoldV2PromptEmbeddingLoader {
    public enum LoaderError: LocalizedError {
        case notABareTensor(URL)
        case unexpectedRank(URL, rank: Int, expected: Int)
        case tokenCountMismatch(embeddings: Int, mask: Int)
        case emptyPrompt(URL)

        public var errorDescription: String? {
            switch self {
            case .notABareTensor(let url):
                return "Marigold V2 prompt file does not hold a single tensor: \(url.path)."
            case .unexpectedRank(let url, let rank, let expected):
                return "Marigold V2 prompt file \(url.path) has rank \(rank), expected \(expected)."
            case .tokenCountMismatch(let embeddings, let mask):
                return "Marigold V2 prompt embeddings cover \(embeddings) tokens "
                    + "but the mask covers \(mask)."
            case .emptyPrompt(let url):
                return "Marigold V2 prompt mask in \(url.path) selects no tokens."
            }
        }
    }

    public static func load(
        embedsURL: URL,
        maskURL: URL,
        dtype: DType = .bfloat16
    ) throws -> MarigoldV2PromptConditioning {
        let embeddings = try loadBareTensor(url: embedsURL, expectedRank: 3, dtype: dtype)
        let mask = try loadBareTensor(url: maskURL, expectedRank: 2, dtype: .int32)

        guard embeddings.dim(1) == mask.dim(1) else {
            throw LoaderError.tokenCountMismatch(embeddings: embeddings.dim(1), mask: mask.dim(1))
        }

        // The published files carry several prompt contexts; a single-image run
        // conditions on the first, matching the reference batch selection.
        let selectedEmbeddings = embeddings[0..<1, 0..., 0...]
        let selectedMask = mask[0..<1, 0...]
        let tokenCount = MLX.sum(selectedMask).item(Int.self)
        guard tokenCount > 0 else {
            throw LoaderError.emptyPrompt(maskURL)
        }

        MLX.eval(selectedEmbeddings, selectedMask)
        return MarigoldV2PromptConditioning(
            embeddings: selectedEmbeddings,
            mask: selectedMask,
            tokenCount: tokenCount
        )
    }

    private static func loadBareTensor(
        url: URL,
        expectedRank: Int,
        dtype: DType
    ) throws -> MLXArray {
        let archive = try PyTorchStateDictArchive(url: url)
        guard let descriptor = archive.bareTensor else {
            throw LoaderError.notABareTensor(url)
        }
        guard descriptor.shape.count == expectedRank else {
            throw LoaderError.unexpectedRank(url, rank: descriptor.shape.count, expected: expectedRank)
        }
        return try archive.loadArray(for: descriptor, dtype: dtype)
    }
}
