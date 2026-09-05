import Foundation
import MLX

/// Selects the execution engine for the compute-heavy Parakeet encoder.
///
/// Schema 1 Core ML artifacts are encoder-only and use a separate Parakeet
/// checkpoint. Schema 2 artifacts also carry Mere's compact MLX decoder and
/// can serve as the complete model root.
public enum ParakeetExecutionProvider: Sendable, Hashable {
    case mlx
    case coreML(artifactURL: URL)

    public var bundledModelURL: URL? {
        guard case .coreML(let artifactURL) = self else { return nil }
        let root = artifactURL.standardizedFileURL
        return ParakeetResources(rootURL: root).validate().isEmpty ? root : nil
    }
}

struct ParakeetEncoderOutput {
    let features: MLXArray
    let lengths: [Int]
}

protocol ParakeetExternalEncoder: AnyObject {
    func encode(_ mel: MLXArray) throws -> ParakeetEncoderOutput
}

protocol ParakeetExternalTDTDecoder: AnyObject {
    var maximumBatchSize: Int { get }

    func decode(
        encoded: MLXArray,
        lengths: [Int]
    ) throws -> [ParakeetAlignedResult]
}
