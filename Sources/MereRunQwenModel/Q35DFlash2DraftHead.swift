import Foundation
import MereRunTensor
import MLX
import MLXNN

/// Proposal-only vocabulary restriction; target verification always uses its full head.
final class Q35DFlash2DraftHead {
    private static let enabledFlag = QwenDFlashDefaults.Flag("MERERUN_Q38_DFLASH2_COMPACT_HEAD")
    @TaskLocal static var enabledOverride: Bool?
    static var enabled: Bool { enabledOverride ?? enabledFlag.enabled }
    static let prefixCount = 98_304
    static let controlStart = 248_032
    static let vocabularySize = 248_320
    static let realCount = prefixCount + vocabularySize - controlStart
    static let paddedCount = 98_624

    private var source: ObjectIdentifier?
    private var projection: Linear?
    private var vocabulary: MLXArray?

    static func rows(_ array: MLXArray) -> MLXArray {
        concatenated([array[0..<prefixCount], array[controlStart..<vocabularySize],
                      array[0..<(paddedCount - realCount)]], axis: 0)
    }

    func callAsFunction(_ hidden: MLXArray, target: Q35Model) -> (logits: MLXArray, vocabulary: MLXArray)? {
        guard target.config.textConfig.vocabSize == Self.vocabularySize, let full = target.lmHead else { return nil }
        if source != ObjectIdentifier(full) {
            let selected: Linear
            if let quantized = full as? PortableQuantizedLinear {
                let compact = PortableQuantizedLinear(
                    weight: Self.rows(quantized.weight), bias: quantized.bias.map(Self.rows),
                    scales: Self.rows(quantized.scales), biases: quantized.biases.map(Self.rows),
                    groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode
                )
                compact.verificationBackend = quantized.verificationBackend
                selected = compact
            } else if type(of: full) == Linear.self {
                selected = Linear(weight: Self.rows(full.weight), bias: full.bias.map(Self.rows))
            } else { return nil }
            projection = selected
            vocabulary = MLXArray((Array(0..<Self.prefixCount) + Array(Self.controlStart..<Self.vocabularySize)).map(Int32.init))
            source = ObjectIdentifier(full)
            FileHandle.standardError.write(Data("[q35] dflash2_compact_draft_vocabulary=\(Self.realCount)\n".utf8))
        }
        guard let projection, let vocabulary else { preconditionFailure("Compact draft head was not prepared.") }
        return (projection(hidden)[0..., 0..., 0..<Self.realCount], vocabulary)
    }
}
