import Foundation
import MLX

struct Flux2LoRAStackInput: @unchecked Sendable {
    let label: String
    let scale: Float
    let weights: LoRAWeights
}

enum Flux2LoRAStacker {
    enum StackError: LocalizedError, Equatable {
        case empty
        case invalidScale(label: String, scale: Float)
        case invalidPair(label: String, path: String, down: [Int], up: [Int])
        case incompatiblePair(
            label: String,
            path: String,
            expectedInput: Int,
            actualInput: Int,
            expectedOutput: Int,
            actualOutput: Int
        )

        var errorDescription: String? {
            switch self {
            case .empty:
                return "At least one FLUX.2 LoRA is required."
            case .invalidScale(let label, let scale):
                return "FLUX.2 LoRA scale must be finite for \(label) (got \(scale))."
            case .invalidPair(let label, let path, let down, let up):
                return "Invalid FLUX.2 LoRA pair in \(label) for \(path): down=\(down), up=\(up)."
            case .incompatiblePair(
                let label,
                let path,
                let expectedInput,
                let actualInput,
                let expectedOutput,
                let actualOutput
            ):
                return "Incompatible stacked FLUX.2 LoRA in \(label) for \(path): "
                    + "expected input/output \(expectedInput)/\(expectedOutput), "
                    + "found \(actualInput)/\(actualOutput)."
            }
        }
    }

    /// Combines an ordered adapter stack into one mathematically equivalent low-rank update.
    /// Each source alpha and user scale is folded into its down matrix before concatenation.
    static func stack(_ inputs: [Flux2LoRAStackInput]) throws -> LoRAWeights {
        guard !inputs.isEmpty else { throw StackError.empty }

        var combined: [String: (down: MLXArray, up: MLXArray)] = [:]
        for input in inputs {
            guard input.scale.isFinite else {
                throw StackError.invalidScale(label: input.label, scale: input.scale)
            }
            for (path, pair) in input.weights.weights {
                let downShape = pair.down.shape
                let upShape = pair.up.shape
                guard downShape.count == 2,
                      upShape.count == 2,
                      downShape[0] > 0,
                      downShape[0] == upShape[1] else {
                    throw StackError.invalidPair(
                        label: input.label,
                        path: path,
                        down: downShape,
                        up: upShape
                    )
                }

                let pairScale = input.scale * input.weights.alpha / Float(downShape[0])
                let scaledDown = pairScale == 1
                    ? pair.down.asType(.float32)
                    : pair.down.asType(.float32) * MLXArray(pairScale)
                let up = pair.up.asType(.float32)

                if let existing = combined[path] {
                    let expectedInput = existing.down.shape[1]
                    let expectedOutput = existing.up.shape[0]
                    guard downShape[1] == expectedInput,
                          upShape[0] == expectedOutput else {
                        throw StackError.incompatiblePair(
                            label: input.label,
                            path: path,
                            expectedInput: expectedInput,
                            actualInput: downShape[1],
                            expectedOutput: expectedOutput,
                            actualOutput: upShape[0]
                        )
                    }
                    combined[path] = (
                        down: MLX.concatenated([existing.down, scaledDown], axis: 0),
                        up: MLX.concatenated([existing.up, up], axis: 1)
                    )
                } else {
                    combined[path] = (down: scaledDown, up: up)
                }
            }
        }

        guard !combined.isEmpty else { throw LoRAError.noWeightPairs }
        let targetRanks = combined.mapValues { $0.down.shape[0] }
        let defaultRank = targetRanks.sorted { $0.key < $1.key }.first?.value ?? 1
        // The injector uses alpha == target rank for every combined layer. Source alpha
        // values are already included in the scaled down matrices.
        return LoRAWeights(
            weights: combined,
            rank: defaultRank,
            alpha: Float(defaultRank),
            targetRanks: targetRanks
        )
    }
}
