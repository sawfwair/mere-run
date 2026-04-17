import Foundation
import MLX
import MLXNN
import MLXRandom

extension Flux2KleinLoRATrainer {
    // MARK: - Deterministic RNG

    struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        init(rawState: UInt64) {
            self.state = rawState
        }

        var rawState: UInt64 {
            state
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextInt(upperBound: Int) -> Int {
            precondition(upperBound > 0)
            return Int(next() % UInt64(upperBound))
        }
    }
}
