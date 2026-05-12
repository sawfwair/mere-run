import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - Timestep Sampling

    /// Compute sampling weights for timesteps based on strategy.
    /// Returns normalized weights that sum to 1.0.
    static func computeTimestepWeights(
        numSteps: Int,
        strategy: TimestepSamplingStrategy
    ) -> [Float] {
        guard numSteps > 0 else { return [] }

        var weights: [Float]

        switch strategy {
        case .uniform:
            weights = [Float](repeating: 1.0 / Float(numSteps), count: numSteps)

        case .bellCurve:
            // Bell-shaped mean-normalized timestep weighting (BSMNTW)
            // Favors middle timesteps where most learning happens
            weights = (0..<numSteps).map { i in
                let x = Float(i)
                let n = Float(numSteps)
                return exp(-2.0 * pow((x - n / 2.0) / n, 2))
            }
            // Shift to start from 0 and normalize
            let minW = weights.min() ?? 0
            weights = weights.map { $0 - minW }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .contentFocused:
            // Cubic distribution favoring earlier timesteps (lower noise, more structure)
            weights = (0..<numSteps).map { i in
                let t = Float(i) / Float(max(numSteps - 1, 1))
                return pow(1.0 - t, 3)
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .styleFocused:
            // Inverse cubic favoring later timesteps (higher noise, more style)
            weights = (0..<numSteps).map { i in
                let t = Float(i) / Float(max(numSteps - 1, 1))
                return pow(t, 3)
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .sigmoid:
            // Sigmoid-based sampling that clusters around middle timesteps
            // t = sigmoid(randn) => values clustered around 0.5
            // This mimics ai-toolkit's default behavior
            weights = (0..<numSteps).map { i in
                let x = Float(i) / Float(max(numSteps - 1, 1))  // 0 to 1
                // Sigmoid-like bell centered at 0.5
                let dist = (x - 0.5) * 4.0  // Scale for steeper curve
                return 1.0 / (1.0 + exp(abs(dist)))
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }
        }

        return weights
    }

    /// Compute loss weights for timesteps (mean-normalized).
    static func computeTimestepLossWeights(
        numSteps: Int,
        strategy: ZImageTimestepLossWeightingStrategy
    ) -> [Float] {
        guard numSteps > 0 else { return [] }

        switch strategy {
        case .none:
            return [Float](repeating: 1.0, count: numSteps)

        case .weighted, .weighted2:
            // Bell-shaped mean-normalized timestep weighting (ai-toolkit `linear_timesteps_weights`)
            let n = Float(numSteps)
            var weights = (0..<numSteps).map { i -> Float in
                let x = Float(i)
                return exp(-2.0 * pow((x - n / 2.0) / n, 2))
            }

            let minW = weights.min() ?? 0
            weights = weights.map { $0 - minW }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 * (n / sum) } // mean-normalized
            }

            guard strategy == .weighted2 else { return weights }

            // weighted2 uses the "half-bell" variant (ai-toolkit `linear_timesteps_weights2`)
            let mid = max(numSteps / 2, 0)
            let maxTail = weights[mid...].max() ?? 1.0
            for i in mid..<numSteps {
                weights[i] = maxTail
            }
            return weights
        }
    }

    /// Build the FlowMatch training sigma table.
    /// Mirrors ai-toolkit's linear/weighted schedule: timesteps = linspace(1000, 1, N), sigma = timesteps / 1000.
    /// Optionally applies sigma shift (mflux-style resolution-dependent or scalar shift).
    /// - Parameters:
    ///   - numSteps: Number of timesteps
    ///   - width: Image width for resolution-dependent shift (optional)
    ///   - height: Image height for resolution-dependent shift (optional)
    ///   - sigmaShift: Fixed scalar shift value (optional, takes precedence over resolution-dependent)
    static func computeTrainingSigmas(
        numSteps: Int,
        width: Int? = nil,
        height: Int? = nil,
        sigmaShift: Float? = nil
    ) -> [Float] {
        guard numSteps > 0 else { return [] }
        if numSteps == 1 { return [1.0] }

        // Build linear sigmas: linspace(1.0, 1/numSteps, numSteps)
        let step = (1.0 - 1.0 / Float(numSteps)) / Float(numSteps - 1)
        var sigmas = (0..<numSteps).map { i in
            1.0 - Float(i) * step
        }

        // Apply sigma shift if requested
        if let shift = sigmaShift {
            // Scalar shift: shift * sigma / (1 + (shift - 1) * sigma)
            // Matches ZImageTurboLinearScheduler.applyScalarSigmaShift
            sigmas = sigmas.map { sigma in
                (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
            }
        } else if let w = width, let h = height {
            // Resolution-dependent shift (mflux formula)
            // Matches ZImageTurboLinearScheduler.applySigmaShift
            let y1: Float = 0.5
            let x1: Float = 256
            let m = (1.15 - y1) / (4096 - x1)
            let b = y1 - m * x1
            let mu = m * Float(w * h) / 256.0 + b
            let expMu = exp(mu)
            sigmas = sigmas.map { sigma in
                guard sigma > 0 else { return 0 }
                return expMu / (expMu + (1.0 / sigma - 1.0))
            }
        }

        return sigmas
    }

    private static func resampleWeights(_ base: [Float], toCount count: Int, normalizeMean: Bool) -> [Float] {
        guard count > 0 else { return [] }
        guard base.count > 1 else { return [Float](repeating: base.first ?? 1.0, count: count) }
        if count == base.count {
            return base
        }

        var out: [Float] = []
        out.reserveCapacity(count)
        let maxIndex = Float(base.count - 1)
        let denom = Float(max(count - 1, 1))
        for i in 0..<count {
            let pos = (Float(i) / denom) * maxIndex
            let lo = Int(pos.rounded(.down))
            let hi = min(lo + 1, base.count - 1)
            let t = pos - Float(lo)
            let value = base[lo] * (1 - t) + base[hi] * t
            out.append(value)
        }

        guard normalizeMean else { return out }
        let mean = out.reduce(0, +) / Float(out.count)
        guard mean > 0 else { return out }
        return out.map { $0 / mean }
    }

    /// Sample a timestep index using weighted probabilities.
    /// - Parameters:
    ///   - weights: Full weight array for all timesteps
    ///   - rng: Random number generator
    ///   - low: Minimum timestep index (inclusive), default 0
    ///   - high: Maximum timestep index (exclusive), default weights.count
    static func sampleWeightedTimestep(
        weights: [Float],
        rng: inout SplitMix64,
        low: Int = 0,
        high: Int? = nil
    ) -> Int {
        guard !weights.isEmpty else { return 0 }

        let effectiveHigh = min(high ?? weights.count, weights.count)
        let effectiveLow = max(low, 0)
        guard effectiveLow < effectiveHigh else { return effectiveLow }

        // If range is restricted, re-normalize weights for just that range
        let rangeWeights = Array(weights[effectiveLow..<effectiveHigh])
        let sum = rangeWeights.reduce(0, +)
        guard sum > 0 else { return effectiveLow }

        // Generate random value in [0, 1)
        let r = Float(rng.next() % 1_000_000) / 1_000_000.0

        // Find the bucket within the range
        var cumulative: Float = 0
        for (i, w) in rangeWeights.enumerated() {
            cumulative += w / sum
            if r < cumulative {
                return effectiveLow + i
            }
        }

        return effectiveHigh - 1
    }

    /// Sample sigma using ai-toolkit's sigmoid method: sigma = 1 - sigmoid(randn)
    /// This clusters timesteps around the middle (sigma ~0.5)
    /// - Parameters:
    ///   - rng: Random number generator
    ///   - low: Minimum timestep index (inclusive), default 0
    ///   - high: Maximum timestep index (exclusive), default numSteps
    ///   - numSteps: Total number of timesteps (for sigma conversion)
    static func sampleSigmoidTimestep(
        rng: inout SplitMix64,
        low: Int = 0,
        high: Int? = nil,
        numSteps: Int = 1000
    ) -> MLXArray {
        // Generate a random normal value using Box-Muller transform
        let u1 = max(Float(rng.next() % 1_000_000) / 1_000_000.0, 1e-10)
        let u2 = Float(rng.next() % 1_000_000) / 1_000_000.0
        let z = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)

        // Apply sigmoid: t = 1 / (1 + exp(-z))
        let t = 1.0 / (1.0 + exp(-z))

        // sigma = 1 - t (ai-toolkit: timestep = (1-t)*1000, sigma = timestep/1000 = 1-t)
        var sigma = 1.0 - t

        // Apply timestep range limits by clamping sigma
        // sigma = t / numSteps, so t = sigma * numSteps
        // low corresponds to sigmaLow = low / numSteps
        // high corresponds to sigmaHigh = high / numSteps
        let effectiveHigh = Float(high ?? numSteps)
        let sigmaLow = Float(low) / Float(numSteps)
        let sigmaHigh = effectiveHigh / Float(numSteps)

        // Clamp sigma to the valid range
        sigma = max(min(sigma, sigmaHigh - 0.0001), sigmaLow)

        // Clamp to valid range
        return MLXArray([max(min(sigma, 0.9999), 0.0001)])
    }
}

