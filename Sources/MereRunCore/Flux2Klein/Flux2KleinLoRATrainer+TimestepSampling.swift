import Foundation
import MLX
import MLXNN
import MLXRandom

extension Flux2KleinLoRATrainer {
    // MARK: - Timestep Sampling

    /// Compute sampling weights for timesteps based on strategy.
    static func computeTimestepWeights(
        numSteps: Int,
        strategy: Flux2TimestepSamplingStrategy
    ) -> [Float] {
        guard numSteps > 0 else { return [] }

        var weights: [Float]

        switch strategy {
        case .uniform:
            weights = [Float](repeating: 1.0 / Float(numSteps), count: numSteps)

        case .bellCurve:
            weights = (0..<numSteps).map { i in
                let x = Float(i)
                let n = Float(numSteps)
                return exp(-2.0 * pow((x - n / 2.0) / n, 2))
            }
            let minW = weights.min() ?? 0
            weights = weights.map { $0 - minW }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .contentFocused:
            weights = (0..<numSteps).map { i in
                let t = Float(i) / Float(max(numSteps - 1, 1))
                return pow(1.0 - t, 3)
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .styleFocused:
            weights = (0..<numSteps).map { i in
                let t = Float(i) / Float(max(numSteps - 1, 1))
                return pow(t, 3)
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }

        case .logitNormal:
            // Logit-normal distribution: sample u ~ N(mean, std), then t = sigmoid(u)
            // This concentrates samples in mid-range timesteps (OneTrainer default for FLUX.2)
            let mean: Float = 0.0
            let std: Float = 1.0
            weights = (0..<numSteps).map { i in
                let t = (Float(i) + 0.5) / Float(numSteps)  // center of bucket
                guard t > 0 && t < 1 else { return 0 }
                let logit = log(t / (1 - t))
                // PDF of logit-normal at t
                let pdf = exp(-0.5 * pow((logit - mean) / std, 2)) / (std * sqrt(2 * .pi) * t * (1 - t))
                return pdf
            }
            let sum = weights.reduce(0, +)
            if sum > 0 {
                weights = weights.map { $0 / sum }
            }
        }

        return weights
    }

    /// Build the training sigma table for FlowMatch / rectified flow.
    ///
    /// Applies time shift to match mflux FlowMatchEulerDiscreteScheduler.
    /// Always uses mu=1.0 and terminal stretch (mflux trains all FLUX.2 models this way).
    static func computeTrainingSigmas(
        numSteps: Int,
        numTrainTimesteps: Int
    ) -> [Float] {
        guard numSteps > 0 else { return [] }
        guard numSteps > 1 else { return [1.0] }

        let maxT = Float(max(numTrainTimesteps, 1))
        let minT: Float = 1.0
        let step = (maxT - minT) / Float(numSteps - 1)

        // Compute linear sigmas first (matches mflux)
        let sigmasLinear = (0..<numSteps).map { i in
            let t = maxT - Float(i) * step
            return t / maxT
        }

        // Apply time shift with mu=1.0 (mflux uses this for all FLUX.2 models)
        let mu: Float = 1.0
        let expMu = exp(mu)
        let sigmasShifted = sigmasLinear.map { sigma -> Float in
            guard sigma > 0 else { return 0 }
            return expMu / (expMu + pow(1.0 / sigma - 1.0, 1.0))
        }

        // Apply terminal stretch to 0.02 (mflux does this for all FLUX.2 models)
        let shiftTerminal: Float = 0.02
        let oneMinusSigmas = sigmasShifted.map { 1.0 - $0 }
        guard let lastOms = oneMinusSigmas.last, lastOms > 0 else {
            return sigmasShifted
        }
        let scaleFactor = lastOms / (1.0 - shiftTerminal)
        return oneMinusSigmas.map { 1.0 - ($0 / scaleFactor) }
    }

    /// Compute per-timestep *loss* weights (mean-normalized) used by ai-toolkit when
    /// `timestep_type: weighted`.
    static func computeTimestepLossWeights(
        numSteps: Int,
        strategy: Flux2TimestepLossWeightingStrategy
    ) -> [Float] {
        guard numSteps > 0 else { return [] }
        guard strategy != .none else { return [Float](repeating: 1.0, count: numSteps) }

        // Bell-shaped mean-normalized weighting (bsmntw).
        let n = Float(numSteps)
        var weights = (0..<numSteps).map { i in
            let x = Float(i)
            return exp(-2.0 * pow((x - n / 2.0) / n, 2))
        }
        let minW = weights.min() ?? 0
        weights = weights.map { $0 - minW }
        let sum = weights.reduce(0, +)
        guard sum > 0 else { return [Float](repeating: 1.0, count: numSteps) }

        // Scale so mean(weight)=1.
        let scale = n / sum
        return weights.map { $0 * scale }
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
}

