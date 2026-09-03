import Foundation
import MLX

public struct Flux1Scheduler: Sendable, Hashable {
    public let sigmas: [Float]

    public init(
        steps: Int,
        imageSequenceLength: Int,
        configuration: Flux1SchedulerConfiguration
    ) {
        let count = max(steps, 1)
        let base = (0..<count).map { 1 - Float($0) / Float(count) }
        if configuration.useDynamicShifting {
            let slope = (configuration.maxShift - configuration.baseShift)
                / Float(configuration.maxImageSequenceLength - configuration.baseImageSequenceLength)
            let intercept = configuration.baseShift
                - slope * Float(configuration.baseImageSequenceLength)
            let mu = slope * Float(imageSequenceLength) + intercept
            let exponential = exp(mu)
            self.sigmas = base.map { sigma in
                exponential / (exponential + (1 / sigma - 1))
            } + [0]
        } else {
            self.sigmas = base.map { sigma in
                configuration.shift * sigma / (1 + (configuration.shift - 1) * sigma)
            } + [0]
        }
    }

    public func step(modelOutput: MLXArray, index: Int, sample: MLXArray) -> MLXArray {
        sample + modelOutput.asType(sample.dtype) * MLXArray(sigmas[index + 1] - sigmas[index])
    }
}

enum Flux1SampleBuilder {
    static let vaeCompression = 8
    static let patch = 2

    static func alignedResolution(width: Int, height: Int) -> (width: Int, height: Int) {
        let divisor = vaeCompression * patch
        return (
            max(divisor, width / divisor * divisor),
            max(divisor, height / divisor * divisor)
        )
    }

    static func pack(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let channels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        return latents
            .reshaped(batch, channels, height / 2, 2, width / 2, 2)
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped(batch, height / 2 * width / 2, channels * 4)
    }

    static func unpack(_ tokens: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = tokens.dim(0)
        let channels = tokens.dim(2) / 4
        return tokens
            .reshaped(batch, height / 2, width / 2, channels, 2, 2)
            .transposed(0, 3, 1, 4, 2, 5)
            .reshaped(batch, channels, height, width)
    }

    static func textIDs(length: Int) -> MLXArray {
        MLX.zeros([length, 3], dtype: .float32)
    }

    static func imageIDs(height: Int, width: Int) -> MLXArray {
        let tokenHeight = height / 2
        let tokenWidth = width / 2
        var values = [Float](repeating: 0, count: tokenHeight * tokenWidth * 3)
        for y in 0..<tokenHeight {
            for x in 0..<tokenWidth {
                let offset = (y * tokenWidth + x) * 3
                values[offset + 1] = Float(y)
                values[offset + 2] = Float(x)
            }
        }
        return MLXArray(values).reshaped(tokenHeight * tokenWidth, 3)
    }
}
