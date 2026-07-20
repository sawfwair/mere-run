import Foundation
import MLX

public enum Wan2GenerationError: LocalizedError, Sendable {
    case invalidResolution(width: Int, height: Int)
    case invalidFrameCount(Int)
    case invalidStepCount(Int)
    case sourceImageRequired

    public var errorDescription: String? {
        switch self {
        case .invalidResolution(let width, let height):
            return "Wan2.2 TI2V resolution must be positive, divisible by 32, and at most 901120 pixels; received \(width)x\(height)."
        case .invalidFrameCount(let count):
            return "Wan2.2 TI2V frame count must be 4n+1 and at least 5; received \(count)."
        case .invalidStepCount(let count):
            return "Wan2.2 TI2V step count must be positive; received \(count)."
        case .sourceImageRequired:
            return "Wan2.2 TI2V generation requires a source image."
        }
    }
}

public struct Wan2GenerationOptions: Hashable, Sendable {
    public let prompt: String
    public let negativePrompt: String
    public let sourceImageURL: URL
    public let outputURL: URL
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let steps: Int
    public let guidanceScale: Float
    public let shift: Float
    public let seed: UInt64
    public let fps: Int
    public let cameraConditioning: Wan2ProjectiveCameraConditioning?

    public init(
        prompt: String,
        negativePrompt: String,
        sourceImageURL: URL,
        outputURL: URL,
        width: Int = 1_280,
        height: Int = 704,
        numFrames: Int = 41,
        steps: Int = 40,
        guidanceScale: Float = 5,
        shift: Float = 5,
        seed: UInt64 = 42,
        fps: Int = 24,
        cameraConditioning: Wan2ProjectiveCameraConditioning? = nil
    ) throws {
        guard width > 0,
              height > 0,
              width % 32 == 0,
              height % 32 == 0,
              width * height <= 704 * 1_280 else {
            throw Wan2GenerationError.invalidResolution(width: width, height: height)
        }
        guard numFrames >= 5, (numFrames - 1) % 4 == 0 else {
            throw Wan2GenerationError.invalidFrameCount(numFrames)
        }
        guard steps > 0 else {
            throw Wan2GenerationError.invalidStepCount(steps)
        }
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.sourceImageURL = sourceImageURL
        self.outputURL = outputURL
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.seed = seed
        self.fps = fps
        self.cameraConditioning = cameraConditioning
    }

    public var latentShape: [Int] {
        [48, (numFrames - 1) / 4 + 1, height / 16, width / 16]
    }

    public var patchGridShape: [Int] {
        let latent = latentShape
        return [latent[1], latent[2] / 2, latent[3] / 2]
    }

    public var sequenceLength: Int {
        patchGridShape.reduce(1, *)
    }
}

public struct Wan2FlowMatchEulerScheduler: Sendable {
    public let timesteps: [Float]
    public let sigmas: [Float]
    private var stepIndex = 0

    public init(steps: Int, shift: Float = 5, trainTimesteps: Int = 1_000) {
        precondition(steps > 0)
        precondition(trainTimesteps > 1)
        precondition(shift > 0)
        var shifted: [Float] = []
        shifted.reserveCapacity(steps + 1)
        let trainSigmaMin = 1 / Float(trainTimesteps)
        let sigmaMin = shift * trainSigmaMin / (1 + (shift - 1) * trainSigmaMin)
        for index in 0..<steps {
            let fraction = steps == 1 ? 0 : Float(index) / Float(steps - 1)
            let sigma = 1 + (sigmaMin - 1) * fraction
            shifted.append(shift * sigma / (1 + (shift - 1) * sigma))
        }
        shifted.append(0)
        self.sigmas = shifted
        self.timesteps = shifted.dropLast().map { $0 * Float(trainTimesteps) }
    }

    public init(
        denoisingStepList: [Int],
        shift: Float = 5,
        trainTimesteps: Int = 1_000
    ) {
        precondition(!denoisingStepList.isEmpty)
        precondition(trainTimesteps > 1)
        precondition(shift > 0)
        precondition(denoisingStepList.allSatisfy { (1...trainTimesteps).contains($0) })
        precondition(zip(
            denoisingStepList,
            denoisingStepList.dropFirst()
        ).allSatisfy { $0.0 > $0.1 })
        let scheduled = denoisingStepList.map { trainingStep -> Float in
            let sigma = Float(trainingStep) / Float(trainTimesteps)
            return shift * sigma / (1 + (shift - 1) * sigma)
        }
        self.sigmas = scheduled + [0]
        self.timesteps = scheduled.map { $0 * Float(trainTimesteps) }
    }

    public mutating func step(velocity: MLXArray, sample: MLXArray) -> MLXArray {
        precondition(stepIndex < timesteps.count)
        let delta = sigmas[stepIndex + 1] - sigmas[stepIndex]
        stepIndex += 1
        return sample + delta * velocity
    }

    public mutating func reset() {
        stepIndex = 0
    }
}

public struct Wan2CausalForcingScheduler: Sendable {
    public let timesteps: [Float]

    public init(shift: Float = 5) {
        precondition(shift > 0)
        let trainingIndices: [Float] = [1, 0.75, 0.5, 0.25]
        self.timesteps = trainingIndices.map { sigma in
            1_000 * shift * sigma / (1 + (shift - 1) * sigma)
        }
    }

    public func predictClean(flow: MLXArray, sample: MLXArray, timestep: Float) -> MLXArray {
        (sample.asType(.float32) - (timestep / 1_000) * flow.asType(.float32))
            .asType(flow.dtype)
    }

    public func addNoise(clean: MLXArray, noise: MLXArray, timestep: Float) -> MLXArray {
        let sigma = timestep / 1_000
        return ((1 - sigma) * clean.asType(.float32) + sigma * noise.asType(.float32))
            .asType(noise.dtype)
    }
}

public struct Wan2UniPCScheduler {
    public let timesteps: [Float]
    public let sigmas: [Float]
    private var modelOutputs: [MLXArray?] = [nil, nil]
    private var lowerOrderCount = 0
    private var lastSample: MLXArray?
    private var stepIndex = 0
    private var currentOrder = 1

    public init(steps: Int, shift: Float = 5, trainTimesteps: Int = 1_000) {
        precondition(steps > 0)
        precondition(shift > 0)
        var shifted: [Float] = []
        var scheduledTimesteps: [Float] = []
        shifted.reserveCapacity(steps + 1)
        scheduledTimesteps.reserveCapacity(steps)
        for index in 0..<steps {
            // Matches upstream `np.linspace(sigma_max, sigma_min,
            // steps + 1)[:-1]`. Its training schedule stores sigma_max
            // `(trainTimesteps - 1) / trainTimesteps` as float32 before NumPy
            // builds the inference grid in float64. Upstream then derives
            // integer timesteps before storing the shifted sigmas as float32.
            let sigmaMaximum = Double(
                Float(Double(trainTimesteps - 1) / Double(trainTimesteps))
            )
            let sigma = sigmaMaximum * (1 - Double(index) / Double(steps))
            let shift64 = Double(shift)
            let scheduled = shift64 * sigma / (1 + (shift64 - 1) * sigma)
            shifted.append(Float(scheduled))
            scheduledTimesteps.append(Float(Int(scheduled * Double(trainTimesteps))))
        }
        shifted.append(0)
        self.sigmas = shifted
        self.timesteps = scheduledTimesteps
    }

    public mutating func step(modelOutput: MLXArray, sample initialSample: MLXArray) -> MLXArray {
        precondition(stepIndex < timesteps.count)
        let converted = initialSample.asType(.float32) - sigmas[stepIndex] * modelOutput.asType(.float32)
        var sample = initialSample.asType(.float32)
        if stepIndex > 0, let lastSample {
            sample = correct(
                currentModelOutput: converted,
                lastSample: lastSample,
                currentSample: sample,
                order: currentOrder
            )
        }

        modelOutputs[0] = modelOutputs[1]
        modelOutputs[1] = converted
        currentOrder = min(2, timesteps.count - stepIndex, lowerOrderCount + 1)
        lastSample = sample
        let previous = predict(sample: sample, order: currentOrder)
        lowerOrderCount = min(lowerOrderCount + 1, 2)
        stepIndex += 1
        return previous.asType(.float32)
    }

    private func predict(sample: MLXArray, order: Int) -> MLXArray {
        let sigmaTarget = Double(sigmas[stepIndex + 1])
        let sigmaSource = Double(sigmas[stepIndex])
        let alphaTarget = 1 - sigmaTarget
        let alphaSource = 1 - sigmaSource
        let lambdaTarget = log(alphaTarget) - log(sigmaTarget)
        let lambdaSource = log(alphaSource) - log(sigmaSource)
        let h = lambdaTarget - lambdaSource
        let phi = expm1(-h)
        guard let current = modelOutputs[1] else { preconditionFailure("Missing UniPC model output") }

        var result = Float(sigmaTarget / sigmaSource) * sample
            - Float(alphaTarget * phi) * current
        if order == 2, stepIndex > 0, let previous = modelOutputs[0] {
            let sigmaHistory = Double(sigmas[stepIndex - 1])
            let alphaHistory = 1 - sigmaHistory
            let lambdaHistory = log(alphaHistory) - log(sigmaHistory)
            let ratio = (lambdaHistory - lambdaSource) / h
            let firstDifference = (previous - current) / Float(ratio)
            result = result - Float(alphaTarget * phi * 0.5) * firstDifference
        }
        return result
    }

    private func correct(
        currentModelOutput: MLXArray,
        lastSample: MLXArray,
        currentSample: MLXArray,
        order: Int
    ) -> MLXArray {
        let sigmaTarget = Double(sigmas[stepIndex])
        let sigmaSource = Double(sigmas[stepIndex - 1])
        let alphaTarget = 1 - sigmaTarget
        let alphaSource = 1 - sigmaSource
        let lambdaTarget = log(alphaTarget) - log(sigmaTarget)
        let lambdaSource = log(alphaSource) - log(sigmaSource)
        let h = lambdaTarget - lambdaSource
        let phi = expm1(-h)
        guard let previousModelOutput = modelOutputs[1] else {
            preconditionFailure("Missing UniPC previous output")
        }

        var historicalCorrection = MLX.zeros(currentSample.shape, dtype: .float32)
        let coefficients: [Float]
        if order == 2, stepIndex > 1, let historicalOutput = modelOutputs[0] {
            let sigmaHistory = Double(sigmas[stepIndex - 2])
            let alphaHistory = 1 - sigmaHistory
            let lambdaHistory = log(alphaHistory) - log(sigmaHistory)
            let ratio = (lambdaHistory - lambdaSource) / h
            let firstDifference = (historicalOutput - previousModelOutput) / Float(ratio)
            coefficients = Self.correctorCoefficients(ratio: ratio, h: -h, phi: phi)
            historicalCorrection = coefficients[0] * firstDifference
        } else {
            coefficients = [0.5]
        }
        let currentDifference = currentModelOutput - previousModelOutput
        return Float(sigmaTarget / sigmaSource) * lastSample
            - Float(alphaTarget * phi) * previousModelOutput
            - Float(alphaTarget * phi) * (historicalCorrection + coefficients.last! * currentDifference)
    }

    private static func correctorCoefficients(ratio: Double, h: Double, phi: Double) -> [Float] {
        let firstPhi = expm1(h)
        var higherPhi = firstPhi / h - 1
        let b0 = higherPhi / phi
        higherPhi = higherPhi / h - 0.5
        let b1 = higherPhi * 2 / phi
        let first = (b0 - b1) / (1 - ratio)
        return [Float(first), Float(b0 - first)]
    }
}

public enum Wan2TI2VConditioning {
    public static func latentMask(shape: [Int], dtype: DType = .float32) -> MLXArray {
        precondition(shape.count == 4)
        let firstFrame = MLX.zeros([shape[0], 1, shape[2], shape[3]], dtype: dtype)
        let remaining = MLX.ones([shape[0], max(shape[1] - 1, 0), shape[2], shape[3]], dtype: dtype)
        return shape[1] == 1 ? firstFrame : MLX.concatenated([firstFrame, remaining], axis: 1)
    }

    public static func tokenMask(
        latentShape: [Int],
        patchSize: [Int] = [1, 2, 2],
        dtype: DType = .float32
    ) -> MLXArray {
        precondition(latentShape.count == 4)
        precondition(patchSize.count == 3)
        let temporal = latentShape[1] / patchSize[0]
        let height = latentShape[2] / patchSize[1]
        let width = latentShape[3] / patchSize[2]
        let firstFrameTokens = height * width
        let total = temporal * firstFrameTokens
        let frozen = MLX.zeros([1, firstFrameTokens], dtype: dtype)
        let denoised = MLX.ones([1, max(total - firstFrameTokens, 0)], dtype: dtype)
        return total == firstFrameTokens ? frozen : MLX.concatenated([frozen, denoised], axis: 1)
    }

    public static func blend(imageLatent: MLXArray, noise: MLXArray, mask: MLXArray) -> MLXArray {
        (1 - mask) * imageLatent + mask * noise
    }
}
