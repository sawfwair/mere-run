import Foundation
@preconcurrency import MLX
import MLXRandom

public struct UniverSREnhancementOptions: Equatable, Sendable {
    public let inputRateHz: Int
    public let odeMethod: UniverSRODEMethod
    public let odeSteps: Int
    public let guidanceScale: Float
    public let seed: UInt64
    public let chunkSeconds: Int

    public init(
        inputRateHz: Int,
        odeMethod: UniverSRODEMethod = .midpoint,
        odeSteps: Int = 4,
        guidanceScale: Float = 1.5,
        seed: UInt64 = 42,
        chunkSeconds: Int = 10
    ) {
        self.inputRateHz = inputRateHz
        self.odeMethod = odeMethod
        self.odeSteps = odeSteps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.chunkSeconds = chunkSeconds
    }
}

public struct UniverSREnhancementResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let channels: Int
    public let frameCount: Int
    public let chunkCount: Int
    public let elapsedSeconds: Double
}

public enum UniverSRAudio {
    /// Windowed-sinc interpolation for the integer 8/12/16/24 kHz to 48 kHz ratios.
    public static func upsampleTo48k(_ samples: [Float], inputRate: Int) throws -> [Float] {
        guard !samples.isEmpty else { return [] }
        guard [8_000, 12_000, 16_000, 24_000].contains(inputRate) else {
            throw UniverSRError.unsupportedInputRate(inputRate)
        }
        let factor = 48_000 / inputRate
        let radius = 32
        var output = [Float](repeating: 0, count: samples.count * factor)
        for outputIndex in output.indices {
            let position = Double(outputIndex) / Double(factor)
            let center = Int(floor(position))
            var sum = 0.0
            var normalization = 0.0
            for sourceIndex in (center - radius + 1)...(center + radius) {
                let distance = position - Double(sourceIndex)
                guard abs(distance) < Double(radius) else { continue }
                let coefficient = sinc(distance) * sinc(distance / Double(radius))
                let reflected = reflectedIndex(sourceIndex, count: samples.count)
                sum += Double(samples[reflected]) * coefficient
                normalization += coefficient
            }
            output[outputIndex] = Float(sum / max(abs(normalization), 1e-12))
        }
        return output
    }

    private static func sinc(_ value: Double) -> Double {
        if abs(value) < 1e-12 { return 1 }
        let argument = Double.pi * value
        return sin(argument) / argument
    }

    private static func reflectedIndex(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        let upper = count - 1
        let period = upper * 2
        var value = index % period
        if value < 0 { value += period }
        return value > upper ? period - value : value
    }
}

public final class UniverSREnhancer {
    public typealias ProgressHandler = @Sendable (_ completedChunks: Int, _ totalChunks: Int) -> Void

    public let checkpoint: UniverSRCheckpoint
    public let dtype: DType
    private let model: UniverSRModel

    private init(checkpoint: UniverSRCheckpoint, dtype: DType, model: UniverSRModel) {
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.model = model
    }

    public static func load(
        resources: UniverSRResources,
        dtype: DType = .float32
    ) throws -> UniverSREnhancer {
        let checkpoint = try resources.resolve()
        let model = try UniverSRModel.load(checkpoint: checkpoint, dtype: dtype)
        return UniverSREnhancer(checkpoint: checkpoint, dtype: dtype, model: model)
    }

    public func enhance(
        lowResolution48kSamples: [Float],
        options: UniverSREnhancementOptions,
        progress: ProgressHandler? = nil
    ) throws -> UniverSREnhancementResult {
        guard !lowResolution48kSamples.isEmpty else { throw UniverSRError.invalidAudioBuffer }
        try validate(options)
        let started = ContinuousClock.now
        let configuration = checkpoint.configuration
        let chunkSize = options.chunkSeconds * configuration.transform.samplingRate
        let chunkCount = max(1, Int(ceil(Double(lowResolution48kSamples.count) / Double(chunkSize))))
        var output: [Float] = []
        output.reserveCapacity(lowResolution48kSamples.count)

        for chunkIndex in 0..<chunkCount {
            let start = chunkIndex * chunkSize
            let end = min(start + chunkSize, lowResolution48kSamples.count)
            let source = Array(lowResolution48kSamples[start..<end])
            let requiredLength = max(source.count, configuration.inference.minimumSamples)
            var padded = source
            if padded.count < requiredLength {
                padded.append(contentsOf: repeatElement(0, count: requiredLength - padded.count))
            }
            let enhanced = try enhanceChunk(
                padded,
                inputRateKHz: options.inputRateHz / 1_000,
                method: options.odeMethod,
                steps: options.odeSteps,
                guidanceScale: options.guidanceScale,
                seed: options.seed &+ UInt64(chunkIndex)
            )
            output.append(contentsOf: enhanced.prefix(source.count))
            MLX.Memory.clearCache()
            progress?(chunkIndex + 1, chunkCount)
        }

        let elapsed = started.duration(to: .now)
        return UniverSREnhancementResult(
            samples: output,
            sampleRate: configuration.transform.samplingRate,
            channels: 1,
            frameCount: output.count,
            chunkCount: chunkCount,
            elapsedSeconds: Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        )
    }

    private func validate(_ options: UniverSREnhancementOptions) throws {
        guard checkpoint.configuration.supportedInputRates.contains(options.inputRateHz) else {
            throw UniverSRError.unsupportedInputRate(options.inputRateHz)
        }
        guard options.odeSteps > 0 else {
            throw UniverSRError.invalidInferenceOption("ode_steps must be positive")
        }
        guard options.guidanceScale >= 0, options.guidanceScale.isFinite else {
            throw UniverSRError.invalidInferenceOption("guidance_scale must be finite and non-negative")
        }
        guard options.chunkSeconds >= 3 else {
            throw UniverSRError.invalidInferenceOption("chunk_seconds must be at least 3")
        }
    }

    private func enhanceChunk(
        _ samples: [Float],
        inputRateKHz: Int,
        method: UniverSRODEMethod,
        steps: Int,
        guidanceScale: Float,
        seed: UInt64
    ) throws -> [Float] {
        let configuration = checkpoint.configuration
        let waveform = MLXArray(samples).reshaped(1, 1, samples.count).asType(dtype)
        let representation = preprocess(waveform)
        guard let lowBinCount = configuration.model.frequencyBins(for: inputRateKHz) else {
            throw UniverSRError.unsupportedInputRate(inputRateKHz * 1_000)
        }
        let highStart = configuration.model.totalFrequencyBins - configuration.model.highFrequencyBins
        let condition = representation[0..., 0..<lowBinCount, 0..., 0...]
        let highShape = representation[
            0..., highStart..<configuration.model.totalFrequencyBins, 0..., 0...
        ].shape
        var state = MLXRandom.normal(highShape, key: MLXRandom.key(seed)).asType(dtype)
        let stepSize = 1 / Float(steps)

        for index in 0..<steps {
            let time = Float(index) * stepSize
            switch method {
            case .euler:
                state = state + stepSize * vectorField(
                    state,
                    time: time,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
            case .midpoint:
                let first = vectorField(
                    state,
                    time: time,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                let midpoint = state + (stepSize * 0.5) * first
                let second = vectorField(
                    midpoint,
                    time: time + stepSize * 0.5,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                state = state + stepSize * second
            case .rk4:
                let first = vectorField(
                    state,
                    time: time,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                let second = vectorField(
                    state + (stepSize * 0.5) * first,
                    time: time + stepSize * 0.5,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                let third = vectorField(
                    state + (stepSize * 0.5) * second,
                    time: time + stepSize * 0.5,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                let fourth = vectorField(
                    state + stepSize * third,
                    time: time + stepSize,
                    condition: condition,
                    inputRateKHz: inputRateKHz,
                    guidanceScale: guidanceScale
                )
                state = state + (stepSize / 6) * (first + 2 * second + 2 * third + fourth)
            }
            MLX.eval(state)
        }

        let overlap = max(0, lowBinCount - highStart)
        let generated = state[0..., overlap..<configuration.model.highFrequencyBins, 0..., 0...]
        let fullSpectrum = MLX.concatenated([condition, generated], axis: 1)
        let reconstructed = postprocess(fullSpectrum, length: samples.count)
            .reshaped(samples.count)
            .asType(.float32)
        MLX.eval(reconstructed)
        let values = reconstructed.asArray(Float.self)
        guard values.allSatisfy(\.isFinite) else { throw UniverSRError.nonFiniteOutput }
        return values
    }

    private func vectorField(
        _ state: MLXArray,
        time: Float,
        condition: MLXArray,
        inputRateKHz: Int,
        guidanceScale: Float
    ) -> MLXArray {
        let timeArray = MLXArray([time]).asType(dtype)
        let guided = model(
            state,
            time: timeArray,
            condition: condition,
            inputRateKHz: inputRateKHz
        )
        guard guidanceScale > 0, guidanceScale != 1 else { return guided }
        let unguided = model(
            state,
            time: timeArray,
            condition: nil,
            inputRateKHz: inputRateKHz
        )
        return (1 - guidanceScale) * unguided + guidanceScale * guided
    }

    private func preprocess(_ waveform: MLXArray) -> MLXArray {
        let transform = checkpoint.configuration.transform
        let spectrum = RoFormerDSP.stft(
            waveform,
            nFFT: transform.nFFT,
            hopLength: transform.hopLength,
            window: symmetricHannWindow(length: transform.nFFT, dtype: waveform.dtype)
        )
        let real = spectrum[0..., 0..<transform.nFFT / 2, 0..., 0] + transform.compressionEpsilon
        let imaginary = spectrum[0..., 0..<transform.nFFT / 2, 0..., 1]
        let magnitude = MLX.sqrt(real.square() + imaginary.square())
        let compressed = MLX.pow(magnitude, transform.alpha) * transform.beta
        let phase = MLX.atan2(imaginary, real)
        return MLX.stacked([compressed * MLX.cos(phase), compressed * MLX.sin(phase)], axis: -1)
    }

    private func postprocess(_ representation: MLXArray, length: Int) -> MLXArray {
        let transform = checkpoint.configuration.transform
        let real = representation[.ellipsis, 0]
        let imaginary = representation[.ellipsis, 1]
        let magnitude = MLX.sqrt(real.square() + imaginary.square()) / transform.beta
        let amplitude = MLX.pow(magnitude, 1 / transform.alpha)
        let phase = MLX.atan2(imaginary, real)
        let restored = MLX.stacked([amplitude * MLX.cos(phase), amplitude * MLX.sin(phase)], axis: -1)
        let nyquist = MLX.zeros(
            [restored.dim(0), 1, restored.dim(2), 2],
            dtype: restored.dtype
        )
        let fullSpectrum = MLX.concatenated([restored, nyquist], axis: 1).expandedDimensions(axis: 1)
        return RoFormerDSP.istft(
            fullSpectrum,
            channels: 1,
            length: length,
            nFFT: transform.nFFT,
            hopLength: transform.hopLength,
            window: symmetricHannWindow(length: transform.nFFT, dtype: representation.dtype),
            zeroDC: false
        ).squeezed(axis: 1).squeezed(axis: 1)
    }

    private func symmetricHannWindow(length: Int, dtype: DType) -> MLXArray {
        let positions = MLXArray((0..<length).map(Float.init)).asType(dtype)
        return MLXArray(0.5).asType(dtype)
            - MLXArray(0.5).asType(dtype)
                * MLX.cos(positions * (2 * Float.pi / Float(length - 1)))
    }
}
