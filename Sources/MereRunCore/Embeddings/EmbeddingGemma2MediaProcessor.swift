import Foundation
import MediaIO
import MLX

enum EmbeddingGemma2MediaProcessor {
    struct Image {
        let pixels: MLXArray
        let positions: [[Int]]
        let softTokens: Int
    }
    struct Audio {
        let features: MLXArray
        let validFrames: [Bool]
        var softTokens: Int { stride(from: 0, to: validFrames.count, by: 4).filter { validFrames[$0] }.count }
    }

    static func image(_ url: URL, config: EmbeddingGemma2ProcessorConfig.Vision) throws -> Image {
        try image(MediaImageIO.decode(url), config: config)
    }

    static func image(_ source: MediaImage, config: EmbeddingGemma2ProcessorConfig.Vision) throws -> Image {
        let patch = config.patchSize, kernel = config.poolingKernelSize, multiple = patch * kernel
        let budget = config.maxSoftTokens * kernel * kernel
        let factor = sqrt(Double(budget * patch * patch) / Double(source.height * source.width))
        var height = Int(floor(Double(source.height) * factor / Double(multiple))) * multiple
        var width = Int(floor(Double(source.width) * factor / Double(multiple))) * multiple
        if height == 0 {
            height = multiple
            width = min(Int(floor(Double(source.width) / Double(source.height))) * multiple, config.maxSoftTokens * multiple)
        } else if width == 0 {
            width = multiple
            height = min(Int(floor(Double(source.height) / Double(source.width))) * multiple, config.maxSoftTokens * multiple)
        }
        guard height > 0, width > 0, height * width <= budget * patch * patch else {
            throw EmbeddingGemma2Error.invalidInput("Image cannot fit the configured patch budget.")
        }
        let resized = try MediaImageIO.bicubicResizedRGB(source, width: width, height: height)
        let rows = height / patch, columns = width / patch, patchDim = patch * patch * 3
        var pixels = [Float](repeating: 0, count: budget * patchDim)
        var positions = [[Int]](repeating: [-1, -1], count: budget)
        for y in 0..<rows {
            for x in 0..<columns {
                let index = y * columns + x
                positions[index] = [x, y]
                for py in 0..<patch {
                    for px in 0..<patch {
                        let sourceIndex = ((y * patch + py) * width + x * patch + px) * 4
                        let destination = index * patchDim + (py * patch + px) * 3
                        for channel in 0..<3 { pixels[destination + channel] = Float(resized.rgba8[sourceIndex + channel]) * config.rescaleFactor }
                    }
                }
            }
        }
        return Image(pixels: MLXArray(pixels, [1, budget, patchDim]), positions: positions,
                     softTokens: rows * columns / (kernel * kernel))
    }

    static func audio(_ url: URL, config: EmbeddingGemma2ProcessorConfig.Audio) throws -> Audio {
        let metadata = try MediaAudioIO.probe(url)
        guard metadata.durationSeconds > 0, metadata.durationSeconds <= 30 else {
            throw EmbeddingGemma2Error.invalidInput("Each audio segment must contain at most 30 seconds. Split longer audio into ordered segments.")
        }
        let decoded = try MediaAudioIO.decode(url, targetSampleRate: config.samplingRate, channels: 1)
        return try audio(samples: decoded.samples, config: config)
    }

    static func audio(samples: [Float], config: EmbeddingGemma2ProcessorConfig.Audio) throws -> Audio {
        guard samples.count > config.frameLength / 2, samples.count <= config.samplingRate * 30,
              samples.allSatisfy(\.isFinite) else {
            throw EmbeddingGemma2Error.invalidInput("Audio must contain finite samples and be between 10 milliseconds and 30 seconds.")
        }
        let sampleCount = ((samples.count + 127) / 128) * 128
        let padded = [Float](repeating: 0, count: config.frameLength / 2) + samples
            + [Float](repeating: 0, count: sampleCount - samples.count)
        let count = (padded.count - config.frameLength - 1) / config.hopLength + 1
        let window = (0..<config.frameLength).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(config.frameLength))) }
        var frames = [Float](repeating: 0, count: count * config.fftLength)
        var valid: [Bool] = []
        for index in 0..<count {
            let start = index * config.hopLength
            valid.append(start + config.frameLength < config.frameLength / 2 + samples.count)
            for offset in 0..<config.frameLength { frames[index * config.fftLength + offset] = padded[start + offset] * window[offset] }
        }
        let frequencies = config.fftLength / 2 + 1
        let melMin = 2595 * log10(1 + config.minFrequency / 700)
        let melMax = 2595 * log10(1 + config.maxFrequency / 700)
        let edges = (0..<(config.featureSize + 2)).map { index in
            700 * (pow(10, (melMin + (melMax - melMin) * Double(index) / Double(config.featureSize + 1)) / 2595) - 1)
        }
        var filters = [Float](repeating: 0, count: frequencies * config.featureSize)
        for frequency in 0..<frequencies {
            let hz = Double(frequency) * Double(config.samplingRate) / Double(config.fftLength)
            for mel in 0..<config.featureSize {
                filters[frequency * config.featureSize + mel] = Float(max(0, min((hz - edges[mel]) / (edges[mel + 1] - edges[mel]),
                    (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]))))
            }
        }
        let spectra = abs(MLXFFT.rfft(MLXArray(frames, [count, config.fftLength]), axis: -1))
        let features = log(matmul(spectra, MLXArray(filters, [frequencies, config.featureSize])) + MLXArray(Float(config.melFloor)))
        return Audio(features: (features * MLXArray(valid.map { Float($0 ? 1 : 0) }, [count, 1])).expandedDimensions(axis: 0), validFrames: valid)
    }
}
