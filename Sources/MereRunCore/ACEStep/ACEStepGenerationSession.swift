import Foundation
import MLX

public struct ACEStepSessionRequest {
    public var caption: String
    public var lyrics: String
    public var config: ACEStepInferenceConfig
    public var lmConfig: ACEStep5HzLMGenerationConfig
    public var lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata
    public var lmCodeGenerationContext: ACEStepLMCodeGenerationContext?
    public var sourceLatents25Hz: MLXArray?
    public var sourceAudio48kHz: MLXArray?
    public var referenceTimbreLatents25Hz: [MLXArray]?
    public var referenceTimbreAudio48kHz: [MLXArray]?
    public var audioCoverStrength: Float
    public var vocalLanguage: String
    public var instruction: String?
    public var task: ACEStepTask
    public var repaintConfiguration: ACEStepRepaintConfiguration?
    public var flowEditConfiguration: ACEStepFlowEditConfiguration?
    public var useLanguageModel: Bool

    public init(
        caption: String,
        lyrics: String = "",
        config: ACEStepInferenceConfig = .init(),
        lmConfig: ACEStep5HzLMGenerationConfig = .init(),
        lmUserMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata = .init(),
        lmCodeGenerationContext: ACEStepLMCodeGenerationContext? = nil,
        sourceLatents25Hz: MLXArray? = nil,
        sourceAudio48kHz: MLXArray? = nil,
        referenceTimbreLatents25Hz: [MLXArray]? = nil,
        referenceTimbreAudio48kHz: [MLXArray]? = nil,
        audioCoverStrength: Float = 1,
        vocalLanguage: String = "en",
        instruction: String? = nil,
        task: ACEStepTask = .textToMusic,
        repaintConfiguration: ACEStepRepaintConfiguration? = nil,
        flowEditConfiguration: ACEStepFlowEditConfiguration? = nil,
        useLanguageModel: Bool = false
    ) {
        self.caption = caption
        self.lyrics = lyrics
        self.config = config
        self.lmConfig = lmConfig
        self.lmUserMetadata = lmUserMetadata
        self.lmCodeGenerationContext = lmCodeGenerationContext
        self.sourceLatents25Hz = sourceLatents25Hz
        self.sourceAudio48kHz = sourceAudio48kHz
        self.referenceTimbreLatents25Hz = referenceTimbreLatents25Hz
        self.referenceTimbreAudio48kHz = referenceTimbreAudio48kHz
        self.audioCoverStrength = audioCoverStrength
        self.vocalLanguage = vocalLanguage
        self.instruction = instruction
        self.task = task
        self.repaintConfiguration = repaintConfiguration
        self.flowEditConfiguration = flowEditConfiguration
        self.useLanguageModel = useLanguageModel
    }
}

public struct ACEStepCandidateMetrics: Codable, Hashable, Sendable {
    public var peak: Float
    public var rms: Float
    public var crestFactorDB: Float
    public var silenceRatio: Float
    public var clippingRatio: Float
    public var dcOffset: Float
    public var finiteRatio: Float
    public var spectralFlatness: Float?
    public var frameEnergyVariation: Float?
    public var periodicity: Float?
    public var temporalSpectralVariation: Float?
    public var tailEnergyRatio: Float?

    public init(
        peak: Float,
        rms: Float,
        crestFactorDB: Float,
        silenceRatio: Float,
        clippingRatio: Float,
        dcOffset: Float,
        finiteRatio: Float,
        spectralFlatness: Float? = nil,
        frameEnergyVariation: Float? = nil,
        periodicity: Float? = nil,
        temporalSpectralVariation: Float? = nil,
        tailEnergyRatio: Float? = nil
    ) {
        self.peak = peak
        self.rms = rms
        self.crestFactorDB = crestFactorDB
        self.silenceRatio = silenceRatio
        self.clippingRatio = clippingRatio
        self.dcOffset = dcOffset
        self.finiteRatio = finiteRatio
        self.spectralFlatness = spectralFlatness
        self.frameEnergyVariation = frameEnergyVariation
        self.periodicity = periodicity
        self.temporalSpectralVariation = temporalSpectralVariation
        self.tailEnergyRatio = tailEnergyRatio
    }
}

public struct ACEStepGeneratedCandidate {
    public var index: Int
    public var seed: UInt64
    public var audio: MLXArray
    public var score: Float
    public var metrics: ACEStepCandidateMetrics
    public var lmAudioCodeCount: Int?

    public init(
        index: Int,
        seed: UInt64,
        audio: MLXArray,
        score: Float,
        metrics: ACEStepCandidateMetrics,
        lmAudioCodeCount: Int? = nil
    ) {
        self.index = index
        self.seed = seed
        self.audio = audio
        self.score = score
        self.metrics = metrics
        self.lmAudioCodeCount = lmAudioCodeCount
    }
}

public struct ACEStepRankedGeneration {
    public var best: ACEStepGeneratedCandidate
    public var candidates: [ACEStepGeneratedCandidate]

    public init(
        best: ACEStepGeneratedCandidate,
        candidates: [ACEStepGeneratedCandidate]
    ) {
        self.best = best
        self.candidates = candidates
    }
}

/// A warm, reusable ACE-Step runtime with serialized generation and
/// deterministic best-of-N candidate ranking.
public final class ACEStepGenerationSession: @unchecked Sendable {
    public enum SessionError: LocalizedError {
        case invalidCandidateCount(Int)
        case mismatchedBatchCandidateCounts(
            requestCount: Int,
            candidateCount: Int
        )
        case noCandidates

        public var errorDescription: String? {
            switch self {
            case .invalidCandidateCount(let count):
                return "ACE-Step candidate count must be positive; got \(count)."
            case .mismatchedBatchCandidateCounts(
                let requestCount,
                let candidateCount
            ):
                return "ACE-Step batch has \(requestCount) requests but "
                    + "\(candidateCount) candidate counts."
            case .noCandidates:
                return "ACE-Step generation returned no candidates."
            }
        }
    }

    public let pipeline: ACEStepPipeline
    private let generationLock = NSLock()

    public init(pipeline: ACEStepPipeline) {
        self.pipeline = pipeline
    }

    public func generateBest(
        _ request: ACEStepSessionRequest,
        candidateCount: Int
    ) throws -> ACEStepRankedGeneration {
        guard candidateCount > 0 else {
            throw SessionError.invalidCandidateCount(candidateCount)
        }
        generationLock.lock()
        defer { generationLock.unlock() }

        let baseSeed = request.config.seed ?? UInt64.random(in: UInt64.min...UInt64.max)
        var candidates: [ACEStepGeneratedCandidate] = []
        candidates.reserveCapacity(candidateCount)
        for index in 0..<candidateCount {
            let seed = baseSeed &+ UInt64(index)
            var candidateRequest = request
            candidateRequest.config.seed = seed
            let generated = try generate(candidateRequest)
            let evaluation = ACEStepCandidateScorer.evaluate(generated.audio)
            candidates.append(
                ACEStepGeneratedCandidate(
                    index: index,
                    seed: seed,
                    audio: generated.audio,
                    score: evaluation.score,
                    metrics: evaluation.metrics,
                    lmAudioCodeCount: generated.lmAudioCodeCount
                )
            )
            Memory.clearCache()
        }
        guard let best = candidates.max(by: Self.candidateRanksBefore) else {
            throw SessionError.noCandidates
        }
        return ACEStepRankedGeneration(
            best: best,
            candidates: ACEStepCandidateScorer.rank(candidates)
        )
    }

    public func generateBatch(
        _ requests: [ACEStepSessionRequest],
        candidatesPerRequest: Int = 1
    ) throws -> [ACEStepRankedGeneration] {
        try requests.map {
            try generateBest($0, candidateCount: candidatesPerRequest)
        }
    }

    public func generateBatch(
        _ requests: [ACEStepSessionRequest],
        candidateCounts: [Int]
    ) throws -> [ACEStepRankedGeneration] {
        guard requests.count == candidateCounts.count else {
            throw SessionError.mismatchedBatchCandidateCounts(
                requestCount: requests.count,
                candidateCount: candidateCounts.count
            )
        }
        if let invalidCount = candidateCounts.first(where: { $0 <= 0 }) {
            throw SessionError.invalidCandidateCount(invalidCount)
        }
        return try zip(requests, candidateCounts).map {
            try generateBest($0.0, candidateCount: $0.1)
        }
    }

    public func planMusic(
        caption: String,
        lyrics: String,
        instruction: String,
        userMetadata: ACEStep5HzLMConstrainedSampler.UserMetadata,
        lmConfig: ACEStep5HzLMGenerationConfig
    ) throws -> ACEStepMusicPlan {
        generationLock.lock()
        defer { generationLock.unlock() }
        return try pipeline.planMusic(
            caption: caption,
            lyrics: lyrics,
            instruction: instruction,
            userMetadata: userMetadata,
            lmConfig: lmConfig
        )
    }

    private func generate(
        _ request: ACEStepSessionRequest
    ) throws -> (audio: MLXArray, lmAudioCodeCount: Int?) {
        if let flowEdit = request.flowEditConfiguration {
            return (
                try pipeline.generateFlowEdit(
                    targetCaption: request.caption,
                    targetLyrics: request.lyrics,
                    sourceLatents25Hz: request.sourceLatents25Hz,
                    sourceAudio48kHz: request.sourceAudio48kHz,
                    config: request.config,
                    flowEdit: flowEdit,
                    lmUserMetadata: request.lmUserMetadata,
                    referenceTimbreLatents25Hz:
                        request.referenceTimbreLatents25Hz,
                    referenceTimbreAudio48kHz:
                        request.referenceTimbreAudio48kHz,
                    vocalLanguage: request.vocalLanguage
                ),
                nil
            )
        }
        if request.useLanguageModel {
            let result = try pipeline.generatePromptToAudioWithLM(
                caption: request.caption,
                lyrics: request.lyrics,
                config: request.config,
                lmConfig: request.lmConfig,
                lmUserMetadata: request.lmUserMetadata,
                lmCodeGenerationContext: request.lmCodeGenerationContext,
                sourceLatents25Hz: request.sourceLatents25Hz,
                sourceAudio48kHz: request.sourceAudio48kHz,
                referenceTimbreLatents25Hz: request.referenceTimbreLatents25Hz,
                referenceTimbreAudio48kHz: request.referenceTimbreAudio48kHz,
                audioCoverStrength: request.audioCoverStrength,
                vocalLanguage: request.vocalLanguage,
                instruction: request.instruction,
                task: request.task,
                repaintConfiguration: request.repaintConfiguration
            )
            return (result.audio, result.lmResult.audioCodeValues.count)
        }

        return (
            try pipeline.generatePromptToAudio(
                caption: request.caption,
                lyrics: request.lyrics,
                config: request.config,
                lmUserMetadata: request.lmUserMetadata,
                sourceLatents25Hz: request.sourceLatents25Hz,
                sourceAudio48kHz: request.sourceAudio48kHz,
                referenceTimbreLatents25Hz: request.referenceTimbreLatents25Hz,
                referenceTimbreAudio48kHz: request.referenceTimbreAudio48kHz,
                audioCoverStrength: request.audioCoverStrength,
                vocalLanguage: request.vocalLanguage,
                instruction: request.instruction,
                task: request.task,
                repaintConfiguration: request.repaintConfiguration
            ),
            nil
        )
    }

    private static func candidateRanksBefore(
        _ lhs: ACEStepGeneratedCandidate,
        _ rhs: ACEStepGeneratedCandidate
    ) -> Bool {
        if lhs.score == rhs.score {
            return lhs.index > rhs.index
        }
        return lhs.score < rhs.score
    }

}

public enum ACEStepCandidateScorer {
    public static func rank(
        _ candidates: [ACEStepGeneratedCandidate]
    ) -> [ACEStepGeneratedCandidate] {
        candidates.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.index < rhs.index
            }
            return lhs.score > rhs.score
        }
    }

    public static func evaluate(
        _ audio: MLXArray
    ) -> (score: Float, metrics: ACEStepCandidateMetrics) {
        let flattened = audio.asType(.float32).reshaped(-1)
        MLX.eval(flattened)
        let values = flattened.asArray(Float.self)
        guard !values.isEmpty else {
            let metrics = ACEStepCandidateMetrics(
                peak: 0,
                rms: 0,
                crestFactorDB: 0,
                silenceRatio: 1,
                clippingRatio: 0,
                dcOffset: 0,
                finiteRatio: 0,
                spectralFlatness: 1,
                frameEnergyVariation: 0,
                periodicity: 0,
                temporalSpectralVariation: 0,
                tailEnergyRatio: 0
            )
            return (0, metrics)
        }

        var finiteCount = 0
        var silenceCount = 0
        var clippingCount = 0
        var peak: Float = 0
        var sum: Double = 0
        var sumSquares: Double = 0
        for value in values where value.isFinite {
            finiteCount += 1
            let magnitude = abs(value)
            peak = max(peak, magnitude)
            sum += Double(value)
            sumSquares += Double(value) * Double(value)
            if magnitude < 0.000_1 {
                silenceCount += 1
            }
            if magnitude >= 0.999 {
                clippingCount += 1
            }
        }

        let count = max(finiteCount, 1)
        let rms = Float(sqrt(sumSquares / Double(count)))
        let crestFactorDB = rms > 0
            ? 20 * log10(max(peak / rms, 1))
            : 0
        let channelCount = audio.ndim >= 2 ? max(1, audio.dim(audio.ndim - 1)) : 1
        let mono = monoSamples(values, channelCount: channelCount)
        let spectralFlatness = estimateSpectralFlatness(mono)
        let frameEnergyVariation = estimateFrameEnergyVariation(mono)
        let periodicity = estimatePeriodicity(mono)
        let temporalSpectralVariation = estimateTemporalSpectralVariation(mono)
        let tailEnergyRatio = estimateTailEnergyRatio(mono)
        let metrics = ACEStepCandidateMetrics(
            peak: peak,
            rms: rms,
            crestFactorDB: crestFactorDB,
            silenceRatio: Float(silenceCount) / Float(count),
            clippingRatio: Float(clippingCount) / Float(count),
            dcOffset: Float(abs(sum / Double(count))),
            finiteRatio: Float(finiteCount) / Float(values.count),
            spectralFlatness: spectralFlatness,
            frameEnergyVariation: frameEnergyVariation,
            periodicity: periodicity,
            temporalSpectralVariation: temporalSpectralVariation,
            tailEnergyRatio: tailEnergyRatio
        )

        let finiteScore = metrics.finiteRatio
        let activityScore = 1 - min(max(metrics.silenceRatio, 0), 1)
        let levelScore = min(max(rms / 0.02, 0), 1)
        let crestDistance = abs(crestFactorDB - 10)
        let crestScore = max(0, 1 - crestDistance / 18)
        let clippingScore = max(0, 1 - metrics.clippingRatio * 100)
        let dcScore = max(0, 1 - metrics.dcOffset * 20)
        let flatnessScore = 1 - min(max((spectralFlatness - 0.15) / 0.7, 0), 1)
        let variationScore = min(max(frameEnergyVariation / 0.35, 0), 1)
        let periodicityScore = min(max((periodicity - 0.05) / 0.35, 0), 1)
        let temporalStructureScore = min(
            max((temporalSpectralVariation - 0.65) / 0.9, 0),
            1
        )
        let tailContinuityScore = min(
            max((tailEnergyRatio - 0.02) / 0.28, 0),
            1
        )
        let structureScore =
            0.15 * flatnessScore
            + 0.10 * variationScore
            + 0.10 * periodicityScore
            + 0.45 * temporalStructureScore
            + 0.20 * tailContinuityScore
        let score = 100 * (
            0.20 * finiteScore
                + 0.10 * activityScore
                + 0.10 * levelScore
                + 0.10 * crestScore
                + 0.10 * clippingScore
                + 0.05 * dcScore
                + 0.35 * structureScore
        )
        return (score.isFinite ? score : 0, metrics)
    }

    private static func monoSamples(
        _ interleaved: [Float],
        channelCount: Int
    ) -> [Float] {
        guard channelCount > 1 else {
            return interleaved.map { $0.isFinite ? $0 : 0 }
        }
        let frameCount = interleaved.count / channelCount
        return (0..<frameCount).map { frameIndex in
            let start = frameIndex * channelCount
            let sum = interleaved[start..<(start + channelCount)].reduce(0.0) {
                $0 + Double($1.isFinite ? $1 : 0)
            }
            return Float(sum / Double(channelCount))
        }
    }

    private static func estimateFrameEnergyVariation(_ samples: [Float]) -> Float {
        let windowSize = min(2_048, samples.count)
        guard windowSize >= 256 else {
            return 0
        }
        let hopSize = max(1, windowSize / 2)
        var energies: [Double] = []
        var start = 0
        while start + windowSize <= samples.count {
            var sumSquares = 0.0
            for sample in samples[start..<(start + windowSize)] {
                sumSquares += Double(sample) * Double(sample)
            }
            energies.append(sqrt(sumSquares / Double(windowSize)))
            start += hopSize
        }
        guard !energies.isEmpty else {
            return 0
        }
        let mean = energies.reduce(0, +) / Double(energies.count)
        guard mean > 1e-9 else {
            return 0
        }
        let variance = energies.reduce(0) { partial, value in
            partial + (value - mean) * (value - mean)
        } / Double(energies.count)
        return Float(sqrt(variance) / mean)
    }

    private static func estimateSpectralFlatness(_ samples: [Float]) -> Float {
        let windowSize = min(1_024, samples.count)
        guard windowSize >= 256 else {
            return 1
        }
        let starts = analysisWindowStarts(
            sampleCount: samples.count,
            windowSize: windowSize,
            maximumCount: 8
        )
        let binCount = min(128, (windowSize / 2) - 1)
        var flatnessSum = 0.0
        for start in starts {
            var logPowerSum = 0.0
            var powerSum = 0.0
            for bin in 2..<(binCount + 2) {
                let power = goertzelPower(
                    samples,
                    start: start,
                    windowSize: windowSize,
                    bin: bin
                )
                powerSum += power
                logPowerSum += log(power)
            }
            let arithmeticMean = powerSum / Double(binCount)
            let geometricMean = exp(logPowerSum / Double(binCount))
            flatnessSum += min(max(geometricMean / max(arithmeticMean, 1e-20), 0), 1)
        }
        return Float(flatnessSum / Double(starts.count))
    }

    private static func estimatePeriodicity(_ samples: [Float]) -> Float {
        let windowSize = min(4_096, samples.count)
        guard windowSize >= 512 else {
            return 0
        }
        let starts = analysisWindowStarts(
            sampleCount: samples.count,
            windowSize: windowSize,
            maximumCount: 8
        )
        var periodicitySum = 0.0
        for start in starts {
            let window = samples[start..<(start + windowSize)]
            let mean = window.reduce(0.0) { $0 + Double($1) } / Double(windowSize)
            var best = 0.0
            let maximumLag = min(1_200, windowSize / 2)
            for lag in stride(from: 80, through: maximumLag, by: 8) {
                var product = 0.0
                var firstEnergy = 0.0
                var secondEnergy = 0.0
                for offset in 0..<(windowSize - lag) {
                    let first = Double(samples[start + offset]) - mean
                    let second = Double(samples[start + offset + lag]) - mean
                    product += first * second
                    firstEnergy += first * first
                    secondEnergy += second * second
                }
                let correlation = product
                    / sqrt(max(firstEnergy * secondEnergy, Double.leastNonzeroMagnitude))
                best = max(best, correlation)
            }
            periodicitySum += best
        }
        return Float(periodicitySum / Double(starts.count))
    }

    private static func estimateTailEnergyRatio(_ samples: [Float]) -> Float {
        guard samples.count >= 512 else {
            return 0
        }
        let tailCount = max(256, samples.count / 8)
        let bodyEnd = samples.count - tailCount
        guard bodyEnd > 0 else {
            return 0
        }
        let bodyRMS = rootMeanSquare(samples[0..<bodyEnd])
        let tailRMS = rootMeanSquare(samples[bodyEnd..<samples.count])
        guard bodyRMS > 1e-9 else {
            return 0
        }
        return Float(tailRMS / bodyRMS)
    }

    private static func rootMeanSquare(
        _ samples: ArraySlice<Float>
    ) -> Double {
        guard !samples.isEmpty else {
            return 0
        }
        let sumSquares = samples.reduce(0.0) { partial, sample in
            partial + Double(sample) * Double(sample)
        }
        return sqrt(sumSquares / Double(samples.count))
    }

    private static func estimateTemporalSpectralVariation(
        _ samples: [Float]
    ) -> Float {
        let windowSize = min(1_024, samples.count)
        guard windowSize >= 256 else {
            return 0
        }
        let segmentSpan = min(8_192, samples.count)
        let segmentStarts = analysisWindowStarts(
            sampleCount: samples.count,
            windowSize: segmentSpan,
            maximumCount: 8
        )
        let subframeCount = min(4, max(1, segmentSpan / windowSize))
        let maximumSubframeStart = max(0, segmentSpan - windowSize)
        let subframeOffsets = (0..<subframeCount).map { index in
            guard subframeCount > 1 else {
                return 0
            }
            return Int(
                (Double(maximumSubframeStart) * Double(index)
                    / Double(subframeCount - 1)).rounded()
            )
        }
        let binCount = min(64, (windowSize / 2) - 1)
        var spectra: [[Double]] = []
        spectra.reserveCapacity(segmentStarts.count)

        for segmentStart in segmentStarts {
            var spectrum: [Double] = []
            spectrum.reserveCapacity(binCount)
            for bin in 2..<(binCount + 2) {
                let meanPower = subframeOffsets.reduce(0.0) { partial, offset in
                    partial + goertzelPower(
                        samples,
                        start: segmentStart + offset,
                        windowSize: windowSize,
                        bin: bin
                    )
                } / Double(subframeCount)
                spectrum.append(log(max(meanPower, 1e-20)))
            }
            let mean = spectrum.reduce(0, +) / Double(spectrum.count)
            spectra.append(spectrum.map { $0 - mean })
        }

        var variationSum = 0.0
        for bin in 0..<binCount {
            let mean = spectra.reduce(0.0) {
                $0 + $1[bin]
            } / Double(spectra.count)
            let variance = spectra.reduce(0.0) {
                let difference = $1[bin] - mean
                return $0 + difference * difference
            } / Double(spectra.count)
            variationSum += sqrt(variance)
        }
        return Float(variationSum / Double(binCount))
    }

    private static func goertzelPower(
        _ samples: [Float],
        start: Int,
        windowSize: Int,
        bin: Int
    ) -> Double {
        let omega = 2 * Double.pi * Double(bin) / Double(windowSize)
        let coefficient = 2 * cos(omega)
        var previous = 0.0
        var previousPrevious = 0.0
        for offset in 0..<windowSize {
            let hann = 0.5 - 0.5
                * cos(2 * Double.pi * Double(offset) / Double(windowSize - 1))
            let current = Double(samples[start + offset]) * hann
                + coefficient * previous
                - previousPrevious
            previousPrevious = previous
            previous = current
        }
        return max(
            previous * previous
                + previousPrevious * previousPrevious
                - coefficient * previous * previousPrevious,
            1e-20
        )
    }

    private static func analysisWindowStarts(
        sampleCount: Int,
        windowSize: Int,
        maximumCount: Int
    ) -> [Int] {
        guard sampleCount > windowSize, maximumCount > 1 else {
            return [0]
        }
        let lastStart = sampleCount - windowSize
        return (0..<maximumCount).map { index in
            Int(
                (Double(lastStart) * Double(index) / Double(maximumCount - 1))
                    .rounded()
            )
        }
    }
}
