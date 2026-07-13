import AudioCodecs
import Foundation

public struct MuScriptorTempo: Codable, Hashable, Sendable {
    public let bpm: Double
    public let beatOffsetSeconds: Double
    public let confidence: Double

    public init(bpm: Double, beatOffsetSeconds: Double, confidence: Double) {
        self.bpm = bpm
        self.beatOffsetSeconds = beatOffsetSeconds
        self.confidence = confidence
    }
}

public struct MuScriptorTimeSignature: Codable, Hashable, Sendable {
    public let numerator: Int
    public let denominator: Int
    public let downbeatOffsetSeconds: Double
    public let confidence: Double

    public init(
        numerator: Int,
        denominator: Int,
        downbeatOffsetSeconds: Double,
        confidence: Double
    ) {
        self.numerator = numerator
        self.denominator = denominator
        self.downbeatOffsetSeconds = downbeatOffsetSeconds
        self.confidence = confidence
    }

    public var name: String { "\(numerator)/\(denominator)" }
}

public struct MuScriptorKeySignature: Codable, Hashable, Sendable {
    public let name: String
    public let sharpsFlats: Int
    public let isMinor: Bool
    public let confidence: Double

    public init(name: String, sharpsFlats: Int, isMinor: Bool, confidence: Double) {
        self.name = name
        self.sharpsFlats = sharpsFlats
        self.isMinor = isMinor
        self.confidence = confidence
    }
}

public struct MuScriptorBeat: Codable, Hashable, Sendable {
    public let timeSeconds: Double
    public let strength: Double
    public let isDownbeat: Bool

    public init(timeSeconds: Double, strength: Double, isDownbeat: Bool) {
        self.timeSeconds = timeSeconds
        self.strength = strength
        self.isDownbeat = isDownbeat
    }
}

public struct MuScriptorMusicalContext: Codable, Hashable, Sendable {
    public let tempo: MuScriptorTempo?
    public let timeSignature: MuScriptorTimeSignature?
    public let keySignature: MuScriptorKeySignature?
    public let beats: [MuScriptorBeat]

    public init(
        tempo: MuScriptorTempo?,
        timeSignature: MuScriptorTimeSignature?,
        keySignature: MuScriptorKeySignature?,
        beats: [MuScriptorBeat]
    ) {
        self.tempo = tempo
        self.timeSignature = timeSignature
        self.keySignature = keySignature
        self.beats = beats
    }

    public var summary: String {
        var fields: [String] = []
        if let tempo {
            fields.append(String(format: "%.2f BPM (%.2f)", tempo.bpm, tempo.confidence))
        }
        if let timeSignature {
            fields.append("\(timeSignature.name) (\(Self.confidence(timeSignature.confidence)))")
        }
        if let keySignature {
            fields.append("\(keySignature.name) (\(Self.confidence(keySignature.confidence)))")
        }
        return fields.isEmpty ? "no reliable musical context detected" : fields.joined(separator: ", ")
    }

    private static func confidence(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

/// Estimates tempo, beat phase, meter, and key from decoded audio plus MuScriptor notes.
///
/// MuScriptor deliberately predicts absolute note events rather than score-level metadata.
/// This post-processor combines spectral-flux accents with note onsets for rhythm and uses
/// duration-weighted pitch-class profiles for key recognition. Low-evidence fields remain nil.
public struct MuScriptorMusicalContextAnalyzer: Sendable {
    public let minimumBPM: Double
    public let maximumBPM: Double

    public init(minimumBPM: Double = 45, maximumBPM: Double = 220) {
        precondition(minimumBPM > 0 && maximumBPM > minimumBPM)
        self.minimumBPM = minimumBPM
        self.maximumBPM = maximumBPM
    }

    public func analyze(
        samples: [Float],
        sampleRate: Int,
        notes: [MuScriptorNote]
    ) throws -> MuScriptorMusicalContext {
        guard sampleRate > 0 else {
            throw MuScriptorError.invalidAudio("sample rate must be positive")
        }

        let key = estimateKey(notes: notes)
        guard samples.count >= sampleRate * 4 else {
            return MuScriptorMusicalContext(
                tempo: nil,
                timeSignature: nil,
                keySignature: key,
                beats: []
            )
        }

        let hopLength = max(128, sampleRate / 64)
        let audioEnvelope = try spectralFluxEnvelope(
            samples: samples,
            sampleRate: sampleRate,
            hopLength: hopLength
        )
        let noteEnvelope = noteOnsetEnvelope(
            notes: notes,
            frameCount: audioEnvelope.count,
            sampleRate: sampleRate,
            hopLength: hopLength
        )
        let envelope = combine(audio: audioEnvelope, notes: noteEnvelope)
        guard envelope.max() ?? 0 > 1e-6,
              let tempoEstimate = estimateTempo(
                envelope: envelope,
                framesPerSecond: Double(sampleRate) / Double(hopLength)
              ) else {
            return MuScriptorMusicalContext(
                tempo: nil,
                timeSignature: nil,
                keySignature: key,
                beats: []
            )
        }

        let rawBeats = beatGrid(
            envelope: envelope,
            periodFrames: tempoEstimate.periodFrames,
            phaseFrame: tempoEstimate.phaseFrame,
            framesPerSecond: Double(sampleRate) / Double(hopLength),
            durationSeconds: Double(samples.count) / Double(sampleRate)
        )
        let meter = estimateMeter(beats: rawBeats)
        let beats = rawBeats.enumerated().map { index, beat in
            MuScriptorBeat(
                timeSeconds: beat.timeSeconds,
                strength: beat.strength,
                isDownbeat: meter.map {
                    (index - $0.phaseIndex).isMultiple(of: $0.numerator)
                } ?? false
            )
        }
        let timeSignature = meter.flatMap { value -> MuScriptorTimeSignature? in
            guard beats.indices.contains(value.phaseIndex) else { return nil }
            return MuScriptorTimeSignature(
                numerator: value.numerator,
                denominator: 4,
                downbeatOffsetSeconds: beats[value.phaseIndex].timeSeconds,
                confidence: value.confidence
            )
        }
        let tempo = MuScriptorTempo(
            bpm: tempoEstimate.bpm,
            beatOffsetSeconds: beats.first?.timeSeconds ?? 0,
            confidence: tempoEstimate.confidence
        )
        return MuScriptorMusicalContext(
            tempo: tempo,
            timeSignature: timeSignature,
            keySignature: key,
            beats: beats
        )
    }

    private struct TempoEstimate {
        let bpm: Double
        let periodFrames: Double
        let phaseFrame: Int
        let confidence: Double
    }

    private struct MeterEstimate {
        let numerator: Int
        let phaseIndex: Int
        let confidence: Double
    }

    private func spectralFluxEnvelope(
        samples: [Float],
        sampleRate: Int,
        hopLength: Int
    ) throws -> [Double] {
        let fftSize = 1_024
        let plan = try RealFFTPlan(size: fftSize)
        let window = (0..<fftSize).map { index in
            0.5 - 0.5 * cos(2 * Float.pi * Float(index) / Float(fftSize))
        }
        let frameCount = max(1, 1 + (samples.count - fftSize) / hopLength)
        var flux = [Double](repeating: 0, count: frameCount)
        var previous = [Double](repeating: 0, count: fftSize / 2 + 1)
        var frame = [Float](repeating: 0, count: fftSize)

        for frameIndex in 0..<frameCount {
            let start = frameIndex * hopLength
            for sampleIndex in 0..<fftSize {
                frame[sampleIndex] = samples[start + sampleIndex] * window[sampleIndex]
            }
            let power = plan.powerSpectrum(frame)
            var value = 0.0
            for bin in 1..<power.count {
                let current = log1p(Double(max(power[bin], 0)))
                value += max(0, current - previous[bin])
                previous[bin] = current
            }
            flux[frameIndex] = value
        }

        let thresholdRadius = max(2, Int((0.20 * Double(sampleRate) / Double(hopLength)).rounded()))
        var prefix = [Double](repeating: 0, count: flux.count + 1)
        for index in flux.indices {
            prefix[index + 1] = prefix[index] + flux[index]
        }
        var filtered = [Double](repeating: 0, count: flux.count)
        for index in flux.indices {
            let lower = max(0, index - thresholdRadius)
            let upper = min(flux.count, index + thresholdRadius + 1)
            let localMean = (prefix[upper] - prefix[lower]) / Double(upper - lower)
            filtered[index] = max(0, flux[index] - localMean)
        }
        return normalized(filtered)
    }

    private func noteOnsetEnvelope(
        notes: [MuScriptorNote],
        frameCount: Int,
        sampleRate: Int,
        hopLength: Int
    ) -> [Double] {
        var envelope = [Double](repeating: 0, count: frameCount)
        for note in notes {
            let frame = Int((note.onset * Double(sampleRate) / Double(hopLength)).rounded())
            guard envelope.indices.contains(frame) else { continue }
            envelope[frame] += note.isDrum ? 2.0 : 1.0
        }
        envelope = envelope.map(log1p)
        guard envelope.count > 2 else { return normalized(envelope) }
        var smoothed = envelope
        for index in 1..<(envelope.count - 1) {
            smoothed[index] = 0.25 * envelope[index - 1]
                + 0.5 * envelope[index]
                + 0.25 * envelope[index + 1]
        }
        return normalized(smoothed)
    }

    private func combine(audio: [Double], notes: [Double]) -> [Double] {
        zip(audio, notes).map { audioValue, noteValue in
            0.72 * audioValue + 0.28 * noteValue
        }
    }

    private func normalized(_ values: [Double]) -> [Double] {
        guard !values.isEmpty else { return [] }
        let energy = sqrt(values.reduce(0) { $0 + $1 * $1 } / Double(values.count))
        guard energy > 1e-12 else { return values }
        return values.map { $0 / energy }
    }

    private func estimateTempo(
        envelope: [Double],
        framesPerSecond: Double
    ) -> TempoEstimate? {
        let minimumLag = max(2, Int((60 * framesPerSecond / maximumBPM).rounded()))
        let maximumLag = min(
            envelope.count / 3,
            Int((60 * framesPerSecond / minimumBPM).rounded())
        )
        guard maximumLag > minimumLag else { return nil }

        var correlations = [Double](repeating: 0, count: maximumLag + 1)
        for lag in minimumLag...maximumLag {
            correlations[lag] = normalizedCorrelation(envelope, lag: lag)
        }
        var scored: [(lag: Int, score: Double)] = []
        for lag in minimumLag...maximumLag {
            let bpm = 60 * framesPerSecond / Double(lag)
            let doublePeriod = lag * 2 <= maximumLag ? correlations[lag * 2] : 0
            let triplePeriod = lag * 3 <= maximumLag ? correlations[lag * 3] : 0
            let prior = exp(-pow(log(bpm / 120) / 0.75, 2))
            scored.append((lag, correlations[lag] + 0.45 * doublePeriod + 0.20 * triplePeriod + 0.08 * prior))
        }
        guard let rawBest = scored.max(by: { $0.score < $1.score }), rawBest.score > 0.08 else {
            return nil
        }
        var best = rawBest
        let rawBPM = 60 * framesPerSecond / Double(rawBest.lag)
        if rawBPM > 180, rawBest.lag * 2 <= maximumLag {
            let halfTempo = scored[rawBest.lag * 2 - minimumLag]
            if halfTempo.score >= rawBest.score * 0.60 {
                best = halfTempo
            }
        } else if rawBPM < 70, rawBest.lag / 2 >= minimumLag {
            let doubleTempo = scored[rawBest.lag / 2 - minimumLag]
            if doubleTempo.score >= rawBest.score * 0.60 {
                best = doubleTempo
            }
        }
        let competitors = scored.filter { candidate in
            guard abs(candidate.lag - best.lag) > 3 else { return false }
            let halfAlias = abs(candidate.lag * 2 - best.lag) <= 3
            let doubleAlias = abs(candidate.lag - best.lag * 2) <= 3
            return !halfAlias && !doubleAlias
        }
        let second = competitors.max(by: { $0.score < $1.score })?.score ?? 0
        let confidence = min(1, max(0, 0.55 * best.score + 1.8 * (best.score - second)))
        guard confidence >= 0.10 else { return nil }

        let lower = max(minimumLag, best.lag - 1)
        let upper = min(maximumLag, best.lag + 1)
        let weights = (lower...upper).map { max(0, scored[$0 - minimumLag].score) }
        let weightSum = weights.reduce(0, +)
        let period = weightSum > 0
            ? zip(lower...upper, weights).reduce(0) { $0 + Double($1.0) * $1.1 } / weightSum
            : Double(best.lag)
        let phase = strongestPhase(envelope: envelope, periodFrames: period)
        return TempoEstimate(
            bpm: 60 * framesPerSecond / period,
            periodFrames: period,
            phaseFrame: phase,
            confidence: confidence
        )
    }

    private func normalizedCorrelation(_ values: [Double], lag: Int) -> Double {
        guard lag > 0, lag < values.count else { return 0 }
        var dot = 0.0
        var leftEnergy = 0.0
        var rightEnergy = 0.0
        for index in lag..<values.count {
            let left = values[index]
            let right = values[index - lag]
            dot += left * right
            leftEnergy += left * left
            rightEnergy += right * right
        }
        let denominator = sqrt(leftEnergy * rightEnergy)
        return denominator > 1e-12 ? dot / denominator : 0
    }

    private func strongestPhase(envelope: [Double], periodFrames: Double) -> Int {
        let phaseCount = max(1, Int(periodFrames.rounded()))
        var bestPhase = 0
        var bestScore = -Double.infinity
        for phase in 0..<phaseCount {
            var score = 0.0
            var frame = Double(phase)
            while Int(frame.rounded()) < envelope.count {
                let index = Int(frame.rounded())
                score += envelope[index]
                if index > 0 { score += 0.25 * envelope[index - 1] }
                if index + 1 < envelope.count { score += 0.25 * envelope[index + 1] }
                frame += periodFrames
            }
            if score > bestScore {
                bestScore = score
                bestPhase = phase
            }
        }
        return bestPhase
    }

    private func beatGrid(
        envelope: [Double],
        periodFrames: Double,
        phaseFrame: Int,
        framesPerSecond: Double,
        durationSeconds: Double
    ) -> [MuScriptorBeat] {
        let maximum = max(envelope.max() ?? 0, 1e-12)
        var beats: [MuScriptorBeat] = []
        var frame = Double(phaseFrame)
        while frame / framesPerSecond <= durationSeconds {
            let index = min(envelope.count - 1, max(0, Int(frame.rounded())))
            let lower = max(0, index - 1)
            let upper = min(envelope.count - 1, index + 1)
            let strength = envelope[lower...upper].max() ?? 0
            beats.append(MuScriptorBeat(
                timeSeconds: frame / framesPerSecond,
                strength: min(1, strength / maximum),
                isDownbeat: false
            ))
            frame += periodFrames
        }
        return beats
    }

    private func estimateMeter(beats: [MuScriptorBeat]) -> MeterEstimate? {
        guard beats.count >= 8 else { return nil }
        let strengths = beats.map(\.strength)
        var candidates: [(numerator: Int, phase: Int, score: Double)] = []
        for numerator in [2, 3, 4] {
            let repetition = normalizedCorrelation(strengths, lag: numerator)
            for phase in 0..<numerator {
                let downbeats = strengths.enumerated().compactMap { index, value in
                    (index - phase).isMultiple(of: numerator) ? value : nil
                }
                let others = strengths.enumerated().compactMap { index, value in
                    (index - phase).isMultiple(of: numerator) ? nil : value
                }
                guard !downbeats.isEmpty, !others.isEmpty else { continue }
                let downbeatMean = downbeats.reduce(0, +) / Double(downbeats.count)
                let otherMean = others.reduce(0, +) / Double(others.count)
                let contrast = downbeatMean - otherMean
                let prior = numerator == 4 ? 0.04 : (numerator == 3 ? 0.02 : 0)
                candidates.append((numerator, phase, contrast + 0.35 * repetition + prior))
            }
        }
        let ranked = candidates.sorted { $0.score > $1.score }
        guard let best = ranked.first, best.score > 0.03 else { return nil }
        let second = ranked.first { $0.numerator != best.numerator }?.score ?? 0
        let confidence = min(1, max(0, 0.5 * best.score + 2.0 * (best.score - second)))
        guard confidence >= 0.15 else { return nil }
        return MeterEstimate(
            numerator: best.numerator,
            phaseIndex: best.phase,
            confidence: confidence
        )
    }

    private func estimateKey(notes: [MuScriptorNote]) -> MuScriptorKeySignature? {
        let pitched = notes.filter { !$0.isDrum && (0...127).contains($0.pitch) }
        guard pitched.count >= 3 else { return nil }
        var histogram = [Double](repeating: 0, count: 12)
        for note in pitched {
            let duration = min(4, max(0.01, note.offset - note.onset))
            histogram[note.pitch % 12] += sqrt(duration)
        }
        guard histogram.reduce(0, +) > 0 else { return nil }

        let major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        let minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        var candidates: [(tonic: Int, minor: Bool, score: Double)] = []
        for tonic in 0..<12 {
            candidates.append((tonic, false, profileCorrelation(histogram, profile: major, tonic: tonic)))
            candidates.append((tonic, true, profileCorrelation(histogram, profile: minor, tonic: tonic)))
        }
        candidates.sort { $0.score > $1.score }
        guard let best = candidates.first, best.score > 0.10 else { return nil }
        let second = candidates.dropFirst().first?.score ?? -1
        let confidence = min(1, max(0, 0.55 * best.score + 1.5 * (best.score - second)))
        guard confidence >= 0.15 else { return nil }
        let signature = keySignature(tonic: best.tonic, isMinor: best.minor)
        return MuScriptorKeySignature(
            name: signature.name,
            sharpsFlats: signature.sharpsFlats,
            isMinor: best.minor,
            confidence: confidence
        )
    }

    private func profileCorrelation(_ histogram: [Double], profile: [Double], tonic: Int) -> Double {
        let rotated = histogram.indices.map { profile[($0 - tonic + 12) % 12] }
        let histogramMean = histogram.reduce(0, +) / 12
        let profileMean = rotated.reduce(0, +) / 12
        var dot = 0.0
        var histogramEnergy = 0.0
        var profileEnergy = 0.0
        for index in histogram.indices {
            let left = histogram[index] - histogramMean
            let right = rotated[index] - profileMean
            dot += left * right
            histogramEnergy += left * left
            profileEnergy += right * right
        }
        let denominator = sqrt(histogramEnergy * profileEnergy)
        return denominator > 1e-12 ? dot / denominator : 0
    }

    private func keySignature(tonic: Int, isMinor: Bool) -> (name: String, sharpsFlats: Int) {
        let majorNames = ["C", "D-flat", "D", "E-flat", "E", "F", "F-sharp", "G", "A-flat", "A", "B-flat", "B"]
        let majorSharpsFlats = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5]
        let minorNames = ["C", "C-sharp", "D", "E-flat", "E", "F", "F-sharp", "G", "G-sharp", "A", "B-flat", "B"]
        let minorSharpsFlats = [-3, 4, -1, 6, 1, -4, 3, -2, 5, 0, -5, 2]
        if isMinor {
            return ("\(minorNames[tonic]) minor", minorSharpsFlats[tonic])
        }
        return ("\(majorNames[tonic]) major", majorSharpsFlats[tonic])
    }
}
