import Crypto
import Foundation
import MLX

public enum MiniMaxMusic3FlowStrategy: String, CaseIterable, Codable, Sendable {
    /// Denoise overlapping windows in sequence and carry the overlap forward.
    case sequential
    /// Denoise one song-length latent and average overlapping window velocities.
    case overlapAverage = "overlap-average"
}

public enum MiniMaxMusic3FlowSolver: String, CaseIterable, Codable, Sendable {
    /// Preserve the released first-order flow integration path.
    case euler
    /// Use Euler for the first step, then second-order Adams-Bashforth updates.
    case adamsBashforth2 = "ab2"
}

public enum MiniMaxMusic3SeedStrategy: String, CaseIterable, Codable, Sendable {
    /// Preserve the released recipe: autoregressive sampling and flow noise share one stream.
    case legacy
    /// Give each model stage a stable, independently derived random stream.
    case stageSeparatedV1 = "stage-separated-v1"

    func stageSeeds(seed: UInt64) -> (autoregressive: UInt64, flow: UInt64) {
        switch self {
        case .legacy:
            return (seed, seed)
        case .stageSeparatedV1:
            return (
                Self.derive(seed: seed, domain: "autoregressive"),
                Self.derive(seed: seed, domain: "flow")
            )
        }
    }

    private static func derive(seed: UInt64, domain: String) -> UInt64 {
        var input = (0..<8).map { UInt8(truncatingIfNeeded: seed >> UInt64($0 * 8)) }
        let domainBytes = Array(domain.utf8)
        input.append(contentsOf: (0..<4).map {
            UInt8(truncatingIfNeeded: UInt32(domainBytes.count) >> UInt32($0 * 8))
        })
        input.append(contentsOf: domainBytes)
        let digest = SHA256.hash(data: Data(input))
        return digest.prefix(8).enumerated().reduce(UInt64(0)) { value, element in
            value | (UInt64(element.element) << UInt64(element.offset * 8))
        }
    }
}

public struct MiniMaxMusic3AudioHealthReport: Codable, Equatable, Sendable {
    public let sampleCount: Int
    public let peak: Float
    public let rms: Float
    public let stereoCollapseFraction: Float

    public init(
        sampleCount: Int,
        peak: Float,
        rms: Float,
        stereoCollapseFraction: Float
    ) {
        self.sampleCount = sampleCount
        self.peak = peak
        self.rms = rms
        self.stereoCollapseFraction = stereoCollapseFraction
    }
}

public enum MiniMaxMusic3AudioHealth {
    public static let silenceThreshold: Float = 1e-7
    public static let implausiblePeakThreshold: Float = 8
    public static let stereoCollapseThreshold: Float = 0.75

    public static func validate(
        _ waveform: MLXArray,
        sampleRate: Int
    ) throws -> MiniMaxMusic3AudioHealthReport {
        guard waveform.ndim == 3, waveform.dim(0) == 1, waveform.dim(1) == 2,
              waveform.dim(2) > 0, sampleRate > 0
        else {
            throw MiniMaxMusic3Error.invalidAudio(
                "expected waveform shape [1, 2, samples] and a positive sample rate"
            )
        }
        guard MLX.all(MLX.isFinite(waveform)).item(Bool.self) else {
            throw MiniMaxMusic3Error.invalidAudio("the waveform contains non-finite samples")
        }

        let samples = waveform[0, 0..., 0...].asType(.float32)
        let peak = MLX.max(MLX.abs(samples)).item(Float.self)
        let rms = MLX.sqrt(MLX.mean(MLX.square(samples))).item(Float.self)
        guard rms > silenceThreshold else {
            throw MiniMaxMusic3Error.invalidAudio("the waveform is silent")
        }
        guard peak <= implausiblePeakThreshold else {
            throw MiniMaxMusic3Error.invalidAudio(
                String(format: "the waveform has an implausible %.3f peak", peak)
            )
        }

        let blockCount = waveform.dim(2) / sampleRate
        let collapseFraction: Float
        if blockCount == 0 {
            collapseFraction = 0
        } else {
            let framed = samples[0..., 0..<(blockCount * sampleRate)]
                .reshaped(2, blockCount, sampleRate)
            let blockRMS = MLX.sqrt(MLX.mean(MLX.square(framed), axis: 2))
            let lower = MLX.min(blockRMS, axis: 0)
            let upper = MLX.max(blockRMS, axis: 0)
            let collapsed = MLX.logicalAnd(
                upper .> MLXArray(silenceThreshold),
                lower .< upper * MLXArray(Float(0.1))
            )
            collapseFraction = MLX.mean(collapsed.asType(.float32)).item(Float.self)
        }
        guard collapseFraction < stereoCollapseThreshold else {
            throw MiniMaxMusic3Error.invalidAudio(
                String(
                    format: "the DAV decoder collapsed one stereo channel in %.0f%% of the audio",
                    collapseFraction * 100
                )
            )
        }
        return MiniMaxMusic3AudioHealthReport(
            sampleCount: waveform.dim(2),
            peak: peak,
            rms: rms,
            stereoCollapseFraction: collapseFraction
        )
    }
}

struct MiniMaxMusic3DAVDecodeSlice: Equatable {
    let context: Range<Int>
    let retained: Range<Int>

    static func plan(
        frameCount: Int,
        maximumChunkFrames: Int = 1_024,
        overlapFrames: Int = 64
    ) -> [Self] {
        precondition(frameCount > 0)
        precondition(overlapFrames >= 0 && maximumChunkFrames > 2 * overlapFrames)
        guard frameCount > maximumChunkFrames else {
            return [Self(context: 0..<frameCount, retained: 0..<frameCount)]
        }
        let retainedFrameCount = maximumChunkFrames - 2 * overlapFrames
        var retainedStart = 0
        var slices: [Self] = []
        while retainedStart < frameCount {
            let retainedEnd = min(frameCount, retainedStart + retainedFrameCount)
            let contextStart = max(0, retainedStart - overlapFrames)
            let contextEnd = min(frameCount, retainedEnd + overlapFrames)
            let localStart = retainedStart - contextStart
            slices.append(Self(
                context: contextStart..<contextEnd,
                retained: localStart..<(localStart + retainedEnd - retainedStart)
            ))
            retainedStart = retainedEnd
        }
        return slices
    }
}
