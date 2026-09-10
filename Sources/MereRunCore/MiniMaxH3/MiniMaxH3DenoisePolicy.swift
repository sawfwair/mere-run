import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

struct MiniMaxH3BlockReusePolicy {
    static let maximumSigmaDelta: Float = 0.12

    let cacheDepth: Double
    let window: ClosedRange<Float>
    let maximumConsecutiveCachedSteps: Int

    init(
        cacheDepth: Double,
        window: ClosedRange<Float> = 0.1...0.9,
        maximumConsecutiveCachedSteps: Int = 2
    ) {
        precondition((0..<1).contains(cacheDepth))
        precondition((0...1).contains(window.lowerBound))
        precondition((0...1).contains(window.upperBound))
        precondition(maximumConsecutiveCachedSteps >= 0)
        self.cacheDepth = cacheDepth
        self.window = window
        self.maximumConsecutiveCachedSteps = maximumConsecutiveCachedSteps
    }

    func warmBlockCount(totalBlockCount: Int) -> Int {
        precondition(totalBlockCount > 0)
        let count = Int((Double(totalBlockCount) * (1 - cacheDepth)).rounded())
        return max(0, min(totalBlockCount - 1, count))
    }

    func shouldReuseTail(
        stepIndex: Int,
        stepCount: Int,
        videoSigmas: [Float],
        audioSigmas: [Float],
        hasCachedResidual: Bool,
        consecutiveCachedSteps: Int
    ) -> Bool {
        guard stepIndex >= 2,
              stepIndex < stepCount,
              videoSigmas.count == stepCount + 1,
              audioSigmas.count == stepCount + 1,
              hasCachedResidual,
              consecutiveCachedSteps < maximumConsecutiveCachedSteps else { return false }
        let position = Float(stepIndex) / Float(stepCount)
        guard window.contains(position) else { return false }
        let videoDelta = abs(videoSigmas[stepIndex - 1] - videoSigmas[stepIndex])
        let audioDelta = abs(audioSigmas[stepIndex - 1] - audioSigmas[stepIndex])
        return max(videoDelta, audioDelta) < Self.maximumSigmaDelta
    }

}

enum MiniMaxH3DenoiseExecutionMode: Equatable {
    case compiledStep
    case eagerStep
    case blockwiseCompiled

    var usesLayerwiseEvaluation: Bool {
        self == .eagerStep
    }
}

enum MiniMaxH3DenoiseExecutionPolicy {
    static let blockwiseSequenceThreshold = 12_000
    static let practicalAttentionUpperBound = 13_500
    static let largeSequenceAttentionThreshold = 32_768
    static let veryLargeSequenceAttentionThreshold = 65_536

    static func attentionKernelSchedule(
        sequenceLength: Int
    ) -> (
        maximumQueryTokens: Int,
        maximumHeadsPerKernel: Int?,
        maximumKernelsPerEvaluation: Int
    ) {
        if sequenceLength >= veryLargeSequenceAttentionThreshold {
            return (
                maximumQueryTokens: 640,
                maximumHeadsPerKernel: 8,
                maximumKernelsPerEvaluation: 1
            )
        }
        if sequenceLength >= largeSequenceAttentionThreshold {
            return (
                maximumQueryTokens: 768,
                maximumHeadsPerKernel: nil,
                maximumKernelsPerEvaluation: 1
            )
        }
        if sequenceLength > blockwiseSequenceThreshold,
           sequenceLength <= practicalAttentionUpperBound {
            return (
                maximumQueryTokens: 640,
                maximumHeadsPerKernel: nil,
                maximumKernelsPerEvaluation: 1
            )
        }
        return (
            maximumQueryTokens: 1_024,
            maximumHeadsPerKernel: nil,
            maximumKernelsPerEvaluation: 1
        )
    }

    static func mode(
        usesResidentBF16: Bool,
        sequenceLength: Int,
        usesBlockProfiling: Bool,
        denoiseStepCount: Int = 2,
        profilingOverride: String? = nil
    ) -> MiniMaxH3DenoiseExecutionMode {
        if !usesBlockProfiling {
            switch profilingOverride?.lowercased() {
            case "compiled": return .compiledStep
            case "eager": return .eagerStep
            case "blockwise": return .blockwiseCompiled
            default: break
            }
        }
        if sequenceLength > blockwiseSequenceThreshold {
            return .blockwiseCompiled
        }
        if usesBlockProfiling || (usesResidentBF16 && denoiseStepCount == 1) {
            return .eagerStep
        }
        return .compiledStep
    }

}

public enum MiniMaxH3StepPolicy {
    public static let practicalPointCount = 9
    public static let extendedPointCount = 16
    public static let maximumQualityPointCount = 21
    public static let maximumSpeedPointCount = 12
    public static let practicalPackedRowLimit = 13_500
    public static let extendedPackedRowLimit = 26_000

    public static func recommendedPointCount(
        width: Int,
        height: Int,
        numFrames: Int,
        keyframeCount: Int = 0,
        referenceKinds: [MiniMaxH3ReferenceKind] = [],
        accelerationMode: MiniMaxH3AccelerationMode = .quality
    ) throws -> Int {
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: numFrames)
        let rowsPerVideoFrame = (height / 32) * (width / 32)
        let targetVideoRows = latentFrames * rowsPerVideoFrame
        let targetAudioRows = MiniMaxH3Geometry.audioLatentFrameCount(for: numFrames) * 2
        var estimatedRows = 128 + targetVideoRows + targetAudioRows
        estimatedRows += keyframeCount * rowsPerVideoFrame
        for kind in referenceKinds {
            switch kind {
            case .image:
                estimatedRows += rowsPerVideoFrame
            case .video:
                estimatedRows += targetVideoRows + targetAudioRows
            case .audio:
                estimatedRows += targetAudioRows
            }
        }
        let geometryPointCount: Int
        if estimatedRows <= practicalPackedRowLimit {
            geometryPointCount = practicalPointCount
        } else if estimatedRows <= extendedPackedRowLimit {
            geometryPointCount = extendedPointCount
        } else {
            geometryPointCount = maximumQualityPointCount
        }
        switch accelerationMode {
        case .quality, .balanced, .layers45, .layers40, .velocityReuse2, .tokenReduction:
            return geometryPointCount
        case .maximum:
            return min(geometryPointCount, maximumSpeedPointCount)
        }
    }
}

enum MiniMaxH3ResidentBF16Policy {
    static let gibibyte = UInt64(1_073_741_824)
    static let minimumPackedRows = 2_048
    static let minimumPortableMemoryBytes = 96 * gibibyte

    static func automaticReserveBytes(sequenceLength: Int) -> UInt64 {
        let base = 16 * gibibyte
        let rowsBeyondPracticalTier = UInt64(max(0, sequenceLength - 13_000))
        let geometryReserve = rowsBeyondPracticalTier * 384 * 1_024
        return min(32 * gibibyte, base + geometryReserve)
    }

    static func shouldMaterialize(
        mode: MiniMaxH3TransformerWeightMode,
        physicalMemoryBytes: UInt64,
        estimatedResidentBytes: UInt64,
        sequenceLength: Int,
        hasAdaLNCache: Bool,
        isPortableMac: Bool
    ) throws -> Bool {
        switch mode {
        case .quantized:
            return false
        case .automatic:
            let portableMemoryQualified = !isPortableMac
                || physicalMemoryBytes >= minimumPortableMemoryBytes
            guard portableMemoryQualified,
                  hasAdaLNCache,
                  estimatedResidentBytes > 0,
                  sequenceLength >= minimumPackedRows else { return false }
            let reserve = automaticReserveBytes(sequenceLength: sequenceLength)
            return estimatedResidentBytes <= physicalMemoryBytes - min(physicalMemoryBytes, reserve)
        case .residentBF16:
            guard hasAdaLNCache else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "resident-bf16 requires a compatible AdaLN cache"
                )
            }
            let reserve = 8 * gibibyte
            guard estimatedResidentBytes <= physicalMemoryBytes - min(physicalMemoryBytes, reserve) else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "resident-bf16 does not fit physical memory with the required runtime reserve"
                )
            }
            return true
        }
    }
}

enum MiniMaxH3Host {
    static var isPortableMac: Bool {
        #if canImport(IOKit.ps)
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as NSArray
        return sources.contains { source in
            guard let description = IOPSGetPowerSourceDescription(
                snapshot,
                source as CFTypeRef
            )?.takeUnretainedValue() as NSDictionary? else {
                return false
            }
            return description[kIOPSTypeKey] as? String == kIOPSInternalBatteryType
        }
        #else
        return false
        #endif
    }
}
