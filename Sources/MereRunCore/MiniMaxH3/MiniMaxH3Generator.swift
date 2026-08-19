import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

@inline(__always)
private func withMiniMaxH3AutoreleasePool<T>(_ body: () throws -> T) rethrows -> T {
    #if canImport(ObjectiveC)
    return try autoreleasepool(invoking: body)
    #else
    return try body()
    #endif
}

public enum MiniMaxH3GeneratorError: LocalizedError {
    case missingModelFiles([URL])
    case invalidOptions(String)
    case imageDecodeFailed(URL)
    case mediaDecodeFailed(URL, String)

    public var errorDescription: String? {
        switch self {
        case .missingModelFiles(let files):
            return "MiniMax-H3 model root is missing: \(files.map(\.lastPathComponent).joined(separator: ", "))"
        case .invalidOptions(let reason): return "Invalid MiniMax-H3 request: \(reason)"
        case .imageDecodeFailed(let url): return "MiniMax-H3 could not decode image: \(url.path)"
        case .mediaDecodeFailed(let url, let reason):
            return "MiniMax-H3 could not decode reference \(url.path): \(reason)"
        }
    }
}

extension MiniMaxH3ExactKernelMode {
    func requiresEagerExecution(sequenceLength: Int) -> Bool {
        switch self {
        case .disabled:
            false
        case .boundaryLayout:
            sequenceLength > MiniMaxH3DenoiseExecutionPolicy.blockwiseSequenceThreshold
        case .affineQ8:
            true
        }
    }

    static func resolve(environmentValue: String?) throws -> Self {
        switch environmentValue?.lowercased() {
        case nil, "", "disabled": .disabled
        case MiniMaxH3ExactKernelMode.boundaryLayout.rawValue: .boundaryLayout
        case MiniMaxH3ExactKernelMode.affineQ8.rawValue: .affineQ8
        case .some(let value):
            throw MiniMaxH3GeneratorError.invalidOptions(
                "MERERUN_H3_EXACT_KERNELS must be disabled, boundary-layout, "
                    + "or affine-q8, not \(value)"
            )
        }
    }
}

public struct MiniMaxH3ReferenceInput: Sendable, Hashable {
    public let kind: MiniMaxH3ReferenceKind
    public let url: URL

    public init(kind: MiniMaxH3ReferenceKind, url: URL) {
        self.kind = kind
        self.url = url.standardizedFileURL
    }
}

public struct MiniMaxH3FrameInput: Sendable, Hashable {
    public let frameIndex: Int
    public let url: URL

    public init(frameIndex: Int, url: URL) {
        self.frameIndex = frameIndex
        self.url = url.standardizedFileURL
    }
}

public enum MiniMaxH3TransformerWeightMode: String, Sendable, Hashable {
    case automatic = "auto"
    case quantized
    case residentBF16 = "resident-bf16"
}

public enum MiniMaxH3AccelerationMode: String, Sendable, Hashable {
    case quality
    case balanced
    case maximum
    case layers45 = "layers-45"
    case layers40 = "layers-40"
    case velocityReuse2 = "velocity-reuse-2"
    case tokenReduction = "token-reduction"

    var adaptiveFirstBlockCachePolicy: MiniMaxH3AdaptiveFirstBlockCachePolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            MiniMaxH3AdaptiveFirstBlockCachePolicy(
                globalThreshold: 0.08,
                temporalThreshold: 0.12,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 2,
                requiredFinalFullSteps: 2
            )
        case .maximum:
            MiniMaxH3AdaptiveFirstBlockCachePolicy(
                globalThreshold: 0.30,
                temporalThreshold: 0.40,
                window: 0.1...0.95,
                maximumConsecutiveCachedSteps: 4,
                requiredFinalFullSteps: 1
            )
        }
    }

    var dynamicSparseAttentionPolicy: DynamicSparseAttentionPolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            DynamicSparseAttentionPolicy(
                thresholdStandardDeviations: 0.75
            )
        case .maximum:
            DynamicSparseAttentionPolicy(
                thresholdStandardDeviations: 1
            )
        }
    }

    // Retained as an internal benchmark baseline. Production acceleration
    // selects the adaptive first-block cache unless explicitly overridden.
    var blockReusePolicy: MiniMaxH3BlockReusePolicy? {
        switch self {
        case .quality, .layers45, .layers40, .velocityReuse2, .tokenReduction: nil
        case .balanced:
            MiniMaxH3BlockReusePolicy(
                cacheDepth: 0.5,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 2
            )
        case .maximum:
            MiniMaxH3BlockReusePolicy(
                cacheDepth: 0.82,
                window: 0.1...0.9,
                maximumConsecutiveCachedSteps: 4
            )
        }
    }

    var velocityReusePolicy: MiniMaxH3VelocityReusePolicy? {
        switch self {
        case .quality, .balanced, .maximum, .layers45, .layers40, .tokenReduction: nil
        case .velocityReuse2: MiniMaxH3VelocityReusePolicy(interval: 2)
        }
    }

    var layerThinningPolicy: MiniMaxH3LayerThinningPolicy? {
        switch self {
        case .quality, .balanced, .maximum, .velocityReuse2, .tokenReduction: nil
        case .layers45: MiniMaxH3LayerThinningPolicy(activeBlockCount: 45)
        case .layers40: MiniMaxH3LayerThinningPolicy(activeBlockCount: 40)
        }
    }

    var tokenReductionPolicy: MiniMaxH3TokenReductionPolicy? {
        switch self {
        case .quality, .balanced, .maximum, .layers45, .layers40, .velocityReuse2: nil
        case .tokenReduction: MiniMaxH3TokenReductionPolicy()
        }
    }
}

struct MiniMaxH3TokenReductionPolicy: Sendable, Equatable {
    let beginBlock: Int
    let endBlock: Int
    let earlyStepCount: Int
    let earlyEndBlock: Int
    let updateScale: Float

    init(
        beginBlock: Int = 4,
        endBlock: Int = 30,
        earlyStepCount: Int = 10,
        earlyEndBlock: Int = 40,
        updateScale: Float = 1
    ) {
        precondition(beginBlock >= 0)
        precondition(beginBlock < endBlock)
        precondition(endBlock < earlyEndBlock)
        precondition(earlyStepCount > 0)
        precondition(updateScale >= 0 && updateScale <= 2)
        self.beginBlock = beginBlock
        self.endBlock = endBlock
        self.earlyStepCount = earlyStepCount
        self.earlyEndBlock = earlyEndBlock
        self.updateScale = updateScale
    }

    func restoreBeforeBlock(stepIndex: Int) -> Int {
        stepIndex < earlyStepCount ? earlyEndBlock : endBlock
    }
}

struct MiniMaxH3LayerThinningPolicy: Sendable, Equatable {
    let activeBlockCount: Int

    init(activeBlockCount: Int) {
        precondition(activeBlockCount >= 3)
        self.activeBlockCount = activeBlockCount
    }

    func activeBlockIndices(blockModulations: [MLXArray]) -> [Int] {
        precondition(activeBlockCount <= blockModulations.count)
        guard activeBlockCount < blockModulations.count else {
            return Array(blockModulations.indices)
        }
        let finalIndex = blockModulations.index(before: blockModulations.endIndex)
        let candidates = blockModulations.indices.filter { index in
            index >= 2 && index < finalIndex
        }
        let scoreArrays = candidates.map { index in
            let parts = MLX.split(blockModulations[index], parts: 6, axis: -1)
            return MLX.mean(MLX.abs(MLX.concatenated([parts[2], parts[5]], axis: -1)).asType(.float32))
        }
        let scoreValues = MLX.stacked(scoreArrays).asArray(Float.self)
        let ranked = zip(candidates, scoreValues).sorted { lhs, rhs in
            lhs.1 == rhs.1 ? lhs.0 < rhs.0 : lhs.1 < rhs.1
        }
        let skippedCount = blockModulations.count - activeBlockCount
        let skipped = Set(ranked.prefix(skippedCount).map(\.0))
        return blockModulations.indices.filter { !skipped.contains($0) }
    }
}

struct MiniMaxH3VelocityReusePolicy: Sendable, Equatable {
    let interval: Int
    let requiredFinalFullSteps: Int

    init(interval: Int, requiredFinalFullSteps: Int = 1) {
        precondition(interval >= 2)
        precondition(requiredFinalFullSteps >= 1)
        self.interval = interval
        self.requiredFinalFullSteps = requiredFinalFullSteps
    }

    func shouldReuse(stepIndex: Int, stepCount: Int, hasCachedVelocity: Bool) -> Bool {
        guard stepIndex > 0,
              stepIndex < stepCount,
              hasCachedVelocity,
              stepIndex < stepCount - requiredFinalFullSteps else { return false }
        return !stepIndex.isMultiple(of: interval)
    }
}

struct MiniMaxH3AdaptiveFirstBlockCachePolicy: Sendable, Equatable {
    let globalThreshold: Float
    let temporalThreshold: Float
    let window: ClosedRange<Float>
    let maximumConsecutiveCachedSteps: Int
    let minimumFullSteps: Int
    let requiredFinalFullSteps: Int

    init(
        globalThreshold: Float,
        temporalThreshold: Float,
        window: ClosedRange<Float> = 0.1...0.9,
        maximumConsecutiveCachedSteps: Int = 2,
        minimumFullSteps: Int = 2,
        requiredFinalFullSteps: Int = 1
    ) {
        precondition(globalThreshold > 0)
        precondition(temporalThreshold > 0)
        precondition((0...1).contains(window.lowerBound))
        precondition((0...1).contains(window.upperBound))
        precondition(maximumConsecutiveCachedSteps >= 0)
        precondition(minimumFullSteps >= 1)
        precondition(requiredFinalFullSteps >= 1)
        self.globalThreshold = globalThreshold
        self.temporalThreshold = temporalThreshold
        self.window = window
        self.maximumConsecutiveCachedSteps = maximumConsecutiveCachedSteps
        self.minimumFullSteps = minimumFullSteps
        self.requiredFinalFullSteps = requiredFinalFullSteps
    }

    func canConsiderReuse(
        stepIndex: Int,
        stepCount: Int,
        fullStepCount: Int,
        consecutiveCachedSteps: Int,
        hasCachedState: Bool
    ) -> Bool {
        guard stepIndex >= 0,
              stepIndex < stepCount,
              fullStepCount >= minimumFullSteps,
              hasCachedState,
              consecutiveCachedSteps < maximumConsecutiveCachedSteps,
              stepIndex < stepCount - requiredFinalFullSteps else { return false }
        return window.contains(Float(stepIndex) / Float(stepCount))
    }

    func shouldReuse(change: MiniMaxH3FirstBlockChange) -> Bool {
        change.isFinite
            && change.videoGlobal <= globalThreshold
            && change.audioGlobal <= globalThreshold
            && change.videoTemporalMaximum <= temporalThreshold
            && change.audioTemporalMaximum <= temporalThreshold
    }
}

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

public struct MiniMaxH3GenerationOptions: Sendable, Hashable {
    public let prompt: String
    public let width: Int
    public let height: Int
    public let renderWidth: Int?
    public let renderHeight: Int?
    public let numFrames: Int
    public let steps: Int
    public let seed: UInt64
    public let transformerWeightMode: MiniMaxH3TransformerWeightMode
    public let accelerationMode: MiniMaxH3AccelerationMode
    public let adapterURL: URL?
    public let adapterStrength: Float
    public let adapterInferenceRecipe: MiniMaxH3TurboAdapter.InferenceRecipe?
    public let firstFrameURL: URL?
    public let lastFrameURL: URL?
    public let frameInputs: [MiniMaxH3FrameInput]
    public let references: [MiniMaxH3ReferenceInput]

    public init(
        prompt: String,
        width: Int = 768,
        height: Int = 768,
        renderWidth: Int? = nil,
        renderHeight: Int? = nil,
        numFrames: Int = 124,
        steps: Int? = nil,
        seed: UInt64 = 42,
        transformerWeightMode: MiniMaxH3TransformerWeightMode = .automatic,
        accelerationMode: MiniMaxH3AccelerationMode = .quality,
        adapterURL: URL? = nil,
        adapterStrength: Float = 1,
        firstFrameURL: URL? = nil,
        lastFrameURL: URL? = nil,
        frameInputs: [MiniMaxH3FrameInput] = [],
        references: [MiniMaxH3ReferenceInput] = []
    ) throws {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MiniMaxH3GeneratorError.invalidOptions("prompt cannot be empty") }
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw MiniMaxH3GeneratorError.invalidOptions("width and height must be positive multiples of 32")
        }
        guard (renderWidth == nil) == (renderHeight == nil) else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "internal render width and height must be set together"
            )
        }
        if let renderWidth, let renderHeight {
            let (leftAspect, leftOverflow) = renderWidth.multipliedReportingOverflow(by: height)
            let (rightAspect, rightOverflow) = renderHeight.multipliedReportingOverflow(by: width)
            guard renderWidth >= 32,
                  renderHeight >= 32,
                  renderWidth.isMultiple(of: 32),
                  renderHeight.isMultiple(of: 32),
                  renderWidth <= width,
                  renderHeight <= height,
                  !leftOverflow,
                  !rightOverflow,
                  leftAspect == rightAspect else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "internal render canvas must be same-aspect multiples of 32 no larger than the output canvas"
                )
            }
        }
        guard numFrames >= 22, numFrames % 17 == 5 else {
            throw MiniMaxH3GeneratorError.invalidOptions("frame count must be at least 22 and have the form 17*n+5")
        }
        guard lastFrameURL == nil || firstFrameURL != nil else {
            throw MiniMaxH3GeneratorError.invalidOptions("a last frame requires a first frame")
        }
        guard frameInputs.count <= 12 else {
            throw MiniMaxH3GeneratorError.invalidOptions("FL2VA accepts at most 12 positioned frame inputs")
        }
        guard frameInputs.allSatisfy({ (0..<numFrames).contains($0.frameIndex) }) else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "positioned frame indices must be inside the output frame range"
            )
        }
        let positionedIndices = frameInputs.map(\.frameIndex)
        guard Set(positionedIndices).count == positionedIndices.count else {
            throw MiniMaxH3GeneratorError.invalidOptions("positioned frame indices must be unique")
        }
        guard firstFrameURL == nil || !positionedIndices.contains(0) else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--image and a positioned frame at index 0 cannot be combined"
            )
        }
        guard lastFrameURL == nil || !positionedIndices.contains(numFrames - 1) else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--end-image and a positioned frame at the final index cannot be combined"
            )
        }
        guard references.count <= 12 else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA accepts at most 12 ordered references")
        }
        guard references.count(where: { $0.kind == .image }) <= 9,
              references.count(where: { $0.kind == .video }) <= 3,
              references.count(where: { $0.kind == .audio }) <= 3 else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA accepts at most 9 images, 3 videos, and 3 audio clips")
        }
        if references.contains(where: { $0.kind == .audio }),
           !references.contains(where: { $0.kind != .audio }) {
            throw MiniMaxH3GeneratorError.invalidOptions("an audio reference must be paired with an image or video")
        }
        let adapterInferenceRecipe = adapterURL.map(MiniMaxH3TurboAdapter.inferenceRecipe(for:))
        if let adapterInferenceRecipe {
            guard adapterStrength > 0 else {
                throw MiniMaxH3GeneratorError.invalidOptions("adapter strength must be greater than zero")
            }
            switch adapterInferenceRecipe.task {
            case .fl2va:
                guard references.isEmpty else {
                    throw MiniMaxH3GeneratorError.invalidOptions(
                        "The selected MiniMax-H3 FL2VA adapter cannot be used with Ref2VA references"
                    )
                }
            case .ref2va:
                guard !references.isEmpty else {
                    throw MiniMaxH3GeneratorError.invalidOptions(
                        "The selected MiniMax-H3 Ref2VA adapter requires ordered references"
                    )
                }
            }
        }
        let resolvedSteps: Int
        if let steps {
            resolvedSteps = steps
        } else if let adapterInferenceRecipe {
            resolvedSteps = adapterInferenceRecipe.defaultSchedulePointCount
        } else {
            resolvedSteps = try MiniMaxH3StepPolicy.recommendedPointCount(
                width: renderWidth ?? width,
                height: renderHeight ?? height,
                numFrames: numFrames,
                keyframeCount: [firstFrameURL, lastFrameURL].compactMap { $0 }.count
                    + frameInputs.count,
                referenceKinds: references.map(\.kind),
                accelerationMode: accelerationMode
            )
        }
        guard resolvedSteps >= 2 else {
            throw MiniMaxH3GeneratorError.invalidOptions("steps must be at least 2")
        }
        if let adapterInferenceRecipe,
           !adapterInferenceRecipe.supports(schedulePointCount: resolvedSteps) {
            let supported = adapterInferenceRecipe.supportedSchedulePointCounts
                .sorted()
                .map(String.init)
                .joined(separator: " or ")
            throw MiniMaxH3GeneratorError.invalidOptions(
                "MiniMax-H3 Turbo recipe \(adapterInferenceRecipe.name) requires \(supported) schedule points"
            )
        }
        self.prompt = trimmed
        self.width = width
        self.height = height
        self.renderWidth = renderWidth
        self.renderHeight = renderHeight
        self.numFrames = numFrames
        self.steps = resolvedSteps
        self.seed = seed
        self.transformerWeightMode = transformerWeightMode
        self.accelerationMode = accelerationMode
        self.adapterURL = adapterURL
        self.adapterStrength = adapterStrength
        self.adapterInferenceRecipe = adapterInferenceRecipe
        self.firstFrameURL = firstFrameURL
        self.lastFrameURL = lastFrameURL
        self.frameInputs = frameInputs.sorted { $0.frameIndex < $1.frameIndex }
        self.references = references
    }

    public var internalWidth: Int { renderWidth ?? width }

    public var internalHeight: Int { renderHeight ?? height }

    public var usesReducedRenderCanvas: Bool {
        internalWidth != width || internalHeight != height
    }
}

public enum MiniMaxH3GenerationStage: String, Sendable {
    case loadingTextEncoder = "loading-text-encoder"
    case encodingText = "encoding-text"
    case encodingKeyframes = "encoding-keyframes"
    case encodingReferences = "encoding-references"
    case loadingTransformer = "loading-transformer"
    case interpolatingAdaLNCache = "interpolating-adaln-cache-not-bit-exact"
    case materializingTransformerBF16 = "materializing-transformer-bf16"
    case denoising
    case decodingVideo = "decoding-video"
    case decodingAudio = "decoding-audio"
}

public struct MiniMaxH3GenerationProgress: Sendable {
    public let stage: MiniMaxH3GenerationStage
    public let stepIndex: Int
    public let totalSteps: Int
}

public struct MiniMaxH3GenerationResult: @unchecked Sendable {
    public let frames: MLXArray
    public let audio: MLXArray
    public let seed: UInt64
}

public final class MiniMaxH3Generator: @unchecked Sendable {
    private struct DenoisingRuntimeCacheKey: Hashable {
        let modelRoot: URL
        let modelSourceIdentity: String
        let videoSigmas: [Float]
        let audioSigmas: [Float]
        let weightMode: MiniMaxH3TransformerWeightMode
        let adapterURL: URL?
        let adapterSHA256: String?
        let adapterStrength: Float
    }

    private struct ReferenceCacheKey: Hashable {
        let references: [MiniMaxH3ReferenceInput]
        let maximumFrameCount: Int
        let targetWidth: Int
        let targetHeight: Int
    }

    private struct ConditionerPresentation {
        let tokenIDs: [Int]
        let tokenTags: [Int32]
        let images: [QwenVLEncoder.ConditioningImage]
    }

    private struct PreparedReference {
        let kind: MiniMaxH3ReferenceKind
        let visual: MLXArray?
        let visionBlocks: [MLXArray]
        let blockTimestamps: [Double]
        let waveform: MLXArray?
        let geometry: MiniMaxH3PreparedReferenceGeometry
    }

    private struct FrameCondition {
        let url: URL
        let anchor: MiniMaxH3KeyframeAnchor
    }

    private struct ContinuationConditions {
        let videoRows: MLXArray
        let audioRows: MLXArray
        let videoAnchors: [MiniMaxH3KeyframeAnchor]
        let audioAnchors: [MiniMaxH3AudioConditionAnchor]
    }

    private struct PreparedReferenceRows {
        let video: [MLXArray]
        let audio: [MLXArray]
    }

    private let retainsRuntime: Bool
    private var retainedConditioner: (root: URL, model: QwenVLEncoder)?
    private var retainedVideoVAE: (root: URL, model: MiniMaxH3VideoVAE)?
    private var retainedAudioVAE: (root: URL, model: MiniMaxH3AudioVAE)?
    private var retainedPreparedReferences: (
        key: ReferenceCacheKey,
        values: [PreparedReference]
    )?
    private var retainedPreparedReferenceRows: (
        key: ReferenceCacheKey,
        values: PreparedReferenceRows
    )?
    private var retainedDenoisingRuntime: (
        key: DenoisingRuntimeCacheKey,
        transformer: MiniMaxH3Transformer,
        adaLNCache: MiniMaxH3AdaLNCache?
    )?

    public init(retainsRuntime: Bool = false) {
        self.retainsRuntime = retainsRuntime
    }

    static func mediaFrames(from decodedFrames: MLXArray) -> MLXArray {
        MLX.clip(decodedFrames * 255, min: 0, max: 255).asType(.uint8)
    }

    private func loadDenoisingRuntime(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sequenceLength: Int,
        weightMode: MiniMaxH3TransformerWeightMode,
        adapterURL: URL?,
        adapterStrength: Float,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> (transformer: MiniMaxH3Transformer, adaLNCache: MiniMaxH3AdaLNCache?) {
        let modelSourceIdentity = try resources.adaLNCacheSourceIdentity()
        let cacheKey = DenoisingRuntimeCacheKey(
            modelRoot: resources.rootURL.resolvingSymlinksInPath(),
            modelSourceIdentity: modelSourceIdentity,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            weightMode: weightMode,
            adapterURL: adapterURL?.resolvingSymlinksInPath(),
            adapterSHA256: try adapterURL.map {
                try ModelArtifactPin.fileSHA256($0.resolvingSymlinksInPath())
            },
            adapterStrength: adapterStrength
        )
        if retainsRuntime,
           let retainedDenoisingRuntime,
           retainedDenoisingRuntime.key == cacheKey {
            return (
                retainedDenoisingRuntime.transformer,
                retainedDenoisingRuntime.adaLNCache
            )
        }
        let adaLNCache: MiniMaxH3AdaLNCache?
        let cachedAdaLNForLoading: MiniMaxH3AdaLNCache?
        if resources.usesShardedBF16Transformer {
            // A legacy full BF16 root retains the schedule-only projections.
            // Build an exact table for the requested run; compact managed roots
            // select their source-bound production cache pack below.
            adaLNCache = nil
            cachedAdaLNForLoading = nil
        } else {
            let selection = try Self.loadAdaLNCache(
                resources: resources,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule
            )
            if selection?.exact == false {
                progressHandler?(.init(
                    stage: .interpolatingAdaLNCache,
                    stepIndex: 1,
                    totalSteps: 1
                ))
            }
            adaLNCache = selection?.cache
            cachedAdaLNForLoading = selection?.cache
        }
        let transformer = try MiniMaxH3ModelLoader.loadInferenceTransformer(
            resources: resources,
            configuration: configuration,
            cachedAdaLN: cachedAdaLNForLoading,
            progressHandler: { shard in
                progressHandler?(.init(
                    stage: .loadingTransformer,
                    stepIndex: shard.shardIndex,
                    totalSteps: shard.shardCount
                ))
            }
        )
        let adapterInferenceRecipe = adapterURL.map(MiniMaxH3TurboAdapter.inferenceRecipe(for:))
        if let adapterURL, resources.usesShardedBF16Transformer {
            try MiniMaxH3TurboAdapter.install(
                url: adapterURL,
                into: transformer,
                strength: adapterStrength
            )
        }
        var resolvedAdaLNCache: MiniMaxH3AdaLNCache?
        if resources.usesShardedBF16Transformer {
            resolvedAdaLNCache = transformer.precomputeAdaLN(
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: modelSourceIdentity
            )
            transformer.discardAdaLNWeights()
            Memory.clearCache()
        } else {
            resolvedAdaLNCache = adaLNCache
        }
        let shouldMaterialize = try MiniMaxH3ResidentBF16Policy.shouldMaterialize(
            mode: weightMode,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            estimatedResidentBytes: transformer.estimatedResidentBF16ByteCount,
            sequenceLength: sequenceLength,
            hasAdaLNCache: resolvedAdaLNCache != nil,
            isPortableMac: MiniMaxH3Host.isPortableMac
        )
        if shouldMaterialize {
            progressHandler?(.init(
                stage: .materializingTransformerBF16,
                stepIndex: 0,
                totalSteps: 1
            ))
            _ = transformer.materializeResidentBF16()
        }
        if let adapterURL, !resources.usesShardedBF16Transformer {
            let adapterTask = adapterInferenceRecipe?.task ?? .fl2va
            if adapterTask == .fl2va,
               try !resources.transformerStorage().supportsFL2VATurboAdapters {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "MiniMax-H3 FL2VA Turbo requires compact BF16 or Q8; legacy Q4 is unsupported"
                )
            }
            if adapterTask == .ref2va, !transformer.usesResidentBF16 {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "MiniMax-H3 Ref2VA Turbo requires resident BF16 weights; use --h3-weight-mode resident-bf16 on a machine with sufficient memory"
                )
            }
            let installation = try MiniMaxH3TurboAdapter.installForInference(
                url: adapterURL,
                into: transformer,
                strength: adapterStrength,
                adaLNCache: resolvedAdaLNCache
            )
            resolvedAdaLNCache = installation.adaLNCache
        }
        if retainsRuntime {
            retainedDenoisingRuntime = (cacheKey, transformer, resolvedAdaLNCache)
        }
        return (transformer, resolvedAdaLNCache)
    }

    private func loadConditioner(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws -> QwenVLEncoder {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedConditioner, retainedConditioner.root == root {
            return retainedConditioner.model
        }
        let model = try MiniMaxH3ModelLoader.loadConditioner(
            resources: resources,
            configuration: configuration,
            progressHandler: progressHandler
        )
        if retainsRuntime { retainedConditioner = (root, model) }
        return model
    }

    private func loadVideoVAE(resources: MiniMaxH3Resources) throws -> MiniMaxH3VideoVAE {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedVideoVAE, retainedVideoVAE.root == root {
            return retainedVideoVAE.model
        }
        let model = try MiniMaxH3ModelLoader.loadVideoVAE(resources: resources)
        if retainsRuntime { retainedVideoVAE = (root, model) }
        return model
    }

    private func loadAudioVAE(resources: MiniMaxH3Resources) throws -> MiniMaxH3AudioVAE {
        let root = resources.rootURL.resolvingSymlinksInPath()
        if retainsRuntime, let retainedAudioVAE, retainedAudioVAE.root == root {
            return retainedAudioVAE.model
        }
        let model = try MiniMaxH3ModelLoader.loadAudioVAE(resources: resources)
        if retainsRuntime { retainedAudioVAE = (root, model) }
        return model
    }

    private func denoise(
        transformer: MiniMaxH3Transformer,
        videoRows initialVideoRows: MLXArray,
        audioRows initialAudioRows: MLXArray,
        conditionVideoRows: MLXArray?,
        conditionAudioRows: MLXArray?,
        promptStates: MLXArray,
        layout: MiniMaxH3PackedLayout,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        adaLNCache: MiniMaxH3AdaLNCache?,
        accelerationMode: MiniMaxH3AccelerationMode,
        permitsCacheReuse: Bool,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> (videoRows: MLXArray, audioRows: MLXArray) {
        precondition(videoSchedule.timesteps.count == audioSchedule.timesteps.count)
        let blockProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_BLOCKS"],
            prefix: "[minimax-h3-profile]"
        )
        let stepProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_STEPS"],
            prefix: "[minimax-h3-step-profile]"
        )
        blockProfileLogger?("rows=\(layout.sequenceLength) blocks=\(transformer.configuration.layerCount)")
        stepProfileLogger?(
            "rows=\(layout.sequenceLength) steps=\(videoSchedule.timesteps.count) "
                + "resident_bf16=\(transformer.usesResidentBF16)"
        )
        transformer.blockTimingHandler = blockProfileLogger.map { logger in
            { index, elapsed, memory in
                logger(String(
                    format: "block=%02d seconds=%.3f active_gib=%.2f cache_gib=%.2f peak_gib=%.2f",
                    index,
                    elapsed,
                    Double(memory.activeMemory) / 1_073_741_824,
                    Double(memory.cacheMemory) / 1_073_741_824,
                    Double(memory.peakMemory) / 1_073_741_824
                ))
            }
        }
        let context = transformer.prepare(textStates: promptStates, layout: layout)
        var videoRows = initialVideoRows
        var audioRows = initialAudioRows
        MLX.eval(videoRows, audioRows)
        if let conditionVideoRows { MLX.eval(conditionVideoRows) }
        if let conditionAudioRows { MLX.eval(conditionAudioRows) }

        // The practical H3 tier uses independent compiled block boundaries to
        // bound the graph, reuse intermediate buffers, and hold steadier clocks
        // during a sustained denoise pass. Smaller graphs can still amortize a
        // compiled whole-step transform; very large graphs also stay blockwise
        // to avoid the macOS watchdog.
        let environment = ProcessInfo.processInfo.environment
        let exactKernelMode = try MiniMaxH3ExactKernelMode.resolve(
            environmentValue: environment["MERERUN_H3_EXACT_KERNELS"]
        )
        if exactKernelMode != .disabled {
            guard accelerationMode == .quality else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "exact H3 kernels must be measured with h3 acceleration quality"
                )
            }
        }
        if exactKernelMode == .affineQ8 {
            guard transformer.supportsAffineQ8ExactKernels,
                  !transformer.usesResidentBF16 else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "affine-q8 exact kernels require the unadapted managed Q8/group-64 transformer"
                )
            }
        }
        transformer.exactKernelMode = exactKernelMode
        let usesScheduledTailCache = environment["MERERUN_H3_CACHE_STRATEGY"] == "scheduled-tail"
        let requestedReuseStreak = Int(environment["MERERUN_H3_REUSE_STREAK"] ?? "")
        let adaptiveCachePolicy = !permitsCacheReuse || usesScheduledTailCache
            ? nil
            : accelerationMode.adaptiveFirstBlockCachePolicy.map { policy in
                let globalThreshold = Float(environment["MERERUN_H3_FIRST_BLOCK_THRESHOLD"] ?? "")
                    .flatMap { $0 > 0 ? $0 : nil }
                    ?? policy.globalThreshold
                let temporalThreshold = Float(
                    environment["MERERUN_H3_FIRST_BLOCK_TEMPORAL_THRESHOLD"] ?? ""
                ).flatMap { $0 > 0 ? $0 : nil }
                    ?? policy.temporalThreshold
                let maximumConsecutiveCachedSteps = requestedReuseStreak.flatMap {
                    $0 >= 0 ? $0 : nil
                } ?? policy.maximumConsecutiveCachedSteps
                return MiniMaxH3AdaptiveFirstBlockCachePolicy(
                    globalThreshold: globalThreshold,
                    temporalThreshold: temporalThreshold,
                    window: policy.window,
                    maximumConsecutiveCachedSteps: maximumConsecutiveCachedSteps,
                    minimumFullSteps: policy.minimumFullSteps,
                    requiredFinalFullSteps: policy.requiredFinalFullSteps
                )
            }
        let blockReusePolicy = permitsCacheReuse && usesScheduledTailCache
            ? accelerationMode.blockReusePolicy.map { policy in
                let requestedCacheDepth = Double(environment["MERERUN_H3_REUSE_DEPTH"] ?? "")
                let cacheDepth = requestedCacheDepth.flatMap {
                    (0..<1).contains($0) ? $0 : nil
                } ?? policy.cacheDepth
                let maximumConsecutiveCachedSteps = requestedReuseStreak.flatMap {
                    $0 >= 0 ? $0 : nil
                } ?? policy.maximumConsecutiveCachedSteps
                return MiniMaxH3BlockReusePolicy(
                    cacheDepth: cacheDepth,
                    window: policy.window,
                    maximumConsecutiveCachedSteps: maximumConsecutiveCachedSteps
                )
            } : nil
        let configuredDynamicSparseAttentionPolicy = environment["MERERUN_H3_DYNAMIC_SPARSE"] == "0"
            ? nil
            : accelerationMode.dynamicSparseAttentionPolicy.map { policy in
                DynamicSparseAttentionPolicy(
                    thresholdStandardDeviations: Float(
                        environment["MERERUN_H3_DYNAMIC_SPARSE_TAU"] ?? ""
                    ).flatMap { $0 >= 0 ? $0 : nil } ?? policy.thresholdStandardDeviations,
                    minimumSequenceLength: policy.minimumSequenceLength,
                    denseLeadingStepFraction: policy.denseLeadingStepFraction,
                    denseTrailingStepCount: policy.denseTrailingStepCount,
                    denseLeadingLayerCount: policy.denseLeadingLayerCount
                )
            }
        let dynamicSparseAttentionPolicy = configuredDynamicSparseAttentionPolicy.flatMap { policy in
            layout.sequenceLength >= policy.minimumSequenceLength ? policy : nil
        }
        let velocityReusePolicy = accelerationMode.velocityReusePolicy
        let layerThinningPolicy = accelerationMode.layerThinningPolicy
        let tokenReductionPolicy = accelerationMode.tokenReductionPolicy
        if let layerThinningPolicy {
            guard let adaLNCache else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "layer thinning requires a compatible precomputed AdaLN table"
                )
            }
            transformer.activeBlockIndices = Set(layerThinningPolicy.activeBlockIndices(
                blockModulations: adaLNCache.blockModulations
            ))
        } else {
            transformer.activeBlockIndices = nil
        }
        let tokenReduction = tokenReductionPolicy.map { _ in
            transformer.prepareTokenReduction(context: context)
        }
        let executionMode = exactKernelMode.requiresEagerExecution(
            sequenceLength: layout.sequenceLength
        )
            ? MiniMaxH3DenoiseExecutionMode.eagerStep
            : layerThinningPolicy == nil
            && velocityReusePolicy == nil
            && tokenReductionPolicy == nil
            && adaptiveCachePolicy == nil
            && blockReusePolicy == nil
            && dynamicSparseAttentionPolicy == nil
            ? MiniMaxH3DenoiseExecutionPolicy.mode(
                usesResidentBF16: transformer.usesResidentBF16,
                sequenceLength: layout.sequenceLength,
                usesBlockProfiling: blockProfileLogger != nil,
                denoiseStepCount: videoSchedule.timesteps.count,
                profilingOverride: ProcessInfo.processInfo.environment["MERERUN_H3_EXECUTION_MODE"]
            )
            : .blockwiseCompiled
        let usesCompiledStep = executionMode == .compiledStep
        transformer.usesBlockwiseCompilation = executionMode == .blockwiseCompiled
        let attentionKernelSchedule = MiniMaxH3DenoiseExecutionPolicy.attentionKernelSchedule(
            sequenceLength: layout.sequenceLength
        )
        transformer.maximumAttentionQueryTokensPerKernel = max(
            1,
            Int(environment["MERERUN_H3_ATTENTION_QUERY_TOKENS"] ?? "")
                ?? attentionKernelSchedule.maximumQueryTokens
        )
        transformer.maximumAttentionHeadsPerKernel = Int(
            environment["MERERUN_H3_ATTENTION_HEADS_PER_KERNEL"] ?? ""
        ).map { max(1, $0) } ?? attentionKernelSchedule.maximumHeadsPerKernel
        transformer.maximumAttentionKernelsPerEvaluation = max(
            1,
            Int(environment["MERERUN_H3_ATTENTION_EVALUATION_BATCH"] ?? "")
                ?? attentionKernelSchedule.maximumKernelsPerEvaluation
        )
        transformer.dynamicSparseAttentionPolicy = dynamicSparseAttentionPolicy
        transformer.dynamicSparseAttentionStepCount = videoSchedule.timesteps.count
        transformer.dynamicSparseAttentionLogHandler = stepProfileLogger
        transformer.usesFusedPostAttention = ProcessInfo.processInfo
            .environment["MERERUN_H3_FUSED_POST_ATTENTION"] == "1"
        transformer.usesLayerwiseEvaluation = executionMode.usesLayerwiseEvaluation
        transformer.clearsCacheAfterLayerwiseEvaluation = false
        let attentionHeadsPerKernel = transformer.maximumAttentionHeadsPerKernel
            ?? transformer.configuration.attentionHeadCount
        stepProfileLogger?(
            "execution_mode=\(executionMode) acceleration=\(accelerationMode.rawValue) "
                + "exact_kernels=\(exactKernelMode.rawValue) "
                + "fused_post_attention=\(transformer.usesFusedPostAttention) "
                + "attention_query_tokens=\(transformer.maximumAttentionQueryTokensPerKernel) "
                + "attention_heads_per_kernel=\(attentionHeadsPerKernel) "
                + "attention_evaluation_batch="
                + "\(transformer.maximumAttentionKernelsPerEvaluation) "
                + "dynamic_sparse=\(dynamicSparseAttentionPolicy != nil) "
                + "velocity_reuse_interval=\(velocityReusePolicy?.interval ?? 0) "
                + "token_reduction=\(tokenReductionPolicy != nil) "
                + "active_blocks=\(transformer.activeBlockCount)"
        )
        if let tokenReductionPolicy, let tokenReduction {
            stepProfileLogger?(
                "token_reduction_begin=\(tokenReductionPolicy.beginBlock) "
                    + "token_reduction_end=\(tokenReductionPolicy.endBlock) "
                    + "token_reduction_early_steps=\(tokenReductionPolicy.earlyStepCount) "
                    + "token_reduction_early_end=\(tokenReductionPolicy.earlyEndBlock) "
                    + "full_rows=\(layout.sequenceLength) "
                    + "reduced_rows=\(tokenReduction.reducedContext.layout.sequenceLength)"
            )
        }
        if let dynamicSparseAttentionPolicy {
            stepProfileLogger?(
                "dynamic_sparse_tau=\(dynamicSparseAttentionPolicy.thresholdStandardDeviations) "
                    + "dense_step_fraction=\(dynamicSparseAttentionPolicy.denseLeadingStepFraction) "
                    + "dense_trailing_steps=\(dynamicSparseAttentionPolicy.denseTrailingStepCount) "
                    + "dense_leading_layers=\(dynamicSparseAttentionPolicy.denseLeadingLayerCount) "
                    + "prefix_sink=\(layout.targetVideoRows.lowerBound)"
            )
        }
        if let blockReusePolicy {
            stepProfileLogger?(
                "cache_strategy=scheduled-tail "
                    + "reuse_depth=\(blockReusePolicy.cacheDepth) "
                    + "reuse_streak=\(blockReusePolicy.maximumConsecutiveCachedSteps)"
            )
        }
        if let adaptiveCachePolicy {
            stepProfileLogger?(
                "cache_strategy=adaptive-first-block "
                    + "global_threshold=\(adaptiveCachePolicy.globalThreshold) "
                    + "temporal_threshold=\(adaptiveCachePolicy.temporalThreshold) "
                    + "reuse_streak=\(adaptiveCachePolicy.maximumConsecutiveCachedSteps) "
                    + "final_full_steps=\(adaptiveCachePolicy.requiredFinalFullSteps)"
            )
        }

        let compiledStep = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
            let videoSample = inputs[0]
            let audioSample = inputs[1]
            let timestepValues = inputs[2]
            let videoCoefficients = inputs[3]
            let audioCoefficients = inputs[4]
            let videoInput = conditionVideoRows.map {
                MLX.concatenated([$0, videoSample], axis: 1)
            } ?? videoSample
            let audioInput = conditionAudioRows.map {
                MLX.concatenated([$0, audioSample], axis: 1)
            } ?? audioSample
            let predicted = transformer(
                videoRows: videoInput,
                audioRows: audioInput,
                context: context,
                timesteps: timestepValues,
                cachedAdaLN: adaLNCache.map { _ in
                    let blockStart = 6
                    let blockEnd = blockStart + transformer.configuration.layerCount
                    return MiniMaxH3AdaLNStep(
                        timeEmbedding: inputs[5],
                        blockModulations: Array(inputs[blockStart..<blockEnd]),
                        finalModulation: inputs[blockEnd]
                    )
                }
            )
            return [
                Self.advance(
                    sample: videoSample,
                    velocity: predicted.videoVelocityRows,
                    coefficients: videoCoefficients
                ),
                Self.advance(
                    sample: audioSample,
                    velocity: predicted.audioVelocityRows,
                    coefficients: audioCoefficients
                ),
            ]
        }

        let totalBlockCount = transformer.activeBlockCount
        let warmBlockCount = blockReusePolicy?.warmBlockCount(totalBlockCount: totalBlockCount)
        var cachedTailResidual: MLXArray?
        var previousFirstResidual: MLXArray?
        var cachedTargetTailResidual: MLXArray?
        var consecutiveCachedSteps = 0
        var cachedStepCount = 0
        var fullStepCount = 0
        var executedBlockCount = 0
        var previousVideoVelocity: MLXArray?
        var previousAudioVelocity: MLXArray?
        var cachedVideoVelocity: MLXArray?
        var cachedAudioVelocity: MLXArray?
        var previousVelocityStepIndex: Int?
        var cachedVelocityStepIndex: Int?

        for index in videoSchedule.timesteps.indices {
            transformer.dynamicSparseAttentionStepIndex = index
            let stepStarted = stepProfileLogger.map { _ in CFAbsoluteTimeGetCurrent() }
            progressHandler?(.init(
                stage: .denoising,
                stepIndex: index,
                totalSteps: videoSchedule.timesteps.count
            ))
            let videoTimestep = videoSchedule.timesteps[index]
            let audioTimestep = audioSchedule.timesteps[index]
            let timestepValues = MLXArray([
                videoTimestep,
                audioTimestep,
                max(videoTimestep, 0.999),
            ])
            let videoCoefficients = Self.scheduleCoefficients(videoSchedule, index: index)
            let audioCoefficients = Self.scheduleCoefficients(audioSchedule, index: index)
            let reusesTail = blockReusePolicy?.shouldReuseTail(
                stepIndex: index,
                stepCount: videoSchedule.timesteps.count,
                videoSigmas: videoSchedule.sigmas,
                audioSigmas: audioSchedule.sigmas,
                hasCachedResidual: cachedTailResidual != nil,
                consecutiveCachedSteps: consecutiveCachedSteps
            ) ?? false
            var cacheHitThisStep = false
            var executedBlocksThisStep = totalBlockCount
            var firstBlockChange: MiniMaxH3FirstBlockChange?
            let reusesVelocity = velocityReusePolicy?.shouldReuse(
                stepIndex: index,
                stepCount: videoSchedule.timesteps.count,
                hasCachedVelocity: cachedVideoVelocity != nil && cachedAudioVelocity != nil
            ) ?? false
            if reusesVelocity,
               let cachedVideoVelocity,
               let cachedAudioVelocity,
               let cachedVelocityStepIndex {
                let previousVideoSigma = previousVelocityStepIndex.map {
                    videoSchedule.sigmas[$0]
                }
                let previousAudioSigma = previousVelocityStepIndex.map {
                    audioSchedule.sigmas[$0]
                }
                let videoRatio = MiniMaxH3ServingContract.extrapolationRatio(
                    currentSigma: videoSchedule.sigmas[index],
                    lastSigma: videoSchedule.sigmas[cachedVelocityStepIndex],
                    previousSigma: previousVideoSigma
                )
                let audioRatio = MiniMaxH3ServingContract.extrapolationRatio(
                    currentSigma: audioSchedule.sigmas[index],
                    lastSigma: audioSchedule.sigmas[cachedVelocityStepIndex],
                    previousSigma: previousAudioSigma
                )
                let videoVelocity = previousVideoVelocity.map {
                    cachedVideoVelocity + videoRatio * (cachedVideoVelocity - $0)
                } ?? cachedVideoVelocity
                let audioVelocity = previousAudioVelocity.map {
                    cachedAudioVelocity + audioRatio * (cachedAudioVelocity - $0)
                } ?? cachedAudioVelocity
                videoRows = Self.advance(
                    sample: videoRows,
                    velocity: videoVelocity,
                    coefficients: videoCoefficients
                )
                audioRows = Self.advance(
                    sample: audioRows,
                    velocity: audioVelocity,
                    coefficients: audioCoefficients
                )
                cacheHitThisStep = true
                executedBlocksThisStep = 0
                cachedStepCount += 1
            } else if usesCompiledStep {
                var inputs = [
                    videoRows,
                    audioRows,
                    timestepValues,
                    videoCoefficients,
                    audioCoefficients,
                ]
                if let cacheStep = adaLNCache?.step(at: index) {
                    inputs.append(cacheStep.timeEmbedding)
                    inputs.append(contentsOf: cacheStep.blockModulations)
                    inputs.append(cacheStep.finalModulation)
                }
                let outputs = compiledStep(inputs)
                videoRows = outputs[0]
                audioRows = outputs[1]
                fullStepCount += 1
            } else {
                let videoInput = conditionVideoRows.map {
                    MLX.concatenated([$0, videoRows], axis: 1)
                } ?? videoRows
                let audioInput = conditionAudioRows.map {
                    MLX.concatenated([$0, audioRows], axis: 1)
                } ?? audioRows
                let predicted: MiniMaxH3TransformerOutput
                if let tokenReductionPolicy, let tokenReduction {
                    predicted = transformer.callWithTokenReduction(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        reduction: tokenReduction,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        policy: tokenReductionPolicy,
                        stepIndex: index
                    )
                    fullStepCount += 1
                    stepProfileLogger?(
                        "step_plan=\(index + 1)/\(videoSchedule.timesteps.count) "
                            + "token_reduction_blocks=\(tokenReductionPolicy.beginBlock)..<"
                            + "\(tokenReductionPolicy.restoreBeforeBlock(stepIndex: index)) "
                            + "rows=\(tokenReduction.reducedContext.layout.sequenceLength)"
                    )
                } else if let adaptiveCachePolicy {
                    let canConsiderReuse = adaptiveCachePolicy.canConsiderReuse(
                        stepIndex: index,
                        stepCount: videoSchedule.timesteps.count,
                        fullStepCount: fullStepCount,
                        consecutiveCachedSteps: consecutiveCachedSteps,
                        hasCachedState: previousFirstResidual != nil && cachedTargetTailResidual != nil
                    )
                    let result = transformer.callWithAdaptiveFirstBlockReuse(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        policy: adaptiveCachePolicy,
                        canConsiderReuse: canConsiderReuse,
                        previousFirstResidual: previousFirstResidual,
                        cachedTargetTailResidual: cachedTargetTailResidual
                    )
                    predicted = result.output
                    firstBlockChange = result.change
                    if result.reusedTail {
                        cacheHitThisStep = true
                        executedBlocksThisStep = 1
                        consecutiveCachedSteps += 1
                        cachedStepCount += 1
                    } else if let refreshedFirstResidual = result.refreshedFirstResidual,
                              let refreshedTargetTailResidual = result.refreshedTargetTailResidual {
                        previousFirstResidual = refreshedFirstResidual
                        cachedTargetTailResidual = refreshedTargetTailResidual
                        consecutiveCachedSteps = 0
                        fullStepCount += 1
                    } else {
                        preconditionFailure("adaptive MiniMax-H3 cache did not return refreshed state")
                    }
                } else if let warmBlockCount {
                    let result = transformer.callWithBlockResidualReuse(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index),
                        warmBlockCount: warmBlockCount,
                        cachedTailResidual: reusesTail ? cachedTailResidual : nil
                    )
                    predicted = result.output
                    if let refreshedTailResidual = result.refreshedTailResidual {
                        cachedTailResidual = refreshedTailResidual
                        consecutiveCachedSteps = 0
                        fullStepCount += 1
                    } else {
                        cacheHitThisStep = true
                        executedBlocksThisStep = warmBlockCount
                        consecutiveCachedSteps += 1
                        cachedStepCount += 1
                    }
                } else {
                    predicted = transformer(
                        videoRows: videoInput,
                        audioRows: audioInput,
                        context: context,
                        timesteps: timestepValues,
                        cachedAdaLN: adaLNCache?.step(at: index)
                    )
                    fullStepCount += 1
                }
                if velocityReusePolicy != nil {
                    previousVideoVelocity = cachedVideoVelocity
                    previousAudioVelocity = cachedAudioVelocity
                    previousVelocityStepIndex = cachedVelocityStepIndex
                    cachedVideoVelocity = predicted.videoVelocityRows
                    cachedAudioVelocity = predicted.audioVelocityRows
                    cachedVelocityStepIndex = index
                }
                videoRows = Self.advance(
                    sample: videoRows,
                    velocity: predicted.videoVelocityRows,
                    coefficients: videoCoefficients
                )
                audioRows = Self.advance(
                    sample: audioRows,
                    velocity: predicted.audioVelocityRows,
                    coefficients: audioCoefficients
                )
            }
            executedBlockCount += executedBlocksThisStep
            if adaptiveCachePolicy != nil || blockReusePolicy != nil || velocityReusePolicy != nil {
                var plan = "step_plan=\(index + 1)/\(videoSchedule.timesteps.count) "
                    + "cache_hit=\(cacheHitThisStep) blocks=\(executedBlocksThisStep)"
                if let firstBlockChange {
                    plan += String(
                        format: " video_global=%.5f audio_global=%.5f "
                            + "video_temporal=%.5f audio_temporal=%.5f",
                        firstBlockChange.videoGlobal,
                        firstBlockChange.audioGlobal,
                        firstBlockChange.videoTemporalMaximum,
                        firstBlockChange.audioTemporalMaximum
                    )
                }
                stepProfileLogger?(plan)
            }
            MLX.eval(videoRows, audioRows)
            if let stepStarted, let stepProfileLogger {
                let memory = Memory.snapshot()
                stepProfileLogger(String(
                    format: "step=%02d/%02d seconds=%.3f active_gib=%.2f cache_gib=%.2f peak_gib=%.2f",
                    index + 1,
                    videoSchedule.timesteps.count,
                    CFAbsoluteTimeGetCurrent() - stepStarted,
                    Double(memory.activeMemory) / 1_073_741_824,
                    Double(memory.cacheMemory) / 1_073_741_824,
                    Double(memory.peakMemory) / 1_073_741_824
                ))
            }
        }
        if adaptiveCachePolicy != nil || blockReusePolicy != nil || velocityReusePolicy != nil {
            stepProfileLogger?(
                "cached_steps=\(cachedStepCount)/\(videoSchedule.timesteps.count) "
                    + "full_steps=\(fullStepCount) executed_blocks=\(executedBlockCount) "
                    + "baseline_blocks=\(videoSchedule.timesteps.count * totalBlockCount)"
            )
        }
        return (videoRows, audioRows)
    }

    private static func scheduleCoefficients(_ schedule: MiniMaxH3Schedule, index: Int) -> MLXArray {
        MLXArray([
            schedule.timesteps[index],
            schedule.sigmas[index + 1] / schedule.sigmas[index],
        ])
    }

    private static func advance(
        sample: MLXArray,
        velocity: MLXArray,
        coefficients: MLXArray
    ) -> MLXArray {
        let timestep = coefficients[0]
        let ratio = coefficients[1]
        let denoised = sample + (1 - timestep) * velocity
        return (
            ratio * sample.asType(.float32)
                + (1 - ratio) * denoised.asType(.float32)
        ).asType(sample.dtype)
    }

    private static func loadAdaLNCache(
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) throws -> MiniMaxH3AdaLNCachePack.Selection? {
        do {
            return try MiniMaxH3AdaLNCachePack.load(
                from: resources.rootURL,
                configuration: .init(configuration),
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: resources.adaLNCacheSourceIdentity()
            )
        } catch {
            guard try !resources.requiresAdaLNCache() else {
                throw error
            }
            return nil
        }
    }

    public func generate(
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        continuation: MiniMaxH3ContinuationInput? = nil,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)? = nil
    ) throws -> MiniMaxH3GenerationResult {
        let phaseProfileLogger = MereRunRuntimeDebug.logger(
            keys: ["MERERUN_H3_PROFILE_PHASES"],
            prefix: "[minimax-h3-phase-profile]"
        )
        let generationStarted = CFAbsoluteTimeGetCurrent()
        let missing = resources.validate()
        guard missing.isEmpty else { throw MiniMaxH3GeneratorError.missingModelFiles(missing) }
        let configuration = try resources.loadConfiguration()
        if let adapterInferenceRecipe = options.adapterInferenceRecipe,
           !adapterInferenceRecipe.supports(task: configuration.task) {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "MiniMax-H3 adapter recipe \(adapterInferenceRecipe.name) requires \(adapterInferenceRecipe.task.rawValue), not \(configuration.task)"
            )
        }
        if configuration.task == "ref2va" {
            return try generateRef2VA(
                options: options,
                resources: resources,
                configuration: configuration,
                continuation: continuation,
                progressHandler: progressHandler,
                phaseProfileLogger: phaseProfileLogger,
                generationStarted: generationStarted
            )
        }
        guard options.references.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("--reference requires a Ref2VA model root")
        }
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: options.numFrames)
        let latentHeight = options.internalHeight / 16
        let latentWidth = options.internalWidth / 16
        let audioFrames = MiniMaxH3Geometry.audioLatentFrameCount(for: options.numFrames)
        if let continuation {
            guard !options.usesReducedRenderCanvas else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "reduced internal rendering does not yet support continuation or sliding windows"
                )
            }
            guard continuation.frames.dim(2) == options.height,
                  continuation.frames.dim(3) == options.width else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "continuation dimensions must match the target H3 dimensions"
                )
            }
            guard !options.frameInputs.contains(where: { $0.frameIndex == 0 }),
                  options.firstFrameURL == nil else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "continuation already supplies the first target frame"
                )
            }
        }

        progressHandler?(.init(stage: .loadingTextEncoder, stepIndex: 0, totalSteps: options.steps - 1))
        let conditionerPreparationStarted = CFAbsoluteTimeGetCurrent()
        let tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 262_144)
        let presentation = try conditionerPresentation(
            tokenizer: tokenizer,
            options: options,
            continuationFrame: continuation.map(Self.boundaryFrame)
        )
        guard !presentation.tokenIDs.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("prompt tokenized to zero rows")
        }
        guard presentation.tokenIDs.count <= 262_144 else {
            throw MiniMaxH3GeneratorError.invalidOptions("multimodal presentation exceeds 262144 tokens")
        }
        let inputIDs = MLXArray(presentation.tokenIDs.map(Int32.init)).reshaped(1, presentation.tokenIDs.count)
        let attentionMask = MLXArray.ones([1, presentation.tokenIDs.count], dtype: .int32)
        phaseProfileLogger?(String(
            format: "phase=conditioner_preparation seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - conditionerPreparationStarted
        ))
        let promptStates: MLXArray = try withMiniMaxH3AutoreleasePool {
            let conditionerLoadStarted = CFAbsoluteTimeGetCurrent()
            let encoder = try loadConditioner(
                resources: resources,
                configuration: configuration,
                progressHandler: { shard in
                    progressHandler?(.init(
                        stage: .loadingTextEncoder,
                        stepIndex: shard.shardIndex,
                        totalSteps: shard.shardCount
                    ))
                }
            )
            phaseProfileLogger?(String(
                format: "phase=conditioner_load seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - conditionerLoadStarted
            ))
            progressHandler?(.init(stage: .encodingText, stepIndex: 0, totalSteps: options.steps - 1))
            let textEncodingStarted = CFAbsoluteTimeGetCurrent()
            guard let states = try encoder.forwardMultimodalActivationHiddenState(
                inputIds: inputIDs,
                attentionMask: attentionMask,
                images: presentation.images,
                activationLayer: 49
            ) else {
                throw MiniMaxH3GeneratorError.invalidOptions("text encoder did not return layer-50 states")
            }
            MLX.eval(states)
            phaseProfileLogger?(String(
                format: "phase=text_encoding seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - textEncodingStarted
            ))
            return states
        }
        Memory.clearCache()

        let frameConditions = Self.frameConditions(options: options)
        let keyframeEncodingStarted = CFAbsoluteTimeGetCurrent()
        let continuationConditions = try continuation.map {
            try encodeContinuation(
                $0,
                resources: resources,
                progressHandler: progressHandler
            )
        }
        let keyframeRows = try encodeKeyframes(
            frameConditions.map(\.url),
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        var conditionVideoRows = Self.concatenateRows([
            continuationConditions?.videoRows,
            keyframeRows,
        ])
        var conditionAudioRows = continuationConditions?.audioRows
        phaseProfileLogger?(String(
            format: "phase=keyframe_encoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - keyframeEncodingStarted
        ))
        Memory.clearCache()

        let layout = try MiniMaxH3Geometry.buildFL2VA(
            textTokenTags: presentation.tokenTags,
            videoLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioFrames,
            keyframeAnchors: (continuationConditions?.videoAnchors ?? [])
                + frameConditions.map(\.anchor),
            audioConditionAnchors: continuationConditions?.audioAnchors ?? []
        )
        if let currentConditions = conditionVideoRows {
            conditionVideoRows = MiniMaxH3ServingContract.noisedCondition(
                currentConditions,
                seed: options.seed
            )
        }
        if let currentConditions = conditionAudioRows {
            conditionAudioRows = MiniMaxH3ServingContract.noisedCondition(
                currentConditions,
                seed: options.seed &+ 1
            )
        }
        var video = MiniMaxH3ServingContract.targetVideoNoise(
            seed: options.seed,
            latentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth
        )
        var videoRows = MiniMaxH3Geometry.patchifyVideo(video)
        var audioRows = MiniMaxH3ServingContract.targetAudioNoise(
            seed: options.seed,
            latentFrames: audioFrames
        )
        let videoSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.videoFlowShift ?? configuration.videoFlowShift
        )
        let audioSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.audioFlowShift ?? configuration.audioFlowShift
        )

        progressHandler?(.init(stage: .loadingTransformer, stepIndex: 0, totalSteps: options.steps - 1))
        try withMiniMaxH3AutoreleasePool {
            let transformerPreparationStarted = CFAbsoluteTimeGetCurrent()
            let runtime = try loadDenoisingRuntime(
                resources: resources,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sequenceLength: layout.sequenceLength,
                weightMode: options.transformerWeightMode,
                adapterURL: options.adapterURL,
                adapterStrength: options.adapterStrength,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=transformer_preparation seconds=%.3f resident_bf16=%@",
                CFAbsoluteTimeGetCurrent() - transformerPreparationStarted,
                runtime.transformer.usesResidentBF16 ? "true" : "false"
            ))
            let denoisingStarted = CFAbsoluteTimeGetCurrent()
            (videoRows, audioRows) = try denoise(
                transformer: runtime.transformer,
                videoRows: videoRows,
                audioRows: audioRows,
                conditionVideoRows: conditionVideoRows,
                conditionAudioRows: conditionAudioRows,
                promptStates: promptStates,
                layout: layout,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                adaLNCache: runtime.adaLNCache,
                accelerationMode: options.accelerationMode,
                permitsCacheReuse: true,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=denoising seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - denoisingStarted
            ))
        }
        Memory.clearCache()
        video = MiniMaxH3Geometry.unpatchifyVideo(
            videoRows,
            frames: latentFrames,
            height: latentHeight,
            width: latentWidth
        )
        let audio = MiniMaxH3Geometry.unpackAudio(audioRows[0])
        progressHandler?(.init(stage: .decodingVideo, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let videoDecodingStarted = CFAbsoluteTimeGetCurrent()
        let frames: MLXArray = try withMiniMaxH3AutoreleasePool {
            let vae = try loadVideoVAE(resources: resources)
            let decoded = Self.mediaFrames(from: vae.decode(video))
            let pixels = try MiniMaxH3FrameScaler.scaled(
                decoded,
                width: options.width,
                height: options.height
            )
            MLX.eval(pixels)
            return pixels
        }
        phaseProfileLogger?(String(
            format: "phase=video_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - videoDecodingStarted
        ))
        Memory.clearCache()
        progressHandler?(.init(stage: .decodingAudio, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let audioDecodingStarted = CFAbsoluteTimeGetCurrent()
        let waveform: MLXArray = try withMiniMaxH3AutoreleasePool {
            let vae = try loadAudioVAE(resources: resources)
            let decoded = vae.decode(audio)
            MLX.eval(decoded)
            return decoded
        }
        phaseProfileLogger?(String(
            format: "phase=audio_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - audioDecodingStarted
        ))
        Memory.clearCache()
        phaseProfileLogger?(String(
            format: "phase=generation_total seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - generationStarted
        ))
        return MiniMaxH3GenerationResult(frames: frames, audio: waveform, seed: options.seed)
    }

    private func generateRef2VA(
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        configuration: MiniMaxH3Configuration,
        continuation: MiniMaxH3ContinuationInput?,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?,
        phaseProfileLogger: (@Sendable (String) -> Void)?,
        generationStarted: CFTimeInterval
    ) throws -> MiniMaxH3GenerationResult {
        guard options.firstFrameURL == nil,
              options.lastFrameURL == nil,
              options.frameInputs.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA uses ordered references, not FL2VA keyframes")
        }
        guard !options.references.isEmpty else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA requires at least one --reference")
        }
        let latentFrames = try MiniMaxH3Geometry.videoLatentFrameCount(for: options.numFrames)
        let latentHeight = options.internalHeight / 16
        let latentWidth = options.internalWidth / 16
        let audioFrames = MiniMaxH3Geometry.audioLatentFrameCount(for: options.numFrames)
        if let continuation {
            guard !options.usesReducedRenderCanvas else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "reduced internal rendering does not yet support continuation or sliding windows"
                )
            }
            guard continuation.frames.dim(2) == options.height,
                  continuation.frames.dim(3) == options.width else {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "continuation dimensions must match the target H3 dimensions"
                )
            }
        }
        let prepared = try preparedReferences(options: options)

        progressHandler?(.init(stage: .loadingTextEncoder, stepIndex: 0, totalSteps: options.steps - 1))
        let conditionerPreparationStarted = CFAbsoluteTimeGetCurrent()
        let tokenizer = try QwenTokenizer.load(from: resources.tokenizerURL, maxLengthOverride: 262_144)
        let presentation = try referenceConditionerPresentation(
            tokenizer: tokenizer,
            prompt: options.prompt,
            references: prepared,
            continuationFrame: continuation.map(Self.boundaryFrame)
        )
        guard !presentation.tokenIDs.isEmpty, presentation.tokenIDs.count <= 262_144 else {
            throw MiniMaxH3GeneratorError.invalidOptions("Ref2VA presentation must contain 1...262144 tokens")
        }
        let inputIDs = MLXArray(presentation.tokenIDs.map(Int32.init)).reshaped(1, presentation.tokenIDs.count)
        let attentionMask = MLXArray.ones([1, presentation.tokenIDs.count], dtype: .int32)
        phaseProfileLogger?(String(
            format: "phase=conditioner_preparation seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - conditionerPreparationStarted
        ))
        let promptStates: MLXArray = try withMiniMaxH3AutoreleasePool {
            let conditionerLoadStarted = CFAbsoluteTimeGetCurrent()
            let encoder = try loadConditioner(
                resources: resources,
                configuration: configuration,
                progressHandler: { shard in
                    progressHandler?(.init(
                        stage: .loadingTextEncoder,
                        stepIndex: shard.shardIndex,
                        totalSteps: shard.shardCount
                    ))
                }
            )
            phaseProfileLogger?(String(
                format: "phase=conditioner_load seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - conditionerLoadStarted
            ))
            progressHandler?(.init(stage: .encodingText, stepIndex: 0, totalSteps: options.steps - 1))
            let textEncodingStarted = CFAbsoluteTimeGetCurrent()
            guard let states = try encoder.forwardMultimodalActivationHiddenState(
                inputIds: inputIDs,
                attentionMask: attentionMask,
                images: presentation.images,
                activationLayer: 49
            ) else {
                throw MiniMaxH3GeneratorError.invalidOptions("text encoder did not return layer-50 states")
            }
            MLX.eval(states)
            phaseProfileLogger?(String(
                format: "phase=text_encoding seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - textEncodingStarted
            ))
            return states
        }
        Memory.clearCache()

        let referenceEncodingStarted = CFAbsoluteTimeGetCurrent()
        let referenceRows = try encodeReferences(
            prepared,
            options: options,
            resources: resources,
            progressHandler: progressHandler
        )
        let continuationConditions = try continuation.map {
            try encodeContinuation(
                $0,
                resources: resources,
                progressHandler: progressHandler
            )
        }
        var videoConditionSpans = referenceRows.video
        if let continuationVideo = continuationConditions?.videoRows {
            videoConditionSpans.insert(continuationVideo, at: 0)
        }
        var audioConditionSpans = referenceRows.audio
        if let continuationAudio = continuationConditions?.audioRows {
            audioConditionSpans.insert(continuationAudio, at: 0)
        }
        phaseProfileLogger?(String(
            format: "phase=reference_encoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - referenceEncodingStarted
        ))
        Memory.clearCache()

        let layout = try MiniMaxH3Geometry.buildRef2VA(
            textTokenTags: presentation.tokenTags,
            references: prepared.map(\.geometry),
            videoLatentFrames: latentFrames,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            audioLatentFrames: audioFrames,
            keyframeAnchors: continuationConditions?.videoAnchors ?? [],
            audioConditionAnchors: continuationConditions?.audioAnchors ?? []
        )
        let noisedVideoConditions = Self.concatenateRows(videoConditionSpans.map {
            MiniMaxH3ServingContract.noisedCondition($0, seed: options.seed)
        })
        let noisedAudioConditions = Self.concatenateRows(audioConditionSpans.map {
            MiniMaxH3ServingContract.noisedCondition($0, seed: options.seed &+ 1)
        })
        var videoRows = MiniMaxH3Geometry.patchifyVideo(
            MiniMaxH3ServingContract.targetVideoNoise(
                seed: options.seed,
                latentFrames: latentFrames,
                latentHeight: latentHeight,
                latentWidth: latentWidth
            )
        )
        var audioRows = MiniMaxH3ServingContract.targetAudioNoise(
            seed: options.seed,
            latentFrames: audioFrames
        )
        let videoSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.videoFlowShift ?? configuration.videoFlowShift
        )
        let audioSchedule = try MiniMaxH3Schedule(
            pointCount: options.steps,
            shift: options.adapterInferenceRecipe?.audioFlowShift ?? configuration.audioFlowShift
        )

        progressHandler?(.init(stage: .loadingTransformer, stepIndex: 0, totalSteps: options.steps - 1))
        try withMiniMaxH3AutoreleasePool {
            let transformerPreparationStarted = CFAbsoluteTimeGetCurrent()
            let runtime = try loadDenoisingRuntime(
                resources: resources,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sequenceLength: layout.sequenceLength,
                weightMode: options.transformerWeightMode,
                adapterURL: options.adapterURL,
                adapterStrength: options.adapterStrength,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=transformer_preparation seconds=%.3f resident_bf16=%@",
                CFAbsoluteTimeGetCurrent() - transformerPreparationStarted,
                runtime.transformer.usesResidentBF16 ? "true" : "false"
            ))
            let denoisingStarted = CFAbsoluteTimeGetCurrent()
            (videoRows, audioRows) = try denoise(
                transformer: runtime.transformer,
                videoRows: videoRows,
                audioRows: audioRows,
                conditionVideoRows: noisedVideoConditions,
                conditionAudioRows: noisedAudioConditions,
                promptStates: promptStates,
                layout: layout,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                adaLNCache: runtime.adaLNCache,
                accelerationMode: options.accelerationMode,
                permitsCacheReuse: true,
                progressHandler: progressHandler
            )
            phaseProfileLogger?(String(
                format: "phase=denoising seconds=%.3f",
                CFAbsoluteTimeGetCurrent() - denoisingStarted
            ))
        }
        Memory.clearCache()
        let video = MiniMaxH3Geometry.unpatchifyVideo(
            videoRows,
            frames: latentFrames,
            height: latentHeight,
            width: latentWidth
        )
        let audio = MiniMaxH3Geometry.unpackAudio(audioRows[0])
        progressHandler?(.init(stage: .decodingVideo, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let videoDecodingStarted = CFAbsoluteTimeGetCurrent()
        let frames: MLXArray = try withMiniMaxH3AutoreleasePool {
            let decoded = Self.mediaFrames(
                from: try loadVideoVAE(resources: resources).decode(video)
            )
            let pixels = try MiniMaxH3FrameScaler.scaled(
                decoded,
                width: options.width,
                height: options.height
            )
            MLX.eval(pixels)
            return pixels
        }
        phaseProfileLogger?(String(
            format: "phase=video_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - videoDecodingStarted
        ))
        Memory.clearCache()
        progressHandler?(.init(stage: .decodingAudio, stepIndex: options.steps - 1, totalSteps: options.steps - 1))
        let audioDecodingStarted = CFAbsoluteTimeGetCurrent()
        let waveform: MLXArray = try withMiniMaxH3AutoreleasePool {
            let decoded = try loadAudioVAE(resources: resources).decode(audio)
            MLX.eval(decoded)
            return decoded
        }
        phaseProfileLogger?(String(
            format: "phase=audio_decoding seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - audioDecodingStarted
        ))
        Memory.clearCache()
        phaseProfileLogger?(String(
            format: "phase=generation_total seconds=%.3f",
            CFAbsoluteTimeGetCurrent() - generationStarted
        ))
        return MiniMaxH3GenerationResult(frames: frames, audio: waveform, seed: options.seed)
    }

    private func preparedReferences(
        options: MiniMaxH3GenerationOptions
    ) throws -> [PreparedReference] {
        let key = ReferenceCacheKey(
            references: options.references,
            maximumFrameCount: options.numFrames,
            targetWidth: options.internalWidth,
            targetHeight: options.internalHeight
        )
        if retainsRuntime,
           let retainedPreparedReferences,
           retainedPreparedReferences.key == key {
            return retainedPreparedReferences.values
        }
        let values = try prepareReferences(options: options)
        if retainsRuntime { retainedPreparedReferences = (key, values) }
        return values
    }

    private func encodeReferences(
        _ prepared: [PreparedReference],
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> PreparedReferenceRows {
        let key = ReferenceCacheKey(
            references: options.references,
            maximumFrameCount: options.numFrames,
            targetWidth: options.internalWidth,
            targetHeight: options.internalHeight
        )
        if retainsRuntime,
           let retainedPreparedReferenceRows,
           retainedPreparedReferenceRows.key == key {
            return retainedPreparedReferenceRows.values
        }
        progressHandler?(.init(stage: .encodingReferences, stepIndex: 0, totalSteps: prepared.count))
        let videoRows: [MLXArray] = try withMiniMaxH3AutoreleasePool {
            guard prepared.contains(where: { $0.visual != nil }) else { return [] }
            let vae = try loadVideoVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, reference) in prepared.enumerated() {
                guard let visual = reference.visual else { continue }
                let latent = reference.kind == .image
                    ? vae.encodeKeyframe(visual)
                    : vae.encodeReferenceVideo(visual)
                let packed = MiniMaxH3Geometry.patchifyVideo(latent).asType(.float32)
                MLX.eval(packed)
                rows.append(packed)
                progressHandler?(.init(
                    stage: .encodingReferences,
                    stepIndex: index + 1,
                    totalSteps: prepared.count
                ))
            }
            return rows
        }
        let audioRows: [MLXArray] = try withMiniMaxH3AutoreleasePool {
            guard prepared.contains(where: { $0.waveform != nil }) else { return [] }
            let vae = try loadAudioVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, reference) in prepared.enumerated() {
                guard let waveform = reference.waveform else { continue }
                let packed = MiniMaxH3Geometry.packAudio(
                    vae.encode(waveform)
                ).expandedDimensions(axis: 0)
                MLX.eval(packed)
                rows.append(packed)
                progressHandler?(.init(
                    stage: .encodingReferences,
                    stepIndex: index + 1,
                    totalSteps: prepared.count
                ))
            }
            return rows
        }
        let values = PreparedReferenceRows(video: videoRows, audio: audioRows)
        if retainsRuntime { retainedPreparedReferenceRows = (key, values) }
        return values
    }

    private func prepareReferences(options: MiniMaxH3GenerationOptions) throws -> [PreparedReference] {
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-minimax-h3-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryRoot) }
        let maximumReferenceAudioSamples = MiniMaxH3AudioVAE.samplingRate * 15
        var totalReferenceAudioSamples = 0
        var prepared: [PreparedReference] = []
        for (referenceIndex, input) in options.references.enumerated() {
            switch input.kind {
            case .image:
                let source: MediaImage
                do { source = try MediaImageIO.decode(input.url) } catch {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, error.localizedDescription)
                }
                let size = try MiniMaxH3ServingContract.referenceImageCanvas(
                    width: source.width,
                    height: source.height,
                    targetWidth: options.internalWidth,
                    targetHeight: options.internalHeight
                )
                let image = try MediaImageIO.resized(source, width: size.width, height: size.height)
                let visual = Self.rgbTensor(image).expandedDimensions(axis: 0)
                let vision = try QwenVLImageLoader.pixelValues(image: image, patchSize: 16, spatialMergeSize: 2)
                prepared.append(.init(
                    kind: .image,
                    visual: visual,
                    visionBlocks: [vision],
                    blockTimestamps: [],
                    waveform: nil,
                    geometry: .init(
                        kind: .image,
                        videoLatentFrames: 1,
                        latentHeight: size.height / 16,
                        latentWidth: size.width / 16
                    )
                ))
            case .video:
                let framesDirectory = temporaryRoot.appendingPathComponent("video-\(referenceIndex)", isDirectory: true)
                try FileManager.default.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
                let sequence: VideoFrameSequence
                do { sequence = try MediaVideoIO.extractFrames(from: input.url, into: framesDirectory) } catch {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, error.localizedDescription)
                }
                guard !sequence.frameURLs.isEmpty else {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, "video contains no frames")
                }
                let resampled = Self.resampledFrameIndices(
                    sourceCount: sequence.frameURLs.count,
                    sourceFPS: sequence.fps,
                    maximumCount: options.numFrames
                )
                let selected = resampled.isEmpty ? [0] : resampled
                let alignedCount = Self.trimReferenceFrameCount(selected.count)
                var indices = selected
                if indices.count < alignedCount {
                    indices.append(contentsOf: repeatElement(indices.last!, count: alignedCount - indices.count))
                } else {
                    indices = Array(indices.prefix(alignedCount))
                }
                let canvas = try resolveVideoCanvas(width: sequence.frameWidth, height: sequence.frameHeight)
                var images: [MediaImage] = []
                images.reserveCapacity(indices.count)
                for index in indices {
                    let image = try MediaImageIO.decode(sequence.frameURLs[index])
                    images.append(try MediaImageIO.resized(image, width: canvas.width, height: canvas.height))
                }
                let visual = MLX.stacked(images.map(Self.rgbTensor), axis: 0).expandedDimensions(axis: 0)
                let sampled = stride(from: 0, to: images.count, by: 12).map { images[$0] }
                var blocks: [MLXArray] = []
                var timestamps: [Double] = []
                for start in stride(from: 0, to: sampled.count, by: 2) {
                    let second = min(start + 1, sampled.count - 1)
                    let firstPixels = try QwenVLImageLoader.pixelValues(
                        image: sampled[start], patchSize: 16, spatialMergeSize: 2
                    )
                    let secondPixels = try QwenVLImageLoader.pixelValues(
                        image: sampled[second], patchSize: 16, spatialMergeSize: 2
                    )
                    blocks.append(MLX.concatenated([firstPixels, secondPixels], axis: 0))
                    timestamps.append((Double(start) / 2 + Double(second) / 2) / 2)
                }
                let waveform: MLXArray?
                if MediaVideoIO.hasAudioTrack(input.url) {
                    let maximumSamples = Int(
                        (Double(alignedCount) / Double(MiniMaxH3Geometry.framesPerSecond)
                            * Double(MiniMaxH3AudioVAE.samplingRate)).rounded()
                    )
                    let decoded = try decodeReferenceAudio(
                        input.url,
                        maximumSamples: maximumSamples,
                        truncatesAtLimit: true
                    )
                    guard decoded.dim(1) <= maximumReferenceAudioSamples - totalReferenceAudioSamples else {
                        throw MiniMaxH3GeneratorError.invalidOptions(
                            "ordered reference audio exceeds 15 seconds in total"
                        )
                    }
                    totalReferenceAudioSamples += decoded.dim(1)
                    waveform = decoded
                } else {
                    waveform = nil
                }
                prepared.append(.init(
                    kind: .video,
                    visual: visual,
                    visionBlocks: blocks,
                    blockTimestamps: timestamps,
                    waveform: waveform,
                    geometry: .init(
                        kind: .video,
                        videoLatentFrames: try MiniMaxH3Geometry.videoLatentFrameCount(for: alignedCount),
                        latentHeight: canvas.height / 16,
                        latentWidth: canvas.width / 16,
                        audioLatentFrames: waveform.map { ($0.dim(1) + 799) / 800 } ?? 0
                    )
                ))
            case .audio:
                let waveform = try decodeReferenceAudio(
                    input.url,
                    maximumSamples: maximumReferenceAudioSamples,
                    truncatesAtLimit: false
                )
                guard waveform.dim(1) <= maximumReferenceAudioSamples - totalReferenceAudioSamples else {
                    throw MiniMaxH3GeneratorError.invalidOptions(
                        "ordered reference audio exceeds 15 seconds in total"
                    )
                }
                totalReferenceAudioSamples += waveform.dim(1)
                prepared.append(.init(
                    kind: .audio,
                    visual: nil,
                    visionBlocks: [],
                    blockTimestamps: [],
                    waveform: waveform,
                    geometry: .init(
                        kind: .audio,
                        audioLatentFrames: (waveform.dim(1) + 799) / 800
                    )
                ))
            }
        }
        return prepared
    }

    private func decodeReferenceAudio(
        _ url: URL,
        maximumSamples: Int,
        truncatesAtLimit: Bool
    ) throws -> MLXArray {
        guard maximumSamples >= MiniMaxH3AudioVAE.samplingRate * 2 else {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio requires at least 2 seconds at 32 kHz"
            )
        }
        let buffer: MediaAudioBuffer
        do {
            buffer = try MediaAudioIO.decode(
                url,
                targetSampleRate: MiniMaxH3AudioVAE.samplingRate,
                channels: 2
            )
        } catch {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(url, error.localizedDescription)
        }
        let availableSamples = buffer.samples.count / 2
        if !truncatesAtLimit, availableSamples > maximumSamples {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio exceeds the 15 second total limit"
            )
        }
        let frameCount = min(maximumSamples, availableSamples)
        guard frameCount >= MiniMaxH3AudioVAE.samplingRate * 2 else {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio requires at least 2 seconds at 32 kHz"
            )
        }
        return MLXArray(Array(buffer.samples.prefix(frameCount * 2)))
            .reshaped(1, frameCount, 2)
    }

    private func referenceConditionerPresentation(
        tokenizer: QwenTokenizer,
        prompt: String,
        references: [PreparedReference],
        continuationFrame: MLXArray? = nil
    ) throws -> ConditionerPresentation {
        guard let imageTokenID = tokenizer.imageTokenId,
              let videoTokenID = tokenizer.videoTokenId,
              let visionStartTokenID = tokenizer.visionStartTokenId,
              let visionEndTokenID = tokenizer.visionEndTokenId else {
            throw MiniMaxH3GeneratorError.invalidOptions("Qwen3-VL tokenizer is missing Ref2VA vision tokens")
        }
        var tokenIDs: [Int] = []
        var tokenTags: [Int32] = []
        var images: [QwenVLEncoder.ConditioningImage] = []
        var imageCount = 0
        var videoCount = 0
        var audioCount = 0
        func appendText(_ value: String) {
            let ids = tokenizer.encodeText(value)
            tokenIDs.append(contentsOf: ids)
            tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.text.rawValue, count: ids.count))
        }
        func appendVision(_ pixels: MLXArray, padToken: Int) {
            let patchHeight = pixels.dim(2) / 16
            let patchWidth = pixels.dim(3) / 16
            let temporal = max(1, (pixels.dim(0) + 1) / 2)
            let count = temporal * (patchHeight / 2) * (patchWidth / 2)
            tokenIDs.append(visionStartTokenID)
            tokenTags.append(MiniMaxH3Modality.video.rawValue)
            let range = tokenIDs.count..<(tokenIDs.count + count)
            tokenIDs.append(contentsOf: repeatElement(padToken, count: count))
            tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.video.rawValue, count: count))
            tokenIDs.append(visionEndTokenID)
            tokenTags.append(MiniMaxH3Modality.video.rawValue)
            images.append(.init(
                pixelValues: pixels,
                tokenRange: range,
                temporalPatchCount: temporal,
                heightPatchCount: patchHeight,
                widthPatchCount: patchWidth
            ))
        }
        if let continuationFrame {
            MLX.eval(continuationFrame)
            let frame = continuationFrame[0, 0]
            let image = try MediaImageIO.imageFromRGBHWC(
                frame.asArray(UInt8.self),
                width: continuationFrame.dim(3),
                height: continuationFrame.dim(2)
            )
            let pixels = try QwenVLImageLoader.pixelValues(
                image: image,
                patchSize: 16,
                spatialMergeSize: 2
            )
            imageCount += 1
            appendText("<Picture \(imageCount)>: ")
            appendVision(pixels, padToken: imageTokenID)
        }
        for reference in references {
            if reference.waveform != nil {
                audioCount += 1
                appendText("<Audio \(audioCount)>: ")
            }
            switch reference.kind {
            case .image:
                imageCount += 1
                appendText("<Picture \(imageCount)>: ")
                appendVision(reference.visionBlocks[0], padToken: imageTokenID)
            case .video:
                videoCount += 1
                appendText("<Video \(videoCount)>: ")
                for (index, block) in reference.visionBlocks.enumerated() {
                    let rounded = (reference.blockTimestamps[index] * 10).rounded(.toNearestOrEven) / 10
                    appendText(String(format: "<%.1f seconds>", rounded))
                    appendVision(block, padToken: videoTokenID)
                }
            case .audio:
                break
            }
        }
        appendText(prompt)
        return ConditionerPresentation(tokenIDs: tokenIDs, tokenTags: tokenTags, images: images)
    }

    private static func rgbTensor(_ image: MediaImage) -> MLXArray {
        MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
            .reshaped(3, image.height, image.width)
            .transposed(1, 2, 0)
    }

    private func resolveVideoCanvas(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, width <= 4 * height, height <= 4 * width else {
            throw MiniMaxH3GeneratorError.invalidOptions("reference video aspect ratio must be between 1:4 and 4:1")
        }
        let ratio = Double(width) / Double(height)
        var resolvedWidth = ratio >= 1 ? 768 * ratio : 768
        var resolvedHeight = ratio >= 1 ? 768.0 : 768 / ratio
        let maximumArea = Double(768 * 1_344)
        if resolvedWidth * resolvedHeight > maximumArea {
            let scale = sqrt(maximumArea / (resolvedWidth * resolvedHeight))
            resolvedWidth *= scale
            resolvedHeight *= scale
        }
        return (
            max(32, Int((resolvedWidth / 32).rounded()) * 32),
            max(32, Int((resolvedHeight / 32).rounded()) * 32)
        )
    }

    private static func resampledFrameIndices(
        sourceCount: Int,
        sourceFPS: Double,
        maximumCount: Int
    ) -> [Int] {
        guard sourceCount > 0, sourceFPS > 0 else { return [] }
        if sourceFPS == Double(MiniMaxH3Geometry.framesPerSecond) {
            return Array(0..<min(sourceCount, maximumCount))
        }
        let scale = Double(MiniMaxH3Geometry.framesPerSecond) / sourceFPS
        let slots = (0..<sourceCount).map { Int((Double($0) * scale).rounded()) }
        let end = Int((Double(sourceCount) * scale).rounded())
        var result: [Int] = []
        for index in 0..<sourceCount {
            let next = index + 1 < sourceCount ? slots[index + 1] : end
            if next > slots[index] {
                result.append(contentsOf: repeatElement(index, count: next - slots[index]))
            }
            if result.count >= maximumCount { return Array(result.prefix(maximumCount)) }
        }
        return result
    }

    private static func trimReferenceFrameCount(_ count: Int) -> Int {
        max(1, (count - 5) / 17) * 17 + 5
    }

    private func encodeKeyframes(
        _ urls: [URL],
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> MLXArray? {
        guard !urls.isEmpty else { return nil }
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 0, totalSteps: urls.count))
        return try withMiniMaxH3AutoreleasePool {
            let vae = try loadVideoVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, url) in urls.enumerated() {
                let image: MediaImage
                do {
                    image = try MediaImageIO.resized(
                        try MediaImageIO.decode(url),
                        width: options.internalWidth,
                        height: options.internalHeight
                    )
                } catch {
                    throw MiniMaxH3GeneratorError.imageDecodeFailed(url)
                }
                let chw = MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
                    .reshaped(1, 3, options.internalHeight, options.internalWidth)
                let rgb = chw.transposed(0, 2, 3, 1)
                rows.append(MiniMaxH3Geometry.patchifyVideo(vae.encodeKeyframe(rgb)).asType(.float32))
                progressHandler?(.init(stage: .encodingKeyframes, stepIndex: index + 1, totalSteps: urls.count))
            }
            let result = MLX.concatenated(rows, axis: 1)
            MLX.eval(result)
            return result
        }
    }

    private func encodeContinuation(
        _ continuation: MiniMaxH3ContinuationInput,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> ContinuationConditions {
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 0, totalSteps: 2))
        let videoVAE = try loadVideoVAE(resources: resources)
        let rgb = continuation.frames.asType(.float32) / 255
        var videoRows: [MLXArray] = []
        var videoAnchors: [MiniMaxH3KeyframeAnchor] = []
        if continuation.frameCount > 1 {
            let history = rgb[
                0...,
                0..<(continuation.frameCount - 1),
                0...,
                0...,
                0...
            ]
            let historyLatent = videoVAE.encodeReferenceVideo(history)
            videoRows.append(MiniMaxH3Geometry.patchifyVideo(historyLatent).asType(.float32))
            videoAnchors.append(.history(latentFrameCount: historyLatent.dim(2)))
        }
        let boundary = rgb[
            0...,
            (continuation.frameCount - 1)..<continuation.frameCount,
            0...,
            0...,
            0...
        ]
        let boundaryLatent = videoVAE.encodeKeyframe(boundary.squeezed(axis: 1))
        videoRows.append(MiniMaxH3Geometry.patchifyVideo(boundaryLatent).asType(.float32))
        videoAnchors.append(.first)
        let packedVideo = MLX.concatenated(videoRows, axis: 1)
        MLX.eval(packedVideo)
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 1, totalSteps: 2))

        let audioVAE = try loadAudioVAE(resources: resources)
        let audioLatent = audioVAE.encode(continuation.audio.asType(.float32))
        let boundaryLatentCount = min(
            audioLatent.dim(3),
            max(
                1,
                Int((Double(MiniMaxH3Geometry.audioLatentsPerSecond)
                    / Double(MiniMaxH3Geometry.framesPerSecond)).rounded())
            )
        )
        let historyLatentCount = audioLatent.dim(3) - boundaryLatentCount
        var audioRows: [MLXArray] = []
        var audioAnchors: [MiniMaxH3AudioConditionAnchor] = []
        if historyLatentCount > 0 {
            let history = audioLatent[0..., 0..., 0..., 0..<historyLatentCount]
            audioRows.append(MiniMaxH3Geometry.packAudio(history).expandedDimensions(axis: 0))
            audioAnchors.append(.history(latentFrameCount: historyLatentCount))
        }
        let first = audioLatent[
            0...,
            0...,
            0...,
            historyLatentCount..<audioLatent.dim(3)
        ]
        audioRows.append(MiniMaxH3Geometry.packAudio(first).expandedDimensions(axis: 0))
        audioAnchors.append(.first(latentFrameCount: boundaryLatentCount))
        let packedAudio = MLX.concatenated(audioRows, axis: 1)
        MLX.eval(packedAudio)
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 2, totalSteps: 2))
        return ContinuationConditions(
            videoRows: packedVideo,
            audioRows: packedAudio,
            videoAnchors: videoAnchors,
            audioAnchors: audioAnchors
        )
    }

    private static func concatenateRows(_ arrays: [MLXArray?]) -> MLXArray? {
        concatenateRows(arrays.compactMap { $0 })
    }

    private static func concatenateRows(_ arrays: [MLXArray]) -> MLXArray? {
        let values = arrays
        guard !values.isEmpty else { return nil }
        return values.count == 1 ? values[0] : MLX.concatenated(values, axis: 1)
    }

    private static func boundaryFrame(_ continuation: MiniMaxH3ContinuationInput) -> MLXArray {
        continuation.frames[
            0...,
            (continuation.frameCount - 1)..<continuation.frameCount,
            0...,
            0...,
            0...
        ]
    }

    private func conditionerPresentation(
        tokenizer: QwenTokenizer,
        options: MiniMaxH3GenerationOptions,
        continuationFrame: MLXArray? = nil
    ) throws -> ConditionerPresentation {
        let keyframeURLs = Self.frameConditions(options: options).map(\.url)
        var tokenIDs: [Int] = []
        var tokenTags: [Int32] = []
        var images: [QwenVLEncoder.ConditioningImage] = []

        if continuationFrame != nil || !keyframeURLs.isEmpty {
            guard let imageTokenID = tokenizer.imageTokenId,
                  let visionStartTokenID = tokenizer.visionStartTokenId,
                  let visionEndTokenID = tokenizer.visionEndTokenId else {
                throw MiniMaxH3GeneratorError.invalidOptions("Qwen3-VL tokenizer is missing vision tokens")
            }
            var preparedImages: [MediaImage] = []
            if let continuationFrame {
                MLX.eval(continuationFrame)
                let frame = continuationFrame[0, 0]
                preparedImages.append(try MediaImageIO.imageFromRGBHWC(
                    frame.asArray(UInt8.self),
                    width: continuationFrame.dim(3),
                    height: continuationFrame.dim(2)
                ))
            }
            for url in keyframeURLs {
                do {
                    preparedImages.append(try MediaImageIO.resized(
                        try MediaImageIO.decode(url),
                        width: options.internalWidth,
                        height: options.internalHeight
                    ))
                } catch {
                    throw MiniMaxH3GeneratorError.imageDecodeFailed(url)
                }
            }
            for (index, prepared) in preparedImages.enumerated() {
                let pixelValues = try QwenVLImageLoader.pixelValues(
                    image: prepared,
                    patchSize: 16,
                    spatialMergeSize: 2
                )
                let patchHeight = pixelValues.dim(2) / 16
                let patchWidth = pixelValues.dim(3) / 16
                let imageTokenCount = QwenVLEncoder.imageTokenCount(
                    imageHeight: pixelValues.dim(2),
                    imageWidth: pixelValues.dim(3),
                    patchSize: 16,
                    spatialMergeSize: 2
                )
                let labelIDs = tokenizer.encodeText("<Picture \(index + 1)>: ")
                tokenIDs.append(contentsOf: labelIDs)
                tokenTags.append(contentsOf: repeatElement(
                    MiniMaxH3Modality.text.rawValue,
                    count: labelIDs.count
                ))
                tokenIDs.append(visionStartTokenID)
                tokenTags.append(MiniMaxH3Modality.video.rawValue)
                let tokenRange = tokenIDs.count..<(tokenIDs.count + imageTokenCount)
                tokenIDs.append(contentsOf: repeatElement(imageTokenID, count: imageTokenCount))
                tokenTags.append(contentsOf: repeatElement(
                    MiniMaxH3Modality.video.rawValue,
                    count: imageTokenCount
                ))
                tokenIDs.append(visionEndTokenID)
                tokenTags.append(MiniMaxH3Modality.video.rawValue)
                images.append(.init(
                    pixelValues: pixelValues,
                    tokenRange: tokenRange,
                    heightPatchCount: patchHeight,
                    widthPatchCount: patchWidth
                ))
            }
        }
        let promptIDs = tokenizer.encodeText(options.prompt)
        tokenIDs.append(contentsOf: promptIDs)
        tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.text.rawValue, count: promptIDs.count))
        return ConditionerPresentation(tokenIDs: tokenIDs, tokenTags: tokenTags, images: images)
    }

    private static func frameConditions(options: MiniMaxH3GenerationOptions) -> [FrameCondition] {
        var conditions: [(index: Int, condition: FrameCondition)] = []
        if let firstFrameURL = options.firstFrameURL {
            conditions.append((0, .init(url: firstFrameURL, anchor: .first)))
        }
        conditions.append(contentsOf: options.frameInputs.map { input in
            let anchor: MiniMaxH3KeyframeAnchor
            if input.frameIndex == 0 {
                anchor = .first
            } else if input.frameIndex == options.numFrames - 1 {
                anchor = .last
            } else {
                anchor = .frame(input.frameIndex)
            }
            return (input.frameIndex, .init(url: input.url, anchor: anchor))
        })
        if let lastFrameURL = options.lastFrameURL {
            conditions.append((
                options.numFrames - 1,
                .init(url: lastFrameURL, anchor: .last)
            ))
        }
        return conditions.sorted { $0.index < $1.index }.map(\.condition)
    }
}
