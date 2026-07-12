import Foundation

public struct VideoDepthAnythingPreprocessingPlan: Equatable, Sendable {
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let requestedInputSize: Int
    public let effectiveInputSize: Int
    public let networkWidth: Int
    public let networkHeight: Int

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        requestedInputSize: Int = VideoDepthAnythingLimits.defaultInputSize
    ) throws {
        try VideoDepthAnythingLimits.validateDecodedSequence(
            width: sourceWidth,
            height: sourceHeight,
            frameCount: 1
        )
        _ = try VideoDepthAnythingLimits.validateRequest(
            inputSize: requestedInputSize,
            maximumFrameCount: 1
        )
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.requestedInputSize = requestedInputSize

        let ratio = Double(max(sourceWidth, sourceHeight)) / Double(min(sourceWidth, sourceHeight))
        let adjusted: Int
        if ratio > 1.78 {
            let reducedValue = Double(requestedInputSize) * 1.777 / ratio
            guard reducedValue.isFinite,
                  reducedValue >= 0,
                  reducedValue <= Double(Int.max) else {
                throw VideoDepthAnythingLimitError.invalidNetworkDimensions(
                    width: sourceWidth,
                    height: sourceHeight
                )
            }
            let reduced = Int(reducedValue)
            adjusted = max(14, Self.nearestMultiple(reduced, of: 14))
        } else {
            adjusted = requestedInputSize
        }
        effectiveInputSize = adjusted

        let scale = max(
            Double(adjusted) / Double(sourceWidth),
            Double(adjusted) / Double(sourceHeight)
        )
        let plannedWidth = try Self.lowerBoundMultiple(
            Double(sourceWidth) * scale,
            minimum: adjusted,
            multiple: 14
        )
        let plannedHeight = try Self.lowerBoundMultiple(
            Double(sourceHeight) * scale,
            minimum: adjusted,
            multiple: 14
        )
        try VideoDepthAnythingLimits.validateNetworkDimensions(
            width: plannedWidth,
            height: plannedHeight
        )
        networkWidth = plannedWidth
        networkHeight = plannedHeight
    }

    private static func nearestMultiple(_ value: Int, of multiple: Int) -> Int {
        Int((Double(value) / Double(multiple)).rounded(.toNearestOrEven)) * multiple
    }

    private static func lowerBoundMultiple(
        _ value: Double,
        minimum: Int,
        multiple: Int
    ) throws -> Int {
        guard value.isFinite,
              value > 0,
              value <= Double(Int.max - multiple) else {
            throw VideoDepthAnythingLimitError.invalidNetworkDimensions(
                width: minimum,
                height: minimum
            )
        }
        var result = Int((value / Double(multiple)).rounded(.toNearestOrEven)) * multiple
        if result < minimum {
            result = Int(ceil(value / Double(multiple))) * multiple
        }
        return result
    }
}

public struct VideoDepthAnythingWindowPlan: Equatable, Sendable {
    public let windowIndex: Int
    public let outputStartFrame: Int
    /// Source-frame indices after final-frame padding and keyframe substitution.
    public let sourceFrameIndices: [Int]

    public init(windowIndex: Int, outputStartFrame: Int, sourceFrameIndices: [Int]) {
        self.windowIndex = windowIndex
        self.outputStartFrame = outputStartFrame
        self.sourceFrameIndices = sourceFrameIndices
    }
}

public struct VideoDepthAnythingAffineAlignment: Equatable, Sendable {
    public let scale: Float
    public let shift: Float

    public init(scale: Float, shift: Float) {
        self.scale = scale
        self.shift = shift
    }
}

public struct VideoDepthAnythingAlignedWindows: Equatable, Sendable {
    public let frames: [[Float]]
    /// One entry per raw window; the first is the identity transform.
    public let alignments: [VideoDepthAnythingAffineAlignment]

    public init(frames: [[Float]], alignments: [VideoDepthAnythingAffineAlignment]) {
        self.frames = frames
        self.alignments = alignments
    }
}

public struct VideoDepthAnythingAlignedChunk: Equatable, Sendable {
    /// Frames that can no longer be changed by a later temporal window.
    public let finalizedFrames: [[Float]]
    public let alignment: VideoDepthAnythingAffineAlignment

    public init(
        finalizedFrames: [[Float]],
        alignment: VideoDepthAnythingAffineAlignment
    ) {
        self.finalizedFrames = finalizedFrames
        self.alignment = alignment
    }
}

public enum VideoDepthAnythingWindowingError: Error, Equatable, LocalizedError, Sendable {
    case invalidOriginalFrameCount(Int)
    case invalidWindowLength(window: Int, expected: Int, actual: Int)
    case inconsistentDepthElementCount(expected: Int, actual: Int)
    case affineFrameCountMismatch(prediction: Int, target: Int)
    case affineDepthElementCountMismatch(frame: Int, prediction: Int, target: Int)
    case streamingAlignmentAlreadyFinished

    public var errorDescription: String? {
        switch self {
        case .invalidOriginalFrameCount(let count):
            "Video Depth Anything requires at least one source frame; found \(count)."
        case .invalidWindowLength(let window, let expected, let actual):
            "Depth window \(window) has \(actual) frames; expected \(expected)."
        case .inconsistentDepthElementCount(let expected, let actual):
            "A depth frame has \(actual) values; expected \(expected)."
        case let .affineFrameCountMismatch(prediction, target):
            "Affine alignment received \(prediction) prediction frames and \(target) target frames."
        case let .affineDepthElementCountMismatch(frame, prediction, target):
            "Affine alignment frame \(frame) has \(prediction) prediction values and \(target) target values."
        case .streamingAlignmentAlreadyFinished:
            "The Video Depth Anything streaming alignment has already finished."
        }
    }
}

public enum VideoDepthAnythingWindowing {
    public static let windowLength = 32
    public static let overlap = 10
    public static let frameStep = 22
    public static let interpolationLength = 8
    public static let alignmentLength = 2
    public static let keyframes = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]

    /// Reproduces the reference input schedule, including final-frame padding
    /// and replacement of each later window's overlap with prior keyframes.
    public static func plans(originalFrameCount: Int) throws -> [VideoDepthAnythingWindowPlan] {
        guard originalFrameCount > 0 else {
            throw VideoDepthAnythingWindowingError.invalidOriginalFrameCount(originalFrameCount)
        }
        let appendCount = (frameStep - (originalFrameCount % frameStep)) % frameStep
            + (windowLength - frameStep)
        let paddedCount = originalFrameCount + appendCount
        var plans: [VideoDepthAnythingWindowPlan] = []
        var previousIndices: [Int]?
        var start = 0
        while start < originalFrameCount {
            var indices = (0..<windowLength).map { offset in
                min(originalFrameCount - 1, min(paddedCount - 1, start + offset))
            }
            if let previousIndices {
                for overlapIndex in 0..<overlap {
                    indices[overlapIndex] = previousIndices[keyframes[overlapIndex]]
                }
            }
            plans.append(
                VideoDepthAnythingWindowPlan(
                    windowIndex: plans.count,
                    outputStartFrame: start,
                    sourceFrameIndices: indices
                )
            )
            previousIndices = indices
            start += frameStep
        }
        return plans
    }

    /// Reproduces the reference affine alignment and eight-frame crossfade.
    /// `windows` contains the source-resolution depth from each 32-frame pass.
    public static func align(
        windows: [[[Float]]],
        originalFrameCount: Int,
        semantics: DepthSemantics
    ) throws -> VideoDepthAnythingAlignedWindows {
        guard originalFrameCount > 0 else {
            throw VideoDepthAnythingWindowingError.invalidOriginalFrameCount(originalFrameCount)
        }
        guard !windows.isEmpty else {
            return VideoDepthAnythingAlignedWindows(frames: [], alignments: [])
        }
        var aligner = try StreamingAligner(
            originalFrameCount: originalFrameCount,
            semantics: semantics
        )
        var aligned: [[Float]] = []
        var transforms: [VideoDepthAnythingAffineAlignment] = []
        for (windowIndex, window) in windows.enumerated() {
            let chunk = try aligner.append(
                window: window,
                isFinal: windowIndex == windows.count - 1
            )
            aligned.append(contentsOf: chunk.finalizedFrames)
            transforms.append(chunk.alignment)
        }
        return VideoDepthAnythingAlignedWindows(frames: aligned, alignments: transforms)
    }

    /// Stateful upstream-equivalent alignment. It retains only the eight
    /// frames that a following window may crossfade, so sequence memory is
    /// bounded independently of source duration.
    public struct StreamingAligner: Sendable {
        public let originalFrameCount: Int
        public let semantics: DepthSemantics

        private var bufferedFrames: [[Float]] = []
        private var referenceAlignment: [[Float]] = []
        private var elementCount: Int?
        private var windowIndex = 0
        private var paddedOutputPosition = 0
        private var isFinished = false

        public init(originalFrameCount: Int, semantics: DepthSemantics) throws {
            guard originalFrameCount > 0 else {
                throw VideoDepthAnythingWindowingError.invalidOriginalFrameCount(originalFrameCount)
            }
            self.originalFrameCount = originalFrameCount
            self.semantics = semantics
        }

        public var retainedFrameCount: Int { bufferedFrames.count }

        public mutating func append(
            window: [[Float]],
            isFinal: Bool
        ) throws -> VideoDepthAnythingAlignedChunk {
            guard !isFinished else {
                throw VideoDepthAnythingWindowingError.streamingAlignmentAlreadyFinished
            }
            guard window.count == windowLength else {
                throw VideoDepthAnythingWindowingError.invalidWindowLength(
                    window: windowIndex,
                    expected: windowLength,
                    actual: window.count
                )
            }
            let expectedElementCount = elementCount ?? window.first?.count ?? 0
            for frame in window where frame.count != expectedElementCount {
                throw VideoDepthAnythingWindowingError.inconsistentDepthElementCount(
                    expected: expectedElementCount,
                    actual: frame.count
                )
            }
            elementCount = expectedElementCount

            let transform: VideoDepthAnythingAffineAlignment
            if windowIndex == 0 {
                transform = VideoDepthAnythingAffineAlignment(scale: 1, shift: 0)
                bufferedFrames = window
                referenceAlignment = [window[keyframes[0]], window[keyframes[1]]]
            } else {
                switch semantics {
                case .metricMeters:
                    transform = VideoDepthAnythingAffineAlignment(scale: 1, shift: 0)
                case .affineRelative:
                    transform = try solveAffine(
                        prediction: [window[0], window[1]],
                        target: referenceAlignment
                    )
                }

                let priorStart = bufferedFrames.count - interpolationLength
                precondition(priorStart >= 0)
                for index in 0..<interpolationLength {
                    let transformed = transformedAndClamped(
                        window[index + alignmentLength],
                        transform: transform
                    )
                    let postWeight = Float(index) / Float(interpolationLength - 1)
                    bufferedFrames[priorStart + index] = blend(
                        bufferedFrames[priorStart + index],
                        transformed,
                        postWeight: postWeight
                    )
                }
                for index in overlap..<windowLength {
                    bufferedFrames.append(
                        transformedAndClamped(window[index], transform: transform)
                    )
                }
                referenceAlignment = [
                    referenceAlignment[0],
                    transformedAndClamped(window[keyframes[1]], transform: transform),
                ]
            }

            let flushCount = isFinal
                ? bufferedFrames.count
                : max(0, bufferedFrames.count - interpolationLength)
            let originalFramesRemaining = max(0, originalFrameCount - paddedOutputPosition)
            let outputCount = min(flushCount, originalFramesRemaining)
            let finalized = Array(bufferedFrames.prefix(outputCount))
            if flushCount > 0 {
                bufferedFrames.removeFirst(flushCount)
            }
            paddedOutputPosition += flushCount
            windowIndex += 1
            if isFinal {
                isFinished = true
                bufferedFrames.removeAll(keepingCapacity: false)
            }
            return VideoDepthAnythingAlignedChunk(
                finalizedFrames: finalized,
                alignment: transform
            )
        }
    }

    public static func solveAffine(
        prediction: [[Float]],
        target: [[Float]]
    ) throws -> VideoDepthAnythingAffineAlignment {
        guard prediction.count == target.count else {
            throw VideoDepthAnythingWindowingError.affineFrameCountMismatch(
                prediction: prediction.count,
                target: target.count
            )
        }
        var a00: Float = 0
        var a01: Float = 0
        var a11: Float = 0
        var b0: Float = 0
        var b1: Float = 0
        for (frameIndex, frames) in zip(prediction, target).enumerated() {
            let (predictionFrame, targetFrame) = frames
            guard predictionFrame.count == targetFrame.count else {
                throw VideoDepthAnythingWindowingError.affineDepthElementCountMismatch(
                    frame: frameIndex,
                    prediction: predictionFrame.count,
                    target: targetFrame.count
                )
            }
            for (value, reference) in zip(predictionFrame, targetFrame) {
                a00 += value * value
                a01 += value
                a11 += 1
                b0 += value * reference
                b1 += reference
            }
        }
        let determinant = a00 * a11 - a01 * a01
        guard determinant != 0, determinant.isFinite else {
            return VideoDepthAnythingAffineAlignment(scale: 1, shift: 0)
        }
        let scale = (a11 * b0 - a01 * b1) / determinant
        let shift = (-a01 * b0 + a00 * b1) / determinant
        guard scale.isFinite, shift.isFinite else {
            return VideoDepthAnythingAffineAlignment(scale: 1, shift: 0)
        }
        return VideoDepthAnythingAffineAlignment(scale: scale, shift: shift)
    }

    private static func transformedAndClamped(
        _ frame: [Float],
        transform: VideoDepthAnythingAffineAlignment
    ) -> [Float] {
        frame.map { max(0, $0 * transform.scale + transform.shift) }
    }

    private static func blend(_ before: [Float], _ after: [Float], postWeight: Float) -> [Float] {
        zip(before, after).map { prior, next in
            prior * (1 - postWeight) + next * postWeight
        }
    }
}
