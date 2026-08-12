import Foundation
import MLX

public struct MiniMaxH3SlidingWindowOptions: Sendable, Hashable {
    public let totalFrameCount: Int
    public let windowFrameCount: Int
    public let overlapFrameCount: Int

    public init(
        totalFrameCount: Int,
        windowFrameCount: Int,
        overlapFrameCount: Int
    ) throws {
        guard totalFrameCount >= 22, totalFrameCount % 17 == 5 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "sliding output frame count must be at least 22 and have the form 17*n+5"
            )
        }
        guard windowFrameCount >= 22, windowFrameCount % 17 == 5 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-frames must be at least 22 and have the form 17*n+5"
            )
        }
        guard overlapFrameCount >= 1, overlapFrameCount % 17 == 1 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-overlap must have the form 17*n+1"
            )
        }
        guard overlapFrameCount < windowFrameCount else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "--h3-window-overlap must be smaller than --h3-window-frames"
            )
        }
        let continuationTargetCount = windowFrameCount - overlapFrameCount + 1
        guard continuationTargetCount >= 22 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "sliding windows must leave at least 22 target frames after overlap"
            )
        }
        self.totalFrameCount = totalFrameCount
        self.windowFrameCount = windowFrameCount
        self.overlapFrameCount = overlapFrameCount
    }
}

public struct MiniMaxH3SlidingWindowPlan: Sendable, Hashable {
    public struct Window: Sendable, Hashable {
        public let index: Int
        public let boundaryFrameIndex: Int?
        public let generatedFrameCount: Int
        public let appendedFrameRange: Range<Int>
        public let outputFrameRange: Range<Int>

        public var appendedFrameCount: Int { appendedFrameRange.count }

        public func localFrameIndex(for globalFrameIndex: Int) -> Int? {
            let origin = boundaryFrameIndex ?? outputFrameRange.lowerBound
            let local = globalFrameIndex - origin
            return (0..<generatedFrameCount).contains(local) ? local : nil
        }
    }

    public let options: MiniMaxH3SlidingWindowOptions
    public let windows: [Window]

    public init(options: MiniMaxH3SlidingWindowOptions) {
        self.options = options
        var windows: [Window] = []
        let firstCount = min(options.totalFrameCount, options.windowFrameCount)
        windows.append(.init(
            index: 0,
            boundaryFrameIndex: nil,
            generatedFrameCount: firstCount,
            appendedFrameRange: 0..<firstCount,
            outputFrameRange: 0..<firstCount
        ))
        let stride = options.windowFrameCount - options.overlapFrameCount
        let targetCount = stride + 1
        var emitted = firstCount
        while emitted < options.totalFrameCount {
            let appendCount = min(stride, options.totalFrameCount - emitted)
            windows.append(.init(
                index: windows.count,
                boundaryFrameIndex: emitted - 1,
                generatedFrameCount: targetCount,
                appendedFrameRange: 1..<(appendCount + 1),
                outputFrameRange: emitted..<(emitted + appendCount)
            ))
            emitted += appendCount
        }
        self.windows = windows
    }
}

public struct MiniMaxH3ContinuationInput: @unchecked Sendable {
    public let frames: MLXArray
    public let audio: MLXArray
    public let frameCount: Int

    public init(frames: MLXArray, audio: MLXArray, frameCount: Int) throws {
        guard frameCount >= 1, frameCount % 17 == 1 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "continuation overlap must have the form 17*n+1"
            )
        }
        guard frames.ndim == 5,
              frames.dim(0) == 1,
              frames.dim(1) == frameCount,
              frames.dim(4) == 3 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "continuation video must have shape [1, overlap, height, width, 3]"
            )
        }
        guard audio.ndim == 3,
              audio.dim(0) == 1,
              audio.dim(2) == 2 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "continuation audio must have shape [1, samples, 2]"
            )
        }
        self.frames = frames
        self.audio = audio
        self.frameCount = frameCount
    }
}

extension MiniMaxH3Generator {
    public func generateSlidingWindows(
        options: MiniMaxH3GenerationOptions,
        slidingWindowOptions: MiniMaxH3SlidingWindowOptions,
        resources: MiniMaxH3Resources,
        windowHandler: (@Sendable (_ windowIndex: Int, _ windowCount: Int) -> Void)? = nil,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)? = nil
    ) throws -> MiniMaxH3GenerationResult {
        guard options.numFrames == slidingWindowOptions.totalFrameCount else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "sliding total frame count must match the generation request"
            )
        }
        guard !options.usesReducedRenderCanvas else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "reduced internal rendering does not yet support continuation or sliding windows"
            )
        }
        let plan = MiniMaxH3SlidingWindowPlan(options: slidingWindowOptions)
        var assembledFrames: MLXArray?
        var assembledAudio: MLXArray?
        let sampleRate = MiniMaxH3AudioVAE.samplingRate
        let framesPerSecond = MiniMaxH3Geometry.framesPerSecond

        for window in plan.windows {
            windowHandler?(window.index, plan.windows.count)
            let frameInputs = try positionedFrames(
                options: options,
                window: window,
                totalFrameCount: slidingWindowOptions.totalFrameCount
            )
            let windowOptions = try MiniMaxH3GenerationOptions(
                prompt: options.prompt,
                width: options.width,
                height: options.height,
                renderWidth: options.renderWidth,
                renderHeight: options.renderHeight,
                numFrames: window.generatedFrameCount,
                steps: options.steps,
                seed: options.seed &+ UInt64(window.index),
                transformerWeightMode: options.transformerWeightMode,
                accelerationMode: options.accelerationMode,
                adapterURL: options.adapterURL,
                adapterStrength: options.adapterStrength,
                firstFrameURL: window.index == 0 ? options.firstFrameURL : nil,
                frameInputs: frameInputs,
                references: options.references
            )
            let continuation: MiniMaxH3ContinuationInput?
            if window.index == 0 {
                continuation = nil
            } else {
                guard let assembledFrames, let assembledAudio else {
                    preconditionFailure("sliding continuation requires the previous window")
                }
                let emittedFrameCount = window.outputFrameRange.lowerBound
                let overlap = slidingWindowOptions.overlapFrameCount
                let audioStart = Self.sampleOffset(
                    frameIndex: emittedFrameCount - overlap,
                    sampleRate: sampleRate,
                    framesPerSecond: framesPerSecond
                )
                let audioEnd = Self.sampleOffset(
                    frameIndex: emittedFrameCount,
                    sampleRate: sampleRate,
                    framesPerSecond: framesPerSecond
                )
                continuation = try MiniMaxH3ContinuationInput(
                    frames: assembledFrames[
                        0...,
                        (emittedFrameCount - overlap)..<emittedFrameCount,
                        0...,
                        0...,
                        0...
                    ],
                    audio: assembledAudio[0..., audioStart..<audioEnd, 0...],
                    frameCount: overlap
                )
            }
            let result = try generate(
                options: windowOptions,
                resources: resources,
                continuation: continuation,
                progressHandler: progressHandler
            )
            let frameSlice = result.frames[
                0...,
                window.appendedFrameRange,
                0...,
                0...,
                0...
            ]
            let localAudioStart = Self.sampleOffset(
                frameIndex: window.appendedFrameRange.lowerBound,
                sampleRate: sampleRate,
                framesPerSecond: framesPerSecond
            )
            let localAudioEnd = Self.sampleOffset(
                frameIndex: window.appendedFrameRange.upperBound,
                sampleRate: sampleRate,
                framesPerSecond: framesPerSecond
            )
            let fittedWindowAudio = Self.fitAudio(
                result.audio,
                sampleCount: localAudioEnd
            )
            let audioSlice = fittedWindowAudio[0..., localAudioStart..<localAudioEnd, 0...]
            assembledFrames = assembledFrames.map {
                MLX.concatenated([$0, frameSlice], axis: 1)
            } ?? frameSlice
            assembledAudio = assembledAudio.map {
                MLX.concatenated([$0, audioSlice], axis: 1)
            } ?? audioSlice
            if let assembledFrames, let assembledAudio {
                MLX.eval(assembledFrames, assembledAudio)
            }
            Memory.clearCache()
        }
        guard let assembledFrames, let assembledAudio else {
            preconditionFailure("sliding plan must contain at least one window")
        }
        let exactSampleCount = Self.sampleOffset(
            frameIndex: slidingWindowOptions.totalFrameCount,
            sampleRate: sampleRate,
            framesPerSecond: framesPerSecond
        )
        let exactAudio = Self.fitAudio(assembledAudio, sampleCount: exactSampleCount)
        MLX.eval(assembledFrames, exactAudio)
        return MiniMaxH3GenerationResult(
            frames: assembledFrames,
            audio: exactAudio,
            seed: options.seed
        )
    }

    private func positionedFrames(
        options: MiniMaxH3GenerationOptions,
        window: MiniMaxH3SlidingWindowPlan.Window,
        totalFrameCount: Int
    ) throws -> [MiniMaxH3FrameInput] {
        var globalFrames = options.frameInputs
        if let lastFrameURL = options.lastFrameURL {
            globalFrames.append(.init(
                frameIndex: totalFrameCount - 1,
                url: lastFrameURL
            ))
        }
        return globalFrames.compactMap { input in
            guard window.outputFrameRange.contains(input.frameIndex),
                  let localIndex = window.localFrameIndex(for: input.frameIndex) else {
                return nil
            }
            return MiniMaxH3FrameInput(frameIndex: localIndex, url: input.url)
        }
    }

    private static func sampleOffset(
        frameIndex: Int,
        sampleRate: Int,
        framesPerSecond: Int
    ) -> Int {
        Int((Double(frameIndex) * Double(sampleRate) / Double(framesPerSecond)).rounded())
    }

    private static func fitAudio(_ audio: MLXArray, sampleCount: Int) -> MLXArray {
        if audio.dim(1) == sampleCount { return audio }
        if audio.dim(1) > sampleCount {
            return audio[0..., 0..<sampleCount, 0...]
        }
        return MLX.padded(
            audio,
            widths: [[0, 0], [0, sampleCount - audio.dim(1)], [0, 0]]
        )
    }
}
