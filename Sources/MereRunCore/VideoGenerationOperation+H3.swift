import Foundation
import MLX

extension NativeVideoGeneration {
    func runH3(_ h3Options: MiniMaxH3GenerationOptions, request: VideoGenerationPreparedRequest) async throws -> VideoGenerationOutcome {
        let options = request.plan.options
        let plan = request.plan
        let resolvedRootURL = request.modelRoot
        let outputURL = options.outputURL
        let h3Resources = MiniMaxH3Resources(rootURL: resolvedRootURL)
        let h3Configuration = try h3Resources.loadConfiguration()
        let resolvedH3Adapter = h3Options.adapterURL
        let h3AdapterStrength = options.h3AdapterStrength
        let h3Width = plan.width
        let h3Height = plan.height
        let width = options.resolvedOutputWidth
        let height = options.resolvedOutputHeight
        let h3RenderWidth = options.h3RenderWidth
        let h3RenderHeight = options.h3RenderHeight
        let h3Frames = plan.numFrames
        let requestedH3Frames = plan.requestedFrames
        let slidingWindowOptions = plan.h3SlidingWindowOptions
        if reportsDiagnostics {
            diagnostic("Engine: native MiniMax-H3 \(h3Configuration.task.uppercased())\n")
            diagnostic("Model root: \(resolvedRootURL.path)\n")
            if let resolvedH3Adapter {
                diagnostic("Adapter: \(resolvedH3Adapter.path) (strength \(h3AdapterStrength))\n")
            }
            if h3Width != width || h3Height != height {
                diagnostic("Adjusted MiniMax-H3 size to \(h3Width)x\(h3Height) (must be divisible by 32)\n")
            }
            if let h3RenderWidth, let h3RenderHeight {
                diagnostic(
                    "MiniMax-H3 internal render: \(h3RenderWidth)x\(h3RenderHeight); "
                        + "high-quality upscale to \(h3Width)x\(h3Height)\n"
                )
            }
            if h3Frames != requestedH3Frames {
                diagnostic("Adjusted MiniMax-H3 frame count to \(h3Frames) (must have form 17*n+5)\n")
            }
            if let slidingWindowOptions {
                let plan = MiniMaxH3SlidingWindowPlan(options: slidingWindowOptions)
                diagnostic(
                    "Sliding windows: \(plan.windows.count) x "
                        + "\(slidingWindowOptions.windowFrameCount) frames, "
                        + "overlap \(slidingWindowOptions.overlapFrameCount)\n"
                )
            }
        }

        let generator = MiniMaxH3Generator(retainsRuntime: slidingWindowOptions != nil)
        let progressHandler: @Sendable (MiniMaxH3GenerationProgress) -> Void = { progress in
            eventHandler?(.progress(stage: progress.stage.rawValue, step: progress.stepIndex, totalSteps: progress.totalSteps))
        }
        let wiredMemoryTicket = MiniMaxH3WiredMemoryPolicy().ticket(
            size: miniMaxH3WiredMemoryTargetBytes()
        )
        let result = try await wiredMemoryTicket.withWiredLimit {
            try Stream.withNewDefaultStream {
                if let slidingWindowOptions {
                    return try generator.generateSlidingWindows(
                        options: h3Options,
                        slidingWindowOptions: slidingWindowOptions,
                        resources: h3Resources,
                        windowHandler: { index, count in
                            eventHandler?(.progress(stage: "window", step: index, totalSteps: count))
                        },
                        progressHandler: progressHandler
                    )
                }
                return try generator.generate(
                    options: h3Options,
                    resources: h3Resources,
                    progressHandler: progressHandler
                )
            }
        }
        try Task.checkCancellation()
        eventHandler?(.progressFinished)
        try LTXVideoMP4Writer.writeMP4(
            frames: result.frames,
            fps: MiniMaxH3Geometry.framesPerSecond,
            to: outputURL,
            audioWaveform: result.audio,
            audioSampleRate: MiniMaxH3AudioVAE.samplingRate
        )
        return VideoGenerationOutcome(primaryURL: outputURL)
    }
}

private struct MiniMaxH3WiredMemoryPolicy: WiredMemoryPolicy, Hashable {
    func limit(baseline: Int, activeSizes: [Int]) -> Int {
        max(baseline, activeSizes.max() ?? baseline)
    }
}

private func miniMaxH3WiredMemoryTargetBytes() -> Int {
    let gibibyte = 1_073_741_824
    let desired = ProcessInfo.processInfo.physicalMemory >= UInt64(96 * gibibyte)
        ? 64 * gibibyte
        : 50 * gibibyte
#if os(macOS)
    let safetyMargin = 1_048_576
    guard let recommended = GPU.maxRecommendedWorkingSetBytes() else {
        return min(desired, Int(ProcessInfo.processInfo.physicalMemory / 2))
    }
    return min(desired, max(0, recommended - safetyMargin))
#else
    return min(desired, Int(ProcessInfo.processInfo.physicalMemory / 2))
#endif
}
