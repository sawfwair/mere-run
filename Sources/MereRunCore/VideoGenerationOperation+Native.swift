import Foundation
import MediaIO
import MLX
import MereRunContract

struct NativeVideoGeneration: Sendable {
    let eventHandler: VideoGenerationOperation.EventHandler?
    var reportsDiagnostics: Bool { eventHandler != nil }

    func diagnostic(_ message: String) { eventHandler?(.diagnostic(message)) }

    func execute(_ request: VideoGenerationPreparedRequest) async throws -> VideoGenerationOutcome {
        try Task.checkCancellation()
        let plan = request.plan
        let options = plan.options
        let outputURL = options.outputURL
        switch request.input {
        case .h3(let native):
            return try await runH3(native, request: request)
        case .wan(let native):
            reportGeometry(plan, family: "Wan", spatialMultiple: 32, temporalMultiple: 4)
            return try await runNativeWanGenerate(options: native, modelRoot: request.modelRoot, outputURL: outputURL)
        case .audioToVideo(let native):
            return try await Stream.withNewDefaultStream {
                try await runNativeAudioToVideoGenerate(
                    options: native, videoDecoder: plan.videoDecoder, modelRoot: request.modelRoot, outputURL: outputURL
                )
            }
        case .ltx(let native):
            reportGeometry(plan, family: "LTX", spatialMultiple: 64, temporalMultiple: 8)
            if options.variant == .unifiedAV, options.fps != 24 {
                diagnostic("Warning: LTX unified AV is trained for 24 fps; --fps \(options.fps) can make generated motion look time-stretched relative to audio.\n")
            }
            return try await Stream.withNewDefaultStream {
                try await runNativeGenerate(request: native, modelRoot: request.modelRoot, outputURL: outputURL)
            }
        }
    }

    private func reportGeometry(_ plan: VideoGenerationPlan, family: String, spatialMultiple: Int, temporalMultiple: Int) {
        let options = plan.options
        let prefix = family == "Wan" ? "Wan " : ""
        if plan.width != options.resolvedOutputWidth || plan.height != options.resolvedOutputHeight {
            diagnostic("Adjusted \(prefix)size to \(plan.width)x\(plan.height) (must be divisible by \(spatialMultiple))\n")
        }
        if let duration = options.duration {
            let seconds = Double(plan.numFrames) / options.fps
            diagnostic(
                "Resolved \(prefix)duration \(String(format: "%.2f", duration))s to \(plan.numFrames) frames at \(options.fps) fps "
                    + "(~\(String(format: "%.2f", seconds))s; must satisfy \(temporalMultiple)n+1)\n"
            )
        } else if let requested = options.numFrames, plan.numFrames != requested {
            diagnostic("Adjusted \(prefix)frame count to \(plan.numFrames) (must satisfy \(temporalMultiple)n+1)\n")
        }
    }

    func resolveAutoDuration(
        _ range: LTX25AutoDuration?,
        prompt: String,
        fps: Double,
        generator: LTXUnifiedAVGenerator,
        fallback: Int
    ) async throws -> Int {
        guard let range else { return fallback }
        let frameCount = try await generator.predictFrameCount(
            prompt: prompt,
            frameRate: fps,
            range: range,
            conditioning: .audioVideo
        )
        if reportsDiagnostics {
            let seconds = Double(frameCount) / fps
            diagnostic(
                "DurationHead selected \(frameCount) frames at \(fps) fps "
                    + "(~\(String(format: "%.2f", seconds))s)\n"
            )
        }
        return frameCount
    }

    func mediaHasAudioTrack(at url: URL) async -> Bool {
        MediaVideoIO.hasAudioTrack(url)
    }

}

func videoOperationMonotonicSeconds() -> Double { ProcessInfo.processInfo.systemUptime }

func shapeString(_ shape: [Int]) -> String {
    "[" + shape.map(String.init).joined(separator: ", ") + "]"
}
