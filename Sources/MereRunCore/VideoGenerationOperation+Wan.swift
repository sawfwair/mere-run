import Foundation

extension NativeVideoGeneration {
    func runNativeWanGenerate(
        options: Wan2GenerationOptions,
        modelRoot: URL,
        outputURL: URL
    ) async throws -> VideoGenerationOutcome {

        if reportsDiagnostics {
            diagnostic("Engine: native Wan2.2 TI2V\n")
            diagnostic("Model root: \(modelRoot.path)\n")
            diagnostic("Mode: image-to-video\n")
        }
        let generator = Wan2TI2VGenerator()
        let result = try await generator.generate(
            options: options,
            resources: Wan2Resources(rootURL: modelRoot),
            progressHandler: { progress in
                eventHandler?(.progress(stage: progress.stage.rawValue, step: progress.stepIndex, totalSteps: progress.totalSteps))
            }
        )
        if reportsDiagnostics {
            diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
            diagnostic("Writing MP4...\n")
        }
        try Task.checkCancellation()
        eventHandler?(.progressFinished)
        try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: options.fps, to: outputURL)
        return VideoGenerationOutcome(primaryURL: outputURL)
    }

}
