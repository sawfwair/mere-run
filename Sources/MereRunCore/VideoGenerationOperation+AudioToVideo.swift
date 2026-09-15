import Foundation

extension NativeVideoGeneration {
    func runNativeAudioToVideoGenerate(
        options: LTXAudioToVideoGenerationOptions,
        videoDecoder: LTXVideoDecoderKind,
        modelRoot: URL,
        outputURL: URL
    ) async throws -> VideoGenerationOutcome {
        var timingReport: LTXVideoTimingReport?
        let audioURL = options.audioURL
        let audioStartTime = options.audioStartTime
        let fps = options.fps
        let sourceImageURL = options.sourceImageURL
        let hdrColorSpace = options.hdrColorSpace
        let endToEndStart = videoOperationMonotonicSeconds()

        if reportsDiagnostics {
            let version = isLTX25FullModelRoot(modelRoot) ? "LTX 2.5" : "LTX 2.3"
            diagnostic("Engine: native \(version) A2Vid\n")
            diagnostic("Model root: \(modelRoot.path)\n")
            diagnostic(
                "Mode: \(sourceImageURL == nil ? "audio-to-video" : "audio-and-image-to-video")\n"
            )
            diagnostic("Source audio: \(audioURL.path) at \(audioStartTime)s\n")
            diagnostic("Loading \(version) dev + distilled-LoRA model...\n")
        }

        let generator = LTXUnifiedAVGenerator()
        do {
            let loadTimings: LTXLoadTimings
            if isLTX23FullModelRoot(modelRoot) || isLTX25FullModelRoot(modelRoot) {
                loadTimings = try await generator.loadFull(
                    modelRoot: modelRoot,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            } else {
                loadTimings = try await generator.loadAudioToVideo(
                    modelRoot: modelRoot,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                )
            }
            if reportsDiagnostics {
                diagnostic("Running guided stage 1 and distilled-LoRA stage 2...\n")
            }
            let result = try await generator.generateAudioToVideo(
                options: options
            )
            let unloadStart = videoOperationMonotonicSeconds()
            await generator.unload()
            let unloadSeconds = videoOperationMonotonicSeconds() - unloadStart

            if reportsDiagnostics {
                diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                diagnostic("Writing MP4 with the original source-audio segment...\n")
            }
            try Task.checkCancellation()
            let writeStart = videoOperationMonotonicSeconds()
            if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                try LTXHDRVideoWriter.write(
                    hdrOutput,
                    colorSpace: hdrColorSpace,
                    fps: fps,
                    to: outputURL,
                    sourceAudio: result.sourceAudio
                )
            } else {
                try LTXVideoMP4Writer.writeMP4(
                    frames: result.frames,
                    fps: fps,
                    to: outputURL,
                    sourceAudio: result.sourceAudio
                )
            }
            guard await mediaHasAudioTrack(at: outputURL) else {
                throw VideoGenerationError.invalidInput("A2Vid output has no audio track at \(outputURL.path)")
            }
            let mp4WriteSeconds = videoOperationMonotonicSeconds() - writeStart
            timingReport = LTXVideoTimingReport(
                mode: "audio-to-video",
                modelRoot: modelRoot.path,
                residentModelReused: false,
                load: loadTimings,
                generation: result.timings,
                unloadSeconds: unloadSeconds,
                mp4WriteSeconds: mp4WriteSeconds,
                totalSeconds: videoOperationMonotonicSeconds() - endToEndStart
            )
        } catch {
            await generator.unload()
            throw error
        }

        return VideoGenerationOutcome(primaryURL: outputURL, timings: timingReport, includesTimings: true)
    }

}
