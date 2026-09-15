import Foundation
import MLX
import MereRunContract

extension NativeVideoGeneration {
    func runNativeGenerate(
        request: VideoGenerationLTXRequest,
        modelRoot: URL,
        outputURL: URL
    ) async throws -> VideoGenerationOutcome {
        var timingReport: LTXVideoTimingReport?
        let nativeOptions = request.unifiedOptions()
        let prompt = nativeOptions.prompt
        let numFrames = nativeOptions.numFrames
        let fps = nativeOptions.fps
        let outputMode = request.settings.effectiveOutputMode
        let autoDurationRange = request.plan.autoDuration
        let pipeline = nativeOptions.pipeline
        let distilledLoRAStrengthStage1 = nativeOptions.distilledLoRAStrengthStage1
        let sourceImageURL = nativeOptions.sourceImageURL
        let dfrOptions = nativeOptions.dfr
        let precomputedTextEmbeddingsURL = nativeOptions.precomputedTextEmbeddingsURL
        let hdrColorSpace = nativeOptions.hdrColorSpace
        let hdrICLoRA = nativeOptions.hdrICLoRA
        let skipHDRMP4 = request.settings.skipHDRMP4
        let videoDecoder = request.plan.videoDecoder
        let rootURL = modelRoot
        let savedArtifactURL = skipHDRMP4
            ? outputURL.deletingLastPathComponent().appendingPathComponent(
                "\(outputURL.deletingPathExtension().lastPathComponent)_exr",
                isDirectory: true
            )
            : outputURL
        guard let route = request.plan.ltxRoute else {
            throw VideoGenerationError.invalidInput("The effective plan does not select an LTX runtime.")
        }
        if route == .unifiedAV,
           isLTX23AudioToVideoModelRoot(rootURL),
           !isLTX23FullModelRoot(rootURL) {
            throw VideoGenerationError.invalidInput(
                "This legacy A2Vid root has no vocoder for generated audio. Pull \(ModelResolver.ModelID.ltxVideo23FullMLX.rawValue)."
            )
        }


        if reportsDiagnostics {
            diagnostic("Engine: native\n")
            let isFinalQuality = isLTX23AudioToVideoModelRoot(rootURL) || isLTX25ModelRoot(rootURL)
            diagnostic("Quality: \(isFinalQuality ? LTXVideoQuality.final.rawValue : LTXVideoQuality.draft.rawValue)\n")
            diagnostic("Output mode: \(outputMode.rawValue)\n")
            diagnostic("Runtime lane: \(route.rawValue)\n")
            diagnostic("Model root: \(rootURL.path)\n")
            if isLTX25ModelRoot(rootURL) {
                diagnostic("Video decoder: \(videoDecoder.rawValue)\n")
            }
            diagnostic("Mode: \(sourceImageURL == nil ? "text-to-video" : "image-to-video")\n")
        }

        let makeUnifiedOptions: (Int) -> LTXUnifiedAVGenerationOptions = { frames in
            request.unifiedOptions(numFrames: frames)
        }

        switch route {
        case .legacyDistilledVideo:
            if reportsDiagnostics {
                diagnostic("Loading native distilled model...\n")
            }
            let generator = LTXDistilledLatentGenerator()
            do {
                try await generator.load(modelRoot: rootURL)
                if reportsDiagnostics {
                    diagnostic("Running native denoising + decode...\n")
                }
                let result = try await generator.generateVideo(
                    options: request.distilledOptions()
                )
                await generator.unload()

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.latents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if reportsDiagnostics {
                    diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    diagnostic("Writing MP4...\n")
                }
                try Task.checkCancellation()
                try LTXVideoMP4Writer.writeMP4(
                    frames: result.frames,
                    fps: fps,
                    to: outputURL
                )
            } catch {
                await generator.unload()
                throw error
            }

        case .splitDistilledVideo:
            let endToEndStart = videoOperationMonotonicSeconds()
            if reportsDiagnostics {
                let version = isLTX25ModelRoot(rootURL) ? "LTX 2.5" : "LTX 2.3"
                diagnostic("Loading native \(version) distilled model for video-only output...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings = try await generator.loadVideoOnly(
                    modelRoot: rootURL,
                    videoDecoder: videoDecoder,
                    videoDecoderDType: hdrColorSpace == nil ? nil : .float32,
                    loadTextEncoder: precomputedTextEmbeddingsURL == nil
                )
                if reportsDiagnostics {
                    diagnostic(
                        "Running standalone distilled joint denoising + video decode (audio output disabled)...\n"
                    )
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generateVideoOnly(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoOperationMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoOperationMonotonicSeconds() - unloadStart

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.videoLatents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if reportsDiagnostics {
                    diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    diagnostic("Writing video-only MP4...\n")
                }
                try Task.checkCancellation()
                let writeStart = videoOperationMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                }
                let mp4WriteSeconds = videoOperationMonotonicSeconds() - writeStart
                timingReport = LTXVideoTimingReport(
                    mode: "standalone-distilled-video-only",
                    modelRoot: rootURL.path,
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

        case .fullQualityVideo:
            let endToEndStart = videoOperationMonotonicSeconds()
            if reportsDiagnostics {
                let version = isLTX25FullModelRoot(rootURL) ? "LTX 2.5" : "LTX 2.3"
                diagnostic("Loading native \(version) full-quality model for video-only output...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings: LTXLoadTimings
                if hdrICLoRA != nil {
                    loadTimings = try await generator.loadVideoOnly(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: .float32,
                        loadTextEncoder: precomputedTextEmbeddingsURL == nil
                    )
                } else if dfrOptions != nil {
                    loadTimings = try await generator.loadDFR(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if distilledLoRAStrengthStage1 != 0 {
                    loadTimings = try await generator.loadFullReusable(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else {
                    loadTimings = try await generator.loadFullVideoOnly(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                }
                if reportsDiagnostics {
                    diagnostic(
                        dfrOptions != nil
                            ? "Running LTX 2.5 DFR spatial detail and temporal refinement (audio output disabled)...\n"
                            : (pipeline == .devOneStage
                                ? "Running target-resolution dev one-stage generation (audio output disabled)...\n"
                                : "Running guided dev stage 1 + distilled-LoRA stage 2 (audio output disabled)...\n")
                    )
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generateVideoOnly(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoOperationMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoOperationMonotonicSeconds() - unloadStart

                if let debugPrefix = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_DEBUG_SAVE_PREFIX"], !debugPrefix.isEmpty {
                    let base = URL(fileURLWithPath: debugPrefix).standardizedFileURL
                    let parent = base.deletingLastPathComponent()
                    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
                    let stem = base.lastPathComponent
                    try MLX.save(array: result.frames, url: parent.appendingPathComponent("\(stem)_frames.npy"))
                    try MLX.save(array: result.videoLatents, url: parent.appendingPathComponent("\(stem)_latents.npy"))
                }

                if reportsDiagnostics {
                    diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    diagnostic(
                        skipHDRMP4
                            ? "Writing full-quality half-float EXR sequence...\n"
                            : "Writing full-quality video-only MP4...\n"
                    )
                }
                try Task.checkCancellation()
                let writeStart = videoOperationMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL,
                        writeHLGMaster: !skipHDRMP4
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL
                    )
                }
                let mp4WriteSeconds = videoOperationMonotonicSeconds() - writeStart
                timingReport = LTXVideoTimingReport(
                    mode: dfrOptions != nil ? "ltx25-dfr-video-only" : "full-video-only",
                    modelRoot: rootURL.path,
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

        case .unifiedAV:
            let endToEndStart = videoOperationMonotonicSeconds()
            if reportsDiagnostics {
                diagnostic("Loading native unified AV model...\n")
            }
            let generator = LTXUnifiedAVGenerator()
            do {
                let loadTimings: LTXLoadTimings
                if dfrOptions != nil {
                    loadTimings = try await generator.loadDFR(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if distilledLoRAStrengthStage1 != 0 {
                    loadTimings = try await generator.loadFullReusable(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else if isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL) {
                    loadTimings = try await generator.loadFull(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                } else {
                    loadTimings = try await generator.load(
                        modelRoot: rootURL,
                        videoDecoder: videoDecoder,
                        videoDecoderDType: hdrColorSpace == nil ? nil : .float32
                    )
                }
                if reportsDiagnostics {
                    let lane = dfrOptions != nil
                        ? "LTX 2.5 DFR spatial detail and temporal refinement"
                        : (isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL)
                            ? (pipeline == .devOneStage
                                ? "target-resolution dev one-stage"
                                : "guided dev stage 1 + distilled-LoRA stage 2")
                            : "standalone distilled two-stage")
                    diagnostic("Running \(lane) unified AV denoising + decode...\n")
                }
                let resolvedFrames = try await resolveAutoDuration(
                    autoDurationRange,
                    prompt: prompt,
                    fps: fps,
                    generator: generator,
                    fallback: numFrames
                )
                let result = try await generator.generate(
                    options: makeUnifiedOptions(resolvedFrames)
                )
                let unloadStart = videoOperationMonotonicSeconds()
                await generator.unload()
                let unloadSeconds = videoOperationMonotonicSeconds() - unloadStart

                if reportsDiagnostics {
                    diagnostic("Decoded frames shape: \(shapeString(result.frames.shape))\n")
                    diagnostic("Audio waveform shape: \(shapeString(result.audioWaveform.shape))\n")
                    diagnostic("Writing MP4 with audio...\n")
                }
                try Task.checkCancellation()
                let writeStart = videoOperationMonotonicSeconds()
                if let hdrOutput = result.hdrOutput, let hdrColorSpace {
                    try LTXHDRVideoWriter.write(
                        hdrOutput,
                        colorSpace: hdrColorSpace,
                        fps: result.playbackFPS,
                        to: outputURL,
                        audioWaveform: result.audioWaveform,
                        audioSampleRate: result.audioSampleRate
                    )
                } else {
                    try LTXVideoMP4Writer.writeMP4(
                        frames: result.frames,
                        fps: result.playbackFPS,
                        to: outputURL,
                        audioWaveform: result.audioWaveform,
                        audioSampleRate: result.audioSampleRate
                    )
                }

                guard await mediaHasAudioTrack(at: outputURL) else {
                    throw VideoGenerationError.invalidInput("Unified AV output has no audio track at \(outputURL.path)")
                }
                let mp4WriteSeconds = videoOperationMonotonicSeconds() - writeStart
                timingReport = LTXVideoTimingReport(
                    mode: dfrOptions != nil
                        ? "ltx25-dfr-unified-av"
                        : (isLTX23FullModelRoot(rootURL) || isLTX25FullModelRoot(rootURL)
                            ? "full-unified-av"
                            : "standalone-distilled-unified-av"),
                    modelRoot: rootURL.path,
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
        }

        return VideoGenerationOutcome(
            primaryURL: savedArtifactURL, isDirectory: skipHDRMP4,
            timings: timingReport, includesTimings: true
        )
    }
}
