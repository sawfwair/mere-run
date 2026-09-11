import Foundation
import MereRunContract

/// Effective geometry, seed, decoder, and sampler recipe before tensor loading.
/// Duration prediction and source-audio inspection remain runtime preparation.
public struct VideoGenerationPlan: Sendable {
    public let options: VideoGenerationOptions
    public let profile: VideoGenerationModelProfile
    public let width: Int
    public let height: Int
    public let requestedFrames: Int
    public let numFrames: Int
    public let fps: Double
    public let seed: Int
    public let autoDuration: LTX25AutoDuration?
    public let videoDecoder: LTXVideoDecoderKind
    public let ltxRoute: LTXVideoGenerationRoute?
    public let ltxRecipe: VideoGenerationLTXRecipe
    public let h3SlidingWindowOptions: MiniMaxH3SlidingWindowOptions?

    public init(
        options: VideoGenerationOptions,
        profile: VideoGenerationModelProfile,
        preparation: VideoGenerationLTXPreparation? = nil
    ) throws {
        if let issue = options.basicValidationIssues(profile: profile).first(where: { $0.severity == .blocker }) {
            throw issue
        }
        self.options = options
        self.profile = profile
        let preservesHDRDimensions = preparation?.hdrICLoRA != nil
        let spatialMultiple = profile.isH3 || profile == .wan ? 32 : 64
        width = preservesHDRDimensions ? options.resolvedOutputWidth
            : max(spatialMultiple, options.resolvedOutputWidth / spatialMultiple * spatialMultiple)
        height = preservesHDRDimensions ? options.resolvedOutputHeight
            : max(spatialMultiple, options.resolvedOutputHeight / spatialMultiple * spatialMultiple)
        fps = profile.isH3 ? Double(MiniMaxH3Geometry.framesPerSecond) : options.fps
        guard fps.isFinite, fps >= 1, fps < Double(Int.max) else {
            throw VideoGenerationIssue(id: "fps_invalid", title: "FPS is invalid",
                                       message: "--fps must be finite, >= 1, and representable as an integer frame rate.")
        }
        let minimum = profile.isH3 ? 22 : (profile == .wan ? 5 : 9)
        let multiple = profile == .wan ? 4 : 8
        if let duration = options.duration {
            guard duration.isFinite, duration > 0 else {
                throw VideoGenerationIssue(id: "duration_invalid", title: "Duration is invalid",
                                           message: "--duration must be finite and > 0.")
            }
            if profile.isH3 {
                guard let frames = Int(exactly: (duration * fps).rounded()) else {
                    throw Self.frameCountOverflow
                }
                requestedFrames = frames
            } else {
                requestedFrames = try Self.nearestFrameCount(duration: duration, fps: fps, multiple: multiple)
            }
        } else {
            requestedFrames = options.numFrames
                ?? (options.hasSourceAudio && profile.isLTX25 ? 121 : 65)
        }
        let frames = max(minimum, requestedFrames)
        if profile.isH3 {
            let remainder = (5 - frames % 17 + 17) % 17
            let (aligned, overflow) = frames.addingReportingOverflow(remainder)
            guard !overflow else { throw Self.frameCountOverflow }
            numFrames = aligned
        } else {
            numFrames = (frames - 1) / multiple * multiple + 1
        }
        seed = options.seed ?? (profile.isLTX25 ? 10 : 42)
        if options.numFrames != nil {
            autoDuration = nil
        } else if let explicit = options.autoDurationRange {
            _ = try Self.nearestFrameCount(duration: explicit.maximumSeconds, fps: fps, multiple: 8)
            autoDuration = explicit
        } else if profile.isLTX25, options.duration == nil, !options.hasSourceAudio {
            autoDuration = LTX25AutoDuration(minimumSeconds: 1, maximumSeconds: 20)
        } else {
            autoDuration = nil
        }
        videoDecoder = options.videoDecoder ?? (profile == .ltx25Full ? .diffusion : .convolutional)
        ltxRoute = profile.ltxRoute(outputMode: options.effectiveOutputMode)
        ltxRecipe = VideoGenerationLTXRecipe(options: options)
        if profile.isH3, let renderWidth = options.h3RenderWidth, let renderHeight = options.h3RenderHeight {
            let (leftAspect, leftOverflow) = renderWidth.multipliedReportingOverflow(by: height)
            let (rightAspect, rightOverflow) = renderHeight.multipliedReportingOverflow(by: width)
            guard renderWidth >= 32, renderHeight >= 32,
                  renderWidth.isMultiple(of: 32), renderHeight.isMultiple(of: 32),
                  renderWidth <= width, renderHeight <= height,
                  !leftOverflow, !rightOverflow, leftAspect == rightAspect else {
                throw VideoGenerationIssue(
                    id: "h3_render_canvas_invalid", title: "MiniMax-H3 internal render canvas is invalid",
                    message: "Internal render dimensions must preserve output aspect, use 32px multiples, and not exceed the resolved output canvas."
                )
            }
            if options.h3WindowFrames != nil, renderWidth != width || renderHeight != height {
                throw VideoGenerationIssue(
                    id: "h3_render_canvas_sliding_window_unsupported", title: "Reduced H3 rendering cannot use sliding windows yet",
                    message: "Run a single H3 window or remove --h3-render-width and --h3-render-height."
                )
            }
        }
        if profile.isH3, let windowFrames = options.h3WindowFrames {
            do {
                h3SlidingWindowOptions = try MiniMaxH3SlidingWindowOptions(
                    totalFrameCount: numFrames, windowFrameCount: windowFrames, overlapFrameCount: options.h3WindowOverlap
                )
            } catch {
                throw VideoGenerationIssue(id: "h3_sliding_window_invalid", title: "MiniMax-H3 sliding window is invalid", message: error.localizedDescription)
            }
        } else {
            h3SlidingWindowOptions = nil
        }
    }

    public static func nearestFrameCount(duration: Double, fps: Double, multiple: Int) throws -> Int {
        let target = max(Double(multiple + 1), duration * max(1, fps))
        guard duration.isFinite, fps.isFinite, target.isFinite,
              let chunks = Int(exactly: max(1, ((target - 1) / Double(multiple)).rounded())) else {
            throw frameCountOverflow
        }
        let (frames, overflow) = chunks.multipliedReportingOverflow(by: multiple)
        let (result, additionOverflow) = frames.addingReportingOverflow(1)
        guard !overflow, !additionOverflow else { throw frameCountOverflow }
        return result
    }

    private static var frameCountOverflow: VideoGenerationIssue {
        VideoGenerationIssue(id: "frame_count_overflow", title: "Frame count is too large",
                             message: "The requested duration or frame count exceeds the supported integer range.")
    }
}

public struct VideoGenerationLTXRecipe: Sendable {
    public let inferenceSteps: Int
    public let distilledLoRAStrengthStage1: Float
    public let distilledLoRAStrengthStage2: Float
    public let sampler: LTXSamplerConfiguration
    public let videoGuidance: LTXMultiModalGuidance
    public let audioGuidance: LTXMultiModalGuidance

    public init(options: VideoGenerationOptions) {
        let hq = options.ltxPreset == .hq
        inferenceSteps = hq ? 15 : options.a2vSteps
        distilledLoRAStrengthStage1 = options.distilledLoRAStrengthStage1
            ?? (options.dfr ? 1 : (hq ? 0.25 : 0))
        distilledLoRAStrengthStage2 = options.distilledLoRAStrengthStage2
            ?? (hq ? 0.5 : (options.ltxPipeline == .devOneStage ? 0 : 1))
        sampler = LTXSamplerConfiguration(
            mode: options.ltxSampler ?? (hq ? .res2s : .euler),
            eta: options.ltxSamplerEta,
            noiseSeedOffset: hq ? LTXSamplerConfiguration.hq.noiseSeedOffset : 10_000,
            substepNoiseSeedOffset: hq ? LTXSamplerConfiguration.hq.substepNoiseSeedOffset : 20_000,
            res2sBongMath: !options.noRes2sBongMath,
            res2sBongMathMaxIterations: options.res2sBongMaxIterations,
            gradientEstimationGamma: options.gradientEstimationGamma
        )
        videoGuidance = LTXMultiModalGuidance(
            classifierFreeScale: options.videoCFGGuidanceScale,
            spatioTemporalScale: hq ? 0 : options.videoSTGScale,
            rescale: hq ? 0.45 : options.videoGuidanceRescale,
            modalityScale: options.a2vGuidanceScale,
            spatioTemporalBlocks: Set(hq ? [] : (options.videoSTGBlocks.isEmpty ? [28] : options.videoSTGBlocks)),
            skipStep: options.videoGuidanceSkipStep
        )
        audioGuidance = LTXMultiModalGuidance(
            classifierFreeScale: options.audioCFGGuidanceScale,
            spatioTemporalScale: hq ? 0 : options.audioSTGScale,
            rescale: hq ? 1 : options.audioGuidanceRescale,
            modalityScale: options.v2aGuidanceScale,
            spatioTemporalBlocks: Set(hq ? [] : (options.audioSTGBlocks.isEmpty ? [28] : options.audioSTGBlocks)),
            skipStep: options.audioGuidanceSkipStep
        )
    }
}
