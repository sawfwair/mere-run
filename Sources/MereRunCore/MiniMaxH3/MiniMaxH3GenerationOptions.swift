import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

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
            if adapterInferenceRecipe.requiresTextOnlyConditioning,
               firstFrameURL != nil || lastFrameURL != nil || !frameInputs.isEmpty {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "FastH3 Preview v1 supports text-to-audio/video only; frame conditioning is unsupported"
                )
            }
            if adapterInferenceRecipe.requiresFastH3VSA,
               accelerationMode != .quality {
                throw MiniMaxH3GeneratorError.invalidOptions(
                    "FastH3 VSA cannot be combined with additional H3 approximation modes; use --h3-acceleration quality"
                )
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
