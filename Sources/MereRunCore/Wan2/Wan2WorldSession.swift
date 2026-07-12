import Foundation
import MediaIO
import MLX

public enum Wan2WorldMotion: String, Codable, Hashable, Sendable {
    case hold
    case forward
    case backward
    case strafeLeft
    case strafeRight
    case yawLeft
    case yawRight
    case custom
}

public struct Wan2WorldCameraControl: Codable, Hashable, Sendable {
    public let motion: Wan2WorldMotion
    public let translationMeters: [Float]
    public let rotationDegrees: [Float]

    public init(
        motion: Wan2WorldMotion,
        translationMeters: [Float] = [0, 0, 0],
        rotationDegrees: [Float] = [0, 0, 0]
    ) {
        precondition(translationMeters.count == 3)
        precondition(rotationDegrees.count == 3)
        self.motion = motion
        self.translationMeters = translationMeters
        self.rotationDegrees = rotationDegrees
    }

    public static func forward(meters: Float = 1) -> Self {
        Self(motion: .forward, translationMeters: [0, 0, meters])
    }

    public static func backward(meters: Float = 1) -> Self {
        Self(motion: .backward, translationMeters: [0, 0, -meters])
    }

    public static func yawLeft(degrees: Float = 30) -> Self {
        Self(motion: .yawLeft, rotationDegrees: [0, -degrees, 0])
    }

    public static func yawRight(degrees: Float = 30) -> Self {
        Self(motion: .yawRight, rotationDegrees: [0, degrees, 0])
    }

    var promptClause: String {
        switch motion {
        case .hold:
            return "The camera remains stationary with no translation or rotation."
        case .forward:
            return "The first-person camera translates straight forward through the existing three-dimensional space without yaw or sideways movement."
        case .backward:
            return "The first-person camera translates straight backward through the existing three-dimensional space without yaw or sideways movement."
        case .strafeLeft:
            return "The first-person camera sidesteps left while preserving its viewing direction, with no yaw."
        case .strafeRight:
            return "The first-person camera sidesteps right while preserving its viewing direction, with no yaw."
        case .yawLeft:
            return "The first-person camera pivots left in place with no forward translation."
        case .yawRight:
            return "The first-person camera pivots right in place with no forward translation."
        case .custom:
            let translation = translationMeters.map { String(format: "%.3f", $0) }.joined(separator: ",")
            let rotation = rotationDegrees.map { String(format: "%.3f", $0) }.joined(separator: ",")
            return "The camera follows translation [\(translation)] meters and XYZ rotation [\(rotation)] degrees while preserving scene identity."
        }
    }
}

public enum Wan2WorldConditioningMode: String, Codable, Hashable, Sendable {
    case textAndFirstFrame
    case projectiveCameraLatents
    case causalCameraLatents
}

public struct Wan2WorldTransitionRequest: Hashable, Sendable {
    public let requestID: UUID
    public let prompt: String
    public let negativePrompt: String
    public let camera: Wan2WorldCameraControl
    public let sourceImageURL: URL?
    public let outputURL: URL?
    public let width: Int
    public let height: Int
    public let numFrames: Int
    public let steps: Int
    public let guidanceScale: Float
    public let shift: Float
    public let seed: UInt64
    public let fps: Int

    public init(
        requestID: UUID = UUID(),
        prompt: String,
        negativePrompt: String = Wan2Resources.defaultNegativePrompt,
        camera: Wan2WorldCameraControl,
        sourceImageURL: URL? = nil,
        outputURL: URL? = nil,
        width: Int = 512,
        height: Int = 320,
        numFrames: Int = 17,
        steps: Int = 40,
        guidanceScale: Float = 5,
        shift: Float = 5,
        seed: UInt64 = 42,
        fps: Int = 24
    ) {
        self.requestID = requestID
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.camera = camera
        self.sourceImageURL = sourceImageURL
        self.outputURL = outputURL
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.shift = shift
        self.seed = seed
        self.fps = fps
    }
}

public enum Wan2WorldSessionPhase: String, Codable, Hashable, Sendable {
    case cold
    case ready
    case generating
}

public struct Wan2WorldSessionSnapshot: Codable, Hashable, Sendable {
    public let sessionID: UUID
    public let phase: Wan2WorldSessionPhase
    public let transitionCount: Int
    public let currentStateID: UUID?
    public let currentFrameURL: URL?
    public let conditioningMode: Wan2WorldConditioningMode
    public let keepsModelsWarm: Bool
    public let keepsTerminalLatent: Bool
}

public struct Wan2WorldTransitionReceipt: Codable, Hashable, Sendable {
    public let requestID: UUID
    public let previousStateID: UUID?
    public let stateID: UUID
    public let transitionIndex: Int
    public let outputURL: URL
    public let terminalFrameURL: URL
    public let camera: Wan2WorldCameraControl
    public let conditioningMode: Wan2WorldConditioningMode
    public let seed: UInt64
}

public enum Wan2WorldSessionError: LocalizedError, Sendable {
    case busy
    case sourceImageRequired
    case terminalLatentMissing

    public var errorDescription: String? {
        switch self {
        case .busy:
            return "The Wan world session is already generating a transition."
        case .sourceImageRequired:
            return "The first world transition requires a source image."
        case .terminalLatentMissing:
            return "Wan generation did not return the terminal-frame latent required for chaining."
        }
    }
}

public actor Wan2WorldSession {
    public let sessionID: UUID
    public let resources: Wan2Resources
    public let stateDirectory: URL

    private let generator: Wan2TI2VGenerator
    private let causalGenerator: Wan2CausalWorldGenerator?
    private var phase: Wan2WorldSessionPhase = .cold
    private var transitionCount = 0
    private var currentStateID: UUID?
    private var currentFrameURL: URL?

    public init(
        resources: Wan2Resources,
        stateDirectory: URL,
        cameraWeightsURL: URL? = nil,
        causalWeightsURL: URL? = nil,
        sessionID: UUID = UUID()
    ) {
        precondition(cameraWeightsURL == nil || causalWeightsURL == nil)
        self.resources = resources
        self.stateDirectory = stateDirectory.standardizedFileURL
        self.sessionID = sessionID
        self.generator = Wan2TI2VGenerator(cameraWeightsURL: cameraWeightsURL)
        self.causalGenerator = causalWeightsURL.map(Wan2CausalWorldGenerator.init(weightsURL:))
    }

    private var conditioningMode: Wan2WorldConditioningMode {
        causalGenerator == nil ? generator.conditioningMode : .causalCameraLatents
    }

    private var isWarm: Bool {
        causalGenerator?.isWarm ?? generator.isWarm
    }

    public func prepare(progress: (@Sendable (String) -> Void)? = nil) throws {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        try FileManager.default.createDirectory(at: stateDirectory, withIntermediateDirectories: true)
        if let causalGenerator {
            try causalGenerator.prepare(resources: resources, progress: progress)
        } else {
            try generator.prepare(resources: resources, progress: progress)
        }
        phase = .ready
    }

    public func transition(
        _ request: Wan2WorldTransitionRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> Wan2WorldTransitionReceipt {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        try FileManager.default.createDirectory(at: stateDirectory, withIntermediateDirectories: true)

        let usesExplicitSource = request.sourceImageURL != nil
        guard let sourceImageURL = request.sourceImageURL ?? currentFrameURL else {
            throw Wan2WorldSessionError.sourceImageRequired
        }
        let nextIndex = transitionCount + 1
        let outputURL = request.outputURL ?? stateDirectory
            .appendingPathComponent(String(format: "transition-%04d.mp4", nextIndex))
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let usesProjectiveCamera = conditioningMode == .projectiveCameraLatents
        let effectivePrompt = usesProjectiveCamera
            ? request.prompt
            : request.prompt + "\n\nCamera behavior: " + request.camera.promptClause
        let cameraConditioning = usesProjectiveCamera
            ? Wan2DreamXCameraTrajectory.compile(
                control: request.camera,
                pixelFrameCount: request.numFrames
            )
            : nil
        phase = .generating
        do {
            let result: Wan2VideoGenerationResult
            if let causalGenerator {
                if usesExplicitSource, transitionCount > 0 {
                    causalGenerator.reset()
                }
                result = try await causalGenerator.generateBlock(
                    prompt: request.prompt,
                    camera: request.camera,
                    sourceImageURL: causalGenerator.latentFrameCount == 0 ? sourceImageURL : nil,
                    resources: resources,
                    width: request.width,
                    height: request.height,
                    seed: request.seed,
                    progressHandler: progressHandler
                )
            } else {
                let options = try Wan2GenerationOptions(
                    prompt: effectivePrompt,
                    negativePrompt: request.negativePrompt,
                    sourceImageURL: sourceImageURL,
                    outputURL: outputURL,
                    width: request.width,
                    height: request.height,
                    numFrames: request.numFrames,
                    steps: request.steps,
                    guidanceScale: request.guidanceScale,
                    shift: request.shift,
                    seed: request.seed,
                    fps: request.fps,
                    cameraConditioning: cameraConditioning
                )
                result = try await generator.generate(
                    options: options,
                    resources: resources,
                    useChainedSourceLatent: !usesExplicitSource,
                    captureTerminalFrameLatent: true,
                    progressHandler: progressHandler
                )
            }
            guard result.terminalFrameLatent != nil else {
                throw Wan2WorldSessionError.terminalLatentMissing
            }
            try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: request.fps, to: outputURL)

            let terminalFrameURL = stateDirectory
                .appendingPathComponent(String(format: "state-%04d.png", nextIndex))
            let terminal = result.frames[0, result.frames.dim(1) - 1]
            eval(terminal)
            let image = try MediaImageIO.imageFromRGBHWC(
                terminal.asArray(UInt8.self),
                width: result.frames.dim(3),
                height: result.frames.dim(2)
            )
            try MediaImageIO.writePNG(image, to: terminalFrameURL)

            let previousStateID = currentStateID
            let nextStateID = UUID()
            currentFrameURL = terminalFrameURL
            currentStateID = nextStateID
            transitionCount = nextIndex
            phase = .ready
            return Wan2WorldTransitionReceipt(
                requestID: request.requestID,
                previousStateID: previousStateID,
                stateID: nextStateID,
                transitionIndex: nextIndex,
                outputURL: outputURL,
                terminalFrameURL: terminalFrameURL,
                camera: request.camera,
                conditioningMode: conditioningMode,
                seed: result.seed
            )
        } catch {
            phase = isWarm ? .ready : .cold
            throw error
        }
    }

    public func reset(sourceImageURL: URL? = nil) throws {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        currentFrameURL = sourceImageURL
        if let causalGenerator {
            causalGenerator.reset()
        } else {
            generator.clearChain()
        }
        currentStateID = sourceImageURL == nil ? nil : UUID()
        transitionCount = 0
        phase = isWarm ? .ready : .cold
    }

    public func unload() throws {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        if let causalGenerator {
            causalGenerator.unload()
        } else {
            generator.unload()
        }
        phase = .cold
    }

    public func snapshot() -> Wan2WorldSessionSnapshot {
        Wan2WorldSessionSnapshot(
            sessionID: sessionID,
            phase: phase,
            transitionCount: transitionCount,
            currentStateID: currentStateID,
            currentFrameURL: currentFrameURL,
            conditioningMode: conditioningMode,
            keepsModelsWarm: isWarm,
            keepsTerminalLatent: causalGenerator.map { $0.latentFrameCount > 0 }
                ?? generator.hasChainedTerminalFrameLatent
        )
    }
}
