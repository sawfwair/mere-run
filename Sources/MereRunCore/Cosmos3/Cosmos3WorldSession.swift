import Foundation
import MediaIO
import MLX

public enum Cosmos3WorldSessionPhase: String, Codable, Hashable, Sendable {
    case cold
    case ready
    case generating
}

public struct Cosmos3WorldTransitionRequest: Hashable, Sendable {
    public let requestID: UUID
    public let prompt: String
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
    public let modelSpaceActions: [[Float]]?

    public init(
        requestID: UUID = UUID(),
        prompt: String,
        camera: Wan2WorldCameraControl,
        sourceImageURL: URL? = nil,
        outputURL: URL? = nil,
        width: Int = 320,
        height: Int = 176,
        numFrames: Int = 17,
        steps: Int = 30,
        guidanceScale: Float = 1,
        shift: Float = 3,
        seed: UInt64 = 0,
        fps: Int = 30,
        modelSpaceActions: [[Float]]? = nil
    ) {
        self.requestID = requestID
        self.prompt = prompt
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
        self.modelSpaceActions = modelSpaceActions
    }
}

public struct Cosmos3WorldSessionSnapshot: Codable, Hashable, Sendable {
    public let sessionID: UUID
    public let phase: Cosmos3WorldSessionPhase
    public let transitionCount: Int
    public let currentStateID: UUID?
    public let currentFrameURL: URL?
    public let conditioningMode: String
    public let keepsModelsWarm: Bool
    public let keepsTerminalLatent: Bool
}

public struct Cosmos3WorldTransitionReceipt: Codable, Hashable, Sendable {
    public let requestID: UUID
    public let previousStateID: UUID?
    public let stateID: UUID
    public let transitionIndex: Int
    public let outputURL: URL
    public let terminalFrameURL: URL
    public let camera: Wan2WorldCameraControl
    public let conditioningMode: String
    public let actionDomain: Cosmos3ActionDomain
    public let actionSpace: String
    public let modelSpaceActions: [[Float]]
    public let seed: UInt64

    @available(*, deprecated, renamed: "modelSpaceActions")
    public var rawActions: [[Float]] { modelSpaceActions }
}

public enum Cosmos3WorldSessionError: LocalizedError, Sendable {
    case busy
    case sourceImageRequired
    case invalidFrameCount(Int)
    case invalidModelSpaceActions(String)

    public var errorDescription: String? {
        switch self {
        case .busy:
            return "The Cosmos3 world session is already generating a transition."
        case .sourceImageRequired:
            return "The first Cosmos3 world transition requires a source image."
        case .invalidFrameCount(let count):
            return "Cosmos3 world transitions require a 4n+1 frame count; received \(count)."
        case .invalidModelSpaceActions(let reason):
            return "Invalid Cosmos3 normalized model-space camera trajectory: \(reason)"
        }
    }
}

public enum Cosmos3CameraActionCompiler {
    public static func compile(
        control: Wan2WorldCameraControl,
        actionCount: Int,
        startingAction: [Float]? = nil
    ) -> [[Float]] {
        precondition(actionCount > 0)
        precondition(startingAction == nil || startingAction?.count == 9)
        // `camera_pose` rows are frame-to-frame deltas. A prior chunk's final
        // delta is not an absolute pose and must never be accumulated into the
        // next chunk.
        _ = startingAction
        let (direction, scale) = semanticTranslation(for: control)
        var actions = if let direction {
            Cosmos3CameraModelSpaceTrajectory.translated(
                direction: direction,
                scale: scale,
                actionCount: actionCount
            )
        } else {
            Cosmos3CameraModelSpaceTrajectory.stationary(actionCount: actionCount)
        }

        if control.rotationDegrees.contains(where: { abs($0) > 0.0001 }) {
            let radiansPerAction = control.rotationDegrees.map {
                ($0 / Float(actionCount)) * .pi / 180
            }
            let rotation = rotationMatrixXYZ(
                x: radiansPerAction[0],
                y: radiansPerAction[1],
                z: radiansPerAction[2]
            )
            let deltaRotation = rotation6D(rotation)
            for index in actions.indices {
                actions[index].replaceSubrange(3..<9, with: deltaRotation)
            }
        }
        return actions
    }

    private static func semanticTranslation(
        for control: Wan2WorldCameraControl
    ) -> (direction: [Float]?, scale: Float) {
        let requestedMagnitude = sqrt(
            control.translationMeters.reduce(Float.zero) { $0 + $1 * $1 }
        )
        let standardScale = requestedMagnitude > 0.0001 ? requestedMagnitude : 1
        switch control.motion {
        case .forward:
            return ([0, 0, 1], standardScale)
        case .backward:
            return ([0, 0, -1], standardScale)
        case .strafeLeft:
            return ([-1, 0, 0], standardScale)
        case .strafeRight:
            return ([1, 0, 0], standardScale)
        case .custom where requestedMagnitude > 0.0001:
            return (control.translationMeters, requestedMagnitude)
        case .hold, .custom, .yawLeft, .yawRight:
            return (nil, 0)
        }
    }

    private static func rotation6D(_ rotation: [[Float]]) -> [Float] {
        [
            rotation[0][0], rotation[1][0], rotation[2][0],
            rotation[0][1], rotation[1][1], rotation[2][1],
        ]
    }

    private static func rotationMatrixXYZ(x: Float, y: Float, z: Float) -> [[Float]] {
        let cx = cos(x)
        let sx = sin(x)
        let cy = cos(y)
        let sy = sin(y)
        let cz = cos(z)
        let sz = sin(z)
        return [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    }
}

enum Cosmos3WorldFrameStitching {
    static func replaceFirstFrame(
        in frames: MLXArray,
        with source: MediaImage
    ) throws -> MLXArray {
        precondition(frames.ndim == 5 && frames.dim(0) == 1 && frames.dim(4) == 3)
        let width = frames.dim(3)
        let height = frames.dim(2)
        let resized = try MediaImageIO.resized(source, width: width, height: height)
        var rgb = [UInt8]()
        rgb.reserveCapacity(width * height * 3)
        for offset in stride(from: 0, to: resized.rgba8.count, by: 4) {
            rgb.append(resized.rgba8[offset])
            rgb.append(resized.rgba8[offset + 1])
            rgb.append(resized.rgba8[offset + 2])
        }
        let first = MLXArray(rgb).reshaped(1, 1, height, width, 3)
        guard frames.dim(1) > 1 else { return first }
        return MLX.concatenated([first, frames[0..., 1...]], axis: 1)
    }
}

enum Cosmos3AutoregressiveSeedSequence {
    static func seed(baseSeed: UInt64, chunkIndex: Int) -> UInt64 {
        precondition(chunkIndex >= 0)
        return baseSeed &+ UInt64(chunkIndex)
    }
}

enum Cosmos3WorldGraphCache {
    static func areInverse(
        _ lhs: Wan2WorldCameraControl,
        _ rhs: Wan2WorldCameraControl,
        tolerance: Float = 0.0001
    ) -> Bool {
        let inverseMotions: Set<[Wan2WorldMotion]> = [
            [.forward, .backward],
            [.backward, .forward],
            [.strafeLeft, .strafeRight],
            [.strafeRight, .strafeLeft],
            [.yawLeft, .yawRight],
            [.yawRight, .yawLeft],
        ]
        guard inverseMotions.contains([lhs.motion, rhs.motion]) else { return false }
        return zip(lhs.translationMeters, rhs.translationMeters).allSatisfy {
            abs($0 + $1) <= tolerance
        } && zip(lhs.rotationDegrees, rhs.rotationDegrees).allSatisfy {
            abs($0 + $1) <= tolerance
        }
    }

    static func reversedFrames(
        _ bytes: [UInt8],
        frameCount: Int,
        width: Int,
        height: Int
    ) -> [UInt8] {
        precondition(frameCount > 0 && width > 0 && height > 0)
        let frameSize = width * height * 3
        precondition(bytes.count == frameCount * frameSize)
        var output = [UInt8](repeating: 0, count: bytes.count)
        for outputFrame in 0..<frameCount {
            let inputFrame = frameCount - outputFrame - 1
            let source = (inputFrame * frameSize)..<((inputFrame + 1) * frameSize)
            let target = (outputFrame * frameSize)..<((outputFrame + 1) * frameSize)
            output[target] = bytes[source]
        }
        return output
    }
}

private struct Cosmos3WorldUndoCandidate {
    let camera: Wan2WorldCameraControl
    let sourceStateID: UUID?
    let sourceModelSpaceAction: [Float]?
    let sourceGenerationDepth: Int
    let frameBytes: [UInt8]
    let frameCount: Int
    let width: Int
    let height: Int
    let fps: Int
    let modelSpaceActions: [[Float]]
    let seed: UInt64
}

public actor Cosmos3WorldSession {
    public let sessionID: UUID
    public let resources: Cosmos3Resources
    public let stateDirectory: URL

    private let generator = Cosmos3EdgeGenerator()
    private var phase: Cosmos3WorldSessionPhase = .cold
    private var transitionCount = 0
    private var generationDepth = 0
    private var currentStateID: UUID?
    private var currentFrameURL: URL?
    private var currentModelSpaceAction: [Float]?
    private var edgeHistory: [Cosmos3WorldUndoCandidate] = []

    public init(
        resources: Cosmos3Resources,
        stateDirectory: URL,
        sessionID: UUID = UUID()
    ) {
        self.resources = resources
        self.stateDirectory = stateDirectory.standardizedFileURL
        self.sessionID = sessionID
    }

    public func prepare(progress: (@Sendable (String) -> Void)? = nil) async throws {
        guard phase != .generating else { throw Cosmos3WorldSessionError.busy }
        try FileManager.default.createDirectory(
            at: stateDirectory,
            withIntermediateDirectories: true
        )
        try await generator.prepare(resources: resources, progress: progress)
        phase = .ready
    }

    public func transition(
        _ request: Cosmos3WorldTransitionRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil
    ) async throws -> Cosmos3WorldTransitionReceipt {
        guard phase != .generating else { throw Cosmos3WorldSessionError.busy }
        guard request.numFrames > 1, (request.numFrames - 1).isMultiple(of: 4) else {
            throw Cosmos3WorldSessionError.invalidFrameCount(request.numFrames)
        }
        let usesExplicitSource = request.sourceImageURL != nil
        guard let sourceImageURL = request.sourceImageURL ?? currentFrameURL else {
            throw Cosmos3WorldSessionError.sourceImageRequired
        }
        try FileManager.default.createDirectory(
            at: stateDirectory,
            withIntermediateDirectories: true
        )
        let nextIndex = transitionCount + 1
        let outputURL = request.outputURL ?? stateDirectory
            .appendingPathComponent(String(format: "transition-%04d.mp4", nextIndex))
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let previousStateID = currentStateID
        let previousModelSpaceAction = currentModelSpaceAction
        let previousGenerationDepth = usesExplicitSource ? 0 : generationDepth
        if !usesExplicitSource,
           request.modelSpaceActions == nil,
           let candidate = edgeHistory.last,
           candidate.frameCount == request.numFrames,
           candidate.width == request.width,
           candidate.height == request.height,
           candidate.fps == request.fps,
           Cosmos3WorldGraphCache.areInverse(request.camera, candidate.camera) {
            phase = .generating
            do {
                let reversedBytes = Cosmos3WorldGraphCache.reversedFrames(
                    candidate.frameBytes,
                    frameCount: candidate.frameCount,
                    width: candidate.width,
                    height: candidate.height
                )
                let outputFrames = MLXArray(reversedBytes).reshaped(
                    1,
                    candidate.frameCount,
                    candidate.height,
                    candidate.width,
                    3
                )
                eval(outputFrames)
                try LTXVideoMP4Writer.writeMP4(
                    frames: outputFrames,
                    fps: request.fps,
                    to: outputURL
                )
                let terminalFrameURL = stateDirectory
                    .appendingPathComponent(String(format: "state-%04d.png", nextIndex))
                let terminal = outputFrames[0, outputFrames.dim(1) - 1]
                eval(terminal)
                let image = try MediaImageIO.imageFromRGBHWC(
                    terminal.asArray(UInt8.self),
                    width: candidate.width,
                    height: candidate.height
                )
                try MediaImageIO.writePNG(image, to: terminalFrameURL)

                let nextStateID = candidate.sourceStateID ?? UUID()
                let reversedActions = Array(candidate.modelSpaceActions.reversed())
                edgeHistory.removeLast()
                currentFrameURL = terminalFrameURL
                currentStateID = nextStateID
                currentModelSpaceAction = candidate.sourceModelSpaceAction
                generationDepth = candidate.sourceGenerationDepth
                transitionCount = nextIndex
                phase = .ready
                return Cosmos3WorldTransitionReceipt(
                    requestID: request.requestID,
                    previousStateID: previousStateID,
                    stateID: nextStateID,
                    transitionIndex: nextIndex,
                    outputURL: outputURL,
                    terminalFrameURL: terminalFrameURL,
                    camera: request.camera,
                    conditioningMode: "cosmos3_camera_pose_cached_inverse",
                    actionDomain: .cameraPose,
                    actionSpace: "normalized_model_space_reversed",
                    modelSpaceActions: reversedActions,
                    seed: candidate.seed
                )
            } catch {
                phase = generator.isWarm ? .ready : .cold
                throw error
            }
        }
        // NVIDIA advances the base seed by the autoregressive chunk index.
        // Graph-cache returns restore the source node's depth and do not
        // consume another stochastic generation chunk.
        let effectiveSeed = Cosmos3AutoregressiveSeedSequence.seed(
            baseSeed: request.seed,
            chunkIndex: previousGenerationDepth
        )
        let actionCount = request.numFrames - 1
        let modelSpaceActions: [[Float]]
        if let explicit = request.modelSpaceActions {
            guard !explicit.isEmpty else {
                throw Cosmos3WorldSessionError.invalidModelSpaceActions("the trajectory is empty")
            }
            if let invalid = explicit.first(where: {
                $0.count != Cosmos3CameraModelSpaceTrajectory.actionDimension
            }) {
                throw Cosmos3WorldSessionError.invalidModelSpaceActions(
                    "expected 9 values per action; received \(invalid.count)"
                )
            }
            guard explicit.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
                throw Cosmos3WorldSessionError.invalidModelSpaceActions(
                    "all values must be finite"
                )
            }
            modelSpaceActions = Cosmos3CameraModelSpaceTrajectory.fitted(
                explicit,
                actionCount: actionCount
            )
        } else {
            modelSpaceActions = Cosmos3CameraActionCompiler.compile(
                control: request.camera,
                actionCount: actionCount,
                startingAction: usesExplicitSource ? nil : currentModelSpaceAction
            )
        }
        let action = try Cosmos3ActionCondition(
            mode: .forwardDynamics,
            chunkSize: actionCount,
            domain: .cameraPose,
            resolutionTier: Self.resolutionTier(width: request.width, height: request.height),
            rawActions: modelSpaceActions,
            imageURL: sourceImageURL,
            viewpoint: .egoView
        )
        let options = try Cosmos3GenerationOptions(
            prompt: request.prompt,
            action: action,
            width: request.width,
            height: request.height,
            numFrames: request.numFrames,
            steps: request.steps,
            guidanceScale: request.guidanceScale,
            shift: request.shift,
            seed: effectiveSeed,
            fps: request.fps
        )
        phase = .generating
        do {
            let result = try await generator.generate(
                options: options,
                resources: resources,
                // NVIDIA's autoregressive recipe extracts the terminal public
                // frame and re-encodes it as the next chunk's conditioning
                // image. A terminal latent from the end of one causal VAE
                // timeline is not a valid frame-zero latent for another.
                useChainedSourceLatent: false,
                progressHandler: progressHandler
            )
            generator.clearChain()
            // The Wan decoder is temporally causal: the same terminal latent
            // decodes differently when it moves from the end of one clip to
            // the start of the next because its preceding decoder context is
            // absent. Preserve the latent for model conditioning, but splice
            // the public first pixel frame to the exact prior world state.
            let outputFrames = try Cosmos3WorldFrameStitching.replaceFirstFrame(
                in: result.frames,
                with: MediaImageIO.decode(sourceImageURL)
            )
            eval(outputFrames)
            try LTXVideoMP4Writer.writeMP4(
                frames: outputFrames,
                fps: request.fps,
                to: outputURL
            )
            let terminalFrameURL = stateDirectory
                .appendingPathComponent(String(format: "state-%04d.png", nextIndex))
            let terminal = outputFrames[0, outputFrames.dim(1) - 1]
            eval(terminal)
            let image = try MediaImageIO.imageFromRGBHWC(
                terminal.asArray(UInt8.self),
                width: result.width,
                height: result.height
            )
            try MediaImageIO.writePNG(image, to: terminalFrameURL)

            let nextStateID = UUID()
            let frameBytes = outputFrames.asArray(UInt8.self)
            let generatedEdge = Cosmos3WorldUndoCandidate(
                camera: request.camera,
                sourceStateID: previousStateID,
                sourceModelSpaceAction: previousModelSpaceAction,
                sourceGenerationDepth: previousGenerationDepth,
                frameBytes: frameBytes,
                frameCount: outputFrames.dim(1),
                width: result.width,
                height: result.height,
                fps: request.fps,
                modelSpaceActions: modelSpaceActions,
                seed: result.seed
            )
            if usesExplicitSource {
                edgeHistory = [generatedEdge]
            } else {
                edgeHistory.append(generatedEdge)
            }
            currentFrameURL = terminalFrameURL
            currentStateID = nextStateID
            currentModelSpaceAction = modelSpaceActions.last
            generationDepth = previousGenerationDepth + 1
            transitionCount = nextIndex
            phase = .ready
            return Cosmos3WorldTransitionReceipt(
                requestID: request.requestID,
                previousStateID: previousStateID,
                stateID: nextStateID,
                transitionIndex: nextIndex,
                outputURL: outputURL,
                terminalFrameURL: terminalFrameURL,
                camera: request.camera,
                conditioningMode: "cosmos3_camera_pose_media_reencode",
                actionDomain: .cameraPose,
                actionSpace: "normalized_model_space",
                modelSpaceActions: modelSpaceActions,
                seed: result.seed
            )
        } catch {
            phase = generator.isWarm ? .ready : .cold
            throw error
        }
    }

    public func reset(sourceImageURL: URL? = nil) throws {
        guard phase != .generating else { throw Cosmos3WorldSessionError.busy }
        currentFrameURL = sourceImageURL
        generator.clearChain()
        currentModelSpaceAction = nil
        edgeHistory.removeAll(keepingCapacity: true)
        generationDepth = 0
        currentStateID = sourceImageURL == nil ? nil : UUID()
        transitionCount = 0
        phase = generator.isWarm ? .ready : .cold
    }

    public func unload() throws {
        guard phase != .generating else { throw Cosmos3WorldSessionError.busy }
        generator.unload()
        phase = .cold
    }

    public func snapshot() -> Cosmos3WorldSessionSnapshot {
        Cosmos3WorldSessionSnapshot(
            sessionID: sessionID,
            phase: phase,
            transitionCount: transitionCount,
            currentStateID: currentStateID,
            currentFrameURL: currentFrameURL,
            conditioningMode: "cosmos3_camera_pose_media_reencode",
            keepsModelsWarm: generator.isWarm,
            keepsTerminalLatent: false
        )
    }

    private static func resolutionTier(
        width: Int,
        height: Int
    ) -> Cosmos3ActionResolutionTier {
        let shortSide = min(width, height)
        if shortSide <= 320 { return .compact }
        if shortSide <= 640 { return .medium }
        if shortSide <= 704 { return .large }
        return .hd
    }
}
