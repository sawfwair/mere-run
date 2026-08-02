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

public struct Wan2WorldRolloutRequest: Hashable, Sendable {
    public let requestID: UUID
    public let prompt: String
    public let actionSequence: [Wan2DreamXARTrajectorySegment]
    public let sourceImageURL: URL?
    public let outputURL: URL?
    public let width: Int
    public let height: Int
    public let latentFrameCount: Int
    public let speed: Float
    public let seed: UInt64
    public let fps: Int

    public init(
        requestID: UUID = UUID(),
        prompt: String,
        actionSequence: [Wan2DreamXARTrajectorySegment],
        sourceImageURL: URL? = nil,
        outputURL: URL? = nil,
        width: Int = 1_280,
        height: Int = 704,
        latentFrameCount: Int = 21,
        speed: Float = Wan2DreamXARTrajectory.defaultSpeed,
        seed: UInt64 = 42,
        fps: Int = 16
    ) {
        self.requestID = requestID
        self.prompt = prompt
        self.actionSequence = actionSequence
        self.sourceImageURL = sourceImageURL
        self.outputURL = outputURL
        self.width = width
        self.height = height
        self.latentFrameCount = latentFrameCount
        self.speed = speed
        self.seed = seed
        self.fps = fps
    }

    public var expectedPixelFrameCount: Int {
        (latentFrameCount - 1) * 4 + 1
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
    public let generatedLatentFrameCount: Int
    public let retainedLatentFrameCount: Int
    public let causalCheckpointCount: Int
    public let currentWorldPose: Wan2DreamXWorldPose?
    public let sceneMemoryMode: Wan2DreamXSceneMemoryMode
    public let sceneMemoryFrameCount: Int
    public let sceneMemoryRetrievalCount: Int
    public let sceneMemoryRecycledFrameCount: Int
}

public struct Wan2WorldCausalCheckpointReceipt: Codable, Hashable, Sendable {
    public let checkpointID: UUID
    public let name: String?
    public let sessionID: UUID
    public let stateID: UUID
    public let currentFrameURL: URL
    public let transitionIndex: Int
    public let generatedLatentFrameCount: Int
    public let retainedLatentFrameCount: Int
    public let currentWorldPose: Wan2DreamXWorldPose
    public let sceneMemoryFrameCount: Int
    public let createdAt: Date
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
    public let terminalWorldPose: Wan2DreamXWorldPose?
    public let sceneMemoryMode: Wan2DreamXSceneMemoryMode
    public let sceneMemoryRetrievalCount: Int
    public let sceneMemoryRecycledFrameCount: Int
    public let seed: UInt64
}

public struct Wan2WorldRolloutChunkReceipt: Codable, Hashable, Sendable {
    public let blockIndex: Int
    public let blockCount: Int
    public let pixelFrameStart: Int
    public let pixelFrameCount: Int
    public let outputURL: URL
}

public struct Wan2WorldRolloutReceipt: Codable, Hashable, Sendable {
    public let requestID: UUID
    public let previousStateID: UUID?
    public let stateID: UUID
    public let transitionIndex: Int
    public let outputURL: URL
    public let terminalFrameURL: URL
    public let actionSequence: [Wan2DreamXARTrajectorySegment]
    public let latentFrameCount: Int
    public let pixelFrameCount: Int
    public let speed: Float
    public let conditioningMode: Wan2WorldConditioningMode
    public let chunks: [Wan2WorldRolloutChunkReceipt]
    public let terminalWorldPose: Wan2DreamXWorldPose
    public let sceneMemoryMode: Wan2DreamXSceneMemoryMode
    public let sceneMemoryRetrievalCount: Int
    public let sceneMemoryRecycledFrameCount: Int
    public let seed: UInt64
    public let fps: Int
}

public enum Wan2WorldSessionError: LocalizedError, Sendable {
    case busy
    case sourceImageRequired
    case terminalLatentMissing
    case causalRolloutRequiresDreamX
    case causalCheckpointRequiresDreamX
    case causalCheckpointStateUnavailable
    case causalCheckpointNotFound(UUID)

    public var errorDescription: String? {
        switch self {
        case .busy:
            return "The Wan world session is already generating a transition."
        case .sourceImageRequired:
            return "The first world transition requires a source image."
        case .terminalLatentMissing:
            return "Wan generation did not return the terminal-frame latent required for chaining."
        case .causalRolloutRequiresDreamX:
            return "Long causal rollouts require the DreamX autoregressive checkpoint."
        case .causalCheckpointRequiresDreamX:
            return "Exact causal checkpoints require the DreamX autoregressive checkpoint."
        case .causalCheckpointStateUnavailable:
            return "A causal checkpoint requires an active world state."
        case .causalCheckpointNotFound(let checkpointID):
            return "Causal checkpoint not found: \(checkpointID.uuidString)"
        }
    }
}

private struct Wan2StoredCausalCheckpoint: @unchecked Sendable {
    let receipt: Wan2WorldCausalCheckpointReceipt
    let runtime: Wan2CausalWorldCheckpoint
}

struct Wan2WorldArtifactSequence: Sendable {
    private(set) var lastAllocatedIndex = 0

    mutating func next(in stateDirectory: URL, fileManager: FileManager = .default) -> Int {
        repeat {
            lastAllocatedIndex += 1
        } while Self.hasArtifact(
            index: lastAllocatedIndex,
            in: stateDirectory,
            fileManager: fileManager
        )
        return lastAllocatedIndex
    }

    private static func hasArtifact(
        index: Int,
        in stateDirectory: URL,
        fileManager: FileManager
    ) -> Bool {
        let suffix = String(format: "%04d", index)
        return [
            "transition-\(suffix).mp4",
            "rollout-\(suffix).mp4",
            "rollout-\(suffix).chunks",
            "state-\(suffix).png",
        ].contains { name in
            fileManager.fileExists(
                atPath: stateDirectory.appendingPathComponent(name).path
            )
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
    private var causalCheckpoints: [UUID: Wan2StoredCausalCheckpoint] = [:]
    private var artifactSequence = Wan2WorldArtifactSequence()

    public init(
        resources: Wan2Resources,
        stateDirectory: URL,
        cameraWeightsURL: URL? = nil,
        causalWeightsURL: URL? = nil,
        sceneMemoryPolicy: Wan2DreamXSceneMemoryPolicy = .init(),
        sessionID: UUID = UUID()
    ) {
        precondition(cameraWeightsURL == nil || causalWeightsURL == nil)
        self.resources = resources
        self.stateDirectory = stateDirectory.standardizedFileURL
        self.sessionID = sessionID
        self.generator = Wan2TI2VGenerator(cameraWeightsURL: cameraWeightsURL)
        self.causalGenerator = causalWeightsURL.map {
            Wan2CausalWorldGenerator(weightsURL: $0, sceneMemoryPolicy: sceneMemoryPolicy)
        }
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
        let artifactIndex = artifactSequence.next(in: stateDirectory)
        let outputURL = request.outputURL ?? stateDirectory
            .appendingPathComponent(String(format: "transition-%04d.mp4", artifactIndex))
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
        if usesExplicitSource, transitionCount > 0 {
            causalGenerator?.reset()
        }
        phase = .generating
        do {
            let result: Wan2VideoGenerationResult
            let retrievalsBefore = causalGenerator?.sceneMemoryRetrievalCount ?? 0
            let recycledBefore = causalGenerator?.sceneMemoryRecycledFrameCount ?? 0
            if let causalGenerator {
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
                .appendingPathComponent(String(format: "state-%04d.png", artifactIndex))
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
                terminalWorldPose: causalGenerator?.worldPose,
                sceneMemoryMode: causalGenerator?.sceneMemoryMode ?? .disabled,
                sceneMemoryRetrievalCount: (causalGenerator?.sceneMemoryRetrievalCount ?? 0) - retrievalsBefore,
                sceneMemoryRecycledFrameCount: (causalGenerator?.sceneMemoryRecycledFrameCount ?? 0) - recycledBefore,
                seed: result.seed
            )
        } catch {
            phase = isWarm ? .ready : .cold
            throw error
        }
    }

    public func rollout(
        _ request: Wan2WorldRolloutRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)? = nil,
        chunkHandler: (@Sendable (Wan2WorldRolloutChunkReceipt) async -> Void)? = nil
    ) async throws -> Wan2WorldRolloutReceipt {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        guard let causalGenerator else { throw Wan2WorldSessionError.causalRolloutRequiresDreamX }
        try FileManager.default.createDirectory(at: stateDirectory, withIntermediateDirectories: true)

        let usesExplicitSource = request.sourceImageURL != nil
        guard let sourceImageURL = request.sourceImageURL ?? currentFrameURL else {
            throw Wan2WorldSessionError.sourceImageRequired
        }
        if usesExplicitSource, transitionCount > 0 {
            causalGenerator.reset()
        }

        let nextIndex = transitionCount + 1
        let artifactIndex = artifactSequence.next(in: stateDirectory)
        let outputURL = request.outputURL ?? stateDirectory
            .appendingPathComponent(String(format: "rollout-%04d.mp4", artifactIndex))
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let chunkDirectory = outputURL.deletingPathExtension()
            .appendingPathExtension("chunks")
        try FileManager.default.createDirectory(at: chunkDirectory, withIntermediateDirectories: true)

        phase = .generating
        do {
            var chunkReceipts: [Wan2WorldRolloutChunkReceipt] = []
            let retrievalsBefore = causalGenerator.sceneMemoryRetrievalCount
            let recycledBefore = causalGenerator.sceneMemoryRecycledFrameCount
            let result = try await causalGenerator.generateRollout(
                prompt: request.prompt,
                actionSequence: request.actionSequence,
                latentFrameCount: request.latentFrameCount,
                speed: request.speed,
                sourceImageURL: causalGenerator.latentFrameCount == 0 ? sourceImageURL : nil,
                resources: resources,
                width: request.width,
                height: request.height,
                seed: request.seed,
                progressHandler: progressHandler
            ) { chunk in
                let chunkURL = chunkDirectory.appendingPathComponent(
                    String(format: "chunk-%04d.mp4", chunk.blockIndex + 1)
                )
                try LTXVideoMP4Writer.writeMP4(frames: chunk.frames, fps: request.fps, to: chunkURL)
                let receipt = Wan2WorldRolloutChunkReceipt(
                    blockIndex: chunk.blockIndex,
                    blockCount: chunk.blockCount,
                    pixelFrameStart: chunk.pixelFrameStart,
                    pixelFrameCount: chunk.frames.dim(1),
                    outputURL: chunkURL
                )
                chunkReceipts.append(receipt)
                await chunkHandler?(receipt)
            }
            guard result.terminalFrameLatent != nil else {
                throw Wan2WorldSessionError.terminalLatentMissing
            }
            try LTXVideoMP4Writer.writeMP4(frames: result.frames, fps: request.fps, to: outputURL)

            let terminalFrameURL = stateDirectory
                .appendingPathComponent(String(format: "state-%04d.png", artifactIndex))
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
            return Wan2WorldRolloutReceipt(
                requestID: request.requestID,
                previousStateID: previousStateID,
                stateID: nextStateID,
                transitionIndex: nextIndex,
                outputURL: outputURL,
                terminalFrameURL: terminalFrameURL,
                actionSequence: request.actionSequence,
                latentFrameCount: request.latentFrameCount,
                pixelFrameCount: result.frames.dim(1),
                speed: request.speed,
                conditioningMode: conditioningMode,
                chunks: chunkReceipts,
                terminalWorldPose: causalGenerator.worldPose,
                sceneMemoryMode: causalGenerator.sceneMemoryMode,
                sceneMemoryRetrievalCount: causalGenerator.sceneMemoryRetrievalCount - retrievalsBefore,
                sceneMemoryRecycledFrameCount: causalGenerator.sceneMemoryRecycledFrameCount - recycledBefore,
                seed: result.seed,
                fps: request.fps
            )
        } catch {
            phase = isWarm ? .ready : .cold
            throw error
        }
    }

    public func createCausalCheckpoint(
        name: String? = nil,
        checkpointID: UUID = UUID(),
        createdAt: Date = Date()
    ) throws -> Wan2WorldCausalCheckpointReceipt {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        guard let causalGenerator else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        guard let stateID = currentStateID, let currentFrameURL else {
            throw Wan2WorldSessionError.causalCheckpointStateUnavailable
        }
        let runtime = causalGenerator.checkpoint()
        let receipt = Wan2WorldCausalCheckpointReceipt(
            checkpointID: checkpointID,
            name: name,
            sessionID: sessionID,
            stateID: stateID,
            currentFrameURL: currentFrameURL,
            transitionIndex: transitionCount,
            generatedLatentFrameCount: runtime.generatedLatentFrameCount,
            retainedLatentFrameCount: runtime.retainedLatentFrameCount,
            currentWorldPose: runtime.currentWorldPose,
            sceneMemoryFrameCount: runtime.sceneMemoryFrameCount,
            createdAt: createdAt
        )
        causalCheckpoints[checkpointID] = Wan2StoredCausalCheckpoint(
            receipt: receipt,
            runtime: runtime
        )
        return receipt
    }

    public func restoreCausalCheckpoint(
        _ checkpointID: UUID
    ) throws -> Wan2WorldCausalCheckpointReceipt {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        guard let causalGenerator else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        guard let checkpoint = causalCheckpoints[checkpointID] else {
            throw Wan2WorldSessionError.causalCheckpointNotFound(checkpointID)
        }
        causalGenerator.restore(checkpoint.runtime)
        currentStateID = checkpoint.receipt.stateID
        currentFrameURL = checkpoint.receipt.currentFrameURL
        phase = isWarm ? .ready : .cold
        return checkpoint.receipt
    }

    public func causalCheckpointReceipts() -> [Wan2WorldCausalCheckpointReceipt] {
        causalCheckpoints.values.map(\.receipt).sorted { lhs, rhs in
            if lhs.createdAt == rhs.createdAt {
                return lhs.checkpointID.uuidString < rhs.checkpointID.uuidString
            }
            return lhs.createdAt < rhs.createdAt
        }
    }

    public func discardCausalCheckpoint(
        _ checkpointID: UUID
    ) throws -> Wan2WorldCausalCheckpointReceipt {
        guard phase != .generating else { throw Wan2WorldSessionError.busy }
        guard let checkpoint = causalCheckpoints.removeValue(forKey: checkpointID) else {
            throw Wan2WorldSessionError.causalCheckpointNotFound(checkpointID)
        }
        return checkpoint.receipt
    }

    public func causalCheckpointFrameURL(_ checkpointID: UUID) throws -> URL {
        guard let checkpoint = causalCheckpoints[checkpointID] else {
            throw Wan2WorldSessionError.causalCheckpointNotFound(checkpointID)
        }
        return checkpoint.receipt.currentFrameURL
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
        causalCheckpoints.removeAll(keepingCapacity: true)
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
        causalCheckpoints.removeAll(keepingCapacity: true)
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
                ?? generator.hasChainedTerminalFrameLatent,
            generatedLatentFrameCount: causalGenerator?.latentFrameCount ?? 0,
            retainedLatentFrameCount: causalGenerator?.retainedLatentFrameCount ?? 0,
            causalCheckpointCount: causalCheckpoints.count,
            currentWorldPose: causalGenerator?.worldPose,
            sceneMemoryMode: causalGenerator?.sceneMemoryMode ?? .disabled,
            sceneMemoryFrameCount: causalGenerator?.sceneMemoryFrameCount ?? 0,
            sceneMemoryRetrievalCount: causalGenerator?.sceneMemoryRetrievalCount ?? 0,
            sceneMemoryRecycledFrameCount: causalGenerator?.sceneMemoryRecycledFrameCount ?? 0
        )
    }
}
