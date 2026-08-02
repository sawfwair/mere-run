import ArgumentParser
import Foundation
import Hummingbird
import MediaIO
import MereRunCore
import NIOCore
import NIOPosix

enum WorldBackend: String, CaseIterable, ExpressibleByArgument {
    case dreamx
    case cosmos3
}

struct World: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "world",
        abstract: "Run persistent local conditioned-video world sessions.",
        subcommands: [WorldServe.self]
    )
}

struct WorldServe: AsyncParsableCommand {
    private static let apiKeyEnvironmentKey = "MERERUN_API_KEY"

    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Serve one warm native world-model session over HTTP."
    )

    @Option(name: [.long], help: "Host to bind to.")
    var host = "127.0.0.1"

    @Option(name: [.long], help: "Port to listen on.")
    var port = 8791

    @Option(name: [.long], help: "Bearer token required by world endpoints. Also read from MERERUN_API_KEY.")
    var apiKey: String?

    @Option(name: [.long], help: "World runtime backend: dreamx or cosmos3.")
    var backend: WorldBackend = .dreamx

    @Option(name: [.long], help: "Wan2.2 TI2V base resource model id or local root.")
    var baseModel = Wan2Resources.modelID

    @Option(name: [.long], help: "Converted DreamX-World-5B AR model id or local root.")
    var model = Wan2DreamXCausalResources.modelID

    @Option(name: [.long], help: "Directory for transition videos and terminal state frames.")
    var stateDirectory: String?

    @Flag(name: [.long], help: "Load and warm all models before accepting requests.")
    var prepare = false

    @Flag(name: [.long], help: "Disable DreamX geometry-guided revisit memory.")
    var disableSceneMemory = false

    @Option(name: [.long], help: "Clean-latent recycling strength for DreamX revisits (0...1).")
    var sceneMemoryStrength: Float = 0.08

    @Option(name: [.long], help: "Maximum DreamX scene-memory latent frames retained in RAM.")
    var sceneMemoryMaxFrames = 96

    @Option(name: [.long], help: "Minimum latent-frame gap before a DreamX view can be retrieved.")
    var sceneMemoryMinimumGap = 3

    @Option(name: [.long], help: "Maximum yaw distance in degrees for a DreamX revisit match.")
    var sceneMemoryMaxYaw: Float = 2

    @Option(name: [.long], help: "Maximum model-space translation distance for a DreamX revisit match.")
    var sceneMemoryMaxTranslation: Float = 0.1

    @Option(name: [.long], help: "Yaw tolerance in degrees for an exact DreamX latent restore.")
    var sceneMemoryExactYaw: Float = 0.01

    @Option(name: [.long], help: "Model-space translation tolerance for an exact DreamX latent restore.")
    var sceneMemoryExactTranslation: Float = 0.001

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)
        let resolvedKey = resolvedAPIKey()
        if !Self.isLoopback(host), resolvedKey?.isEmpty != false {
            throw ValidationError("Non-loopback world servers require --api-key or MERERUN_API_KEY.")
        }
        let stateURL = stateDirectory.map { URL(fileURLWithPath: $0).standardizedFileURL }
            ?? Self.defaultStateDirectory()
        let sessionBackend: WorldSessionBackend
        switch backend {
        case .dreamx:
            let baseRoot = try resolveRoot(baseModel)
            let causalRoot = try resolveRoot(model)
            let baseResources = Wan2Resources(rootURL: baseRoot)
            let missingBase = baseResources.validate()
            guard missingBase.isEmpty else {
                throw ValidationError(
                    "Wan2.2 base resources are incomplete: "
                        + missingBase.map(\.path).joined(separator: ", ")
                )
            }
            let causalResources = Wan2DreamXCausalResources(rootURL: causalRoot)
            let missingCausal = causalResources.validate()
            guard missingCausal.isEmpty else {
                throw ValidationError(
                    "DreamX causal resources are incomplete: "
                        + missingCausal.map(\.path).joined(separator: ", ")
                )
            }
            sessionBackend = .dreamx(Wan2WorldSession(
                resources: baseResources,
                stateDirectory: stateURL,
                causalWeightsURL: causalResources.weightsURL,
                sceneMemoryPolicy: try resolvedSceneMemoryPolicy()
            ))
        case .cosmos3:
            let requestedModel = model == Wan2DreamXCausalResources.modelID
                ? Cosmos3Resources.modelID
                : model
            let root = try resolveRoot(requestedModel)
            let resources = Cosmos3Resources(rootURL: root)
            let missing = resources.validate()
            guard missing.isEmpty else {
                throw ValidationError(
                    "Cosmos3-Edge resources are incomplete: "
                        + missing.map(\.path).joined(separator: ", ")
                )
            }
            sessionBackend = .cosmos3(Cosmos3WorldSession(
                resources: resources,
                stateDirectory: stateURL
            ))
        }
        let runtime = WorldHTTPRuntime(session: sessionBackend)
        if prepare {
            try await runtime.prepare()
        }
        let server = WorldHTTPServer(runtime: runtime, apiKey: resolvedKey)
        try await server.run(host: host, port: port)
    }

    func resolvedAPIKey(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String? {
        if let apiKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        if let apiKey = environment[Self.apiKeyEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !apiKey.isEmpty {
            return apiKey
        }
        return nil
    }

    func resolvedSceneMemoryPolicy() throws -> Wan2DreamXSceneMemoryPolicy {
        if disableSceneMemory { return .disabled }
        guard sceneMemoryStrength.isFinite,
              (0...1).contains(sceneMemoryStrength) else {
            throw ValidationError("--scene-memory-strength must be finite and between 0 and 1.")
        }
        guard sceneMemoryMaxFrames >= 0 else {
            throw ValidationError("--scene-memory-max-frames must be non-negative.")
        }
        guard sceneMemoryMinimumGap >= 0 else {
            throw ValidationError("--scene-memory-minimum-gap must be non-negative.")
        }
        guard sceneMemoryMaxYaw.isFinite, sceneMemoryMaxYaw >= 0 else {
            throw ValidationError("--scene-memory-max-yaw must be finite and non-negative.")
        }
        guard sceneMemoryMaxTranslation.isFinite, sceneMemoryMaxTranslation >= 0 else {
            throw ValidationError("--scene-memory-max-translation must be finite and non-negative.")
        }
        guard sceneMemoryExactYaw.isFinite,
              sceneMemoryExactYaw >= 0,
              sceneMemoryExactYaw <= sceneMemoryMaxYaw else {
            throw ValidationError(
                "--scene-memory-exact-yaw must be finite, non-negative, and no greater than --scene-memory-max-yaw."
            )
        }
        guard sceneMemoryExactTranslation.isFinite,
              sceneMemoryExactTranslation >= 0,
              sceneMemoryExactTranslation <= sceneMemoryMaxTranslation else {
            throw ValidationError(
                "--scene-memory-exact-translation must be finite, non-negative, and no greater than --scene-memory-max-translation."
            )
        }
        return Wan2DreamXSceneMemoryPolicy(
            maximumFrameCount: sceneMemoryMaxFrames,
            minimumFrameGap: sceneMemoryMinimumGap,
            maximumYawDistanceDegrees: sceneMemoryMaxYaw,
            maximumTranslationDistance: sceneMemoryMaxTranslation,
            exactRevisitMaximumYawDistanceDegrees: sceneMemoryExactYaw,
            exactRevisitMaximumTranslationDistance: sceneMemoryExactTranslation,
            recyclingStrength: sceneMemoryStrength
        )
    }

    private func resolveRoot(_ value: String) throws -> URL {
        let expanded = NSString(string: value).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory), isDirectory.boolValue {
            return URL(fileURLWithPath: expanded).standardizedFileURL
        }
        guard let modelID = ModelResolver.ModelID(rawValue: value) else {
            throw ValidationError("World model is neither an installed model id nor a local directory: \(value)")
        }
        return try ModelResolver().resolve(modelID).rootURL
    }

    private static func isLoopback(_ host: String) -> Bool {
        let normalized = host.lowercased()
        return normalized == "localhost" || normalized == "127.0.0.1" || normalized == "::1"
    }

    private static func defaultStateDirectory() -> URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("world-sessions", isDirectory: true)
            .appendingPathComponent("default", isDirectory: true)
    }
}

private struct WorldRequestContext: RequestContext, RemoteAddressRequestContext {
    var coreContext: CoreRequestContextStorage
    let remoteAddress: SocketAddress?

    init(source: Source) {
        coreContext = CoreRequestContextStorage(source: source)
        remoteAddress = source.channel.remoteAddress
    }
}

struct WorldTransitionPayload: Codable, Sendable {
    let prompt: String
    let camera: Wan2WorldCameraControl
    let sourceImage: String?
    let output: String?
    let width: Int?
    let height: Int?
    let numFrames: Int?
    let steps: Int?
    let guidanceScale: Float?
    let shift: Float?
    let seed: UInt64?
    let fps: Int?
    let modelSpaceActions: [[Float]]?

    fileprivate func request(defaults: WorldTransitionDefaults) -> WorldTransitionRuntimeRequest {
        return WorldTransitionRuntimeRequest(
            base: Wan2WorldTransitionRequest(
                prompt: prompt,
                camera: camera,
                sourceImageURL: sourceImage.map { URL(fileURLWithPath: $0).standardizedFileURL },
                outputURL: output.map { URL(fileURLWithPath: $0).standardizedFileURL },
                width: width ?? defaults.width,
                height: height ?? defaults.height,
                numFrames: numFrames ?? defaults.numFrames,
                steps: steps ?? defaults.steps,
                guidanceScale: guidanceScale ?? defaults.guidanceScale,
                shift: shift ?? defaults.shift,
                seed: seed ?? defaults.seed,
                fps: fps ?? defaults.fps
            ),
            modelSpaceActions: modelSpaceActions
        )
    }

}

struct WorldRolloutPayload: Codable, Sendable {
    let prompt: String
    let actionSeq: [String]
    let actionSpeedList: [Float]?
    let sourceImage: String?
    let output: String?
    let width: Int?
    let height: Int?
    let numOutputFrames: Int?
    let speed: Float?
    let seed: UInt64?
    let fps: Int?

    func request() throws -> Wan2WorldRolloutRequest {
        let prompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { throw WorldRolloutValidationError.emptyPrompt }
        guard !actionSeq.isEmpty else { throw WorldRolloutValidationError.emptyActionSequence }
        let weights = actionSpeedList ?? Array(repeating: 1, count: actionSeq.count)
        guard weights.count == actionSeq.count else {
            throw WorldRolloutValidationError.mismatchedActionWeights(
                actions: actionSeq.count,
                weights: weights.count
            )
        }
        for weight in weights where !weight.isFinite || weight <= 0 {
            throw WorldRolloutValidationError.invalidActionWeight(weight)
        }
        let totalWeight = weights.reduce(Float(0), +)
        guard totalWeight.isFinite, totalWeight > 0 else {
            throw WorldRolloutValidationError.invalidActionWeight(totalWeight)
        }
        let normalizedActions = try actionSeq.map(Self.validatedAction)
        let width = width ?? 1_280
        let height = height ?? 704
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            throw WorldRolloutValidationError.invalidResolution(width: width, height: height)
        }
        guard width <= 1_280, height <= 704 else {
            throw WorldRolloutValidationError.resolutionExceedsMaximum(width: width, height: height)
        }
        let latentFrameCount = numOutputFrames ?? 21
        guard latentFrameCount > 0, latentFrameCount.isMultiple(of: 3) else {
            throw WorldRolloutValidationError.invalidLatentFrameCount(latentFrameCount)
        }
        guard latentFrameCount <= Wan2CausalWorldGenerator.maximumRolloutLatentFrameCount else {
            throw WorldRolloutValidationError.latentFrameCountExceedsMaximum(latentFrameCount)
        }
        guard normalizedActions.count <= (latentFrameCount - 1) * 4 + 1 else {
            throw WorldRolloutValidationError.tooManyActions(
                actions: normalizedActions.count,
                pixelFrames: (latentFrameCount - 1) * 4 + 1
            )
        }
        let speed = speed ?? Wan2DreamXARTrajectory.defaultSpeed
        guard speed.isFinite, speed > 0 else {
            throw WorldRolloutValidationError.invalidSpeed(speed)
        }
        let fps = fps ?? 16
        guard fps > 0 else { throw WorldRolloutValidationError.invalidFPS(fps) }

        return Wan2WorldRolloutRequest(
            prompt: prompt,
            actionSequence: zip(normalizedActions, weights).map {
                Wan2DreamXARTrajectorySegment(action: $0.0, weight: $0.1)
            },
            sourceImageURL: sourceImage.map { URL(fileURLWithPath: $0).standardizedFileURL },
            outputURL: output.map { URL(fileURLWithPath: $0).standardizedFileURL },
            width: width,
            height: height,
            latentFrameCount: latentFrameCount,
            speed: speed,
            seed: seed ?? 42,
            fps: fps
        )
    }

    private static func validatedAction(_ raw: String) throws -> String {
        let action = raw.lowercased()
        let allowed = Set("wasdijkl ")
        guard !action.isEmpty, action.allSatisfy(allowed.contains) else {
            throw WorldRolloutValidationError.invalidAction(raw)
        }
        let contradictoryPairs: [(Character, Character)] = [
            ("w", "s"), ("a", "d"), ("i", "k"), ("j", "l"),
        ]
        guard !contradictoryPairs.contains(where: { action.contains($0.0) && action.contains($0.1) }) else {
            throw WorldRolloutValidationError.contradictoryAction(raw)
        }
        return action
    }
}

enum WorldRolloutValidationError: LocalizedError, Equatable {
    case emptyPrompt
    case emptyActionSequence
    case mismatchedActionWeights(actions: Int, weights: Int)
    case invalidActionWeight(Float)
    case invalidAction(String)
    case contradictoryAction(String)
    case invalidResolution(width: Int, height: Int)
    case resolutionExceedsMaximum(width: Int, height: Int)
    case invalidLatentFrameCount(Int)
    case latentFrameCountExceedsMaximum(Int)
    case tooManyActions(actions: Int, pixelFrames: Int)
    case invalidSpeed(Float)
    case invalidFPS(Int)

    var errorDescription: String? {
        switch self {
        case .emptyPrompt:
            "DreamX rollout prompt must not be empty."
        case .emptyActionSequence:
            "DreamX rollout action_seq must contain at least one action."
        case .mismatchedActionWeights(let actions, let weights):
            "DreamX rollout action_speed_list count (\(weights)) must match action_seq count (\(actions))."
        case .invalidActionWeight(let weight):
            "DreamX rollout action weights must be finite and positive; received \(weight)."
        case .invalidAction(let action):
            "DreamX action '\(action)' contains unsupported keys; use composed WASD and IJKL controls."
        case .contradictoryAction(let action):
            "DreamX action '\(action)' contains opposing controls on the same axis."
        case .invalidResolution(let width, let height):
            "DreamX rollout resolution must be positive and divisible by 32; received \(width)x\(height)."
        case .resolutionExceedsMaximum(let width, let height):
            "DreamX rollout resolution may not exceed the official 1280x704 geometry; received \(width)x\(height)."
        case .invalidLatentFrameCount(let count):
            "DreamX num_output_frames is a latent-frame count and must be positive and divisible by 3; received \(count)."
        case .latentFrameCountExceedsMaximum(let count):
            "DreamX num_output_frames may not exceed \(Wan2CausalWorldGenerator.maximumRolloutLatentFrameCount); received \(count)."
        case .tooManyActions(let actions, let pixelFrames):
            "DreamX action_seq has \(actions) entries but the rollout has only \(pixelFrames) pixel frames."
        case .invalidSpeed(let speed):
            "DreamX rollout speed must be finite and positive; received \(speed)."
        case .invalidFPS(let fps):
            "DreamX rollout fps must be positive; received \(fps)."
        }
    }
}

private struct WorldTransitionRuntimeRequest: Sendable {
    let base: Wan2WorldTransitionRequest
    let modelSpaceActions: [[Float]]?
}

private struct WorldTransitionDefaults: Sendable {
    let width: Int
    let height: Int
    let numFrames: Int
    let steps: Int
    let guidanceScale: Float
    let shift: Float
    let seed: UInt64
    let fps: Int
}

private struct WorldResetPayload: Codable, Sendable {
    let sourceImage: String?
}

private struct WorldCausalCheckpointRequest: Codable, Sendable {
    let name: String?
}

private enum WorldJobStatus: String, Codable, Sendable {
    case queued
    case generating
    case cancelling
    case completed
    case cancelled
    case failed

    var isActive: Bool {
        self == .queued || self == .generating || self == .cancelling
    }
}

private struct WorldJobProgress: Codable, Sendable {
    let stage: String
    let stepIndex: Int
    let totalSteps: Int
    let updatedAt: Date

    enum CodingKeys: String, CodingKey {
        case stage
        case stepIndex = "step_index"
        case totalSteps = "total_steps"
        case updatedAt = "updated_at"
    }
}

private struct WorldTransitionReceiptPayload: Codable, Sendable {
    let requestID: UUID
    let previousStateID: UUID?
    let stateID: UUID
    let transitionIndex: Int
    let outputURL: URL
    let terminalFrameURL: URL
    let camera: Wan2WorldCameraControl
    let conditioningMode: String
    let actionDomain: Cosmos3ActionDomain?
    let actionSpace: String?
    let modelSpaceActions: [[Float]]?
    let rawActions: [[Float]]?
    let terminalWorldPose: Wan2DreamXWorldPose?
    let sceneMemoryMode: String?
    let sceneMemoryRetrievalCount: Int?
    let sceneMemoryRecycledFrameCount: Int?
    let seed: UInt64

    enum CodingKeys: String, CodingKey {
        case camera, seed
        case requestID = "request_id"
        case previousStateID = "previous_state_id"
        case stateID = "state_id"
        case transitionIndex = "transition_index"
        case outputURL = "output_url"
        case terminalFrameURL = "terminal_frame_url"
        case conditioningMode = "conditioning_mode"
        case actionDomain = "action_domain"
        case actionSpace = "action_space"
        case modelSpaceActions = "model_space_actions"
        case rawActions = "raw_actions"
        case terminalWorldPose = "terminal_world_pose"
        case sceneMemoryMode = "scene_memory_mode"
        case sceneMemoryRetrievalCount = "scene_memory_retrieval_count"
        case sceneMemoryRecycledFrameCount = "scene_memory_recycled_frame_count"
    }
}

private struct WorldRolloutChunkPayload: Codable, Sendable {
    let blockIndex: Int
    let blockCount: Int
    let pixelFrameStart: Int
    let pixelFrameCount: Int
    let outputURL: URL
    let mediaPath: String

    enum CodingKeys: String, CodingKey {
        case blockIndex = "block_index"
        case blockCount = "block_count"
        case pixelFrameStart = "pixel_frame_start"
        case pixelFrameCount = "pixel_frame_count"
        case outputURL = "output_url"
        case mediaPath = "media_path"
    }
}

private struct WorldRolloutReceiptPayload: Codable, Sendable {
    let requestID: UUID
    let previousStateID: UUID?
    let stateID: UUID
    let transitionIndex: Int
    let outputURL: URL
    let outputMediaPath: String
    let terminalFrameURL: URL
    let terminalFrameMediaPath: String
    let actionSeq: [String]
    let actionSpeedList: [Float]
    let latentFrameCount: Int
    let pixelFrameCount: Int
    let speed: Float
    let conditioningMode: String
    let chunks: [WorldRolloutChunkPayload]
    let terminalWorldPose: Wan2DreamXWorldPose
    let sceneMemoryMode: String
    let sceneMemoryRetrievalCount: Int
    let sceneMemoryRecycledFrameCount: Int
    let seed: UInt64
    let fps: Int

    enum CodingKeys: String, CodingKey {
        case speed, chunks, seed, fps
        case requestID = "request_id"
        case previousStateID = "previous_state_id"
        case stateID = "state_id"
        case transitionIndex = "transition_index"
        case outputURL = "output_url"
        case outputMediaPath = "output_media_path"
        case terminalFrameURL = "terminal_frame_url"
        case terminalFrameMediaPath = "terminal_frame_media_path"
        case actionSeq = "action_seq"
        case actionSpeedList = "action_speed_list"
        case latentFrameCount = "latent_frame_count"
        case pixelFrameCount = "pixel_frame_count"
        case conditioningMode = "conditioning_mode"
        case terminalWorldPose = "terminal_world_pose"
        case sceneMemoryMode = "scene_memory_mode"
        case sceneMemoryRetrievalCount = "scene_memory_retrieval_count"
        case sceneMemoryRecycledFrameCount = "scene_memory_recycled_frame_count"
    }
}

private struct WorldJobSnapshot: Codable, Sendable {
    let jobID: UUID
    let status: WorldJobStatus
    let createdAt: Date
    let startedAt: Date?
    let completedAt: Date?
    let progress: WorldJobProgress?
    let progressEvents: [WorldJobProgress]
    let chunks: [WorldRolloutChunkPayload]
    let receipt: WorldTransitionReceiptPayload?
    let rolloutReceipt: WorldRolloutReceiptPayload?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status, progress, chunks, receipt, error
        case progressEvents = "progress_events"
        case rolloutReceipt = "rollout_receipt"
        case jobID = "job_id"
        case createdAt = "created_at"
        case startedAt = "started_at"
        case completedAt = "completed_at"
    }
}

private struct WorldSessionSnapshotPayload: Codable, Sendable {
    let sessionID: UUID
    let phase: String
    let transitionCount: Int
    let currentStateID: UUID?
    let currentFrameURL: URL?
    let conditioningMode: String
    let keepsModelsWarm: Bool
    let keepsTerminalLatent: Bool
    let generatedLatentFrameCount: Int?
    let retainedLatentFrameCount: Int?
    let causalCheckpointCount: Int?
    let currentWorldPose: Wan2DreamXWorldPose?
    let sceneMemoryMode: String?
    let sceneMemoryPolicy: Wan2DreamXSceneMemoryPolicy?
    let sceneMemoryFrameCount: Int?
    let sceneMemoryRetrievalCount: Int?
    let sceneMemoryRecycledFrameCount: Int?

    enum CodingKeys: String, CodingKey {
        case phase
        case sessionID = "session_id"
        case transitionCount = "transition_count"
        case currentStateID = "current_state_id"
        case currentFrameURL = "current_frame_url"
        case conditioningMode = "conditioning_mode"
        case keepsModelsWarm = "keeps_models_warm"
        case keepsTerminalLatent = "keeps_terminal_latent"
        case generatedLatentFrameCount = "generated_latent_frame_count"
        case retainedLatentFrameCount = "retained_latent_frame_count"
        case causalCheckpointCount = "causal_checkpoint_count"
        case currentWorldPose = "current_world_pose"
        case sceneMemoryMode = "scene_memory_mode"
        case sceneMemoryPolicy = "scene_memory_policy"
        case sceneMemoryFrameCount = "scene_memory_frame_count"
        case sceneMemoryRetrievalCount = "scene_memory_retrieval_count"
        case sceneMemoryRecycledFrameCount = "scene_memory_recycled_frame_count"
    }
}

private struct WorldRuntimeSnapshot: Codable, Sendable {
    let object = "world.session"
    let backend: String
    let session: WorldSessionSnapshotPayload
    let activeJob: WorldJobSnapshot?

    enum CodingKeys: String, CodingKey {
        case object, backend, session
        case activeJob = "active_job"
    }
}

private struct WorldHealth: Encodable, Sendable {
    let status = "ok"
    let object = "world.health"
}

private struct WorldErrorPayload: Codable, Sendable {
    let error: String
}

private enum WorldHTTPRuntimeError: LocalizedError {
    case busy
    case jobNotFound
    case rolloutRequiresDreamX
    case mediaNotFound

    var errorDescription: String? {
        switch self {
        case .busy: "A world transition is already active."
        case .jobNotFound: "World transition job not found."
        case .rolloutRequiresDreamX: "The action_seq rollout endpoint requires the DreamX backend."
        case .mediaNotFound: "World job media not found."
        }
    }
}

private enum WorldSessionBackend: Sendable {
    case dreamx(Wan2WorldSession)
    case cosmos3(Cosmos3WorldSession)

    var identifier: String {
        switch self {
        case .dreamx: "dreamx"
        case .cosmos3: "cosmos3"
        }
    }

    func stateDirectory() async -> URL {
        switch self {
        case .dreamx(let session): await session.stateDirectory
        case .cosmos3(let session): await session.stateDirectory
        }
    }

    var transitionDefaults: WorldTransitionDefaults {
        switch self {
        case .dreamx:
            return WorldTransitionDefaults(
                width: 512,
                height: 320,
                numFrames: 17,
                steps: 40,
                guidanceScale: 5,
                shift: 5,
                seed: 42,
                fps: 24
            )
        case .cosmos3:
            return WorldTransitionDefaults(
                width: 320,
                height: 176,
                numFrames: 17,
                steps: 30,
                guidanceScale: 1,
                shift: 3,
                seed: 0,
                fps: 30
            )
        }
    }

    func prepare() async throws {
        switch self {
        case .dreamx(let session):
            try await session.prepare()
        case .cosmos3(let session):
            try await session.prepare()
        }
    }

    func transition(
        _ runtimeRequest: WorldTransitionRuntimeRequest,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> WorldTransitionReceiptPayload {
        let request = runtimeRequest.base
        switch self {
        case .dreamx(let session):
            let receipt = try await session.transition(
                request,
                progressHandler: progressHandler
            )
            return WorldTransitionReceiptPayload(
                requestID: receipt.requestID,
                previousStateID: receipt.previousStateID,
                stateID: receipt.stateID,
                transitionIndex: receipt.transitionIndex,
                outputURL: receipt.outputURL,
                terminalFrameURL: receipt.terminalFrameURL,
                camera: receipt.camera,
                conditioningMode: receipt.conditioningMode.rawValue,
                actionDomain: nil,
                actionSpace: nil,
                modelSpaceActions: nil,
                rawActions: nil,
                terminalWorldPose: receipt.terminalWorldPose,
                sceneMemoryMode: receipt.sceneMemoryMode.rawValue,
                sceneMemoryRetrievalCount: receipt.sceneMemoryRetrievalCount,
                sceneMemoryRecycledFrameCount: receipt.sceneMemoryRecycledFrameCount,
                seed: receipt.seed
            )
        case .cosmos3(let session):
            let receipt = try await session.transition(
                Cosmos3WorldTransitionRequest(
                    requestID: request.requestID,
                    prompt: request.prompt,
                    camera: request.camera,
                    sourceImageURL: request.sourceImageURL,
                    outputURL: request.outputURL,
                    width: request.width,
                    height: request.height,
                    numFrames: request.numFrames,
                    steps: request.steps,
                    guidanceScale: request.guidanceScale,
                    shift: request.shift,
                    seed: request.seed,
                    fps: request.fps,
                    modelSpaceActions: runtimeRequest.modelSpaceActions
                ),
                progressHandler: progressHandler
            )
            return WorldTransitionReceiptPayload(
                requestID: receipt.requestID,
                previousStateID: receipt.previousStateID,
                stateID: receipt.stateID,
                transitionIndex: receipt.transitionIndex,
                outputURL: receipt.outputURL,
                terminalFrameURL: receipt.terminalFrameURL,
                camera: receipt.camera,
                conditioningMode: receipt.conditioningMode,
                actionDomain: receipt.actionDomain,
                actionSpace: receipt.actionSpace,
                modelSpaceActions: receipt.modelSpaceActions,
                rawActions: receipt.modelSpaceActions,
                terminalWorldPose: nil,
                sceneMemoryMode: nil,
                sceneMemoryRetrievalCount: nil,
                sceneMemoryRecycledFrameCount: nil,
                seed: receipt.seed
            )
        }
    }

    func rollout(
        _ request: Wan2WorldRolloutRequest,
        jobID: UUID,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?,
        chunkHandler: (@Sendable (WorldRolloutChunkPayload) async -> Void)?
    ) async throws -> WorldRolloutReceiptPayload {
        guard case .dreamx(let session) = self else {
            throw WorldHTTPRuntimeError.rolloutRequiresDreamX
        }
        let mediaBase = "/v1/world/jobs/\(jobID.uuidString.lowercased())/media"
        let receipt = try await session.rollout(
            request,
            progressHandler: progressHandler
        ) { chunk in
            await chunkHandler?(WorldRolloutChunkPayload(
                blockIndex: chunk.blockIndex,
                blockCount: chunk.blockCount,
                pixelFrameStart: chunk.pixelFrameStart,
                pixelFrameCount: chunk.pixelFrameCount,
                outputURL: chunk.outputURL,
                mediaPath: "\(mediaBase)/chunks/\(chunk.blockIndex)"
            ))
        }
        return WorldRolloutReceiptPayload(
            requestID: receipt.requestID,
            previousStateID: receipt.previousStateID,
            stateID: receipt.stateID,
            transitionIndex: receipt.transitionIndex,
            outputURL: receipt.outputURL,
            outputMediaPath: "\(mediaBase)/output",
            terminalFrameURL: receipt.terminalFrameURL,
            terminalFrameMediaPath: "\(mediaBase)/terminal-frame",
            actionSeq: receipt.actionSequence.map(\.action),
            actionSpeedList: receipt.actionSequence.map(\.weight),
            latentFrameCount: receipt.latentFrameCount,
            pixelFrameCount: receipt.pixelFrameCount,
            speed: receipt.speed,
            conditioningMode: receipt.conditioningMode.rawValue,
            chunks: receipt.chunks.map { chunk in
                WorldRolloutChunkPayload(
                    blockIndex: chunk.blockIndex,
                    blockCount: chunk.blockCount,
                    pixelFrameStart: chunk.pixelFrameStart,
                    pixelFrameCount: chunk.pixelFrameCount,
                    outputURL: chunk.outputURL,
                    mediaPath: "\(mediaBase)/chunks/\(chunk.blockIndex)"
                )
            },
            terminalWorldPose: receipt.terminalWorldPose,
            sceneMemoryMode: receipt.sceneMemoryMode.rawValue,
            sceneMemoryRetrievalCount: receipt.sceneMemoryRetrievalCount,
            sceneMemoryRecycledFrameCount: receipt.sceneMemoryRecycledFrameCount,
            seed: receipt.seed,
            fps: receipt.fps
        )
    }

    func reset(sourceImageURL: URL?) async throws {
        switch self {
        case .dreamx(let session):
            try await session.reset(sourceImageURL: sourceImageURL)
        case .cosmos3(let session):
            try await session.reset(sourceImageURL: sourceImageURL)
        }
    }

    func createCausalCheckpoint(name: String?) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard case .dreamx(let session) = self else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        return try await session.createCausalCheckpoint(name: name)
    }

    func restoreCausalCheckpoint(
        _ checkpointID: UUID
    ) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard case .dreamx(let session) = self else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        return try await session.restoreCausalCheckpoint(checkpointID)
    }

    func causalCheckpointReceipts() async throws -> [Wan2WorldCausalCheckpointReceipt] {
        guard case .dreamx(let session) = self else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        return await session.causalCheckpointReceipts()
    }

    func discardCausalCheckpoint(
        _ checkpointID: UUID
    ) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard case .dreamx(let session) = self else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        return try await session.discardCausalCheckpoint(checkpointID)
    }

    func causalCheckpointFrameURL(_ checkpointID: UUID) async throws -> URL {
        guard case .dreamx(let session) = self else {
            throw Wan2WorldSessionError.causalCheckpointRequiresDreamX
        }
        return try await session.causalCheckpointFrameURL(checkpointID)
    }

    func unload() async throws {
        switch self {
        case .dreamx(let session):
            try await session.unload()
        case .cosmos3(let session):
            try await session.unload()
        }
    }

    func snapshot() async -> WorldSessionSnapshotPayload {
        switch self {
        case .dreamx(let session):
            let snapshot = await session.snapshot()
            return WorldSessionSnapshotPayload(
                sessionID: snapshot.sessionID,
                phase: snapshot.phase.rawValue,
                transitionCount: snapshot.transitionCount,
                currentStateID: snapshot.currentStateID,
                currentFrameURL: snapshot.currentFrameURL,
                conditioningMode: snapshot.conditioningMode.rawValue,
                keepsModelsWarm: snapshot.keepsModelsWarm,
                keepsTerminalLatent: snapshot.keepsTerminalLatent,
                generatedLatentFrameCount: snapshot.generatedLatentFrameCount,
                retainedLatentFrameCount: snapshot.retainedLatentFrameCount,
                causalCheckpointCount: snapshot.causalCheckpointCount,
                currentWorldPose: snapshot.currentWorldPose,
                sceneMemoryMode: snapshot.sceneMemoryMode.rawValue,
                sceneMemoryPolicy: snapshot.sceneMemoryPolicy,
                sceneMemoryFrameCount: snapshot.sceneMemoryFrameCount,
                sceneMemoryRetrievalCount: snapshot.sceneMemoryRetrievalCount,
                sceneMemoryRecycledFrameCount: snapshot.sceneMemoryRecycledFrameCount
            )
        case .cosmos3(let session):
            let snapshot = await session.snapshot()
            return WorldSessionSnapshotPayload(
                sessionID: snapshot.sessionID,
                phase: snapshot.phase.rawValue,
                transitionCount: snapshot.transitionCount,
                currentStateID: snapshot.currentStateID,
                currentFrameURL: snapshot.currentFrameURL,
                conditioningMode: snapshot.conditioningMode,
                keepsModelsWarm: snapshot.keepsModelsWarm,
                keepsTerminalLatent: snapshot.keepsTerminalLatent,
                generatedLatentFrameCount: nil,
                retainedLatentFrameCount: nil,
                causalCheckpointCount: nil,
                currentWorldPose: nil,
                sceneMemoryMode: nil,
                sceneMemoryPolicy: nil,
                sceneMemoryFrameCount: nil,
                sceneMemoryRetrievalCount: nil,
                sceneMemoryRecycledFrameCount: nil
            )
        }
    }
}

private actor WorldHTTPRuntime {
    private let session: WorldSessionBackend
    private var jobs: [UUID: WorldJobSnapshot] = [:]
    private var tasks: [UUID: Task<Void, Never>] = [:]

    init(session: WorldSessionBackend) {
        self.session = session
    }

    func prepare() async throws {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.prepare()
    }

    func snapshot() async -> WorldRuntimeSnapshot {
        WorldRuntimeSnapshot(
            backend: session.identifier,
            session: await session.snapshot(),
            activeJob: activeJob
        )
    }

    func startTransition(_ payload: WorldTransitionPayload) throws -> WorldJobSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        let id = UUID()
        let job = WorldJobSnapshot(
            jobID: id,
            status: .queued,
            createdAt: Date(),
            startedAt: nil,
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            chunks: [],
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
        jobs[id] = job
        let defaults = session.transitionDefaults
        tasks[id] = Task { [weak self] in
            await self?.runTransition(id: id, request: payload.request(defaults: defaults))
        }
        return job
    }

    func startRollout(_ payload: WorldRolloutPayload) throws -> WorldJobSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        let request = try payload.request()
        let id = UUID()
        let job = WorldJobSnapshot(
            jobID: id,
            status: .queued,
            createdAt: Date(),
            startedAt: nil,
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            chunks: [],
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
        jobs[id] = job
        tasks[id] = Task { [weak self] in
            await self?.runRollout(id: id, request: request)
        }
        return job
    }

    func job(id: UUID) throws -> WorldJobSnapshot {
        guard let job = jobs[id] else { throw WorldHTTPRuntimeError.jobNotFound }
        return job
    }

    func cancel(id: UUID) throws -> WorldJobSnapshot {
        guard let job = jobs[id] else { throw WorldHTTPRuntimeError.jobNotFound }
        guard job.status.isActive else { return job }
        tasks[id]?.cancel()
        let updated = WorldJobSnapshot(
            jobID: job.jobID,
            status: .cancelling,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: nil,
            progress: job.progress,
            progressEvents: job.progressEvents,
            chunks: job.chunks,
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
        jobs[id] = updated
        return updated
    }

    func reset(_ payload: WorldResetPayload?) async throws -> WorldRuntimeSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.reset(sourceImageURL: payload?.sourceImage.map {
            URL(fileURLWithPath: $0).standardizedFileURL
        })
        return await snapshot()
    }

    func createCausalCheckpoint(
        _ payload: WorldCausalCheckpointRequest?
    ) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        return try await session.createCausalCheckpoint(name: payload?.name)
    }

    func restoreCausalCheckpoint(
        _ checkpointID: UUID
    ) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        return try await session.restoreCausalCheckpoint(checkpointID)
    }

    func causalCheckpointReceipts() async throws -> [Wan2WorldCausalCheckpointReceipt] {
        try await session.causalCheckpointReceipts()
    }

    func discardCausalCheckpoint(
        _ checkpointID: UUID
    ) async throws -> Wan2WorldCausalCheckpointReceipt {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        return try await session.discardCausalCheckpoint(checkpointID)
    }

    func causalCheckpointFrameURL(_ checkpointID: UUID) async throws -> URL {
        try await session.causalCheckpointFrameURL(checkpointID)
    }

    func seedSourceImage(_ data: Data) async throws -> WorldRuntimeSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        let image = try MediaImageIO.decode(data: data)
        let stateDirectory = await session.stateDirectory()
        try FileManager.default.createDirectory(
            at: stateDirectory,
            withIntermediateDirectories: true
        )
        let url = stateDirectory
            .appendingPathComponent("source-\(UUID().uuidString.lowercased())")
            .appendingPathExtension("png")
        try MediaImageIO.writePNG(image, to: url)
        try await session.reset(sourceImageURL: url)
        return await snapshot()
    }

    func unload() async throws -> WorldRuntimeSnapshot {
        guard activeJob == nil else { throw WorldHTTPRuntimeError.busy }
        try await session.unload()
        return await snapshot()
    }

    func mediaURL(id: UUID, asset: String, chunkIndex: Int? = nil) throws -> URL {
        guard let job = jobs[id] else { throw WorldHTTPRuntimeError.jobNotFound }
        switch asset {
        case "output":
            guard let url = job.rolloutReceipt?.outputURL else {
                throw WorldHTTPRuntimeError.mediaNotFound
            }
            return url
        case "terminal-frame":
            guard let url = job.rolloutReceipt?.terminalFrameURL else {
                throw WorldHTTPRuntimeError.mediaNotFound
            }
            return url
        case "chunk":
            guard let chunkIndex,
                  let url = job.chunks.first(where: { $0.blockIndex == chunkIndex })?.outputURL else {
                throw WorldHTTPRuntimeError.mediaNotFound
            }
            return url
        default:
            throw WorldHTTPRuntimeError.mediaNotFound
        }
    }

    private var activeJob: WorldJobSnapshot? {
        jobs.values
            .filter { $0.status.isActive }
            .sorted { $0.createdAt < $1.createdAt }
            .first
    }

    private func runTransition(id: UUID, request: WorldTransitionRuntimeRequest) async {
        guard let job = jobs[id] else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: .generating,
            createdAt: job.createdAt,
            startedAt: Date(),
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            chunks: [],
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
        do {
            let receipt = try await session.transition(request) { [weak self] progress in
                Task { await self?.recordProgress(id: id, progress: progress) }
            }
            let current = jobs[id]!
            jobs[id] = WorldJobSnapshot(
                jobID: id,
                status: .completed,
                createdAt: current.createdAt,
                startedAt: current.startedAt,
                completedAt: Date(),
                progress: current.progress,
                progressEvents: current.progressEvents,
                chunks: current.chunks,
                receipt: receipt,
                rolloutReceipt: nil,
                error: nil
            )
        } catch is CancellationError {
            finish(id: id, status: .cancelled, error: nil)
        } catch {
            finish(id: id, status: .failed, error: error.localizedDescription)
        }
        tasks[id] = nil
    }

    private func runRollout(id: UUID, request: Wan2WorldRolloutRequest) async {
        guard let job = jobs[id] else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: .generating,
            createdAt: job.createdAt,
            startedAt: Date(),
            completedAt: nil,
            progress: nil,
            progressEvents: [],
            chunks: [],
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
        do {
            let receipt = try await session.rollout(
                request,
                jobID: id,
                progressHandler: { [weak self] progress in
                    Task { await self?.recordProgress(id: id, progress: progress) }
                },
                chunkHandler: { [weak self] chunk in
                    await self?.recordChunk(id: id, chunk: chunk)
                }
            )
            let current = jobs[id]!
            jobs[id] = WorldJobSnapshot(
                jobID: id,
                status: .completed,
                createdAt: current.createdAt,
                startedAt: current.startedAt,
                completedAt: Date(),
                progress: current.progress,
                progressEvents: current.progressEvents,
                chunks: receipt.chunks,
                receipt: nil,
                rolloutReceipt: receipt,
                error: nil
            )
        } catch is CancellationError {
            finish(id: id, status: .cancelled, error: nil)
        } catch {
            finish(id: id, status: .failed, error: error.localizedDescription)
        }
        tasks[id] = nil
    }

    private func recordProgress(id: UUID, progress: GenerationProgress) {
        guard let job = jobs[id], job.status.isActive else { return }
        let event = WorldJobProgress(
            stage: progress.stage.rawValue,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            updatedAt: Date()
        )
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: job.status == .cancelling ? .cancelling : .generating,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: nil,
            progress: event,
            progressEvents: job.progressEvents + [event],
            chunks: job.chunks,
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
    }

    private func recordChunk(id: UUID, chunk: WorldRolloutChunkPayload) {
        guard let job = jobs[id], job.status.isActive else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: job.status == .cancelling ? .cancelling : .generating,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: nil,
            progress: job.progress,
            progressEvents: job.progressEvents,
            chunks: job.chunks + [chunk],
            receipt: nil,
            rolloutReceipt: nil,
            error: nil
        )
    }

    private func finish(id: UUID, status: WorldJobStatus, error: String?) {
        guard let job = jobs[id] else { return }
        jobs[id] = WorldJobSnapshot(
            jobID: id,
            status: status,
            createdAt: job.createdAt,
            startedAt: job.startedAt,
            completedAt: Date(),
            progress: job.progress,
            progressEvents: job.progressEvents,
            chunks: job.chunks,
            receipt: nil,
            rolloutReceipt: nil,
            error: error
        )
    }
}

private struct WorldHTTPServer: Sendable {
    let runtime: WorldHTTPRuntime
    let apiKey: String?

    func run(host: String, port: Int) async throws {
        let app = Application(
            router: router(),
            configuration: .init(address: .hostname(host, port: port))
        )
        print("Native world server: http://\(host):\(port)/v1/world/session")
        print("Press Ctrl+C to stop.")
        try await app.runService()
    }

    private func router() -> Router<WorldRequestContext> {
        let router = Router(context: WorldRequestContext.self)
        router.add(middleware: CORSMiddleware(
            allowOrigin: .all,
            allowMethods: [.get, .post, .delete, .options]
        ))
        router.get("/health") { _, _ in
            try response(WorldHealth())
        }
        router.get("/v1/world/session") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            return try response(await runtime.snapshot())
        }
        router.post("/v1/world/session/prepare") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                try await runtime.prepare()
                return try response(await runtime.snapshot())
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/source") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let body = try await request.body.collect(upTo: 32 * 1024 * 1024)
                guard body.readableBytes > 0 else {
                    return try response(WorldErrorPayload(error: "Source image body is empty."), status: .badRequest)
                }
                return try response(try await runtime.seedSourceImage(Data(body.readableBytesView)))
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/transitions") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldTransitionPayload = try await decode(request)
                return try response(try await runtime.startTransition(payload), status: .accepted)
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/rollouts") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldRolloutPayload = try await decode(request)
                return try response(try await runtime.startRollout(payload), status: .accepted)
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/session/checkpoints") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                return try response(try await runtime.causalCheckpointReceipts())
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/checkpoints") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldCausalCheckpointRequest? = try await decodeOptional(request)
                return try response(
                    try await runtime.createCausalCheckpoint(payload),
                    status: .created
                )
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/checkpoints/:id/restore") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: raw) else {
                    return try response(
                        WorldErrorPayload(error: "Invalid causal checkpoint id."),
                        status: .badRequest
                    )
                }
                return try response(try await runtime.restoreCausalCheckpoint(id))
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/session/checkpoints/:id/media/frame") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: raw) else {
                    return try response(
                        WorldErrorPayload(error: "Invalid causal checkpoint id."),
                        status: .badRequest
                    )
                }
                let url = try await runtime.causalCheckpointFrameURL(id)
                return try await mediaResponse(url: url, contentType: "image/png", context: context)
            } catch {
                return try errorResponse(error)
            }
        }
        router.delete("/v1/world/session/checkpoints/:id") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: raw) else {
                    return try response(
                        WorldErrorPayload(error: "Invalid causal checkpoint id."),
                        status: .badRequest
                    )
                }
                return try response(try await runtime.discardCausalCheckpoint(id))
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/jobs/:id") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self), let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                return try response(try await runtime.job(id: id))
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/jobs/:id/media/output") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                let url = try await runtime.mediaURL(id: id, asset: "output")
                return try await mediaResponse(url: url, contentType: "video/mp4", context: context)
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/jobs/:id/media/terminal-frame") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                let url = try await runtime.mediaURL(id: id, asset: "terminal-frame")
                return try await mediaResponse(url: url, contentType: "image/png", context: context)
            } catch {
                return try errorResponse(error)
            }
        }
        router.get("/v1/world/jobs/:id/media/chunks/:index") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let rawID = context.parameters.get("id", as: String.self),
                      let id = UUID(uuidString: rawID),
                      let rawIndex = context.parameters.get("index", as: String.self),
                      let index = Int(rawIndex), index >= 0 else {
                    return try response(WorldErrorPayload(error: "Invalid job id or chunk index."), status: .badRequest)
                }
                let url = try await runtime.mediaURL(id: id, asset: "chunk", chunkIndex: index)
                return try await mediaResponse(url: url, contentType: "video/mp4", context: context)
            } catch {
                return try errorResponse(error)
            }
        }
        router.delete("/v1/world/jobs/:id") { request, context in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                guard let raw = context.parameters.get("id", as: String.self), let id = UUID(uuidString: raw) else {
                    return try response(WorldErrorPayload(error: "Invalid job id."), status: .badRequest)
                }
                return try response(try await runtime.cancel(id: id), status: .accepted)
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/reset") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                let payload: WorldResetPayload? = try await decodeOptional(request)
                return try response(try await runtime.reset(payload))
            } catch {
                return try errorResponse(error)
            }
        }
        router.post("/v1/world/session/unload") { request, _ in
            if let denied = unauthorized(request, apiKey: apiKey) { return denied }
            do {
                return try response(try await runtime.unload())
            } catch {
                return try errorResponse(error)
            }
        }
        return router
    }
}

private func decode<T: Decodable>(_ request: Request, as _: T.Type = T.self) async throws -> T {
    let body = try await request.body.collect(upTo: 1024 * 1024)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(T.self, from: Data(body.readableBytesView))
}

private func decodeOptional<T: Decodable>(_ request: Request, as _: T.Type = T.self) async throws -> T? {
    let body = try await request.body.collect(upTo: 1024 * 1024)
    guard body.readableBytes > 0 else { return nil }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(T.self, from: Data(body.readableBytesView))
}

private func response<T: Encodable>(
    _ payload: T,
    status: HTTPResponse.Status = .ok
) throws -> Response {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(payload)
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: ByteBuffer(bytes: data))
    )
}

private func mediaResponse(
    url: URL,
    contentType: String,
    context: WorldRequestContext
) async throws -> Response {
    let body = try await FileIO(threadPool: .singleton).loadFile(path: url.path, context: context)
    return Response(
        status: .ok,
        headers: [
            .contentType: contentType,
            .cacheControl: "no-store",
        ],
        body: body
    )
}

private func errorResponse(_ error: Error) throws -> Response {
    let status: HTTPResponse.Status
    switch error {
    case WorldHTTPRuntimeError.busy, Wan2WorldSessionError.busy:
        status = .conflict
    case WorldHTTPRuntimeError.jobNotFound,
         WorldHTTPRuntimeError.mediaNotFound,
         Wan2WorldSessionError.causalCheckpointNotFound:
        status = .notFound
    default:
        status = .badRequest
    }
    return try response(WorldErrorPayload(error: error.localizedDescription), status: status)
}

private func unauthorized(_ request: Request, apiKey: String?) -> Response? {
    guard let apiKey, !apiKey.isEmpty else { return nil }
    guard request.headers[.authorization] == "Bearer \(apiKey)" else {
        return try? response(WorldErrorPayload(error: "Unauthorized."), status: .unauthorized)
    }
    return nil
}
