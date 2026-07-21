import Foundation

public enum Cosmos3ActionMode: String, Codable, CaseIterable, Sendable {
    case policy
    case forwardDynamics = "forward_dynamics"
    case inverseDynamics = "inverse_dynamics"
}

public enum Cosmos3ActionViewpoint: String, Codable, CaseIterable, Sendable {
    case egoView = "ego_view"
    case thirdPersonView = "third_person_view"
    case wristView = "wrist_view"
    case concatenatedView = "concat_view"

    public var framingPrompt: String {
        switch self {
        case .egoView:
            return "This video is captured from a first-person perspective looking at the scene."
        case .thirdPersonView:
            return "This video is captured from a third-person perspective looking towards the agent from the front."
        case .wristView:
            return "This video is captured from a wrist-mounted camera."
        case .concatenatedView:
            return "This video contains concatenated views from multiple camera perspectives."
        }
    }
}

public enum Cosmos3ActionDomain: String, Codable, CaseIterable, Sendable {
    case av
    case cameraPose = "camera_pose"
    case handPose = "hand_pose"
    case pushT = "pusht"
    case umi
    case bridgeOriginalLeRobot = "bridge_orig_lerobot"
    case droidLeRobot = "droid_lerobot"
    case robomindFranka = "robomind-franka"
    case galbot
    case robomindFrankaDual = "robomind-franka-dual"
    case robomindUR = "robomind-ur"
    case agibotWorld = "agibotworld"
    case agibotGearGripper = "agibot_gear_gripper"
    case agibotGearGripperExtended = "agibot_gear_gripper_ext"
    case fractal

    public var domainID: Int {
        switch self {
        case .av: 1
        case .cameraPose: 2
        case .handPose: 3
        case .pushT: 4
        case .umi: 6
        case .bridgeOriginalLeRobot: 7
        case .droidLeRobot, .robomindFranka: 8
        case .galbot: 9
        case .robomindFrankaDual: 12
        case .robomindUR: 13
        case .agibotWorld, .agibotGearGripper, .agibotGearGripperExtended: 15
        case .fractal: 20
        }
    }

    public var rawActionDimension: Int {
        switch self {
        case .av, .cameraPose: 9
        case .pushT: 2
        case .umi, .bridgeOriginalLeRobot, .droidLeRobot, .robomindFranka, .robomindUR, .fractal: 10
        case .robomindFrankaDual: 20
        case .agibotWorld, .agibotGearGripper, .agibotGearGripperExtended: 29
        case .galbot: 30
        case .handPose: 57
        }
    }
}

public enum Cosmos3ActionResolutionTier: Int, Codable, CaseIterable, Sendable {
    case compact = 256
    case medium = 480
    case large = 704
    case hd = 720

    public var bins: [(height: Int, width: Int)] {
        switch self {
        case .compact:
            return [(256, 256), (256, 320), (320, 256), (192, 320), (320, 192)]
        case .medium:
            return [(640, 640), (544, 736), (736, 544), (480, 832), (832, 480)]
        case .large:
            return [(960, 960), (832, 1_088), (1_088, 832), (704, 1_280), (1_280, 704)]
        case .hd:
            return [(960, 960), (832, 1_104), (1_104, 832), (720, 1_280), (1_280, 720)]
        }
    }

    public func nearestCanvas(sourceHeight: Int, sourceWidth: Int) -> (height: Int, width: Int) {
        precondition(sourceHeight > 0 && sourceWidth > 0)
        let sourceRatio = Double(sourceHeight) / Double(sourceWidth)
        return bins.min { lhs, rhs in
            abs(Double(lhs.height) / Double(lhs.width) - sourceRatio)
                < abs(Double(rhs.height) / Double(rhs.width) - sourceRatio)
        }!
    }
}

public enum Cosmos3ActionConditionError: LocalizedError, Sendable {
    case invalidChunkSize(Int)
    case missingConditioning
    case inverseDynamicsRequiresVideo
    case forwardDynamicsRequiresActions
    case invalidActionCount
    case invalidActionWidth(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidChunkSize(let value):
            return "Cosmos3 action chunk size must be at least 1; received \(value)."
        case .missingConditioning:
            return "Cosmos3 actions require an image or video conditioning source."
        case .inverseDynamicsRequiresVideo:
            return "Cosmos3 inverse dynamics requires video conditioning."
        case .forwardDynamicsRequiresActions:
            return "Cosmos3 forward dynamics requires normalized model-space actions."
        case .invalidActionCount:
            return "Cosmos3 forward dynamics requires at least one normalized model-space action."
        case .invalidActionWidth(let expected, let actual):
            return "Cosmos3 normalized model-space action width must be \(expected); received \(actual)."
        }
    }
}

public struct Cosmos3ActionCondition: Hashable, Sendable {
    public let mode: Cosmos3ActionMode
    public let chunkSize: Int
    public let domain: Cosmos3ActionDomain
    public let resolutionTier: Cosmos3ActionResolutionTier
    public let rawActions: [[Float]]?
    public let imageURL: URL?
    public let videoURL: URL?
    public let viewpoint: Cosmos3ActionViewpoint

    public init(
        mode: Cosmos3ActionMode,
        chunkSize: Int,
        domain: Cosmos3ActionDomain,
        resolutionTier: Cosmos3ActionResolutionTier = .medium,
        rawActions: [[Float]]? = nil,
        imageURL: URL? = nil,
        videoURL: URL? = nil,
        viewpoint: Cosmos3ActionViewpoint = .egoView
    ) throws {
        guard chunkSize >= 1 else { throw Cosmos3ActionConditionError.invalidChunkSize(chunkSize) }
        guard imageURL != nil || videoURL != nil else {
            throw Cosmos3ActionConditionError.missingConditioning
        }
        if mode == .inverseDynamics && videoURL == nil {
            throw Cosmos3ActionConditionError.inverseDynamicsRequiresVideo
        }
        if mode == .forwardDynamics {
            guard let rawActions else {
                throw Cosmos3ActionConditionError.forwardDynamicsRequiresActions
            }
            guard !rawActions.isEmpty else { throw Cosmos3ActionConditionError.invalidActionCount }
            if let width = rawActions.first(where: { $0.count != domain.rawActionDimension })?.count {
                throw Cosmos3ActionConditionError.invalidActionWidth(
                    expected: domain.rawActionDimension,
                    actual: width
                )
            }
        }
        self.mode = mode
        self.chunkSize = chunkSize
        self.domain = domain
        self.resolutionTier = resolutionTier
        self.rawActions = rawActions
        self.imageURL = imageURL
        self.videoURL = videoURL
        self.viewpoint = viewpoint
    }

    public func modelActions(actionDimension: Int = 64) -> [[Float]]? {
        guard let rawActions else { return nil }
        precondition(actionDimension >= domain.rawActionDimension)
        var resized = Array(rawActions.prefix(chunkSize))
        while resized.count < chunkSize {
            resized.append(resized.last!)
        }
        return resized.map { $0 + Array(repeating: 0, count: actionDimension - $0.count) }
    }
}
