import Foundation
import MLX
import MLXNN

public enum MiniMaxH3TurboAdapter {
    public static let format = "minimax-h3-runtime-lora-v1"
    public static let lightX2VFormat = "minimax-h3-peft-fused-lora-v1"
    public static let fastVideoFormat = "fastvideo-lora-v2"
    public static let fastH3PremergedFormat = "mere.run.minimax-h3-fasth3-premerged-v1"
    public static let expectedPairCount = 259
    public static let lightX2VExpectedPairCount = 312
    public static let fastVideoExpectedPairCount = 362
    public static let recommendedSchedulePointCount = 5
    public static let fastH3VSADataFreeFilename =
        "fastvideo_fasth3_4step_v1_vsa_datafree_rank64.safetensors"
    public static let fastH3AdaLNCacheFilename = "fastvideo_fasth3_v1_vsa_datafree_adaln_cache.safetensors"
    public static let fastH3SourceIdentity =
        "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
        + "@b65818d41939b5085451074fe8ca8b799f8d4921:transformer"
    public static let fastVideoExpectedDiffCount = 82
    public static let fastVideoExpectedCompressionGateCount = 50

    public enum Task: String, Sendable, Hashable {
        case fl2va
        case ref2va
    }

    public struct InferenceRecipe: Sendable, Hashable {
        public let name: String
        public let task: Task
        public let defaultSchedulePointCount: Int
        public let supportedSchedulePointCounts: Set<Int>
        public let videoFlowShift: Float?
        public let audioFlowShift: Float?
        public let lightX2VAlpha: Float
        public let baseDenoisingSigmas: [Float]?
        public let requiresFastH3VSA: Bool
        public let requiresTextOnlyConditioning: Bool

        public func supports(schedulePointCount: Int) -> Bool {
            supportedSchedulePointCounts.contains(schedulePointCount)
        }

        public func supports(task value: String) -> Bool {
            task.rawValue == value.lowercased()
        }
    }

    public static let fourEvaluationRecipe = InferenceRecipe(
        name: "four-evaluation",
        task: .fl2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: nil,
        audioFlowShift: nil,
        lightX2VAlpha: 8,
        baseDenoisingSigmas: nil,
        requiresFastH3VSA: false,
        requiresTextOnlyConditioning: false
    )

    public static let lightX2VEightStepV1Recipe = InferenceRecipe(
        name: "lightx2v-v1-8-step",
        task: .fl2va,
        defaultSchedulePointCount: 9,
        supportedSchedulePointCounts: [5, 9],
        videoFlowShift: 12,
        audioFlowShift: 3,
        lightX2VAlpha: 8,
        baseDenoisingSigmas: nil,
        requiresFastH3VSA: false,
        requiresTextOnlyConditioning: false
    )

    public static let lightX2VEightStepV1_768pRecipe = InferenceRecipe(
        name: "lightx2v-v1-8-step-768p",
        task: .fl2va,
        defaultSchedulePointCount: 9,
        supportedSchedulePointCounts: [9],
        videoFlowShift: 6,
        audioFlowShift: 3,
        lightX2VAlpha: 8,
        baseDenoisingSigmas: nil,
        requiresFastH3VSA: false,
        requiresTextOnlyConditioning: false
    )

    public static let lightX2VFourStepV1_768pRecipe = InferenceRecipe(
        name: "lightx2v-v1-4-step-768p",
        task: .fl2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: 6,
        audioFlowShift: 3,
        lightX2VAlpha: 128,
        baseDenoisingSigmas: nil,
        requiresFastH3VSA: false,
        requiresTextOnlyConditioning: false
    )

    public static let lightX2VRef2VFourStepV01Recipe = InferenceRecipe(
        name: "lightx2v-ref2v-v0.1-4-step",
        task: .ref2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: 12,
        audioFlowShift: 3,
        lightX2VAlpha: 8,
        baseDenoisingSigmas: nil,
        requiresFastH3VSA: false,
        requiresTextOnlyConditioning: false
    )

    public static let fastH3VSADataFreeRecipe = InferenceRecipe(
        name: "fastvideo-fasth3-v1-vsa-datafree",
        task: .fl2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: 12,
        audioFlowShift: 3,
        lightX2VAlpha: 1,
        baseDenoisingSigmas: [0.999, 0.749, 0.5, 0.25, 0],
        requiresFastH3VSA: true,
        requiresTextOnlyConditioning: true
    )

    public static func inferenceRecipe(for url: URL) -> InferenceRecipe {
        if let metadata = try? SafetensorsStreamingLoader.fileMetadata(url: url),
           (metadata["format"] == fastVideoFormat
            || metadata["format"] == fastH3PremergedFormat),
           metadata["finetuned_model"] == "FastVideo/FastVideo-FastH3-4-step-v1" {
            return fastH3VSADataFreeRecipe
        }
        return inferenceRecipe(filename: url.lastPathComponent)
    }

    static func isPremergedFastH3Artifact(_ url: URL) -> Bool {
        (try? SafetensorsStreamingLoader.fileMetadata(url: url)["format"])
            == fastH3PremergedFormat
    }

    public static func inferenceRecipe(filename: String) -> InferenceRecipe {
        switch filename.lowercased() {
        case "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors":
            lightX2VRef2VFourStepV01Recipe
        case "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors":
            lightX2VEightStepV1Recipe
        case "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors":
            lightX2VEightStepV1_768pRecipe
        case "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors":
            lightX2VFourStepV1_768pRecipe
        case fastH3VSADataFreeFilename:
            fastH3VSADataFreeRecipe
        default:
            fourEvaluationRecipe
        }
    }

    enum SourceFormat: Equatable {
        case runtime
        case lightX2V
        case fastVideo
        case fastH3Premerged

        var pairSuffixes: (String, String) {
            switch self {
            case .runtime:
                return (".lora_A.weight", ".lora_B.weight")
            case .lightX2V:
                return (".lora_A.default.weight", ".lora_B.default.weight")
            case .fastVideo, .fastH3Premerged:
                return (".lora_A.weight", ".lora_B.weight")
            }
        }

        var expectedPairCount: Int {
            switch self {
            case .runtime:
                return MiniMaxH3TurboAdapter.expectedPairCount
            case .lightX2V:
                return MiniMaxH3TurboAdapter.lightX2VExpectedPairCount
            case .fastVideo:
                return MiniMaxH3TurboAdapter.fastVideoExpectedPairCount
            case .fastH3Premerged:
                return 0
            }
        }
    }

    enum QKVBranch: String, CaseIterable {
        case query
        case key
        case value
    }

    struct LoRAPair {
        let down: MLXArray
        let up: MLXArray
    }

    struct Installation {
        let pairCount: Int
        let adaLNCache: MiniMaxH3AdaLNCache?
    }

    struct LightX2VTarget {
        let modulePath: String
        let qkvBranch: QKVBranch?
    }

    enum AdapterError: LocalizedError {
        case unrecognizedFormat(URL)
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case unsupportedSourceModule(String)
        case missingTargetModule(String)
        case targetIsNotLinear(String)
        case duplicateTarget(String)
        case duplicateQKVBranch(String, QKVBranch)
        case incompleteQKVTarget(String, missing: [QKVBranch])
        case invalidPairShape(String, a: [Int], b: [Int])
        case targetShapeMismatch(String, expected: [Int], actual: [Int])
        case requiresUnitStrength(Float)
        case unexpectedAuxiliaryTensorCount(kind: String, expected: Int, actual: Int)
        case missingTargetParameter(String)

        var errorDescription: String? {
            switch self {
            case .unrecognizedFormat(let url):
                return "Unsupported MiniMax-H3 LoRA tensor format in \(url.path)."
            case .noPairs(let url):
                return "No MiniMax-H3 LoRA tensor pairs were found in \(url.path)."
            case .unexpectedPairCount(let expected, let actual):
                return "MiniMax-H3 LoRA tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .unsupportedSourceModule(let path):
                return "Unsupported MiniMax-H3 LoRA source module: \(path)"
            case .missingTargetModule(let path):
                return "MiniMax-H3 LoRA target module is missing: \(path)"
            case .targetIsNotLinear(let path):
                return "MiniMax-H3 LoRA target \(path) is not a supported Linear layer."
            case .duplicateTarget(let path):
                return "MiniMax-H3 LoRA contains a duplicate target: \(path)"
            case .duplicateQKVBranch(let path, let branch):
                return "MiniMax-H3 LoRA contains a duplicate \(branch.rawValue) branch for \(path)."
            case .incompleteQKVTarget(let path, let missing):
                let names = missing.map(\.rawValue).joined(separator: ", ")
                return "MiniMax-H3 LoRA target \(path) is missing QKV branches: \(names)."
            case .invalidPairShape(let path, let a, let b):
                return "Invalid MiniMax-H3 LoRA pair for \(path): A=\(a), B=\(b)."
            case .targetShapeMismatch(let path, let expected, let actual):
                return "MiniMax-H3 LoRA target \(path) has shape \(actual); expected \(expected)."
            case .requiresUnitStrength(let strength):
                return "FastH3 VSA is a complete student checkpoint and requires adapter strength 1, not \(strength)."
            case .unexpectedAuxiliaryTensorCount(let kind, let expected, let actual):
                return "FastH3 VSA \(kind) tensor count mismatch: expected \(expected), found \(actual)."
            case .missingTargetParameter(let path):
                return "FastH3 VSA target parameter is missing: \(path)"
            }
        }
    }

}
