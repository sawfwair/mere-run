import Foundation
import MLX
import MLXNN

public enum MiniMaxH3TurboAdapter {
    public static let format = "minimax-h3-runtime-lora-v1"
    public static let lightX2VFormat = "minimax-h3-peft-fused-lora-v1"
    public static let expectedPairCount = 259
    public static let lightX2VExpectedPairCount = 312
    public static let recommendedSchedulePointCount = 5

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
        lightX2VAlpha: 8
    )

    public static let lightX2VEightStepV1Recipe = InferenceRecipe(
        name: "lightx2v-v1-8-step",
        task: .fl2va,
        defaultSchedulePointCount: 9,
        supportedSchedulePointCounts: [5, 9],
        videoFlowShift: 12,
        audioFlowShift: 3,
        lightX2VAlpha: 8
    )

    public static let lightX2VFourStepV1_768pRecipe = InferenceRecipe(
        name: "lightx2v-v1-4-step-768p",
        task: .fl2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: 6,
        audioFlowShift: 3,
        lightX2VAlpha: 128
    )

    public static let lightX2VRef2VFourStepV01Recipe = InferenceRecipe(
        name: "lightx2v-ref2v-v0.1-4-step",
        task: .ref2va,
        defaultSchedulePointCount: 5,
        supportedSchedulePointCounts: [5],
        videoFlowShift: 12,
        audioFlowShift: 3,
        lightX2VAlpha: 8
    )

    public static func inferenceRecipe(for url: URL) -> InferenceRecipe {
        inferenceRecipe(filename: url.lastPathComponent)
    }

    public static func inferenceRecipe(filename: String) -> InferenceRecipe {
        switch filename.lowercased() {
        case "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors":
            lightX2VRef2VFourStepV01Recipe
        case "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors":
            lightX2VEightStepV1Recipe
        case "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors":
            lightX2VFourStepV1_768pRecipe
        default:
            fourEvaluationRecipe
        }
    }

    private enum SourceFormat: Equatable {
        case runtime
        case lightX2V

        var pairSuffixes: (String, String) {
            switch self {
            case .runtime:
                return (".lora_A.weight", ".lora_B.weight")
            case .lightX2V:
                return (".lora_A.default.weight", ".lora_B.default.weight")
            }
        }

        var expectedPairCount: Int {
            switch self {
            case .runtime:
                return MiniMaxH3TurboAdapter.expectedPairCount
            case .lightX2V:
                return MiniMaxH3TurboAdapter.lightX2VExpectedPairCount
            }
        }
    }

    private enum QKVBranch: String, CaseIterable {
        case query
        case key
        case value
    }

    private struct LoRAPair {
        let down: MLXArray
        let up: MLXArray
    }

    private struct LightX2VTarget {
        let modulePath: String
        let qkvBranch: QKVBranch?
    }

    private enum AdapterError: LocalizedError {
        case unrecognizedFormat(URL)
        case noPairs(URL)
        case unexpectedPairCount(expected: Int, actual: Int)
        case unsupportedSourceModule(String)
        case missingTargetModule(String)
        case targetIsNotDenseLinear(String)
        case duplicateTarget(String)
        case duplicateQKVBranch(String, QKVBranch)
        case incompleteQKVTarget(String, missing: [QKVBranch])
        case invalidPairShape(String, a: [Int], b: [Int])
        case targetShapeMismatch(String, expected: [Int], actual: [Int])

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
            case .targetIsNotDenseLinear(let path):
                return "MiniMax-H3 Turbo currently requires the BF16 transformer; target \(path) is not a dense Linear layer."
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
            }
        }
    }

    @discardableResult
    static func install(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        expectedPairCount: Int? = nil
    ) throws -> Int {
        let sourceFormat = try sourceFormat(at: url)
        let inferenceRecipe = inferenceRecipe(for: url)
        let suffixes = sourceFormat.pairSuffixes
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]
        var qkvPairs: [String: [QKVBranch: LoRAPair]] = [:]
        var fusedTargets: Set<String> = []

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: suffixes.0,
            secondSuffix: suffixes.1
        ) { sourcePath, rawDown, rawUp in
            let target = try target(for: sourcePath, sourceFormat: sourceFormat)
            guard rawDown.ndim == 2,
                  rawUp.ndim == 2,
                  rawUp.dim(1) == rawDown.dim(0) else {
                throw AdapterError.invalidPairShape(
                    sourcePath,
                    a: rawDown.shape,
                    b: rawUp.shape
                )
            }

            let up = scaledUp(
                rawUp,
                down: rawDown,
                sourceFormat: sourceFormat,
                lightX2VAlpha: inferenceRecipe.lightX2VAlpha
            )
            if let branch = target.qkvBranch {
                guard !fusedTargets.contains(target.modulePath) else {
                    throw AdapterError.duplicateTarget(target.modulePath)
                }
                var branches = qkvPairs[target.modulePath] ?? [:]
                guard branches[branch] == nil else {
                    throw AdapterError.duplicateQKVBranch(target.modulePath, branch)
                }
                branches[branch] = LoRAPair(
                    down: rawDown,
                    up: up
                )
                if branches.count == QKVBranch.allCases.count {
                    guard replacements[target.modulePath] == nil else {
                        throw AdapterError.duplicateTarget(target.modulePath)
                    }
                    let linear = try denseLinear(at: target.modulePath, in: modulesByPath)
                    let query = branches[.query]!
                    let key = branches[.key]!
                    let value = branches[.value]!
                    try validateQKV(
                        target.modulePath,
                        query: query,
                        key: key,
                        value: value,
                        base: linear
                    )
                    fuseQKVLinear(
                        base: linear,
                        query: query,
                        key: key,
                        value: value,
                        strength: strength
                    )
                    fusedTargets.insert(target.modulePath)
                    qkvPairs.removeValue(forKey: target.modulePath)
                } else {
                    qkvPairs[target.modulePath] = branches
                }
                return
            }

            guard replacements[target.modulePath] == nil,
                  !fusedTargets.contains(target.modulePath) else {
                throw AdapterError.duplicateTarget(target.modulePath)
            }
            let linear = try denseLinear(at: target.modulePath, in: modulesByPath)
            let mappedUp = target.modulePath.hasSuffix(".attn.qkv_proj")
                ? MiniMaxH3ModelLoader.deinterleavedQKVOutputRows(
                    up,
                    headCount: transformer.configuration.attentionHeadCount,
                    headDimension: transformer.configuration.attentionHeadDimension
                )
                : up
            try validate(
                target.modulePath,
                down: rawDown,
                up: mappedUp,
                base: linear
            )
            switch sourceFormat {
            case .runtime:
                let runtimeLayer = MiniMaxH3RuntimeLoRALinear(
                    base: linear,
                    loraDown: rawDown,
                    loraUp: mappedUp,
                    strength: strength
                )
                MLX.eval(runtimeLayer.loraDown, runtimeLayer.loraUp)
                replacements[target.modulePath] = runtimeLayer
            case .lightX2V:
                fuseLinear(
                    base: linear,
                    down: rawDown,
                    up: mappedUp,
                    strength: strength
                )
                fusedTargets.insert(target.modulePath)
            }
        }

        guard pairCount > 0 else { throw AdapterError.noPairs(url) }
        let requiredPairCount = expectedPairCount ?? sourceFormat.expectedPairCount
        guard pairCount == requiredPairCount else {
            throw AdapterError.unexpectedPairCount(expected: requiredPairCount, actual: pairCount)
        }

        for (modulePath, branches) in qkvPairs {
            let missing = QKVBranch.allCases.filter { branches[$0] == nil }
            throw AdapterError.incompleteQKVTarget(modulePath, missing: missing)
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        Memory.clearCache()
        return pairCount
    }

    private static func sourceFormat(at url: URL) throws -> SourceFormat {
        let keys = try SafetensorsStreamingLoader.metadata(url: url).keys
        if keys.contains(where: { $0.hasSuffix(".lora_A.default.weight") }) {
            return .lightX2V
        }
        if keys.contains(where: { $0.hasSuffix(".lora_A.weight") }) {
            return .runtime
        }
        throw AdapterError.unrecognizedFormat(url)
    }

    private static func target(
        for sourcePath: String,
        sourceFormat: SourceFormat
    ) throws -> LightX2VTarget {
        guard sourceFormat == .lightX2V else {
            return LightX2VTarget(modulePath: sourcePath, qkvBranch: nil)
        }

        let runtimePrefix: String
        let remainder: Substring
        if sourcePath.hasPrefix("transformer_blocks.") {
            runtimePrefix = "blocks."
            remainder = sourcePath.dropFirst("transformer_blocks.".count)
        } else if sourcePath.hasPrefix("token_refiner.refiner_blocks.") {
            runtimePrefix = "token_refiner.blocks."
            remainder = sourcePath.dropFirst("token_refiner.refiner_blocks.".count)
        } else {
            throw AdapterError.unsupportedSourceModule(sourcePath)
        }
        let normalized = runtimePrefix + remainder

        for (suffix, branch) in [
            (".attn.to_q", QKVBranch.query),
            (".attn.to_k", QKVBranch.key),
            (".attn.to_v", QKVBranch.value),
        ] where normalized.hasSuffix(suffix) {
            return LightX2VTarget(
                modulePath: String(normalized.dropLast(suffix.count)) + ".attn.qkv_proj",
                qkvBranch: branch
            )
        }

        for (suffix, replacement) in [
            (".attn.to_out.0", ".attn.out_proj"),
            (".ff.net.0.proj", ".mlp.fc1"),
            (".ff.net.2", ".mlp.fc2"),
        ] where normalized.hasSuffix(suffix) {
            return LightX2VTarget(
                modulePath: String(normalized.dropLast(suffix.count)) + replacement,
                qkvBranch: nil
            )
        }
        throw AdapterError.unsupportedSourceModule(sourcePath)
    }

    private static func scaledUp(
        _ up: MLXArray,
        down: MLXArray,
        sourceFormat: SourceFormat,
        lightX2VAlpha: Float
    ) -> MLXArray {
        guard sourceFormat == .lightX2V else { return up }
        let scale = MLXArray(lightX2VAlpha / Float(down.dim(0))).asType(up.dtype)
        return up * scale
    }

    private static func denseLinear(
        at path: String,
        in modulesByPath: [String: Module]
    ) throws -> Linear {
        guard let module = modulesByPath[path] else {
            throw AdapterError.missingTargetModule(path)
        }
        guard let linear = module as? Linear,
              !(linear is QuantizedLinear) else {
            throw AdapterError.targetIsNotDenseLinear(path)
        }
        return linear
    }

    private static func validate(
        _ path: String,
        down: MLXArray,
        up: MLXArray,
        base: Linear
    ) throws {
        guard down.dim(1) == base.shape.1,
              up.dim(0) == base.shape.0 else {
            throw AdapterError.targetShapeMismatch(
                path,
                expected: [base.shape.0, base.shape.1],
                actual: [up.dim(0), down.dim(1)]
            )
        }
    }

    private static func validateQKV(
        _ path: String,
        query: LoRAPair,
        key: LoRAPair,
        value: LoRAPair,
        base: Linear
    ) throws {
        guard base.shape.0.isMultiple(of: 3) else {
            throw AdapterError.targetShapeMismatch(
                path,
                expected: [base.shape.0, base.shape.1],
                actual: [base.shape.0, base.shape.1]
            )
        }
        let branchOutputSize = base.shape.0 / 3
        for pair in [query, key, value] {
            guard pair.down.dim(1) == base.shape.1,
                  pair.up.dim(0) == branchOutputSize else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [base.shape.0, base.shape.1],
                    actual: [3 * pair.up.dim(0), pair.down.dim(1)]
                )
            }
        }
    }

    private static func fuseLinear(
        base: Linear,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) {
        let dtype = base.weight.dtype
        let delta = MLX.matmul(up.asType(dtype), down.asType(dtype))
            * MLXArray(strength).asType(dtype)
        let fusedWeight = base.weight + delta
        MLX.eval(fusedWeight)
        base.update(parameters: ModuleParameters.unflattened([("weight", fusedWeight)]))
        Memory.clearCache()
    }

    private static func fuseQKVLinear(
        base: Linear,
        query: LoRAPair,
        key: LoRAPair,
        value: LoRAPair,
        strength: Float
    ) {
        let dtype = base.weight.dtype
        let deltas = [query, key, value].map { pair in
            MLX.matmul(pair.up.asType(dtype), pair.down.asType(dtype))
        }
        let delta = MLX.concatenated(deltas, axis: 0)
            * MLXArray(strength).asType(dtype)
        let fusedWeight = base.weight + delta
        MLX.eval(fusedWeight)
        base.update(parameters: ModuleParameters.unflattened([("weight", fusedWeight)]))
        Memory.clearCache()
    }

    private static func applyModuleReplacements(
        _ replacements: [String: Module],
        leafModules: [(String, Module)],
        to model: Module
    ) {
        var arrayUpdates: [String: [(Int, Module)]] = [:]
        var directUpdates: [(String, Module)] = []

        for (path, replacement) in replacements {
            let components = path.split(separator: ".")
            if let last = components.last, let index = Int(last) {
                let parentPath = components.dropLast().joined(separator: ".")
                arrayUpdates[parentPath, default: []].append((index, replacement))
            } else {
                directUpdates.append((path, replacement))
            }
        }

        var moduleUpdates = directUpdates
        for (parentPath, indexedReplacements) in arrayUpdates {
            let currentModules = leafModules.filter { path, _ in
                let parts = path.split(separator: ".")
                guard parts.count >= 2, Int(parts.last!) != nil else { return false }
                return parts.dropLast().joined(separator: ".") == parentPath
            }
            let replacementMap = Dictionary(uniqueKeysWithValues: indexedReplacements)
            for (modulePath, originalModule) in currentModules {
                let index = Int(modulePath.split(separator: ".").last!)!
                moduleUpdates.append((modulePath, replacementMap[index] ?? originalModule))
            }
        }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }
}

final class MiniMaxH3RuntimeLoRALinear: Linear {
    @ParameterInfo(key: "lora_down") var loraDown: MLXArray
    @ParameterInfo(key: "lora_up") var loraUp: MLXArray
    let strength: Float

    init(base: Linear, loraDown: MLXArray, loraUp: MLXArray, strength: Float) {
        self._loraDown.wrappedValue = loraDown.asType(base.weight.dtype)
        self._loraUp.wrappedValue = loraUp.asType(base.weight.dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MLX.matmul(
            MLX.matmul(input.asType(loraDown.dtype), loraDown.T),
            loraUp.T
        ) * MLXArray(strength).asType(loraDown.dtype)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}
