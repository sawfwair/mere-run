import Foundation
import MLX
import MLXNN

public enum MiniMaxH3TurboAdapter {
    public static let format = "minimax-h3-runtime-lora-v1"
    public static let lightX2VFormat = "minimax-h3-peft-fused-lora-v1"
    public static let fastVideoFormat = "fastvideo-lora-v2"
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
        lightX2VAlpha: 8
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
           metadata["format"] == fastVideoFormat,
           metadata["finetuned_model"] == "FastVideo/FastVideo-FastH3-4-step-v1" {
            return fastH3VSADataFreeRecipe
        }
        return inferenceRecipe(filename: url.lastPathComponent)
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

    private enum SourceFormat: Equatable {
        case runtime
        case lightX2V
        case fastVideo

        var pairSuffixes: (String, String) {
            switch self {
            case .runtime:
                return (".lora_A.weight", ".lora_B.weight")
            case .lightX2V:
                return (".lora_A.default.weight", ".lora_B.default.weight")
            case .fastVideo:
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

    struct Installation {
        let pairCount: Int
        let adaLNCache: MiniMaxH3AdaLNCache?
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

    @discardableResult
    static func install(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        expectedPairCount: Int? = nil
    ) throws -> Int {
        try installForInference(
            url: url,
            into: transformer,
            strength: strength,
            adaLNCache: nil,
            expectedPairCount: expectedPairCount
        ).pairCount
    }

    static func installForInference(
        url: URL,
        into transformer: MiniMaxH3Transformer,
        strength: Float,
        adaLNCache: MiniMaxH3AdaLNCache?,
        expectedPairCount: Int? = nil
    ) throws -> Installation {
        let sourceFormat = try sourceFormat(at: url)
        if sourceFormat == .fastVideo, strength != 1 {
            throw AdapterError.requiresUnitStrength(strength)
        }
        let inferenceRecipe = inferenceRecipe(for: url)
        let suffixes = sourceFormat.pairSuffixes
        let usesNativeFastH3Cache = sourceFormat == .fastVideo
            && adaLNCache?.sourceIdentity == fastH3SourceIdentity
        if sourceFormat == .fastVideo {
            let parameterCount = try applyFastVideoDifferences(
                url: url,
                to: transformer,
                strength: strength,
                omittingCacheCoveredParameters: usesNativeFastH3Cache
            )
            guard parameterCount == fastVideoExpectedDiffCount else {
                throw AdapterError.unexpectedAuxiliaryTensorCount(
                    kind: "difference",
                    expected: fastVideoExpectedDiffCount,
                    actual: parameterCount
                )
            }
        }
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]
        var qkvPairs: [String: [QKVBranch: LoRAPair]] = [:]
        var adaLNPairs: [String: LoRAPair] = [:]

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
            if isAdaLNTarget(target.modulePath) {
                if usesNativeFastH3Cache { return }
                guard adaLNPairs[target.modulePath] == nil else {
                    throw AdapterError.duplicateTarget(target.modulePath)
                }
                adaLNPairs[target.modulePath] = LoRAPair(down: rawDown, up: up)
                return
            }
            if let branch = target.qkvBranch {
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
                    let linear = try targetLinear(at: target.modulePath, in: modulesByPath)
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
                    replacements[target.modulePath] = runtimeQKVLayer(
                        base: linear,
                        query: query,
                        key: key,
                        value: value,
                        strength: strength
                    )
                    qkvPairs.removeValue(forKey: target.modulePath)
                } else {
                    qkvPairs[target.modulePath] = branches
                }
                return
            }

            guard replacements[target.modulePath] == nil else {
                throw AdapterError.duplicateTarget(target.modulePath)
            }
            let linear = try targetLinear(at: target.modulePath, in: modulesByPath)
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
            replacements[target.modulePath] = runtimeLayer(
                base: linear,
                down: rawDown,
                up: mappedUp,
                strength: strength
            )
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

        var resolvedAdaLNCache = adaLNCache
        if !adaLNPairs.isEmpty {
            if let adaLNCache {
                resolvedAdaLNCache = try augmentedAdaLNCache(
                    adaLNCache,
                    pairs: adaLNPairs,
                    configuration: transformer.configuration,
                    strength: strength
                )
            } else {
                for (path, pair) in adaLNPairs {
                    guard replacements[path] == nil else {
                        throw AdapterError.duplicateTarget(path)
                    }
                    let linear = try targetLinear(at: path, in: modulesByPath)
                    try validate(path, down: pair.down, up: pair.up, base: linear)
                    replacements[path] = runtimeLayer(
                        base: linear,
                        down: pair.down,
                        up: pair.up,
                        strength: strength
                    )
                }
            }
        }

        applyModuleReplacements(replacements, leafModules: leafModules, to: transformer)
        if sourceFormat == .fastVideo {
            let gateCount = try installFastVideoCompressionGates(url: url, into: transformer)
            guard gateCount == fastVideoExpectedCompressionGateCount else {
                throw AdapterError.unexpectedAuxiliaryTensorCount(
                    kind: "compression-gate",
                    expected: fastVideoExpectedCompressionGateCount,
                    actual: gateCount
                )
            }
        }
        transformer.exactKernelMode = .disabled
        Memory.clearCache()
        return Installation(pairCount: pairCount, adaLNCache: resolvedAdaLNCache)
    }

    private static func sourceFormat(at url: URL) throws -> SourceFormat {
        if try SafetensorsStreamingLoader.fileMetadata(url: url)["format"] == fastVideoFormat {
            return .fastVideo
        }
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
        guard sourceFormat != .runtime else {
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
            (".adaln_proj.linear", ".adaln_proj.linear"),
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

    private static func applyFastVideoDifferences(
        url: URL,
        to transformer: MiniMaxH3Transformer,
        strength: Float,
        omittingCacheCoveredParameters: Bool
    ) throws -> Int {
        var seenCount = 0
        var targetPaths: Set<String> = []
        let count = try SafetensorsStreamingLoader.forEachTensor(
            url: url,
            where: { $0.hasSuffix(".diff") || $0.hasSuffix(".diff_b") }
        ) { sourceKey, difference in
            seenCount += 1
            let targetPath = try fastVideoDifferenceTarget(sourceKey)
            guard targetPaths.insert(targetPath).inserted else {
                throw AdapterError.duplicateTarget(targetPath)
            }
            if omittingCacheCoveredParameters,
               (targetPath.hasPrefix("time_embedder.")
                || (targetPath.hasPrefix("blocks.") && targetPath.contains(".adaln_proj."))
                || targetPath.hasPrefix("final_layer.adaln_proj.")) {
                return
            }
            let parameters = transformer.parameters().flattened()
            guard let base = parameters.first(where: { $0.0 == targetPath })?.1 else {
                throw AdapterError.missingTargetParameter(targetPath)
            }
            guard base.shape == difference.shape else {
                throw AdapterError.targetShapeMismatch(
                    targetPath,
                    expected: base.shape,
                    actual: difference.shape
                )
            }
            let updated = base + difference.asType(base.dtype)
                * MLXArray(strength).asType(base.dtype)
            transformer.update(parameters: ModuleParameters.unflattened([(targetPath, updated)]))
            MLX.eval(updated)
            Memory.clearCache()
        }
        precondition(count == seenCount)
        return count
    }

    private static func fastVideoDifferenceTarget(_ sourceKey: String) throws -> String {
        let suffix: String
        let parameter: String
        if sourceKey.hasSuffix(".diff_b") {
            suffix = ".diff_b"
            parameter = ".bias"
        } else if sourceKey.hasSuffix(".diff") {
            suffix = ".diff"
            parameter = ".weight"
        } else {
            throw AdapterError.unsupportedSourceModule(sourceKey)
        }
        let source = String(sourceKey.dropLast(suffix.count))
        let mapped: String
        switch source {
        case "proj_in": mapped = "video_patch_proj"
        case "audio_proj_in": mapped = "audio_patch_proj"
        case "context_embedder": mapped = "condition_proj"
        case "proj_out": mapped = "final_layer.video_out"
        case "audio_proj_out": mapped = "final_layer.audio_out"
        case "norm_out.norm": mapped = "final_layer.norm"
        case "norm_out.linear": mapped = "final_layer.adaln_proj.linear"
        case "time_embedder.linear_1": mapped = "time_embedder.proj_in"
        case "time_embedder.linear_2": mapped = "time_embedder.proj_out"
        default:
            if source.hasPrefix("transformer_blocks.") {
                mapped = "blocks." + String(source.dropFirst("transformer_blocks.".count))
            } else {
                throw AdapterError.unsupportedSourceModule(sourceKey)
            }
        }
        return mapped + parameter
    }

    private static func installFastVideoCompressionGates(
        url: URL,
        into transformer: MiniMaxH3Transformer
    ) throws -> Int {
        var blockIndices: Set<Int> = []
        return try SafetensorsStreamingLoader.forEachTensor(
            url: url,
            where: { $0.hasSuffix(".attn.to_gate_compress.set_weight") }
        ) { sourceKey, weight in
            let components = sourceKey.split(separator: ".")
            guard components.count == 5,
                  components[0] == "transformer_blocks",
                  let blockIndex = Int(components[1]),
                  components[2] == "attn",
                  components[3] == "to_gate_compress",
                  components[4] == "set_weight",
                  (0..<transformer.configuration.layerCount).contains(blockIndex) else {
                throw AdapterError.unsupportedSourceModule(sourceKey)
            }
            guard blockIndices.insert(blockIndex).inserted else {
                throw AdapterError.duplicateTarget(sourceKey)
            }
            let expected = [
                transformer.configuration.attentionHeadCount
                    * transformer.configuration.attentionHeadDimension,
                transformer.configuration.hiddenSize,
            ]
            guard weight.shape == expected else {
                throw AdapterError.targetShapeMismatch(
                    sourceKey,
                    expected: expected,
                    actual: weight.shape
                )
            }
            transformer.installFastH3CompressionGate(weight, blockIndex: blockIndex)
        }
    }

    private static func targetLinear(
        at path: String,
        in modulesByPath: [String: Module]
    ) throws -> Linear {
        guard let module = modulesByPath[path] else {
            throw AdapterError.missingTargetModule(path)
        }
        guard let linear = module as? Linear else {
            throw AdapterError.targetIsNotLinear(path)
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

    private static func runtimeLayer(
        base: Linear,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) -> Module {
        if let quantized = base as? QuantizedLinear {
            let layer = MiniMaxH3RuntimeQuantizedLoRALinear(
                base: quantized,
                loraDown: down,
                loraUp: up,
                strength: strength
            )
            MLX.eval(layer.loraDown, layer.loraUp)
            return layer
        }
        let layer = MiniMaxH3RuntimeLoRALinear(
            base: base,
            loraDown: down,
            loraUp: up,
            strength: strength
        )
        MLX.eval(layer.loraDown, layer.loraUp)
        return layer
    }

    private static func runtimeQKVLayer(
        base: Linear,
        query: LoRAPair,
        key: LoRAPair,
        value: LoRAPair,
        strength: Float
    ) -> Module {
        if let quantized = base as? QuantizedLinear {
            let layer = MiniMaxH3RuntimeQuantizedQKVLoRALinear(
                base: quantized,
                queryDown: query.down,
                queryUp: query.up,
                keyDown: key.down,
                keyUp: key.up,
                valueDown: value.down,
                valueUp: value.up,
                strength: strength
            )
            layer.evaluateAdapterParameters()
            return layer
        }
        let layer = MiniMaxH3RuntimeQKVLoRALinear(
            base: base,
            queryDown: query.down,
            queryUp: query.up,
            keyDown: key.down,
            keyUp: key.up,
            valueDown: value.down,
            valueUp: value.up,
            strength: strength
        )
        layer.evaluateAdapterParameters()
        return layer
    }

    private static func isAdaLNTarget(_ path: String) -> Bool {
        path == "final_layer.adaln_proj.linear"
            || (path.hasPrefix("blocks.") && path.hasSuffix(".adaln_proj.linear"))
    }

    private static func augmentedAdaLNCache(
        _ cache: MiniMaxH3AdaLNCache,
        pairs: [String: LoRAPair],
        configuration: MiniMaxH3TransformerConfiguration,
        strength: Float
    ) throws -> MiniMaxH3AdaLNCache {
        let activated = MLXNN.silu(cache.timeEmbeddings)
            .reshaped(cache.stepCount * 3, configuration.timeEmbeddingDimension)
        var blocks = cache.blockModulations
        var final = cache.finalModulations
        for (path, pair) in pairs {
            guard pair.down.dim(1) == configuration.timeEmbeddingDimension else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [pair.up.dim(0), configuration.timeEmbeddingDimension],
                    actual: [pair.up.dim(0), pair.down.dim(1)]
                )
            }
            let delta = MLX.matmul(
                MLX.matmul(activated.asType(pair.down.dtype), pair.down.T),
                pair.up.T
            ) * MLXArray(strength).asType(pair.down.dtype)
            if path == "final_layer.adaln_proj.linear" {
                guard pair.up.dim(0) == 2 * configuration.hiddenSize else {
                    throw AdapterError.targetShapeMismatch(
                        path,
                        expected: [2 * configuration.hiddenSize, configuration.timeEmbeddingDimension],
                        actual: [pair.up.dim(0), pair.down.dim(1)]
                    )
                }
                final = final + delta.reshaped(
                    cache.stepCount,
                    3,
                    2 * configuration.hiddenSize
                ).asType(final.dtype)
                continue
            }
            let components = path.split(separator: ".")
            guard components.count == 4,
                  components[0] == "blocks",
                  let index = Int(components[1]),
                  blocks.indices.contains(index),
                  pair.up.dim(0) == 18 * configuration.hiddenSize else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [18 * configuration.hiddenSize, configuration.timeEmbeddingDimension],
                    actual: [pair.up.dim(0), pair.down.dim(1)]
                )
            }
            blocks[index] = blocks[index] + delta.reshaped(
                cache.stepCount,
                9,
                6 * configuration.hiddenSize
            ).asType(blocks[index].dtype)
        }
        MLX.eval([final] + blocks)
        return MiniMaxH3AdaLNCache(
            timeEmbeddings: cache.timeEmbeddings,
            blockModulations: blocks,
            finalModulations: final,
            videoSigmas: cache.videoSigmas,
            audioSigmas: cache.audioSigmas,
            sourceIdentity: cache.sourceIdentity
        )
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

final class MiniMaxH3RuntimeQuantizedLoRALinear: QuantizedLinear {
    @ParameterInfo(key: "lora_down") var loraDown: MLXArray
    @ParameterInfo(key: "lora_up") var loraUp: MLXArray
    let strength: Float

    init(base: QuantizedLinear, loraDown: MLXArray, loraUp: MLXArray, strength: Float) {
        self._loraDown.wrappedValue = loraDown.asType(base.scales.dtype)
        self._loraUp.wrappedValue = loraUp.asType(base.scales.dtype)
        self.strength = strength
        super.init(
            weight: base.weight,
            bias: base.bias,
            scales: base.scales,
            biases: base.biases,
            groupSize: base.groupSize,
            bits: base.bits,
            mode: base.mode,
            globalScale: base.globalScale
        )
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MiniMaxH3RuntimeLoRAMath.project(
            input,
            down: loraDown,
            up: loraUp,
            strength: strength
        )
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}

final class MiniMaxH3RuntimeQKVLoRALinear: Linear {
    @ParameterInfo(key: "query_down") var queryDown: MLXArray
    @ParameterInfo(key: "query_up") var queryUp: MLXArray
    @ParameterInfo(key: "key_down") var keyDown: MLXArray
    @ParameterInfo(key: "key_up") var keyUp: MLXArray
    @ParameterInfo(key: "value_down") var valueDown: MLXArray
    @ParameterInfo(key: "value_up") var valueUp: MLXArray
    let strength: Float

    init(
        base: Linear,
        queryDown: MLXArray,
        queryUp: MLXArray,
        keyDown: MLXArray,
        keyUp: MLXArray,
        valueDown: MLXArray,
        valueUp: MLXArray,
        strength: Float
    ) {
        let dtype = base.weight.dtype
        self._queryDown.wrappedValue = queryDown.asType(dtype)
        self._queryUp.wrappedValue = queryUp.asType(dtype)
        self._keyDown.wrappedValue = keyDown.asType(dtype)
        self._keyUp.wrappedValue = keyUp.asType(dtype)
        self._valueDown.wrappedValue = valueDown.asType(dtype)
        self._valueUp.wrappedValue = valueUp.asType(dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        return baseOutput + adapterOutput(input).asType(baseOutput.dtype)
    }

    func evaluateAdapterParameters() {
        MLX.eval(queryDown, queryUp, keyDown, keyUp, valueDown, valueUp)
    }

    private func adapterOutput(_ input: MLXArray) -> MLXArray {
        MLX.concatenated([
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: queryDown,
                up: queryUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: keyDown,
                up: keyUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: valueDown,
                up: valueUp,
                strength: strength
            ),
        ], axis: -1)
    }
}

final class MiniMaxH3RuntimeQuantizedQKVLoRALinear: QuantizedLinear {
    @ParameterInfo(key: "query_down") var queryDown: MLXArray
    @ParameterInfo(key: "query_up") var queryUp: MLXArray
    @ParameterInfo(key: "key_down") var keyDown: MLXArray
    @ParameterInfo(key: "key_up") var keyUp: MLXArray
    @ParameterInfo(key: "value_down") var valueDown: MLXArray
    @ParameterInfo(key: "value_up") var valueUp: MLXArray
    let strength: Float

    init(
        base: QuantizedLinear,
        queryDown: MLXArray,
        queryUp: MLXArray,
        keyDown: MLXArray,
        keyUp: MLXArray,
        valueDown: MLXArray,
        valueUp: MLXArray,
        strength: Float
    ) {
        let dtype = base.scales.dtype
        self._queryDown.wrappedValue = queryDown.asType(dtype)
        self._queryUp.wrappedValue = queryUp.asType(dtype)
        self._keyDown.wrappedValue = keyDown.asType(dtype)
        self._keyUp.wrappedValue = keyUp.asType(dtype)
        self._valueDown.wrappedValue = valueDown.asType(dtype)
        self._valueUp.wrappedValue = valueUp.asType(dtype)
        self.strength = strength
        super.init(
            weight: base.weight,
            bias: base.bias,
            scales: base.scales,
            biases: base.biases,
            groupSize: base.groupSize,
            bits: base.bits,
            mode: base.mode,
            globalScale: base.globalScale
        )
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MLX.concatenated([
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: queryDown,
                up: queryUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: keyDown,
                up: keyUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: valueDown,
                up: valueUp,
                strength: strength
            ),
        ], axis: -1)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }

    func evaluateAdapterParameters() {
        MLX.eval(queryDown, queryUp, keyDown, keyUp, valueDown, valueUp)
    }
}

private enum MiniMaxH3RuntimeLoRAMath {
    static func project(
        _ input: MLXArray,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) -> MLXArray {
        MLX.matmul(
            MLX.matmul(input.asType(down.dtype), down.T),
            up.T
        ) * MLXArray(strength).asType(down.dtype)
    }
}
