import Foundation
import MLX
import MLXFast
import MLXNN

public enum WooshResources {
    public static let dflowModelId = "sfx-woosh-dflow"
    public static let flowModelId = "sfx-woosh-flow"
    public static let clapModelId = "sfx-woosh-clap"
    public static let synchformerModelId = "sfx-woosh-synchformer"
    public static let vflow8sModelId = "sfx-woosh-vflow-8s"
    public static let dvflow8sModelId = "sfx-woosh-dvflow-8s"
    public static let upstreamRepoId = "SonyResearch/Woosh"
    public static let upstreamRelease = "v1.0.0"
    public static let huggingFaceMirrorRepoId = "AEmotionStudio/woosh-models"
    public static let synchformerRepoId = "Kijai/MMAudio_safetensors"
    public static let synchformerFilename = "mmaudio_synchformer_fp16.safetensors"
    public static let robertaTokenizerRepoId = "FacebookAI/roberta-large"
    public static let sampleRate = 48_000
    public static let latentChannels = 128
    public static let latentFrames5s = 501
    public static let latentFrames8s = 801

    public static let requiredDFlowComponents = [
        "Woosh-DFlow",
        "Woosh-AE",
        "TextConditionerA",
    ]

    public static let requiredFlowComponents = [
        "Woosh-Flow",
        "Woosh-AE",
        "TextConditionerA",
    ]

    public static let requiredVFlowComponents = [
        "Woosh-VFlow-8s",
        "Woosh-AE",
        "TextConditionerV",
    ]

    public static let requiredDVFlowComponents = [
        "Woosh-DVFlow-8s",
        "Woosh-AE",
        "TextConditionerV",
    ]

    public static func isWooshModel(_ value: String) -> Bool {
        value == dflowModelId
            || value == flowModelId
            || value == clapModelId
            || value == synchformerModelId
            || value == vflow8sModelId
            || value == dvflow8sModelId
    }

    public static func looksLikeWooshRoot(_ rootURL: URL, fileManager: FileManager = .default) -> Bool {
        let checkpoints = normalizeRoot(rootURL, fileManager: fileManager)
        return fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-AE/config.yaml").path)
            && (
                fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-DFlow/config.yaml").path)
                    || fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-Flow/config.yaml").path)
                    || fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-VFlow-8s/config.yaml").path)
                    || fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-DVFlow-8s/config.yaml").path)
            )
    }

    public static func normalizeRoot(_ rootURL: URL, fileManager: FileManager = .default) -> URL {
        let standardized = rootURL.standardizedFileURL
        let checkpoints = standardized.appendingPathComponent("checkpoints", isDirectory: true)
        if fileManager.fileExists(atPath: checkpoints.path) {
            return checkpoints
        }
        return standardized
    }
}

public enum WooshVariant: String, CaseIterable, Sendable, Hashable {
    case dflow = "Woosh-DFlow"
    case flow = "Woosh-Flow"
    case vflow8s = "Woosh-VFlow-8s"
    case dvflow8s = "Woosh-DVFlow-8s"

    public var managedModelId: String {
        switch self {
        case .dflow: WooshResources.dflowModelId
        case .flow: WooshResources.flowModelId
        case .vflow8s: WooshResources.vflow8sModelId
        case .dvflow8s: WooshResources.dvflow8sModelId
        }
    }

    public var requiredComponents: [String] {
        switch self {
        case .dflow: WooshResources.requiredDFlowComponents
        case .flow: WooshResources.requiredFlowComponents
        case .vflow8s: WooshResources.requiredVFlowComponents
        case .dvflow8s: WooshResources.requiredDVFlowComponents
        }
    }

    public var isDistilled: Bool {
        self == .dflow || self == .dvflow8s
    }

    public var defaultSteps: Int {
        switch self {
        case .dflow: 4
        case .flow: 32
        case .vflow8s: 32
        case .dvflow8s: 4
        }
    }

    public var defaultGuidanceScale: Float {
        4.5
    }

    public var defaultRenoiseSchedule: [Float] {
        switch self {
        case .dflow, .dvflow8s:
            [0, 0.5, 0.5, 0.3]
        case .flow, .vflow8s:
            []
        }
    }

    public static func resolve(model: String, rootURL: URL? = nil, fileManager: FileManager = .default) -> WooshVariant? {
        if model == WooshResources.dflowModelId {
            return .dflow
        }
        if model == WooshResources.flowModelId {
            return .flow
        }
        if model == WooshResources.vflow8sModelId {
            return .vflow8s
        }
        if model == WooshResources.dvflow8sModelId {
            return .dvflow8s
        }
        let url = rootURL ?? URL(fileURLWithPath: model)
        let checkpoints = WooshResources.normalizeRoot(url, fileManager: fileManager)
        if fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-DFlow/config.yaml").path) {
            return .dflow
        }
        if fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-Flow/config.yaml").path) {
            return .flow
        }
        if fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-DVFlow-8s/config.yaml").path) {
            return .dvflow8s
        }
        if fileManager.fileExists(atPath: checkpoints.appendingPathComponent("Woosh-VFlow-8s/config.yaml").path) {
            return .vflow8s
        }
        return nil
    }
}

public struct WooshDenoiseConfig: Sendable, Hashable {
    public let durationSeconds: Float
    public let steps: Int
    public let guidanceScale: Float
    public let seed: UInt64?
    public let renoiseSchedule: [Float]

    public init(
        durationSeconds: Float = 5,
        steps: Int,
        guidanceScale: Float = 4.5,
        seed: UInt64? = nil,
        renoiseSchedule: [Float] = []
    ) {
        self.durationSeconds = durationSeconds
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.renoiseSchedule = renoiseSchedule
    }

    public func latentFrames() -> Int {
        max(1, Int((durationSeconds * 100).rounded()) + 1)
    }
}

public struct WooshModelResources: Sendable, Hashable {
    public let checkpointsRootURL: URL
    public let variant: WooshVariant

    public init(checkpointsRootURL: URL, variant: WooshVariant) {
        self.checkpointsRootURL = checkpointsRootURL
        self.variant = variant
    }

    public var generatorURL: URL {
        checkpointsRootURL.appendingPathComponent(variant.rawValue, isDirectory: true)
    }

    public var autoencoderURL: URL {
        checkpointsRootURL.appendingPathComponent("Woosh-AE", isDirectory: true)
    }

    public var textConditionerURL: URL {
        if variant == .vflow8s || variant == .dvflow8s {
            return checkpointsRootURL.appendingPathComponent("TextConditionerV", isDirectory: true)
        }
        return checkpointsRootURL.appendingPathComponent("TextConditionerA", isDirectory: true)
    }

    public var generatorWeightsURL: URL {
        generatorURL.appendingPathComponent("weights.safetensors")
    }

    public var autoencoderWeightsURL: URL {
        autoencoderURL.appendingPathComponent("weights.safetensors")
    }

    public var textConditionerWeightsURL: URL {
        textConditionerURL.appendingPathComponent("weights.safetensors")
    }

    public var tokenizerRootURL: URL {
        let nested = textConditionerURL.appendingPathComponent("tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: nested.path) {
            return nested
        }
        return textConditionerURL
    }

    public func missingFiles(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for component in variant.requiredComponents {
            let root = checkpointsRootURL.appendingPathComponent(component, isDirectory: true)
            for filename in ["config.yaml", "weights.safetensors"] {
                let url = root.appendingPathComponent(filename)
                if !fileManager.fileExists(atPath: url.path) {
                    missing.append(url)
                }
            }
        }

        let tokenizerCandidates = [
            tokenizerRootURL.appendingPathComponent("tokenizer.json"),
            tokenizerRootURL.appendingPathComponent("vocab.json"),
        ]
        if !tokenizerCandidates.contains(where: { fileManager.fileExists(atPath: $0.path) }) {
            missing.append(tokenizerRootURL.appendingPathComponent("tokenizer.json"))
        }
        return missing
    }
}

public struct WooshCLAPResources: Sendable, Hashable {
    public let checkpointsRootURL: URL

    public init(checkpointsRootURL: URL) {
        self.checkpointsRootURL = checkpointsRootURL
    }

    public var clapURL: URL {
        checkpointsRootURL.appendingPathComponent("Woosh-CLAP", isDirectory: true)
    }

    public var tokenizerRootURL: URL {
        let nested = clapURL.appendingPathComponent("tokenizer", isDirectory: true)
        if FileManager.default.fileExists(atPath: nested.path) {
            return nested
        }
        return clapURL
    }

    public var textWeightsURL: URL {
        clapURL.appendingPathComponent("weights_text.safetensors")
    }

    public var audioWeightsURL: URL {
        clapURL.appendingPathComponent("weights_audio.safetensors")
    }

    public func missingFiles(fileManager: FileManager = .default) -> [URL] {
        var missing: [URL] = []
        for filename in ["config.yaml", "weights_text.safetensors", "weights_audio.safetensors"] {
            let url = clapURL.appendingPathComponent(filename)
            if !fileManager.fileExists(atPath: url.path) {
                missing.append(url)
            }
        }

        let tokenizerCandidates = [
            tokenizerRootURL.appendingPathComponent("tokenizer.json"),
            tokenizerRootURL.appendingPathComponent("vocab.json"),
        ]
        if !tokenizerCandidates.contains(where: { fileManager.fileExists(atPath: $0.path) }) {
            missing.append(tokenizerRootURL.appendingPathComponent("tokenizer.json"))
        }
        return missing
    }
}

public struct WooshSynchformerResources: Sendable, Hashable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL
    }

    public var weightsURL: URL {
        if rootURL.pathExtension == "safetensors" {
            return rootURL
        }
        return rootURL.appendingPathComponent(WooshResources.synchformerFilename)
    }

    public func missingFiles(fileManager: FileManager = .default) -> [URL] {
        fileManager.fileExists(atPath: weightsURL.path) ? [] : [weightsURL]
    }
}

public struct WooshDiTConfig: Sendable, Hashable {
    public let maxDescriptionLength: Int
    public let maxSeqLen: Int
    public let ropeLenMultiplier: Int?
    public let dim: Int
    public let interDim: Int
    public let fixedTimestepFeatures: Bool
    public let timestepFeaturesDim: Int
    public let nLayers: Int
    public let nHeads: Int
    public let nMultimodalLayers: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let qkvHeadDim: Int
    public let nMemoryTokensRope: Int
    public let nMemoryTokensDescription: Int
    public let originalSeqLen: Int
    public let ropeTheta: Float
    public let ropeFactor: Float
    public let betaFast: Int
    public let betaSlow: Int
    public let ioChannels: Int
    public let condTokenDim: Int
    public let adalnLastLayer: Bool
    public let adalnLastLayerNomod: Bool
    public let estimateLogvar: Bool
    public let noDescriptionMask: Bool
    public let patchSize: Int
    public let mlpAct: String
    public let maskOutBefore: Int

    public init(
        maxDescriptionLength: Int = 77,
        maxSeqLen: Int = 501,
        ropeLenMultiplier: Int? = 2,
        dim: Int = 1024,
        interDim: Int = 4096,
        fixedTimestepFeatures: Bool = false,
        timestepFeaturesDim: Int = 256,
        nLayers: Int = 12,
        nHeads: Int = 8,
        nMultimodalLayers: Int = 6,
        qkNopeHeadDim: Int = 16,
        qkRopeHeadDim: Int = 112,
        qkvHeadDim: Int = 128,
        nMemoryTokensRope: Int = 1,
        nMemoryTokensDescription: Int = 0,
        originalSeqLen: Int = 501,
        ropeTheta: Float = 10000,
        ropeFactor: Float = 40,
        betaFast: Int = 32,
        betaSlow: Int = 1,
        ioChannels: Int = 128,
        condTokenDim: Int = 1024,
        adalnLastLayer: Bool = true,
        adalnLastLayerNomod: Bool = false,
        estimateLogvar: Bool = true,
        noDescriptionMask: Bool = true,
        patchSize: Int = 1,
        mlpAct: String = "gelu",
        maskOutBefore: Int = -1
    ) {
        self.maxDescriptionLength = maxDescriptionLength
        self.maxSeqLen = maxSeqLen
        self.ropeLenMultiplier = ropeLenMultiplier
        self.dim = dim
        self.interDim = interDim
        self.fixedTimestepFeatures = fixedTimestepFeatures
        self.timestepFeaturesDim = timestepFeaturesDim
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.nMultimodalLayers = nMultimodalLayers
        self.qkNopeHeadDim = qkNopeHeadDim
        self.qkRopeHeadDim = qkRopeHeadDim
        self.qkvHeadDim = qkvHeadDim
        self.nMemoryTokensRope = nMemoryTokensRope
        self.nMemoryTokensDescription = nMemoryTokensDescription
        self.originalSeqLen = originalSeqLen
        self.ropeTheta = ropeTheta
        self.ropeFactor = ropeFactor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.ioChannels = ioChannels
        self.condTokenDim = condTokenDim
        self.adalnLastLayer = adalnLastLayer
        self.adalnLastLayerNomod = adalnLastLayerNomod
        self.estimateLogvar = estimateLogvar
        self.noDescriptionMask = noDescriptionMask
        self.patchSize = patchSize
        self.mlpAct = mlpAct
        self.maskOutBefore = maskOutBefore
    }

    public var headDim: Int {
        qkNopeHeadDim + qkRopeHeadDim
    }
}

public struct WooshRobertaConfig: Sendable, Hashable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let maxPositionEmbeddings: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
    public let padTokenId: Int
    public let lhsIndex: Int
    public let maxSentenceTokens: Int

    public init(
        vocabSize: Int = 50_265,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 4096,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        maxPositionEmbeddings: Int = 514,
        typeVocabSize: Int = 1,
        layerNormEps: Float = 1e-5,
        padTokenId: Int = 1,
        lhsIndex: Int = -2,
        maxSentenceTokens: Int = 77
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.typeVocabSize = typeVocabSize
        self.layerNormEps = layerNormEps
        self.padTokenId = padTokenId
        self.lhsIndex = lhsIndex
        self.maxSentenceTokens = maxSentenceTokens
    }
}

public struct WooshVocosConfig: Sendable, Hashable {
    public let zDim: Int
    public let dModel: Int
    public let intermediateDim: Int
    public let nFFT: Int
    public let hopLength: Int
    public let numLayers: Int
    public let inputLayerNorm: Bool
    public let finalLayerNorm: Bool

    public init(
        zDim: Int = 128,
        dModel: Int = 2048,
        intermediateDim: Int = 3072,
        nFFT: Int = 960,
        hopLength: Int = 480,
        numLayers: Int = 8,
        inputLayerNorm: Bool = true,
        finalLayerNorm: Bool = true
    ) {
        self.zDim = zDim
        self.dModel = dModel
        self.intermediateDim = intermediateDim
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.numLayers = numLayers
        self.inputLayerNorm = inputLayerNorm
        self.finalLayerNorm = finalLayerNorm
    }
}

public enum WooshError: LocalizedError, Sendable {
    case modelNotFound(String)
    case missingFiles([URL])
    case unsupportedVariant(String)
    case unsupportedFlowSampler
    case invalidRenoiseSchedule(expected: Int, actual: Int)
    case invalidTokenizer(URL)
    case invalidAudioShape([Int])
    case missingVideoFeatures
    case invalidVideoFeatureShape([Int])
    case invalidVideoShape([Int])
    case unsupportedVideoInput(URL)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let id):
            return "Woosh model not found: \(id)"
        case .missingFiles(let urls):
            return "Missing Woosh files:\n" + urls.map { "  - \($0.path)" }.joined(separator: "\n")
        case .unsupportedVariant(let value):
            return "Unsupported Woosh variant: \(value)"
        case .unsupportedFlowSampler:
            return "This Woosh sampler is not available in the native runtime."
        case .invalidRenoiseSchedule(let expected, let actual):
            return "Woosh renoise schedule must have \(expected) values; got \(actual)."
        case .invalidTokenizer(let url):
            return "Woosh text conditioner tokenizer files were not found under \(url.path). Add the roberta-large tokenizer files there."
        case .invalidAudioShape(let shape):
            return "Unsupported Woosh audio tensor shape: \(shape)"
        case .missingVideoFeatures:
            return "Woosh V2A generation requires video features with shape [batch, frames, 768]."
        case .invalidVideoFeatureShape(let shape):
            return "Unsupported Woosh video feature shape: \(shape); expected [batch, frames, 768]."
        case .invalidVideoShape(let shape):
            return "Unsupported Woosh video tensor shape: \(shape)."
        case .unsupportedVideoInput(let url):
            return "Unsupported video input for Woosh Synchformer: \(url.path)"
        }
    }
}

enum WooshTensorOps {
    static func geluTanh(_ x: MLXArray) -> MLXArray {
        let c = MLXArray(0.7978845608028654).asType(x.dtype)
        let k = MLXArray(0.044715).asType(x.dtype)
        return MLXArray(0.5).asType(x.dtype) * x * (MLXArray(1).asType(x.dtype) + MLX.tanh(c * (x + k * x * x * x)))
    }

    static func softplus(_ x: MLXArray) -> MLXArray {
        MLX.log(MLXArray(1).asType(x.dtype) + MLX.exp(-MLX.abs(x))) + MLX.maximum(x, MLXArray(0).asType(x.dtype))
    }

    static func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.mean((x - mean) * (x - mean), axis: -1, keepDims: true)
        return (x - mean) / MLX.sqrt(variance + MLXArray(eps).asType(x.dtype))
    }

    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let variance = MLX.mean(x * x, axis: -1, keepDims: true)
        return x * MLX.rsqrt(variance + MLXArray(eps).asType(x.dtype)) * weight
    }

    static func l2Normalize(_ x: MLXArray, eps: Float = 1e-12) -> MLXArray {
        let norm = MLX.sqrt(MLX.sum(x * x, axis: -1, keepDims: true) + MLXArray(eps).asType(x.dtype))
        return x / norm
    }

    static func conv1dWeightOIToOKI(_ value: MLXArray) -> MLXArray {
        guard value.ndim == 3 else { return value }
        let transposed = value.transposed(0, 2, 1)
        return transposed.reshaped(-1).reshaped(transposed.shape)
    }

    static func boolMaskToAttentionBias(_ mask: MLXArray, dtype: DType, queryLength: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let keyMask = mask.asType(dtype).expandedDimensions(axes: [1, 2])
        let zeros = MLXArray.zeros([mask.dim(0), 1, queryLength, mask.dim(1)], dtype: dtype)
        let negative = zeros + MLXArray(-1e9).asType(dtype)
        return .array(MLX.where(keyMask .> MLXArray(0).asType(dtype), zeros, negative))
    }
}
