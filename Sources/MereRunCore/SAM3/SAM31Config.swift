import Foundation

public struct SAM31ModelConfig: Codable, Sendable {
    public var modelType: String
    public var detectorConfig: SAM31DetectorConfig
    public var trackerConfig: SAM31TrackerConfig?
    public var lowResMaskSize: Int
    public var detNMSThresh: Float
    public var assocIouThresh: Float
    public var trkAssocIouThresh: Float
    public var highConfThresh: Float
    public var highIouThresh: Float
    public var newDetThresh: Float
    public var scoreThresholdDetection: Float
    public var fillHoleArea: Int
    public var maxNumObjects: Int

    public init(
        modelType: String = "sam3.1_video",
        detectorConfig: SAM31DetectorConfig,
        trackerConfig: SAM31TrackerConfig? = nil,
        lowResMaskSize: Int = 288,
        detNMSThresh: Float = 0.1,
        assocIouThresh: Float = 0.1,
        trkAssocIouThresh: Float = 0.5,
        highConfThresh: Float = 0.8,
        highIouThresh: Float = 0.8,
        newDetThresh: Float = 0.7,
        scoreThresholdDetection: Float = 0.5,
        fillHoleArea: Int = 16,
        maxNumObjects: Int = 10_000
    ) {
        self.modelType = modelType
        self.detectorConfig = detectorConfig
        self.trackerConfig = trackerConfig
        self.lowResMaskSize = lowResMaskSize
        self.detNMSThresh = detNMSThresh
        self.assocIouThresh = assocIouThresh
        self.trkAssocIouThresh = trkAssocIouThresh
        self.highConfThresh = highConfThresh
        self.highIouThresh = highIouThresh
        self.newDetThresh = newDetThresh
        self.scoreThresholdDetection = scoreThresholdDetection
        self.fillHoleArea = fillHoleArea
        self.maxNumObjects = maxNumObjects
    }

    private enum CodingKeys: String, CodingKey {
        case modelType
        case detectorConfig
        case trackerConfig
        case lowResMaskSize
        case detNMSThresh = "detNmsThresh"
        case assocIouThresh
        case trkAssocIouThresh
        case highConfThresh
        case highIouThresh
        case newDetThresh
        case scoreThresholdDetection
        case fillHoleArea
        case maxNumObjects
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "sam3.1_video"
        self.detectorConfig = try container.decode(SAM31DetectorConfig.self, forKey: .detectorConfig)
        self.trackerConfig = try container.decodeIfPresent(SAM31TrackerConfig.self, forKey: .trackerConfig)
        self.lowResMaskSize = try container.decodeIfPresent(Int.self, forKey: .lowResMaskSize) ?? 288
        self.detNMSThresh = try container.decodeIfPresent(Float.self, forKey: .detNMSThresh) ?? 0.1
        self.assocIouThresh = try container.decodeIfPresent(Float.self, forKey: .assocIouThresh) ?? 0.1
        self.trkAssocIouThresh = try container.decodeIfPresent(Float.self, forKey: .trkAssocIouThresh) ?? 0.5
        self.highConfThresh = try container.decodeIfPresent(Float.self, forKey: .highConfThresh) ?? 0.8
        self.highIouThresh = try container.decodeIfPresent(Float.self, forKey: .highIouThresh) ?? 0.8
        self.newDetThresh = try container.decodeIfPresent(Float.self, forKey: .newDetThresh) ?? 0.7
        self.scoreThresholdDetection = try container.decodeIfPresent(Float.self, forKey: .scoreThresholdDetection) ?? 0.5
        self.fillHoleArea = try container.decodeIfPresent(Int.self, forKey: .fillHoleArea) ?? 16
        self.maxNumObjects = try container.decodeIfPresent(Int.self, forKey: .maxNumObjects) ?? 10_000
    }
}

public struct SAM31DetectorConfig: Codable, Sendable {
    public var modelType: String
    public var visionConfig: SAM31VisionEncoderConfig
    public var textConfig: SAM31TextEncoderConfig
    public var detrEncoderConfig: SAM31DETREncoderConfig
    public var detrDecoderConfig: SAM31DETRDecoderConfig
    public var geometryEncoderConfig: SAM31GeometryEncoderConfig?
    public var maskDecoderConfig: SAM31DetectorMaskDecoderConfig

    public init(
        modelType: String = "sam3.1",
        visionConfig: SAM31VisionEncoderConfig,
        textConfig: SAM31TextEncoderConfig,
        detrEncoderConfig: SAM31DETREncoderConfig,
        detrDecoderConfig: SAM31DETRDecoderConfig,
        geometryEncoderConfig: SAM31GeometryEncoderConfig? = nil,
        maskDecoderConfig: SAM31DetectorMaskDecoderConfig
    ) {
        self.modelType = modelType
        self.visionConfig = visionConfig
        self.textConfig = textConfig
        self.detrEncoderConfig = detrEncoderConfig
        self.detrDecoderConfig = detrDecoderConfig
        self.geometryEncoderConfig = geometryEncoderConfig
        self.maskDecoderConfig = maskDecoderConfig
    }
}

public struct SAM31ViTConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var imageSize: Int
    public var patchSize: Int
    public var numChannels: Int
    public var windowSize: Int
    public var globalAttnIndexes: [Int]
    public var qkvBias: Bool
    public var ropeTheta: Float
    public var pretrainImageSize: Int
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_vit_model",
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 32,
        numAttentionHeads: Int = 16,
        intermediateSize: Int = 4736,
        imageSize: Int = 1008,
        patchSize: Int = 14,
        numChannels: Int = 3,
        windowSize: Int = 24,
        globalAttnIndexes: [Int] = [7, 15, 23, 31],
        qkvBias: Bool = true,
        ropeTheta: Float = 10_000,
        pretrainImageSize: Int = 336,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.windowSize = windowSize
        self.globalAttnIndexes = globalAttnIndexes
        self.qkvBias = qkvBias
        self.ropeTheta = ropeTheta
        self.pretrainImageSize = pretrainImageSize
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31VisionEncoderConfig: Codable, Sendable {
    public var modelType: String
    public var backboneConfig: SAM31ViTConfig
    public var fpnHiddenSize: Int
    public var fpnKernelSize: Int
    public var fpnStride: Int
    public var scaleFactors: [Float]
    public var numFeatureLevels: Int
    public var backboneFeatureSizes: [[Int]]
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_vision_model",
        backboneConfig: SAM31ViTConfig,
        fpnHiddenSize: Int = 256,
        fpnKernelSize: Int = 2,
        fpnStride: Int = 2,
        scaleFactors: [Float] = [4.0, 2.0, 1.0],
        numFeatureLevels: Int = 3,
        backboneFeatureSizes: [[Int]] = [[288, 288], [144, 144], [72, 72]],
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.backboneConfig = backboneConfig
        self.fpnHiddenSize = fpnHiddenSize
        self.fpnKernelSize = fpnKernelSize
        self.fpnStride = fpnStride
        self.scaleFactors = scaleFactors
        self.numFeatureLevels = numFeatureLevels
        self.backboneFeatureSizes = backboneFeatureSizes
        self.layerNormEps = layerNormEps
    }

    private enum CodingKeys: String, CodingKey {
        case modelType
        case backboneConfig
        case fpnHiddenSize
        case fpnKernelSize
        case fpnStride
        case scaleFactors
        case numFeatureLevels
        case backboneFeatureSizes
        case layerNormEps
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "sam3_vision_model"
        self.backboneConfig = try container.decode(SAM31ViTConfig.self, forKey: .backboneConfig)
        self.fpnHiddenSize = try container.decodeIfPresent(Int.self, forKey: .fpnHiddenSize) ?? 256
        self.fpnKernelSize = try container.decodeIfPresent(Int.self, forKey: .fpnKernelSize) ?? 2
        self.fpnStride = try container.decodeIfPresent(Int.self, forKey: .fpnStride) ?? 2
        self.scaleFactors = try container.decodeIfPresent([Float].self, forKey: .scaleFactors) ?? [4.0, 2.0, 1.0]
        self.numFeatureLevels = try container.decodeIfPresent(Int.self, forKey: .numFeatureLevels) ?? 3
        self.backboneFeatureSizes = try container.decodeIfPresent([[Int]].self, forKey: .backboneFeatureSizes) ?? [[288, 288], [144, 144], [72, 72]]
        self.layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

public struct SAM31TextEncoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var maxPositionEmbeddings: Int
    public var projectionDim: Int
    public var layerNormEps: Float
    public var padTokenId: Int

    public init(
        modelType: String = "clip_text_model",
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        intermediateSize: Int = 4096,
        vocabSize: Int = 49_408,
        maxPositionEmbeddings: Int = 32,
        projectionDim: Int = 512,
        layerNormEps: Float = 1e-5,
        padTokenId: Int = 1
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.projectionDim = projectionDim
        self.layerNormEps = layerNormEps
        self.padTokenId = padTokenId
    }
}

public struct SAM31DETREncoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var dropout: Float
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_detr_encoder",
        hiddenSize: Int = 256,
        numLayers: Int = 6,
        numAttentionHeads: Int = 8,
        intermediateSize: Int = 2048,
        dropout: Float = 0.1,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.dropout = dropout
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31DETRDecoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numLayers: Int
    public var numAttentionHeads: Int
    public var numQueries: Int
    public var intermediateSize: Int
    public var dropout: Float
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_detr_decoder",
        hiddenSize: Int = 256,
        numLayers: Int = 6,
        numAttentionHeads: Int = 8,
        numQueries: Int = 200,
        intermediateSize: Int = 2048,
        dropout: Float = 0.1,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.numQueries = numQueries
        self.intermediateSize = intermediateSize
        self.dropout = dropout
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31GeometryEncoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var dropout: Float
    public var roiSize: Int
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_geometry_encoder",
        hiddenSize: Int = 256,
        numLayers: Int = 3,
        numAttentionHeads: Int = 8,
        intermediateSize: Int = 2048,
        dropout: Float = 0.1,
        roiSize: Int = 7,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.dropout = dropout
        self.roiSize = roiSize
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31DetectorMaskDecoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numAttentionHeads: Int
    public var numUpsamplingStages: Int
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_mask_decoder",
        hiddenSize: Int = 256,
        numAttentionHeads: Int = 8,
        numUpsamplingStages: Int = 3,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.numUpsamplingStages = numUpsamplingStages
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31PromptEncoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var imageSize: Int
    public var patchSize: Int
    public var maskInputChannels: Int
    public var numPointEmbeddings: Int
    public var hiddenAct: String
    public var scale: Int
    public var layerNormEps: Float

    public init(
        modelType: String = "sam3_prompt_encoder",
        hiddenSize: Int = 256,
        imageSize: Int = 1008,
        patchSize: Int = 14,
        maskInputChannels: Int = 16,
        numPointEmbeddings: Int = 4,
        hiddenAct: String = "gelu",
        scale: Int = 1,
        layerNormEps: Float = 1e-6
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.maskInputChannels = maskInputChannels
        self.numPointEmbeddings = numPointEmbeddings
        self.hiddenAct = hiddenAct
        self.scale = scale
        self.layerNormEps = layerNormEps
    }
}

public struct SAM31TrackerMaskDecoderConfig: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var attentionDownsampleRate: Int
    public var numMultimaskOutputs: Int
    public var iouHeadDepth: Int
    public var iouHeadHiddenDim: Int
    public var mlpDim: Int
    public var hiddenAct: String
    public var dynamicMultimaskViaStability: Bool
    public var dynamicMultimaskStabilityDelta: Float
    public var dynamicMultimaskStabilityThresh: Float
    public var multiplexCount: Int

    public init(
        modelType: String = "sam3_tracker_mask_decoder",
        hiddenSize: Int = 256,
        numHiddenLayers: Int = 2,
        numAttentionHeads: Int = 8,
        attentionDownsampleRate: Int = 2,
        numMultimaskOutputs: Int = 3,
        iouHeadDepth: Int = 3,
        iouHeadHiddenDim: Int = 256,
        mlpDim: Int = 2048,
        hiddenAct: String = "gelu",
        dynamicMultimaskViaStability: Bool = true,
        dynamicMultimaskStabilityDelta: Float = 0.05,
        dynamicMultimaskStabilityThresh: Float = 0.98,
        multiplexCount: Int = 16
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.attentionDownsampleRate = attentionDownsampleRate
        self.numMultimaskOutputs = numMultimaskOutputs
        self.iouHeadDepth = iouHeadDepth
        self.iouHeadHiddenDim = iouHeadHiddenDim
        self.mlpDim = mlpDim
        self.hiddenAct = hiddenAct
        self.dynamicMultimaskViaStability = dynamicMultimaskViaStability
        self.dynamicMultimaskStabilityDelta = dynamicMultimaskStabilityDelta
        self.dynamicMultimaskStabilityThresh = dynamicMultimaskStabilityThresh
        self.multiplexCount = multiplexCount
    }

    private enum CodingKeys: String, CodingKey {
        case modelType
        case hiddenSize
        case numHiddenLayers
        case numAttentionHeads
        case attentionDownsampleRate
        case numMultimaskOutputs
        case iouHeadDepth
        case iouHeadHiddenDim
        case mlpDim
        case hiddenAct
        case dynamicMultimaskViaStability
        case dynamicMultimaskStabilityDelta
        case dynamicMultimaskStabilityThresh
        case multiplexCount
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "sam3_tracker_mask_decoder"
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 256
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 2
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        self.attentionDownsampleRate = try container.decodeIfPresent(Int.self, forKey: .attentionDownsampleRate) ?? 2
        self.numMultimaskOutputs = try container.decodeIfPresent(Int.self, forKey: .numMultimaskOutputs) ?? 3
        self.iouHeadDepth = try container.decodeIfPresent(Int.self, forKey: .iouHeadDepth) ?? 3
        self.iouHeadHiddenDim = try container.decodeIfPresent(Int.self, forKey: .iouHeadHiddenDim) ?? 256
        self.mlpDim = try container.decodeIfPresent(Int.self, forKey: .mlpDim) ?? 2048
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        self.dynamicMultimaskViaStability = try container.decodeIfPresent(Bool.self, forKey: .dynamicMultimaskViaStability) ?? true
        self.dynamicMultimaskStabilityDelta = try container.decodeIfPresent(Float.self, forKey: .dynamicMultimaskStabilityDelta) ?? 0.05
        self.dynamicMultimaskStabilityThresh = try container.decodeIfPresent(Float.self, forKey: .dynamicMultimaskStabilityThresh) ?? 0.98
        self.multiplexCount = try container.decodeIfPresent(Int.self, forKey: .multiplexCount) ?? 16
    }
}

public struct SAM31TrackerConfig: Codable, Sendable {
    public var modelType: String
    public var imageSize: Int
    public var visionConfig: SAM31VisionEncoderConfig
    public var maskDecoderConfig: SAM31TrackerMaskDecoderConfig
    public var promptEncoderConfig: SAM31PromptEncoderConfig
    public var multiplexCount: Int
    public var memoryAttentionHiddenSize: Int
    public var memoryAttentionNumLayers: Int
    public var memoryAttentionNumAttentionHeads: Int
    public var memoryAttentionFeedForwardHiddenSize: Int
    public var memoryAttentionFeedForwardHiddenAct: String
    public var memoryAttentionDropout: Float
    public var memoryAttentionRopeDropout: Float
    public var memoryAttentionRopeTheta: Float
    public var memoryAttentionRopeFeatSizes: [Int]
    public var memoryAttentionDownsampleRate: Int
    public var memoryEncoderHiddenSize: Int
    public var memoryEncoderOutputChannels: Int
    public var maskDownsamplerEmbedDim: Int
    public var maskDownsamplerKernelSize: Int
    public var maskDownsamplerStride: Int
    public var maskDownsamplerPadding: Int
    public var maskDownsamplerTotalStride: Int
    public var maskDownsamplerHiddenAct: String
    public var maskDownsamplerFirstChannels: Int
    public var memoryFuserEmbedDim: Int
    public var memoryFuserKernelSize: Int
    public var memoryFuserPadding: Int
    public var memoryFuserNumLayers: Int
    public var memoryFuserIntermediateDim: Int
    public var memoryFuserLayerScaleInitValue: Float
    public var memoryFuserHiddenAct: String
    public var numMaskmem: Int
    public var maxCondFrameNum: Int
    public var maxObjectPointersInEncoder: Int
    public var multimaskOutputInSAM: Bool
    public var multimaskOutputForTracking: Bool
    public var multimaskMinPtNum: Int
    public var multimaskMaxPtNum: Int
    public var sigmoidBiasForMemEnc: Float
    public var sigmoidScaleForMemEnc: Float
    public var enableOcclusionSpatialEmbedding: Bool
    public var enableTemporalPosEncodingForObjectPointers: Bool

    public init(
        modelType: String = "sam3.1_tracker_video",
        imageSize: Int = 1008,
        visionConfig: SAM31VisionEncoderConfig,
        maskDecoderConfig: SAM31TrackerMaskDecoderConfig = SAM31TrackerMaskDecoderConfig(),
        promptEncoderConfig: SAM31PromptEncoderConfig = SAM31PromptEncoderConfig(),
        multiplexCount: Int = 16,
        memoryAttentionHiddenSize: Int = 256,
        memoryAttentionNumLayers: Int = 4,
        memoryAttentionNumAttentionHeads: Int = 1,
        memoryAttentionFeedForwardHiddenSize: Int = 2048,
        memoryAttentionFeedForwardHiddenAct: String = "relu",
        memoryAttentionDropout: Float = 0.1,
        memoryAttentionRopeDropout: Float = 0.1,
        memoryAttentionRopeTheta: Float = 10_000,
        memoryAttentionRopeFeatSizes: [Int] = [72, 72],
        memoryAttentionDownsampleRate: Int = 1,
        memoryEncoderHiddenSize: Int = 256,
        memoryEncoderOutputChannels: Int = 64,
        maskDownsamplerEmbedDim: Int = 256,
        maskDownsamplerKernelSize: Int = 3,
        maskDownsamplerStride: Int = 2,
        maskDownsamplerPadding: Int = 1,
        maskDownsamplerTotalStride: Int = 16,
        maskDownsamplerHiddenAct: String = "gelu",
        maskDownsamplerFirstChannels: Int = 16,
        memoryFuserEmbedDim: Int = 256,
        memoryFuserKernelSize: Int = 7,
        memoryFuserPadding: Int = 3,
        memoryFuserNumLayers: Int = 2,
        memoryFuserIntermediateDim: Int = 1024,
        memoryFuserLayerScaleInitValue: Float = 1e-6,
        memoryFuserHiddenAct: String = "gelu",
        numMaskmem: Int = 7,
        maxCondFrameNum: Int = 4,
        maxObjectPointersInEncoder: Int = 16,
        multimaskOutputInSAM: Bool = true,
        multimaskOutputForTracking: Bool = true,
        multimaskMinPtNum: Int = 0,
        multimaskMaxPtNum: Int = 1,
        sigmoidBiasForMemEnc: Float = -10,
        sigmoidScaleForMemEnc: Float = 20,
        enableOcclusionSpatialEmbedding: Bool = true,
        enableTemporalPosEncodingForObjectPointers: Bool = true
    ) {
        self.modelType = modelType
        self.imageSize = imageSize
        self.visionConfig = visionConfig
        self.maskDecoderConfig = maskDecoderConfig
        self.promptEncoderConfig = promptEncoderConfig
        self.multiplexCount = multiplexCount
        self.memoryAttentionHiddenSize = memoryAttentionHiddenSize
        self.memoryAttentionNumLayers = memoryAttentionNumLayers
        self.memoryAttentionNumAttentionHeads = memoryAttentionNumAttentionHeads
        self.memoryAttentionFeedForwardHiddenSize = memoryAttentionFeedForwardHiddenSize
        self.memoryAttentionFeedForwardHiddenAct = memoryAttentionFeedForwardHiddenAct
        self.memoryAttentionDropout = memoryAttentionDropout
        self.memoryAttentionRopeDropout = memoryAttentionRopeDropout
        self.memoryAttentionRopeTheta = memoryAttentionRopeTheta
        self.memoryAttentionRopeFeatSizes = memoryAttentionRopeFeatSizes
        self.memoryAttentionDownsampleRate = memoryAttentionDownsampleRate
        self.memoryEncoderHiddenSize = memoryEncoderHiddenSize
        self.memoryEncoderOutputChannels = memoryEncoderOutputChannels
        self.maskDownsamplerEmbedDim = maskDownsamplerEmbedDim
        self.maskDownsamplerKernelSize = maskDownsamplerKernelSize
        self.maskDownsamplerStride = maskDownsamplerStride
        self.maskDownsamplerPadding = maskDownsamplerPadding
        self.maskDownsamplerTotalStride = maskDownsamplerTotalStride
        self.maskDownsamplerHiddenAct = maskDownsamplerHiddenAct
        self.maskDownsamplerFirstChannels = maskDownsamplerFirstChannels
        self.memoryFuserEmbedDim = memoryFuserEmbedDim
        self.memoryFuserKernelSize = memoryFuserKernelSize
        self.memoryFuserPadding = memoryFuserPadding
        self.memoryFuserNumLayers = memoryFuserNumLayers
        self.memoryFuserIntermediateDim = memoryFuserIntermediateDim
        self.memoryFuserLayerScaleInitValue = memoryFuserLayerScaleInitValue
        self.memoryFuserHiddenAct = memoryFuserHiddenAct
        self.numMaskmem = numMaskmem
        self.maxCondFrameNum = maxCondFrameNum
        self.maxObjectPointersInEncoder = maxObjectPointersInEncoder
        self.multimaskOutputInSAM = multimaskOutputInSAM
        self.multimaskOutputForTracking = multimaskOutputForTracking
        self.multimaskMinPtNum = multimaskMinPtNum
        self.multimaskMaxPtNum = multimaskMaxPtNum
        self.sigmoidBiasForMemEnc = sigmoidBiasForMemEnc
        self.sigmoidScaleForMemEnc = sigmoidScaleForMemEnc
        self.enableOcclusionSpatialEmbedding = enableOcclusionSpatialEmbedding
        self.enableTemporalPosEncodingForObjectPointers = enableTemporalPosEncodingForObjectPointers
    }

    private enum CodingKeys: String, CodingKey {
        case modelType
        case imageSize
        case visionConfig
        case maskDecoderConfig
        case promptEncoderConfig
        case multiplexCount
        case memoryAttentionHiddenSize
        case memoryAttentionNumLayers
        case memoryAttentionNumAttentionHeads
        case memoryAttentionFeedForwardHiddenSize
        case memoryAttentionFeedForwardHiddenAct
        case memoryAttentionDropout
        case memoryAttentionRopeDropout
        case memoryAttentionRopeTheta
        case memoryAttentionRopeFeatSizes
        case memoryAttentionDownsampleRate
        case memoryEncoderHiddenSize
        case memoryEncoderOutputChannels
        case maskDownsamplerEmbedDim
        case maskDownsamplerKernelSize
        case maskDownsamplerStride
        case maskDownsamplerPadding
        case maskDownsamplerTotalStride
        case maskDownsamplerHiddenAct
        case maskDownsamplerFirstChannels
        case memoryFuserEmbedDim
        case memoryFuserKernelSize
        case memoryFuserPadding
        case memoryFuserNumLayers
        case memoryFuserIntermediateDim
        case memoryFuserLayerScaleInitValue
        case memoryFuserHiddenAct
        case numMaskmem
        case maxCondFrameNum
        case maxObjectPointersInEncoder
        case multimaskOutputInSAM
        case multimaskOutputForTracking
        case multimaskMinPtNum
        case multimaskMaxPtNum
        case sigmoidBiasForMemEnc
        case sigmoidScaleForMemEnc
        case enableOcclusionSpatialEmbedding
        case enableTemporalPosEncodingForObjectPointers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "sam3.1_tracker_video"
        self.imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 1008
        self.visionConfig = try container.decodeIfPresent(SAM31VisionEncoderConfig.self, forKey: .visionConfig) ?? SAM31VisionEncoderConfig(backboneConfig: SAM31ViTConfig())
        self.maskDecoderConfig = try container.decodeIfPresent(SAM31TrackerMaskDecoderConfig.self, forKey: .maskDecoderConfig) ?? SAM31TrackerMaskDecoderConfig()
        self.promptEncoderConfig = try container.decodeIfPresent(SAM31PromptEncoderConfig.self, forKey: .promptEncoderConfig) ?? SAM31PromptEncoderConfig()
        self.multiplexCount = try container.decodeIfPresent(Int.self, forKey: .multiplexCount) ?? 16
        self.memoryAttentionHiddenSize = try container.decodeIfPresent(Int.self, forKey: .memoryAttentionHiddenSize) ?? 256
        self.memoryAttentionNumLayers = try container.decodeIfPresent(Int.self, forKey: .memoryAttentionNumLayers) ?? 4
        self.memoryAttentionNumAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .memoryAttentionNumAttentionHeads) ?? 1
        self.memoryAttentionFeedForwardHiddenSize = try container.decodeIfPresent(Int.self, forKey: .memoryAttentionFeedForwardHiddenSize) ?? 2048
        self.memoryAttentionFeedForwardHiddenAct = try container.decodeIfPresent(String.self, forKey: .memoryAttentionFeedForwardHiddenAct) ?? "relu"
        self.memoryAttentionDropout = try container.decodeIfPresent(Float.self, forKey: .memoryAttentionDropout) ?? 0.1
        self.memoryAttentionRopeDropout = try container.decodeIfPresent(Float.self, forKey: .memoryAttentionRopeDropout) ?? 0.1
        self.memoryAttentionRopeTheta = try container.decodeIfPresent(Float.self, forKey: .memoryAttentionRopeTheta) ?? 10_000
        self.memoryAttentionRopeFeatSizes = try container.decodeIfPresent([Int].self, forKey: .memoryAttentionRopeFeatSizes) ?? [72, 72]
        self.memoryAttentionDownsampleRate = try container.decodeIfPresent(Int.self, forKey: .memoryAttentionDownsampleRate) ?? 1
        self.memoryEncoderHiddenSize = try container.decodeIfPresent(Int.self, forKey: .memoryEncoderHiddenSize) ?? 256
        self.memoryEncoderOutputChannels = try container.decodeIfPresent(Int.self, forKey: .memoryEncoderOutputChannels) ?? 64
        self.maskDownsamplerEmbedDim = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerEmbedDim) ?? 256
        self.maskDownsamplerKernelSize = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerKernelSize) ?? 3
        self.maskDownsamplerStride = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerStride) ?? 2
        self.maskDownsamplerPadding = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerPadding) ?? 1
        self.maskDownsamplerTotalStride = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerTotalStride) ?? 16
        self.maskDownsamplerHiddenAct = try container.decodeIfPresent(String.self, forKey: .maskDownsamplerHiddenAct) ?? "gelu"
        self.maskDownsamplerFirstChannels = try container.decodeIfPresent(Int.self, forKey: .maskDownsamplerFirstChannels) ?? 16
        self.memoryFuserEmbedDim = try container.decodeIfPresent(Int.self, forKey: .memoryFuserEmbedDim) ?? 256
        self.memoryFuserKernelSize = try container.decodeIfPresent(Int.self, forKey: .memoryFuserKernelSize) ?? 7
        self.memoryFuserPadding = try container.decodeIfPresent(Int.self, forKey: .memoryFuserPadding) ?? 3
        self.memoryFuserNumLayers = try container.decodeIfPresent(Int.self, forKey: .memoryFuserNumLayers) ?? 2
        self.memoryFuserIntermediateDim = try container.decodeIfPresent(Int.self, forKey: .memoryFuserIntermediateDim) ?? 1024
        self.memoryFuserLayerScaleInitValue = try container.decodeIfPresent(Float.self, forKey: .memoryFuserLayerScaleInitValue) ?? 1e-6
        self.memoryFuserHiddenAct = try container.decodeIfPresent(String.self, forKey: .memoryFuserHiddenAct) ?? "gelu"
        self.numMaskmem = try container.decodeIfPresent(Int.self, forKey: .numMaskmem) ?? 7
        self.maxCondFrameNum = try container.decodeIfPresent(Int.self, forKey: .maxCondFrameNum) ?? 4
        self.maxObjectPointersInEncoder = try container.decodeIfPresent(Int.self, forKey: .maxObjectPointersInEncoder) ?? 16
        self.multimaskOutputInSAM = try container.decodeIfPresent(Bool.self, forKey: .multimaskOutputInSAM) ?? true
        self.multimaskOutputForTracking = try container.decodeIfPresent(Bool.self, forKey: .multimaskOutputForTracking) ?? true
        self.multimaskMinPtNum = try container.decodeIfPresent(Int.self, forKey: .multimaskMinPtNum) ?? 0
        self.multimaskMaxPtNum = try container.decodeIfPresent(Int.self, forKey: .multimaskMaxPtNum) ?? 1
        self.sigmoidBiasForMemEnc = try container.decodeIfPresent(Float.self, forKey: .sigmoidBiasForMemEnc) ?? -10
        self.sigmoidScaleForMemEnc = try container.decodeIfPresent(Float.self, forKey: .sigmoidScaleForMemEnc) ?? 20
        self.enableOcclusionSpatialEmbedding = try container.decodeIfPresent(Bool.self, forKey: .enableOcclusionSpatialEmbedding) ?? true
        self.enableTemporalPosEncodingForObjectPointers = try container.decodeIfPresent(Bool.self, forKey: .enableTemporalPosEncodingForObjectPointers) ?? true
    }
}

extension SAM31ModelConfig {
    public static func load(from url: URL) throws -> SAM31ModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(SAM31ModelConfig.self, from: data)
    }
}
