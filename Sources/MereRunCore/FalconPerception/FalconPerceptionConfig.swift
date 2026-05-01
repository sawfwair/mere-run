import Foundation

public struct FalconPerceptionVisionConfig: Codable, Hashable, Sendable {
    public var modelType: String
    public var spatialPatchSize: Int
    public var temporalPatchSize: Int
    public var channelSize: Int

    public init(
        modelType: String = "falcon_perception",
        spatialPatchSize: Int = 16,
        temporalPatchSize: Int = 1,
        channelSize: Int = 3
    ) {
        self.modelType = modelType
        self.spatialPatchSize = spatialPatchSize
        self.temporalPatchSize = temporalPatchSize
        self.channelSize = channelSize
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case spatialPatchSize = "spatial_patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case channelSize = "channel_size"
    }
}

public struct FalconPerceptionTextConfig: Codable, Hashable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var headDim: Int
    public var numKeyValueHeads: Int
    public var vocabSize: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool

    public init(
        modelType: String = "falcon_perception",
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        headDim: Int = 128,
        numKeyValueHeads: Int = 8,
        vocabSize: Int = 65_536,
        intermediateSize: Int = 3072,
        rmsNormEps: Float = 1e-5,
        maxPositionEmbeddings: Int = 8192,
        ropeTheta: Float = 10_000,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.headDim = headDim
        self.numKeyValueHeads = numKeyValueHeads
        self.vocabSize = vocabSize
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
}

public struct FalconPerceptionModelConfig: Codable, Hashable, Sendable {
    public var textConfig: FalconPerceptionTextConfig
    public var visionConfig: FalconPerceptionVisionConfig
    public var modelType: String
    public var vocabSize: Int

    public var imgID: Int
    public var eosID: Int
    public var imageCLSTokenID: Int
    public var imageReg1TokenID: Int
    public var imageReg2TokenID: Int
    public var imageReg3TokenID: Int
    public var imageReg4TokenID: Int
    public var imgEndID: Int

    public var coordTokenID: Int
    public var sizeTokenID: Int
    public var segTokenID: Int

    public var coordEncDim: Int
    public var coordDecDim: Int
    public var coordOutDim: Int
    public var sizeEncDim: Int
    public var sizeDecDim: Int
    public var sizeOutDim: Int

    public var doSegmentation: Bool
    public var segmOutDim: Int
    public var numSegmLayers: Int

    public init(
        textConfig: FalconPerceptionTextConfig = FalconPerceptionTextConfig(),
        visionConfig: FalconPerceptionVisionConfig = FalconPerceptionVisionConfig(),
        modelType: String = "falcon_perception",
        vocabSize: Int = 65_536,
        imgID: Int = 227,
        eosID: Int = 11,
        imageCLSTokenID: Int = 244,
        imageReg1TokenID: Int = 245,
        imageReg2TokenID: Int = 246,
        imageReg3TokenID: Int = 247,
        imageReg4TokenID: Int = 248,
        imgEndID: Int = 230,
        coordTokenID: Int = 240,
        sizeTokenID: Int = 241,
        segTokenID: Int = 262,
        coordEncDim: Int = 512,
        coordDecDim: Int = 8192,
        coordOutDim: Int = 2048,
        sizeEncDim: Int = 512,
        sizeDecDim: Int = 8192,
        sizeOutDim: Int = 2048,
        doSegmentation: Bool = true,
        segmOutDim: Int = 256,
        numSegmLayers: Int = 3
    ) {
        self.textConfig = textConfig
        self.visionConfig = visionConfig
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.imgID = imgID
        self.eosID = eosID
        self.imageCLSTokenID = imageCLSTokenID
        self.imageReg1TokenID = imageReg1TokenID
        self.imageReg2TokenID = imageReg2TokenID
        self.imageReg3TokenID = imageReg3TokenID
        self.imageReg4TokenID = imageReg4TokenID
        self.imgEndID = imgEndID
        self.coordTokenID = coordTokenID
        self.sizeTokenID = sizeTokenID
        self.segTokenID = segTokenID
        self.coordEncDim = coordEncDim
        self.coordDecDim = coordDecDim
        self.coordOutDim = coordOutDim
        self.sizeEncDim = sizeEncDim
        self.sizeDecDim = sizeDecDim
        self.sizeOutDim = sizeOutDim
        self.doSegmentation = doSegmentation
        self.segmOutDim = segmOutDim
        self.numSegmLayers = numSegmLayers
    }

    private enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case imgID = "img_id"
        case eosID = "eos_id"
        case imageCLSTokenID = "image_cls_token_id"
        case imageReg1TokenID = "image_reg_1_token_id"
        case imageReg2TokenID = "image_reg_2_token_id"
        case imageReg3TokenID = "image_reg_3_token_id"
        case imageReg4TokenID = "image_reg_4_token_id"
        case imgEndID = "img_end_id"
        case coordTokenID = "coord_token_id"
        case sizeTokenID = "size_token_id"
        case segTokenID = "seg_token_id"
        case coordEncDim = "coord_enc_dim"
        case coordDecDim = "coord_dec_dim"
        case coordOutDim = "coord_out_dim"
        case sizeEncDim = "size_enc_dim"
        case sizeDecDim = "size_dec_dim"
        case sizeOutDim = "size_out_dim"
        case doSegmentation = "do_segmentation"
        case segmOutDim = "segm_out_dim"
        case numSegmLayers = "num_segm_layers"
    }

    private enum FlatCodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "dim"
        case numHiddenLayers = "n_layers"
        case numAttentionHeads = "n_heads"
        case headDim = "head_dim"
        case numKeyValueHeads = "n_kv_heads"
        case intermediateSize = "ffn_dim"
        case rmsNormEps = "norm_eps"
        case maxPositionEmbeddings = "max_seq_len"
        case ropeTheta = "rope_theta"
        case channelSize = "channel_size"
        case spatialPatchSize = "spatial_patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imgID = "img_id"
        case eosID = "eos_id"
        case imageCLSTokenID = "image_cls_token_id"
        case imageReg1TokenID = "image_reg_1_token_id"
        case imageReg2TokenID = "image_reg_2_token_id"
        case imageReg3TokenID = "image_reg_3_token_id"
        case imageReg4TokenID = "image_reg_4_token_id"
        case imgEndID = "img_end_id"
        case coordTokenID = "coord_token_id"
        case sizeTokenID = "size_token_id"
        case segTokenID = "seg_token_id"
        case coordEncDim = "coord_enc_dim"
        case coordDecDim = "coord_dec_dim"
        case coordOutDim = "coord_out_dim"
        case sizeEncDim = "size_enc_dim"
        case sizeDecDim = "size_dec_dim"
        case sizeOutDim = "size_out_dim"
        case doSegmentation = "do_segmentation"
        case segmOutDim = "segm_out_dim"
        case numSegmLayers = "num_segm_layers"
    }

    public init(from decoder: Decoder) throws {
        if let nested = try? decoder.container(keyedBy: CodingKeys.self),
           nested.contains(.textConfig) || nested.contains(.visionConfig) {
            let textConfig = try nested.decode(FalconPerceptionTextConfig.self, forKey: .textConfig)
            let visionConfig = try nested.decode(FalconPerceptionVisionConfig.self, forKey: .visionConfig)
            self = FalconPerceptionModelConfig(
                textConfig: textConfig,
                visionConfig: visionConfig,
                modelType: try nested.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_perception",
                vocabSize: try nested.decodeIfPresent(Int.self, forKey: .vocabSize) ?? textConfig.vocabSize,
                imgID: try nested.decodeIfPresent(Int.self, forKey: .imgID) ?? 227,
                eosID: try nested.decodeIfPresent(Int.self, forKey: .eosID) ?? 11,
                imageCLSTokenID: try nested.decodeIfPresent(Int.self, forKey: .imageCLSTokenID) ?? 244,
                imageReg1TokenID: try nested.decodeIfPresent(Int.self, forKey: .imageReg1TokenID) ?? 245,
                imageReg2TokenID: try nested.decodeIfPresent(Int.self, forKey: .imageReg2TokenID) ?? 246,
                imageReg3TokenID: try nested.decodeIfPresent(Int.self, forKey: .imageReg3TokenID) ?? 247,
                imageReg4TokenID: try nested.decodeIfPresent(Int.self, forKey: .imageReg4TokenID) ?? 248,
                imgEndID: try nested.decodeIfPresent(Int.self, forKey: .imgEndID) ?? 230,
                coordTokenID: try nested.decodeIfPresent(Int.self, forKey: .coordTokenID) ?? 240,
                sizeTokenID: try nested.decodeIfPresent(Int.self, forKey: .sizeTokenID) ?? 241,
                segTokenID: try nested.decodeIfPresent(Int.self, forKey: .segTokenID) ?? 262,
                coordEncDim: try nested.decodeIfPresent(Int.self, forKey: .coordEncDim) ?? 512,
                coordDecDim: try nested.decodeIfPresent(Int.self, forKey: .coordDecDim) ?? 8192,
                coordOutDim: try nested.decodeIfPresent(Int.self, forKey: .coordOutDim) ?? 2048,
                sizeEncDim: try nested.decodeIfPresent(Int.self, forKey: .sizeEncDim) ?? 512,
                sizeDecDim: try nested.decodeIfPresent(Int.self, forKey: .sizeDecDim) ?? 8192,
                sizeOutDim: try nested.decodeIfPresent(Int.self, forKey: .sizeOutDim) ?? 2048,
                doSegmentation: try nested.decodeIfPresent(Bool.self, forKey: .doSegmentation) ?? true,
                segmOutDim: try nested.decodeIfPresent(Int.self, forKey: .segmOutDim) ?? 256,
                numSegmLayers: try nested.decodeIfPresent(Int.self, forKey: .numSegmLayers) ?? 3
            )
            return
        }

        let flat = try decoder.container(keyedBy: FlatCodingKeys.self)
        let modelType = try flat.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_perception"
        let vocabSize = try flat.decode(Int.self, forKey: .vocabSize)

        let textConfig = FalconPerceptionTextConfig(
            modelType: modelType,
            hiddenSize: try flat.decode(Int.self, forKey: .hiddenSize),
            numHiddenLayers: try flat.decode(Int.self, forKey: .numHiddenLayers),
            numAttentionHeads: try flat.decode(Int.self, forKey: .numAttentionHeads),
            headDim: try flat.decode(Int.self, forKey: .headDim),
            numKeyValueHeads: try flat.decode(Int.self, forKey: .numKeyValueHeads),
            vocabSize: vocabSize,
            intermediateSize: try flat.decode(Int.self, forKey: .intermediateSize),
            rmsNormEps: try flat.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            maxPositionEmbeddings: try flat.decode(Int.self, forKey: .maxPositionEmbeddings),
            ropeTheta: try flat.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            tieWordEmbeddings: false
        )

        let visionConfig = FalconPerceptionVisionConfig(
            modelType: modelType,
            spatialPatchSize: try flat.decode(Int.self, forKey: .spatialPatchSize),
            temporalPatchSize: try flat.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1,
            channelSize: try flat.decodeIfPresent(Int.self, forKey: .channelSize) ?? 3
        )

        self = FalconPerceptionModelConfig(
            textConfig: textConfig,
            visionConfig: visionConfig,
            modelType: modelType,
            vocabSize: vocabSize,
            imgID: try flat.decodeIfPresent(Int.self, forKey: .imgID) ?? 227,
            eosID: try flat.decodeIfPresent(Int.self, forKey: .eosID) ?? 11,
            imageCLSTokenID: try flat.decodeIfPresent(Int.self, forKey: .imageCLSTokenID) ?? 244,
            imageReg1TokenID: try flat.decodeIfPresent(Int.self, forKey: .imageReg1TokenID) ?? 245,
            imageReg2TokenID: try flat.decodeIfPresent(Int.self, forKey: .imageReg2TokenID) ?? 246,
            imageReg3TokenID: try flat.decodeIfPresent(Int.self, forKey: .imageReg3TokenID) ?? 247,
            imageReg4TokenID: try flat.decodeIfPresent(Int.self, forKey: .imageReg4TokenID) ?? 248,
            imgEndID: try flat.decodeIfPresent(Int.self, forKey: .imgEndID) ?? 230,
            coordTokenID: try flat.decodeIfPresent(Int.self, forKey: .coordTokenID) ?? 240,
            sizeTokenID: try flat.decodeIfPresent(Int.self, forKey: .sizeTokenID) ?? 241,
            segTokenID: try flat.decodeIfPresent(Int.self, forKey: .segTokenID) ?? 262,
            coordEncDim: try flat.decodeIfPresent(Int.self, forKey: .coordEncDim) ?? 512,
            coordDecDim: try flat.decodeIfPresent(Int.self, forKey: .coordDecDim) ?? 8192,
            coordOutDim: try flat.decodeIfPresent(Int.self, forKey: .coordOutDim) ?? 2048,
            sizeEncDim: try flat.decodeIfPresent(Int.self, forKey: .sizeEncDim) ?? 512,
            sizeDecDim: try flat.decodeIfPresent(Int.self, forKey: .sizeDecDim) ?? 8192,
            sizeOutDim: try flat.decodeIfPresent(Int.self, forKey: .sizeOutDim) ?? 2048,
            doSegmentation: try flat.decodeIfPresent(Bool.self, forKey: .doSegmentation) ?? true,
            segmOutDim: try flat.decodeIfPresent(Int.self, forKey: .segmOutDim) ?? 256,
            numSegmLayers: try flat.decodeIfPresent(Int.self, forKey: .numSegmLayers) ?? 3
        )
    }

    public static func load(from url: URL) throws -> FalconPerceptionModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(FalconPerceptionModelConfig.self, from: data)
    }
}
