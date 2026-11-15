import Foundation
import MereRunGemmaModel

package struct SOVTranscribeConfig: Decodable, Sendable {
    package let modelType: String
    package let audioConfig: SOVAudioConfig
    package let textConfig: Gemma4TextConfig
    package let audioTokenID: Int
    package let downsampleFactor: Int
    package let routerDtype: String
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", audioConfig = "audio_config", textConfig = "text_config"
        case audioTokenID = "audio_token_id", downsampleFactor = "downsample_factor", routerDtype = "router_dtype"
    }

    package func validate() throws {
        guard modelType == "sovereignai_transcribe", routerDtype == "float32",
              downsampleFactor > 0, audioConfig.maxSourcePositions > 0,
              audioConfig.maxSourcePositions.isMultiple(of: downsampleFactor),
              audioConfig.dModel > 0, audioConfig.encoderAttentionHeads > 0,
              audioConfig.dModel.isMultiple(of: audioConfig.encoderAttentionHeads),
              audioConfig.numMelBins == 128, audioConfig.encoderLayers > 0,
              audioConfig.encoderFFNDim > 0, textConfig.hiddenSizePerLayerInput == 0,
              textConfig.tieWordEmbeddings, (0..<textConfig.vocabSize).contains(audioTokenID) else {
            throw SOVTranscribeError.invalid("Unsupported SOV-Transcribe audio or text configuration.")
        }
    }
}

package struct SOVAudioConfig: Decodable, Sendable {
    package let dModel: Int
    package let encoderLayers: Int
    package let encoderAttentionHeads: Int
    package let encoderFFNDim: Int
    package let numMelBins: Int
    package let maxSourcePositions: Int
    enum CodingKeys: String, CodingKey {
        case dModel = "d_model", encoderLayers = "encoder_layers", encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim", numMelBins = "num_mel_bins", maxSourcePositions = "max_source_positions"
    }
}

package struct SOVDFlashConfig: Decodable, Sendable {
    package struct Draft: Decodable, Sendable {
        package let maskTokenID: Int
        package let targetLayerIDs: [Int]
        enum CodingKeys: String, CodingKey { case maskTokenID = "mask_token_id", targetLayerIDs = "target_layer_ids" }
    }
    struct Rope: Decodable, Sendable {
        let theta: Float
        let type: String
        enum CodingKeys: String, CodingKey { case theta = "rope_theta", type = "rope_type" }
    }
    package let hiddenSize: Int
    package let intermediateSize: Int
    package let numHiddenLayers: Int
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let headDim: Int
    package let rmsNormEps: Float
    package let vocabSize: Int
    package let slidingWindow: Int
    package let layerTypes: [String]
    package let blockSize: Int
    package let dflash: Draft
    let rope: Rope
    let hiddenAct: String
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", numKeyValueHeads = "num_key_value_heads", headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps", vocabSize = "vocab_size", slidingWindow = "sliding_window"
        case layerTypes = "layer_types", blockSize = "block_size", dflash = "dflash_config"
        case rope = "rope_parameters", hiddenAct = "hidden_act"
    }

    package func validate(target: Gemma4TextConfig) throws {
        guard hiddenSize == target.hiddenSize, vocabSize == target.vocabSize,
              numHiddenLayers > 0, layerTypes.count == numHiddenLayers,
              layerTypes.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }),
              numAttentionHeads > 0, numKeyValueHeads > 0, numAttentionHeads.isMultiple(of: numKeyValueHeads),
              headDim > 0, headDim.isMultiple(of: 2), intermediateSize > 0,
              slidingWindow > 0, blockSize > 1, blockSize <= 16, hiddenAct == "silu", rope.type == "default",
              rope.theta.isFinite, rope.theta > 0, rmsNormEps > 0,
              (0..<vocabSize).contains(dflash.maskTokenID), !dflash.targetLayerIDs.isEmpty,
              Set(dflash.targetLayerIDs).count == dflash.targetLayerIDs.count,
              dflash.targetLayerIDs.allSatisfy({ (0..<target.numHiddenLayers - 1).contains($0) }) else {
            throw SOVTranscribeError.invalid("DFlash configuration does not match the SOV-Transcribe target.")
        }
    }
}

package enum SOVTranscribeError: Error, LocalizedError {
    case invalid(String)
    package var errorDescription: String? { switch self { case .invalid(let message): message } }
}
