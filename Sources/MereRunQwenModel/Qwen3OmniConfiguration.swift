import Foundation

/// The inference-only thinker configuration. Speech-output networks are deliberately not instantiated.
public struct Qwen3OmniConfiguration: Decodable, Sendable {
    public struct Text: Decodable, Sendable {
        public let hiddenSize: Int
        public let vocabSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let moeIntermediateSize: Int
        public let numExperts: Int
        public let numExpertsPerTok: Int
        public let normTopkProb: Bool
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let attentionBias: Bool
        public let decoderSparseStep: Int
        public let mlpOnlyLayers: [Int]
        public let maxPositionEmbeddings: Int
        public let ropeScaling: Rope
        public struct Rope: Decodable, Sendable {
            public let mropeSection: [Int]
            public let mropeInterleaved: Bool
            public let ropeType: String
        }
    }
    public struct Vision: Decodable, Sendable {
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let temporalPatchSize: Int
        public let spatialMergeSize: Int
        public let outHiddenSize: Int
        public let imageSize: Int
        public let deepstackVisualIndexes: [Int]
    }
    public struct Audio: Decodable, Sendable {
        public let dModel: Int
        public let encoderLayers: Int
        public let encoderAttentionHeads: Int
        public let encoderFfnDim: Int
        public let downsampleHiddenSize: Int
        public let outputDim: Int
        public let numMelBins: Int
        public let maxSourcePositions: Int
        public let nWindow: Int
        public let nWindowInfer: Int
        public let convChunksize: Int
    }
    public struct Thinker: Decodable, Sendable {
        public let textConfig: Text
        public let visionConfig: Vision
        public let audioConfig: Audio
        public let imageTokenId: Int
        public let videoTokenId: Int
        public let audioTokenId: Int
        public let visionStartTokenId: Int
        public let visionEndTokenId: Int
        public let audioStartTokenId: Int
        public let audioEndTokenId: Int
        public let positionIdPerSeconds: Int
    }
    public struct ExpertQuantization: Decodable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String
        public let scope: String
    }
    public let quantization: ExpertQuantization?
    public let modelType: String
    public let thinkerConfig: Thinker

    public static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self, from: data)
        try value.validate()
        return value
    }

    public func validate() throws {
        if let quantization {
            guard quantization.bits == 4, quantization.groupSize == 64, quantization.mode == "affine",
                  quantization.scope == "thinker_moe_experts",
                  thinkerConfig.textConfig.hiddenSize.isMultiple(of: quantization.groupSize),
                  thinkerConfig.textConfig.moeIntermediateSize.isMultiple(of: quantization.groupSize) else {
                throw ClefError.invalidConfiguration("Clef Omni quantization requires affine Q4/group-64 routed experts.")
            }
        }
        let text = thinkerConfig.textConfig
        let vision = thinkerConfig.visionConfig
        let audio = thinkerConfig.audioConfig
        guard modelType == "qwen3_omni_moe", (1...128).contains(text.numHiddenLayers),
              (1...8192).contains(text.hiddenSize), (1...512).contains(text.numExperts),
              (1...text.numExperts).contains(text.numExpertsPerTok), text.moeIntermediateSize > 0,
              text.numAttentionHeads > 0, text.numKeyValueHeads > 0,
              text.numAttentionHeads.isMultiple(of: text.numKeyValueHeads), text.headDim.isMultiple(of: 2),
              text.headDim > 0, text.vocabSize > 0, text.maxPositionEmbeddings >= 64_000,
              text.rmsNormEps.isFinite, text.rmsNormEps > 0, text.ropeTheta.isFinite, text.ropeTheta > 0, !text.attentionBias,
              text.decoderSparseStep == 1, text.mlpOnlyLayers.isEmpty,
              text.ropeScaling.ropeType == "default", text.ropeScaling.mropeInterleaved,
              text.ropeScaling.mropeSection.count == 3,
              text.ropeScaling.mropeSection.allSatisfy({ $0 >= 0 }),
              text.ropeScaling.mropeSection.reduce(0, +) == text.headDim / 2,
              vision.patchSize == 16, vision.temporalPatchSize == 2, vision.spatialMergeSize == 2,
              vision.outHiddenSize == text.hiddenSize, vision.depth > 0, vision.numHeads > 0,
              vision.hiddenSize > 0, vision.intermediateSize > 0, vision.imageSize > 0,
              vision.hiddenSize.isMultiple(of: vision.numHeads), vision.imageSize.isMultiple(of: vision.patchSize),
              vision.deepstackVisualIndexes.allSatisfy({ (0..<vision.depth).contains($0) }),
              audio.outputDim == text.hiddenSize, audio.numMelBins == 128,
              audio.encoderLayers > 0, audio.encoderAttentionHeads > 0, audio.dModel > 0,
              audio.encoderFfnDim > 0, audio.downsampleHiddenSize > 0, audio.maxSourcePositions >= 13,
              audio.convChunksize > 0,
              audio.dModel.isMultiple(of: audio.encoderAttentionHeads), audio.nWindow == 50,
              audio.nWindowInfer == 800, thinkerConfig.positionIdPerSeconds == 13 else {
            throw ClefError.invalidConfiguration("Unsupported Clef Omni thinker geometry or attention/processor policy.")
        }
    }
}
