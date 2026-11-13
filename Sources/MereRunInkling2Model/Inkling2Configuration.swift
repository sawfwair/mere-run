import Foundation

public enum Inkling2Error: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return "Inkling 2: \(message)" }
    }
}

public struct Inkling2Configuration: Decodable, Sendable {
    public let modelType: String
    public let eosTokenId: Int
    public let textConfig: Inkling2TextConfiguration
    public let visionConfig: Inkling2VisionConfiguration?

    public static func decode(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        try config.validate()
        return config
    }

    public func validate() throws {
        guard modelType == "inkling2" else { throw Inkling2Error.invalid("expected model_type inkling2") }
        try textConfig.validate()
        if let v = visionConfig {
            guard v.hiddenSize > 0, v.numHiddenLayers > 0, v.numAttentionHeads > 0,
                  v.hiddenSize.isMultiple(of: v.numAttentionHeads),
                  (v.hiddenSize / v.numAttentionHeads).isMultiple(of: 4),
                  v.nativePatchSize > 0, v.adapterMergeSize > 0, v.posembGridSize > 0,
                  v.projectionHiddenDim > 0, v.layerNormEps > 0, v.mlpRatio > 0,
                  !v.projectorBias else { throw Inkling2Error.invalid("unsupported vision geometry") }
        }
    }
}

public struct Inkling2TextConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let unpaddedVocabSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let swaNumAttentionHeads: Int
    public let swaNumKeyValueHeads: Int
    public let swaHeadDim: Int
    public let slidingWindowSize: Int
    public let localLayerIds: [Int]
    public let dRel: Int
    public let relExtent: Int
    public let modelMaxLength: Int
    public let logScalingNFloor: Int
    public let logScalingAlpha: Float
    public let rmsNormEps: Float
    public let sconvKernelSize: Int
    public let qkL2ClipThreshold: Float
    public let relL2ClipThreshold: Float
    public let nLayersPerAttnResBlock: Int
    public let nRoutedExperts: Int
    public let numExpertsPerTok: Int
    public let nSharedExperts: Int
    public let intermediateSize: Int
    public let mlpGateSoftcap: Float
    public let mlpUpSoftcap: Float
    public let routeScale: Float
    public let finalLogitSoftcapping: Float
    public let attnResNormalizeValues: Bool
    public let useValueNormNoGamma: Bool
    public let preNormPreAffine: Bool
    public let finalNormPreAffine: Bool
    public let qBias: Bool
    public let oBias: Bool
    public let useGateBias: Bool

    public func isLocal(_ index: Int) -> Bool { localLayerIds.contains(index) }

    func validate() throws {
        let dimensions = [hiddenSize, numHiddenLayers, vocabSize, unpaddedVocabSize, numAttentionHeads,
                          numKeyValueHeads, headDim, swaNumAttentionHeads, swaNumKeyValueHeads, swaHeadDim,
                          slidingWindowSize, dRel, relExtent, modelMaxLength, logScalingNFloor,
                          sconvKernelSize, nLayersPerAttnResBlock, nRoutedExperts, numExpertsPerTok, intermediateSize]
        guard dimensions.allSatisfy({ $0 > 0 }), unpaddedVocabSize <= vocabSize,
              numAttentionHeads.isMultiple(of: numKeyValueHeads),
              swaNumAttentionHeads.isMultiple(of: swaNumKeyValueHeads), numExpertsPerTok <= nRoutedExperts,
              Set(localLayerIds).count == localLayerIds.count,
              localLayerIds.allSatisfy({ (0..<numHiddenLayers).contains($0) }),
              [rmsNormEps, qkL2ClipThreshold, relL2ClipThreshold, mlpGateSoftcap,
               mlpUpSoftcap, routeScale, finalLogitSoftcapping].allSatisfy({ $0.isFinite && $0 > 0 }),
              logScalingAlpha.isFinite, logScalingAlpha >= 0,
              nSharedExperts == 0, attnResNormalizeValues, useValueNormNoGamma,
              !preNormPreAffine, !finalNormPreAffine, !qBias, !oBias, useGateBias else {
            throw Inkling2Error.invalid("unsupported text architecture or geometry")
        }
    }
}

public struct Inkling2VisionConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let nativePatchSize: Int
    public let adapterMergeSize: Int
    public let projectionHiddenDim: Int
    public let projectorBias: Bool
    public let mlpRatio: Float
    public let posembGridSize: Int
    public let layerNormEps: Float
}
