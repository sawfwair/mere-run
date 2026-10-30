import Foundation

package struct UnlimitedOCRConfiguration: Decodable, Sendable {
    package let modelType: String
    package let hiddenSize: Int
    package let intermediateSize: Int
    package let numHiddenLayers: Int
    package let numAttentionHeads: Int
    package let numKeyValueHeads: Int
    package let vocabSize: Int
    package let nRoutedExperts: Int
    package let nSharedExperts: Int
    package let numExpertsPerTok: Int
    package let moeIntermediateSize: Int
    package let firstKDenseReplace: Int
    package let rmsNormEps: Float
    package let eosTokenId: Int
    package let bosTokenId: Int
    package let sparseAttentionConfig: Sparse
    let normTopkProb: Bool
    let scoringFunc: String
    let topkMethod: String
    let routedScalingFactor: Float
    let tieWordEmbeddings: Bool
    let attentionBias: Bool
    let hiddenAct: String
    let useMla: Bool
    let moeLayerFreq: Int
    let visionConfig: Vision
    let projectorConfig: Projector

    struct Vision: Decodable, Sendable {
        let imageSize: Int
        let modelName: String
        let width: [String: Width]
        struct Width: Decodable, Sendable {
            let heads: Int
            let layers: Int
            let width: Int
            let patchSize: Int?
            let globalAttnIndexes: [Int]?
            let downsampleChannels: [Int]?
        }
    }

    struct Projector: Decodable, Sendable {
        let inputDim: Int
        let nEmbed: Int
        let projectorType: String
    }

    package struct Sparse: Decodable, Sendable {
        package let sparseBlockSize: Int
        package let sparseLocalBlock: Int
        package let sparseIndexDim: Int
        package let dashTopkBlocks: Int
        package let dashStage0: String
        package let dashScalePerLayer: [Float]
        package let dashFloorPerLayer: [Float]
        package let dashSigma: Float
        package let sparseRopePagePairs: Int
        package let sparseRopePageMod: Int
        package let sparseRopeTokTheta: Float
        package let sparseRopePageTheta: Float
        package let imageTokenId: Int
        package let pageTokenId: Int
        let trainingStage: String
        let sparseMainRopeMode: String
        let sparseRopeMode: String
        let sparseMainRopeTokFreqMode: String
        let sparseRopeTokFreqMode: String
        let sparseRopePageSpan: Int
        let imageBucketConstraint: Bool
        let dashImageOnly: Bool
        let sparseAttentionFreq: [Int]
        let sparseNumIndexHeads: Int
    }

    package static func load(_ data: Data) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let config = try decoder.decode(Self.self, from: data)
        try config.validate()
        return config
    }

    package func validate() throws {
        let sparse = sparseAttentionConfig
        guard modelType == "unlimited-ocr", hiddenSize == 1280, intermediateSize == 6848,
              numHiddenLayers == 12, numAttentionHeads == 10, numKeyValueHeads == 10,
              vocabSize == 129280, nRoutedExperts == 64, nSharedExperts == 2,
              numExpertsPerTok == 6, moeIntermediateSize == 896, firstKDenseReplace == 1,
              bosTokenId == 0, eosTokenId == 1, rmsNormEps == 1e-6,
              !normTopkProb, scoringFunc == "softmax", topkMethod == "greedy", routedScalingFactor == 1,
              !tieWordEmbeddings, !attentionBias, hiddenAct == "silu", !useMla, moeLayerFreq == 1,
              visionConfig.imageSize == 1024, visionConfig.modelName == "deeplip_b_l",
              visionConfig.width["sam_vit_b"]?.width == 768,
              visionConfig.width["sam_vit_b"]?.heads == 12,
              visionConfig.width["sam_vit_b"]?.layers == 12,
              visionConfig.width["sam_vit_b"]?.globalAttnIndexes == [2, 5, 8, 11],
              visionConfig.width["sam_vit_b"]?.downsampleChannels == [512, 1024],
              visionConfig.width["clip-l-14-224"]?.width == 1024,
              visionConfig.width["clip-l-14-224"]?.heads == 16,
              visionConfig.width["clip-l-14-224"]?.layers == 24,
              visionConfig.width["clip-l-14-224"]?.patchSize == 14,
              projectorConfig.inputDim == 2048, projectorConfig.nEmbed == 1280, projectorConfig.projectorType == "linear",
              sparse.trainingStage == "dash", sparse.sparseMainRopeMode == "page2d", sparse.sparseRopeMode == "page2d",
              sparse.sparseMainRopeTokFreqMode == "resample", sparse.sparseRopeTokFreqMode == "resample",
              sparse.sparseRopePageSpan == 0, sparse.imageBucketConstraint, sparse.dashImageOnly,
              sparse.sparseAttentionFreq == [Int](repeating: 1, count: 12), sparse.sparseNumIndexHeads == 1,
              sparse.sparseBlockSize == 32, sparse.sparseLocalBlock == 4,
              sparse.sparseIndexDim == 128, sparse.dashTopkBlocks == 12,
              sparse.dashStage0 == "head_cls", sparse.sparseRopePagePairs == 1,
              sparse.sparseRopePageMod == 2, sparse.sparseRopeTokTheta == 10000,
              sparse.sparseRopePageTheta == 256, sparse.dashSigma > 0,
              sparse.dashScalePerLayer.count == numHiddenLayers,
              sparse.dashFloorPerLayer.count == numHiddenLayers,
              sparse.dashFloorPerLayer.allSatisfy({ $0 > 0 }),
              sparse.imageTokenId == 128815, sparse.pageTokenId == 128827 else {
            throw UnlimitedOCRError.unsupportedConfiguration
        }
    }
}

package enum UnlimitedOCRError: Error, LocalizedError {
    case unsupportedConfiguration
    case checkpoint(String)
    case invalidInput(String)

    package var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration: "Unsupported Unlimited-OCR-2 checkpoint configuration."
        case .checkpoint(let detail): "Invalid Unlimited-OCR-2 checkpoint: \(detail)"
        case .invalidInput(let detail): detail
        }
    }
}

/// Coordinates follow the upstream page bucket state machine. Real pages start at 1;
/// bucket 0 contains BOS and the trailing instruction. Response markers restart at page 1.
package struct UnlimitedOCRPageState {
    package private(set) var tokenPositions: [Int32] = []
    package private(set) var pagePositions: [Int32] = []
    package private(set) var queryPages: [Int32] = []
    package private(set) var imagePages: [Int32] = []
    private var prefillCounts: [Int32]
    private var page: Int = 0
    private var token: Int32 = 0
    private var markerCount: Int
    private let promptPages: Int
    private let pageID: Int

    package init(tokens: [Int], imageID: Int = 128815, pageID: Int = 128827) throws {
        guard let lastImage = tokens.lastIndex(of: imageID) else {
            throw UnlimitedOCRError.invalidInput("Unlimited-OCR-2 requires page images.")
        }
        let pages = tokens[...lastImage].filter { $0 == pageID }.count
        guard pages > 0 else { throw UnlimitedOCRError.invalidInput("Missing <PAGE> markers.") }
        self.pageID = pageID
        promptPages = pages
        markerCount = 0
        prefillCounts = [Int32](repeating: 0, count: pages + 1)
        var counts = prefillCounts
        for (index, id) in tokens.enumerated() {
            if id == pageID { markerCount += 1 }
            let prefill = index <= lastImage
            page = max(0, prefill ? markerCount : markerCount - pages)
            token = counts[page]
            counts[page] += 1
            if prefill { prefillCounts[page] += 1 }
            tokenPositions.append(token)
            pagePositions.append(Int32(page))
            queryPages.append(Int32(max(-1, prefill ? markerCount - 1 : markerCount - pages - 1)))
            imagePages.append(id == imageID ? Int32(markerCount - 1) : -1)
        }
    }

    package mutating func append(_ id: Int) {
        if id == pageID {
            markerCount += 1
            page = min(page + 1, promptPages)
            token = prefillCounts[page]
        } else {
            token += 1
        }
        tokenPositions.append(token)
        pagePositions.append(Int32(page))
        queryPages.append(Int32(max(-1, markerCount - 1 - promptPages)))
        imagePages.append(-1)
    }
}
