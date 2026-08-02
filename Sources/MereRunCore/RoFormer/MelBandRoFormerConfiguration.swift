import Foundation

public struct MelBandRoFormerConfiguration: Codable, Equatable, Sendable {
    public let modelID: String
    public let repository: String
    public let revision: String
    public let license: String
    public let sampleRate: Int
    public let audioChannels: Int
    public let chunkSize: Int
    public let overlap: Int
    public let dim: Int
    public let depth: Int
    public let numStems: Int
    public let stemNames: [String]
    public let timeTransformerDepth: Int
    public let frequencyTransformerDepth: Int
    public let numBands: Int
    public let heads: Int
    public let dimHead: Int
    public let dimensionFrequencies: Int
    public let stftNFFT: Int
    public let stftHopLength: Int
    public let stftWindowLength: Int
    public let stftNormalized: Bool
    public let zeroDC: Bool
    public let maskEstimatorDepth: Int
    public let transformerExpansionFactor: Int
    public let mlpExpansionFactor: Int

    var frequencyLayout: MelBandFrequencyLayout {
        MelBandFrequencyLayout(
            sampleRate: sampleRate,
            nFFT: stftNFFT,
            numBands: numBands,
            audioChannels: audioChannels
        )
    }

    public func validate(profile: MelBandRoFormerProfile) throws {
        guard modelID == profile.modelID else {
            throw RoFormerError.invalidConfiguration("unexpected model id \(modelID)")
        }
        guard repository == RoFormerResources.repository, revision == RoFormerResources.revision else {
            throw RoFormerError.invalidConfiguration("source repository or revision is not pinned")
        }
        guard license == "MIT" else {
            throw RoFormerError.invalidConfiguration("expected MIT model license")
        }
        guard sampleRate == 44_100, audioChannels == 2 else {
            throw RoFormerError.invalidConfiguration("MelBand RoFormer requires 44.1 kHz stereo audio")
        }
        guard numStems == 1, stemNames == [profile.stemName] else {
            throw RoFormerError.invalidConfiguration("stem name must match the selected restoration model")
        }
        guard dim > 0, depth > 0, timeTransformerDepth > 0,
              frequencyTransformerDepth > 0, heads > 0, dimHead > 0,
              transformerExpansionFactor > 0, mlpExpansionFactor > 0 else {
            throw RoFormerError.invalidConfiguration("model and attention dimensions must be positive")
        }
        guard numBands == 60, dimensionFrequencies == stftNFFT / 2 + 1 else {
            throw RoFormerError.invalidConfiguration("unsupported mel filter-bank geometry")
        }
        let layout = frequencyLayout
        guard layout.frequencyIndicesByBand.count == numBands,
              layout.bandsPerFrequency.count == dimensionFrequencies,
              layout.bandsPerFrequency.allSatisfy({ $0 > 0 }) else {
            throw RoFormerError.invalidConfiguration("mel bands do not cover every STFT bin")
        }
        guard chunkSize > 0, overlap > 0, chunkSize.isMultiple(of: overlap) else {
            throw RoFormerError.invalidConfiguration("chunk size must be divisible by overlap")
        }
        guard stftWindowLength == stftNFFT, stftHopLength > 0, !stftNormalized else {
            throw RoFormerError.invalidConfiguration("unsupported STFT geometry")
        }
        guard maskEstimatorDepth == 2 else {
            throw RoFormerError.invalidConfiguration("unsupported mask estimator depth")
        }
    }

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case repository
        case revision
        case license
        case sampleRate = "sample_rate"
        case audioChannels = "audio_channels"
        case chunkSize = "chunk_size"
        case overlap
        case dim
        case depth
        case numStems = "num_stems"
        case stemNames = "stem_names"
        case timeTransformerDepth = "time_transformer_depth"
        case frequencyTransformerDepth = "frequency_transformer_depth"
        case numBands = "num_bands"
        case heads
        case dimHead = "dim_head"
        case dimensionFrequencies = "dimension_frequencies"
        case stftNFFT = "stft_n_fft"
        case stftHopLength = "stft_hop_length"
        case stftWindowLength = "stft_window_length"
        case stftNormalized = "stft_normalized"
        case zeroDC = "zero_dc"
        case maskEstimatorDepth = "mask_estimator_depth"
        case transformerExpansionFactor = "transformer_expansion_factor"
        case mlpExpansionFactor = "mlp_expansion_factor"
    }
}

struct MelBandFrequencyLayout: Equatable, Sendable {
    let frequencyIndicesByBand: [[Int]]
    let interleavedFrequencyChannelIndices: [Int]
    let inputDimensions: [Int]
    let bandsPerFrequency: [Int]

    init(sampleRate: Int, nFFT: Int, numBands: Int, audioChannels: Int) {
        let frequencyCount = nFFT / 2 + 1
        let minimumLogHertz = 1_000.0
        let hertzPerMel = 200.0 / 3.0
        let minimumLogMel = minimumLogHertz / hertzPerMel
        let logStep = log(6.4) / 27.0

        func hertzToMel(_ hertz: Double) -> Double {
            if hertz < minimumLogHertz { return hertz / hertzPerMel }
            return minimumLogMel + log(hertz / minimumLogHertz) / logStep
        }
        func melToHertz(_ mel: Double) -> Double {
            if mel < minimumLogMel { return mel * hertzPerMel }
            return minimumLogHertz * exp(logStep * (mel - minimumLogMel))
        }

        let minimumMel = hertzToMel(0)
        let maximumMel = hertzToMel(Double(sampleRate) / 2)
        let melEdges = (0..<(numBands + 2)).map { index in
            let fraction = Double(index) / Double(numBands + 1)
            return melToHertz(minimumMel + (maximumMel - minimumMel) * fraction)
        }
        var bands = (0..<numBands).map { band in
            (0..<frequencyCount).filter { bin in
                let hertz = Double(sampleRate * bin) / Double(nFFT)
                return hertz > melEdges[band] && hertz < melEdges[band + 2]
            }
        }
        if !bands.isEmpty {
            if !bands[0].contains(0) { bands[0].insert(0, at: 0) }
            let finalFrequency = frequencyCount - 1
            if !bands[numBands - 1].contains(finalFrequency) {
                bands[numBands - 1].append(finalFrequency)
            }
        }

        self.frequencyIndicesByBand = bands
        self.interleavedFrequencyChannelIndices = bands.flatMap { band in
            band.flatMap { frequency in
                (0..<audioChannels).map { frequency * audioChannels + $0 }
            }
        }
        self.inputDimensions = bands.map { $0.count * audioChannels * 2 }
        self.bandsPerFrequency = (0..<frequencyCount).map { frequency in
            bands.reduce(0) { $0 + ($1.contains(frequency) ? 1 : 0) }
        }
    }
}
