import Foundation

public enum PhononError: LocalizedError, Sendable {
    case invalidConfiguration(String)
    case invalidRequest(String)
    case invalidAsset(String)
    case frameBudgetExceeded

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason): "Unsupported Phonon configuration: \(reason)"
        case .invalidRequest(let reason): reason
        case .invalidAsset(let reason): "Invalid Phonon asset: \(reason)"
        case .frameBudgetExceeded: "Phonon reached its frame budget before finishing the utterance."
        }
    }
}

/// The original Gradium checkpoint configuration, decoded without untyped JSON.
public struct PhononConfig: Decodable, Sendable {
    struct Flow: Decodable, Sendable {
        let dModel, numHeads, numLayers, dimFeedforward, nBins, lutDim, flowDim, flowDepth, ldim: Int
        let maxPeriod: Float
        let timeRmsNorm, gelu: String
    }
    struct Mimi: Decodable, Sendable {
        let channels, sampleRate, dimension, quantizerDimension, quantizerOutputDimension: Int
        let nFilters, nResidualLayers, kernelSize, lastKernelSize, residualKernelSize, dilationBase, compress: Int
        let ratios: [Int]
        let frameRate: Double
        let transformerDModel, transformerNumHeads, transformerNumLayers, transformerContext: Int
        let transformerDimFeedforward: Int
        let transformerLayerScale, transformerMaxPeriod: Float
        let gelu: String
    }
    struct Conditioner: Decodable, Sendable {
        struct LUT: Decodable, Sendable {
            let nBins, dim: Int
            let tokenizer: String
            let possibleValues: [String]
        }
        struct Continuous: Decodable, Sendable {
            let dim: Int
            let scaleFactor, maxPeriod: Float
        }
        let type, name: String
        let lut: LUT?
        let continuous: Continuous?
    }
    struct Fuser: Decodable, Sendable {
        let sum, prepend, cross, streamingSum: [String]
        let padPrepend: Bool
    }
    let modelType: String
    let languages: [String]
    let embeddingsExt: [String]
    let flowLm: Flow
    let mimi: Mimi
    let lsdDecodeSteps: Int
    let eosThreshold: Float
    let conditioners: [Conditioner]
    let fuser: Fuser
    let prefixSinks: [String: Sink]?
    struct Sink: Decodable, Sendable { let droppedToken: Bool? }

    public var sampleRate: Int { mimi.sampleRate }
    public static func load(from url: URL) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self, from: Data(contentsOf: url))
        try value.validate()
        return value
    }

    func validate() throws {
        let flow = flowLm
        guard modelType == "gradium_phonon_tts", languages == ["en"], prefixSinks?.isEmpty != false,
              flow.dModel > 0, flow.numHeads > 0, flow.dModel % flow.numHeads == 0,
              (flow.dModel / flow.numHeads).isMultiple(of: 2), flow.numLayers > 0,
              flow.lutDim == flow.dModel, flow.ldim > 0, flow.flowDim > 0, flow.flowDepth > 0,
              flow.dimFeedforward > 0, flow.nBins > 0, flow.maxPeriod > 0,
              flow.timeRmsNorm == "rms", flow.gelu == "erf", mimi.gelu == "erf",
              (1...16).contains(lsdDecodeSteps), eosThreshold.isFinite else {
            throw PhononError.invalidConfiguration("expected the English Phonon flow architecture with RMS time embeddings and erf GELU")
        }
        guard mimi.channels == 1, mimi.sampleRate > 0, mimi.frameRate > 0,
              mimi.dimension == mimi.transformerDModel, mimi.quantizerOutputDimension == mimi.dimension,
              mimi.quantizerDimension == flow.ldim, mimi.transformerNumHeads > 0,
              mimi.dimension % mimi.transformerNumHeads == 0,
              (mimi.dimension / mimi.transformerNumHeads).isMultiple(of: 2),
              mimi.transformerNumLayers > 0, mimi.transformerContext > 0, mimi.transformerDimFeedforward > 0,
              mimi.ratios.count == 3, mimi.ratios.allSatisfy({ (2...16).contains($0) }),
              mimi.nFilters > 0, mimi.nResidualLayers == 1, mimi.compress == 2,
              mimi.kernelSize > 0, mimi.lastKernelSize > 0, mimi.residualKernelSize > 0,
              mimi.dilationBase > 0 else {
            throw PhononError.invalidConfiguration("unsupported Mimi decoder")
        }
        let hop = mimi.ratios.reduce(1, *)
        let stride = Double(mimi.sampleRate) / mimi.frameRate / Double(hop)
        guard stride >= 1, stride <= 64, stride == stride.rounded(),
              fuser.sum == ["num_speakers", "duration_delta"],
              fuser.prepend == ["transcript_in_segment", "speaker_wavs"],
              fuser.cross.isEmpty, fuser.streamingSum.isEmpty, !fuser.padPrepend,
              conditioners.count == 2 else {
            throw PhononError.invalidConfiguration("unsupported conditioning or codec rate")
        }
        guard let speakers = conditioners.first(where: { $0.name == "num_speakers" }),
              speakers.type == "lut", let lut = speakers.lut, lut.tokenizer == "noop",
              lut.dim > 0, lut.nBins > 0, lut.possibleValues.contains("1"),
              let duration = conditioners.first(where: { $0.name == "duration_delta" }),
              duration.type == "continuous", let continuous = duration.continuous,
              continuous.dim >= 4, continuous.dim.isMultiple(of: 2),
              continuous.maxPeriod > 0, continuous.scaleFactor.isFinite else {
            throw PhononError.invalidConfiguration("unsupported speaker or duration conditioner")
        }
    }
}
