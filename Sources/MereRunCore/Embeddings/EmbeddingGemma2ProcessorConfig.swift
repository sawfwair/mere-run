import Foundation

struct EmbeddingGemma2ProcessorConfig: Decodable {
    let imageProcessor: Vision
    let videoProcessor: Vision
    let featureExtractor: Audio
    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor", videoProcessor = "video_processor", featureExtractor = "feature_extractor"
    }
    struct Vision: Decodable {
        let patchSize: Int
        let poolingKernelSize: Int
        let maxSoftTokens: Int
        let doResize: Bool
        let doRescale: Bool
        let doNormalize: Bool
        let resample: Int
        let rescaleFactor: Float
        let fps: Double?
        let maxFrames: Int?
        let addTimestamps: Bool?
        let overflowStrategy: String?
        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size", poolingKernelSize = "pooling_kernel_size", maxSoftTokens = "max_soft_tokens"
            case doResize = "do_resize", doRescale = "do_rescale", doNormalize = "do_normalize", resample
            case rescaleFactor = "rescale_factor", fps, maxFrames = "max_frames", addTimestamps = "add_timestamps"
            case overflowStrategy = "overflow_strategy"
        }
    }
    struct Audio: Decodable {
        let samplingRate: Int
        let featureSize: Int
        let frameLength: Int
        let hopLength: Int
        let fftLength: Int
        let melFloor: Double
        let minFrequency: Double
        let maxFrequency: Double
        let dither: Double
        let preemphasis: Double
        let inputScaleFactor: Double
        let perBinMean: [Double]?
        let perBinStddev: [Double]?
        enum CodingKeys: String, CodingKey {
            case samplingRate = "sampling_rate", featureSize = "feature_size", frameLength = "frame_length"
            case hopLength = "hop_length", fftLength = "fft_length", melFloor = "mel_floor"
            case minFrequency = "min_frequency", maxFrequency = "max_frequency", dither, preemphasis
            case inputScaleFactor = "input_scale_factor", perBinMean = "per_bin_mean", perBinStddev = "per_bin_stddev"
        }
    }
    func validate(vision: EmbeddingGemma2VisionConfig) throws {
        for config in [imageProcessor, videoProcessor] {
            guard config.patchSize == vision.patchSize, config.poolingKernelSize == vision.poolingKernelSize,
                  config.maxSoftTokens > 0, config.doResize, config.doRescale, !config.doNormalize,
                  config.resample == 3, config.rescaleFactor == Float(1.0 / 255) else {
                throw EmbeddingGemma2Error.invalidConfiguration("Unsupported vision preprocessing configuration.")
            }
        }
        let audio = featureExtractor
        guard videoProcessor.fps == 1, videoProcessor.maxFrames == 32,
              videoProcessor.addTimestamps == false, videoProcessor.overflowStrategy == "uniform",
              audio.samplingRate == 16_000, audio.featureSize == 128, audio.frameLength == 320,
              audio.hopLength == 160, audio.fftLength == 512, audio.melFloor > 0,
              audio.minFrequency == 0, audio.maxFrequency == 8_000, audio.dither == 0,
              audio.preemphasis == 0, audio.inputScaleFactor == 1,
              audio.perBinMean == nil, audio.perBinStddev == nil else {
            throw EmbeddingGemma2Error.invalidConfiguration("Unsupported audio or video preprocessing configuration.")
        }
    }
}
