import Foundation

public enum MMAudioResources {
    public static let modelID = "sfx-mmaudio-large-44k-v2"
    public static let upstreamRepoID = "hkchengrex/MMAudio"
    public static let upstreamRevision = "974010a026c731054592d8f777218bd9d85a6c24"
    public static let convertedWeightsRepoID = "Kijai/MMAudio_safetensors"
    public static let convertedWeightsRevision = "5984623e6b436818c6ff287ef6eec93e3e05aa3f"
    public static let clipRepoID = "apple/DFN5B-CLIP-ViT-H-14-378"
    public static let clipRevision = "01b771ed0d1395ca5ffdd279897d665ebe00dfd2"
    public static let bigVGANRepoID = "nvidia/bigvgan_v2_44khz_128band_512x"
    public static let bigVGANRevision = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"

    public static let networkFilename = "mmaudio_large_44k_v2_fp16.safetensors"
    public static let clipFilename = "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
    public static let synchformerFilename = "mmaudio_synchformer_fp16.safetensors"
    public static let vaeFilename = "mmaudio_vae_44k_fp16.safetensors"
    public static let bigVGANSafetensorsFilename = "bigvgan_generator.safetensors"
    public static let bigVGANPyTorchFilename = "bigvgan_generator.pt"

    public static let sampleRate = 44_100
    public static let defaultDurationSeconds: Float = 8
    public static let spectrogramFrameRate = 512
    public static let latentDownsampleRate = 2
    public static let latentDimension = 40
    public static let clipDimension = 1_024
    public static let syncDimension = 768
    public static let textDimension = 1_024
    public static let hiddenDimension = 896
    public static let depth = 21
    public static let fusedDepth = 14
    public static let attentionHeads = 14
    public static let textSequenceLength = 77
    public static let clipFramesPerSecond: Float = 8
    public static let syncInputFramesPerSecond: Float = 25
    public static let defaultSteps = 25
    public static let defaultGuidanceScale: Float = 4.5

    /// The upstream source is MIT. The published MMAudio checkpoints are
    /// CC-BY-NC-4.0 and must not be presented as commercially licensed assets.
    public static let architectureLicense = "MIT"
    public static let checkpointLicense = "CC-BY-NC-4.0"
    public static let clipModelLicense = "Apple Machine Learning Research Model License Agreement"
    public static let bigVGANLicense = "MIT"

    public static func latentSequenceLength(durationSeconds: Float) -> Int {
        let spectrogramFrames = ceil(
            durationSeconds * Float(sampleRate) / Float(spectrogramFrameRate)
        )
        return max(1, Int(ceil(spectrogramFrames / Float(latentDownsampleRate))))
    }

    public static func clipSequenceLength(durationSeconds: Float) -> Int {
        max(1, Int(durationSeconds * clipFramesPerSecond))
    }

    public static func syncSequenceLength(durationSeconds: Float) -> Int {
        let frameCount = durationSeconds * syncInputFramesPerSecond
        let segmentCount = floor((frameCount - 16) / 8) + 1
        return max(8, Int(segmentCount * 8))
    }
}

public struct MMAudioModelResources: Sendable, Hashable {
    public let rootURL: URL

    public init(rootURL: URL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var networkWeightsURL: URL {
        rootURL.appendingPathComponent(MMAudioResources.networkFilename)
    }

    public var clipWeightsURL: URL {
        rootURL.appendingPathComponent(MMAudioResources.clipFilename)
    }

    public var synchformerWeightsURL: URL {
        rootURL.appendingPathComponent(MMAudioResources.synchformerFilename)
    }

    public var vaeWeightsURL: URL {
        rootURL.appendingPathComponent(MMAudioResources.vaeFilename)
    }

    public var clipTokenizerURL: URL {
        rootURL.appendingPathComponent("clip", isDirectory: true)
    }

    public var bigVGANURL: URL {
        rootURL.appendingPathComponent("bigvgan", isDirectory: true)
    }

    public var bigVGANConfigURL: URL {
        bigVGANURL.appendingPathComponent("config.json")
    }

    public func bigVGANWeightsURL(fileManager: FileManager = .default) -> URL {
        let safetensors = bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANSafetensorsFilename)
        if fileManager.fileExists(atPath: safetensors.path) {
            return safetensors
        }
        return bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename)
    }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var missing = [
            networkWeightsURL,
            clipWeightsURL,
            synchformerWeightsURL,
            vaeWeightsURL,
            clipTokenizerURL.appendingPathComponent("tokenizer.json"),
            bigVGANConfigURL,
        ].filter { !fileManager.fileExists(atPath: $0.path) }

        let safetensors = bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANSafetensorsFilename)
        let pytorch = bigVGANURL.appendingPathComponent(MMAudioResources.bigVGANPyTorchFilename)
        if !fileManager.fileExists(atPath: safetensors.path)
            && !fileManager.fileExists(atPath: pytorch.path) {
            missing.append(safetensors)
        }
        return missing
    }
}

public struct MMAudioGenerationConfig: Sendable, Hashable {
    public let durationSeconds: Float
    public let steps: Int
    public let guidanceScale: Float
    public let seed: UInt64?

    public init(
        durationSeconds: Float = MMAudioResources.defaultDurationSeconds,
        steps: Int = MMAudioResources.defaultSteps,
        guidanceScale: Float = MMAudioResources.defaultGuidanceScale,
        seed: UInt64? = nil
    ) {
        self.durationSeconds = durationSeconds
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
    }

    public var latentSequenceLength: Int {
        MMAudioResources.latentSequenceLength(durationSeconds: durationSeconds)
    }

    public var clipSequenceLength: Int {
        MMAudioResources.clipSequenceLength(durationSeconds: durationSeconds)
    }

    public var syncSequenceLength: Int {
        MMAudioResources.syncSequenceLength(durationSeconds: durationSeconds)
    }
}
