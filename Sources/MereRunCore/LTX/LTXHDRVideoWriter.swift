import Foundation
import MediaIO
import MLX

public enum LTXHDRVideoWriter {
    public enum WriterError: LocalizedError {
        case invalidShape([Int])

        public var errorDescription: String? {
            switch self {
            case .invalidShape(let shape):
                return "Expected HDR frames shaped [F, H, W, 3], got \(shape)."
            }
        }
    }

    /// Writes the upstream HDR contract: half-float EXR frames under
    /// `<output-stem>_exr/` plus a BT.2020/HLG Main10 HEVC master.
    @discardableResult
    public static func write(
        _ output: LTXHDROutputFrames,
        colorSpace: LTXHDRColorSpace,
        fps: Double,
        to outputURL: URL,
        writeHLGMaster: Bool = true,
        audioWaveform: MLXArray? = nil,
        audioSampleRate: Int = 24_000,
        sourceAudio: MediaAudioBuffer? = nil
    ) throws -> URL {
        let shape = output.exr.shape
        guard output.exr.ndim == 4,
              output.exr.dim(0) > 0,
              output.exr.dim(1) > 0,
              output.exr.dim(2) > 0,
              output.exr.dim(3) == 3,
              output.hlg.shape == shape else {
            throw WriterError.invalidShape(shape)
        }
        let frameCount = output.exr.dim(0)
        let height = output.exr.dim(1)
        let width = output.exr.dim(2)
        let parent = outputURL.deletingLastPathComponent()
        let stem = outputURL.deletingPathExtension().lastPathComponent
        let exrDirectory = parent.appendingPathComponent("\(stem)_exr", isDirectory: true)
        try FileManager.default.createDirectory(at: exrDirectory, withIntermediateDirectories: true)
        for existing in try FileManager.default.contentsOfDirectory(
            at: exrDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) where isGeneratedEXRFrame(existing.lastPathComponent) {
            try FileManager.default.removeItem(at: existing)
        }
        let metadata: MediaEXRColorMetadata = switch colorSpace {
        case .srgbLinear: .rec709Linear
        case .acescg: .acescgLinear
        case .acescct: .acescct
        }
        for frameIndex in 0..<frameCount {
            let values = output.exr[frameIndex, 0..., 0..., 0...]
                .asType(.float32)
                .reshaped(-1)
                .asArray(Float.self)
            let image = try MediaFloatImage(width: width, height: height, rgb: values)
            let frameURL = exrDirectory.appendingPathComponent(
                String(format: "frame_%05d.exr", frameIndex),
                isDirectory: false
            )
            try MediaHDRImageIO.writeEXR(image, metadata: metadata, to: frameURL)
        }

        guard writeHLGMaster else { return exrDirectory }

        let temporaryVideoURL: URL
        if audioWaveform == nil, sourceAudio == nil {
            temporaryVideoURL = outputURL
        } else {
            temporaryVideoURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("mererun-ltx-hlg-\(UUID().uuidString)")
                .appendingPathExtension("mp4")
        }
        defer {
            if temporaryVideoURL != outputURL {
                try? FileManager.default.removeItem(at: temporaryVideoURL)
            }
        }
        try MediaHDRVideoIO.writeHLGMP4(
            rgbFloatFrameAt: { frameIndex in
                output.hlg[frameIndex, 0..., 0..., 0...]
                    .asType(.float32)
                    .reshaped(-1)
                    .asArray(Float.self)
            },
            width: width,
            height: height,
            frameCount: frameCount,
            fps: Double(fps),
            to: temporaryVideoURL
        )

        if audioWaveform != nil || sourceAudio != nil {
            let prepared: (interleaved: [Float], channels: Int, sampleRate: Int)
            if let sourceAudio {
                prepared = try LTXVideoMP4Writer.prepareSourceAudio(sourceAudio)
            } else {
                let generated = try LTXVideoMP4Writer.prepareAudio(audioWaveform!)
                prepared = (generated.interleaved, generated.channels, audioSampleRate)
            }
            let temporaryAudioURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("mererun-ltx-hlg-\(UUID().uuidString)")
                .appendingPathExtension("wav")
            defer { try? FileManager.default.removeItem(at: temporaryAudioURL) }
            try MediaAudioIO.writeFloatWAV(
                samples: prepared.interleaved,
                sampleRate: prepared.sampleRate,
                channels: prepared.channels,
                to: temporaryAudioURL
            )
            try MediaVideoIO.mux(
                videoURL: temporaryVideoURL,
                audioURL: temporaryAudioURL,
                outputURL: outputURL,
                audioBitRate: LTXVideoMP4Writer.defaultAudioBitRate
            )
        }
        return exrDirectory
    }

    private static func isGeneratedEXRFrame(_ filename: String) -> Bool {
        guard filename.hasPrefix("frame_"), filename.hasSuffix(".exr") else {
            return false
        }
        let digits = filename.dropFirst("frame_".count).dropLast(".exr".count)
        return digits.count == 5 && digits.allSatisfy(\.isNumber)
    }
}
