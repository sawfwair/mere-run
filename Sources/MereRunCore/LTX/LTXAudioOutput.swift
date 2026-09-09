import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func matchLTXAudioWaveformDuration(
    _ waveform: MLXArray,
    videoFrames: Int,
    fps: Double,
    sampleRate: Int
) -> MLXArray {
    guard waveform.ndim == 3,
          videoFrames > 0,
          fps > 0,
          sampleRate > 0 else {
        return waveform
    }

    let targetSamples = max(1, Int((Double(videoFrames) * Double(sampleRate) / fps).rounded()))
    let currentSamples = waveform.dim(2)
    if currentSamples == targetSamples {
        return waveform
    }
    if currentSamples > targetSamples {
        return waveform[0..., 0..., 0..<targetSamples]
    }

    let padding = MLX.zeros(
        [waveform.dim(0), waveform.dim(1), targetSamples - currentSamples],
        dtype: waveform.dtype
    )
    return MLX.concatenated([waveform, padding], axis: 2)
}

func saveLTXAVDebugAudio(_ waveform: MLXArray, suffix: String, sampleRate: Int) {
    guard let url = ltxAVDebugURL(suffix: suffix, fileExtension: "wav") else { return }
    guard let prepared = try? LTXVideoMP4Writer.prepareAudio(waveform) else { return }
    try? MediaAudioIO.writeFloatWAV(
        samples: prepared.interleaved,
        sampleRate: sampleRate,
        channels: prepared.channels,
        to: url
    )
}

func resolveLTX23TextEncoderRoot(modelRoot: URL) throws -> URL {
    let fm = FileManager.default
    let localTextEncoder = modelRoot.appendingPathComponent("text_encoder", isDirectory: true)
    if fm.fileExists(atPath: localTextEncoder.appendingPathComponent("config.json").path) {
        return localTextEncoder
    }

    if let override = ProcessInfo.processInfo.environment["MERERUN_VIDEO_LTX_TEXT_ENCODER_ROOT"],
       !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return URL(fileURLWithPath: override).standardizedFileURL
    }

    let companionID = ModelResolver.ModelID.ltxGemma3TwelveB4Bit.rawValue
    if let installed = ManagedModelResolver.resolveInstalledModel(id: companionID) {
        return installed
    }

    throw LTXUnifiedAVGeneratorError.ltx23TextEncoderMissing(companionID)
}
