import AudioQwen3TTSModel
import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCore
import AudioCodecs

extension Qwen3TTSGenerator {
    func emitStreamingAudioDelta(
        generatedCodes: [MLXArray],
        referenceCodesBQT: MLXArray?,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        emittedSampleCount: inout Int,
        onAudioDelta: (([Float]) -> Void)?
    ) throws {
        guard !generatedCodes.isEmpty, let onAudioDelta else { return }

        let generated = MLX.stacked(generatedCodes, axis: 1)
        let fullCodes: MLXArray
        if let referenceCodesBQT {
            let referenceCodesBTQ = referenceCodesBQT.transposed(0, 2, 1)
            fullCodes = MLX.concatenated([referenceCodesBTQ, generated], axis: 1)
        } else {
            fullCodes = generated
        }

        let (wav, lengths) = speechTokenizer.decode(fullCodes)
        var audio = wav.squeezed(axis: 0)
        let validLength = lengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.size {
            audio = audio[0..<validLength]
        }

        if let referenceCodesBQT {
            let refLength = referenceCodesBQT.dim(2)
            let totalLength = fullCodes.dim(1)
            let cut = Int(Double(refLength) / Double(max(totalLength, 1)) * Double(audio.size))
            if cut > 0 && cut < audio.size {
                audio = audio[cut..<audio.size]
            }
        }

        MLX.eval(audio)
        let fullSamples = audio.asType(.float32).reshaped(-1).asArray(Float.self)
        let delta = streamingAudioNewTail(fullSamples: fullSamples, emittedSampleCount: emittedSampleCount)
        emittedSampleCount = delta.updatedSampleCount
        if !delta.samples.isEmpty {
            onAudioDelta(delta.samples)
        }
    }

}
