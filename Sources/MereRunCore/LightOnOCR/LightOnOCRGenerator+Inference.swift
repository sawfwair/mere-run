import Foundation
import MediaIO
import MLX
import MLXNN
import MLXFast

/// Owns OCR image preprocessing and autoregressive decoding.
/// Keeping these helpers together makes the vision -> prompt -> text path
/// easier to follow than when they are mixed with model loading.
extension LightOnOCRGenerator {
    func loadAndPreprocessImage(
        _ url: URL,
        patchSize: Int,
        spatialMergeSize: Int
    ) throws -> MLXArray {
        let image: MediaImage
        do {
            image = try MediaImageIO.decode(url)
        } catch {
            throw LightOnOCRError.imageLoadFailed(url)
        }

        let originalWidth = image.width
        let originalHeight = image.height
        let maxEdge = 1540
        let mergeUnit = max(1, patchSize * spatialMergeSize)
        let longestEdge = max(originalWidth, originalHeight)
        let scale: Float = longestEdge > maxEdge ? Float(maxEdge) / Float(longestEdge) : 1.0

        func roundUpToMultiple(_ value: Int, _ multiple: Int) -> Int {
            guard multiple > 0 else { return value }
            return ((value + multiple - 1) / multiple) * multiple
        }

        var targetWidth = roundUpToMultiple(Int(Float(originalWidth) * scale), mergeUnit)
        var targetHeight = roundUpToMultiple(Int(Float(originalHeight) * scale), mergeUnit)
        targetWidth = min(maxEdge, max(mergeUnit, targetWidth))
        targetHeight = min(maxEdge, max(mergeUnit, targetHeight))

        let pixelArray = try QwenImageIO.resizedPixelArray(
            from: image,
            width: targetWidth,
            height: targetHeight,
            addBatchDimension: true,
            dtype: .float32
        )

        let mean = MLXArray([Float(0.48145466), Float(0.4578275), Float(0.40821073)]).reshaped(1, 3, 1, 1)
        let std = MLXArray([Float(0.26862954), Float(0.26130258), Float(0.27577711)]).reshaped(1, 3, 1, 1)
        return ((pixelArray - mean) / std).asType(.float32)
    }

    func buildOCRPromptParts(tokenizer: QwenTokenizer) -> (pre: [Int], post: [Int]) {
        let instruction = "Please transcribe the text in the image."
        let prompt =
            "<|im_start|>system<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<|image_pad|>" + instruction + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"

        let tokens = tokenizer.encodeText(prompt)
        let imageId = tokenizer.imageTokenId ?? imageTokenId
        guard let imagePos = tokens.firstIndex(of: imageId) else {
            let preTokens = [151644, 8948, 151645, 198, 151644, 872, 198]
            let postTokens = [151645, 198, 151644, 77091, 198]
            return (preTokens, postTokens)
        }

        return (Array(tokens[..<imagePos]), Array(tokens[(imagePos + 1)...]))
    }

    func generateFromEmbeddings(
        inputEmbeds: MLXArray,
        positionIds: MLXArray?,
        promptSeqLen: Int,
        ropeDelta: Int,
        textDecoder: QwenTextEncoder,
        config: Config
    ) throws -> [Int] {
        let cache: [KVCache] = (0..<textDecoder.configuration.numHiddenLayers).map { _ in
            KVCacheSimple(step: 256)
        }

        var logits = textDecoder.encoder.forwardCausal(
            embeddings: inputEmbeds,
            cache: cache,
            positionIds: positionIds
        )
        MLX.eval(logits)
        log("[Gen] Prefill logits shape: \(logits.shape)")
        if logProgress {
            // Debug-only: full-vocabulary sort plus a GPU readback.
            let lastLogitsPrefill = logits[0, -1, 0...]
            let topIndices = MLX.argSort(lastLogitsPrefill, axis: -1)[(-5)...]
            eval(topIndices)
            log("[Gen] Top 5 token indices: \(topIndices.asArray(Int32.self))")
        }

        // Depth-1 pipelined decode: the sampled token stays on GPU and feeds
        // the next forward directly; the previous step's token is read back
        // while the current step executes. The legacy loop synchronized
        // twice per token (softmax eval inside sampling plus a blocking
        // logits eval), which dominates OCR's long page transcriptions.
        var generatedTokens: [Int] = []
        var pendingToken: MLXArray?
        var scheduledCount = 0
        for _ in 0..<config.maxNewTokens {
            let tokenArray = sampleTokenArray(logits: logits[0, -1, 0...], temperature: config.temperature)
            scheduledCount += 1
            let nextInput = tokenArray.reshaped(1, 1)
            if positionIds != nil {
                let posValue = Int32(promptSeqLen + scheduledCount - 1 + ropeDelta)
                let posIds = MLXArray([posValue]).reshaped(1, 1)
                logits = textDecoder.encoder.forwardCausal(inputIds: nextInput, cache: cache, positionIds: posIds)
            } else {
                logits = textDecoder.encoder.forwardCausal(inputIds: nextInput, cache: cache)
            }
            asyncEval([logits, tokenArray])

            if let previous = pendingToken {
                pendingToken = nil
                let value = previous.item(Int.self)
                if value == eosTokenId || value == padTokenId {
                    return generatedTokens
                }
                generatedTokens.append(value)
            }
            pendingToken = tokenArray
        }
        if let previous = pendingToken {
            let value = previous.item(Int.self)
            if value != eosTokenId && value != padTokenId {
                generatedTokens.append(value)
            }
        }

        return generatedTokens
    }

    func computeExpandedRopePositions(
        originalSeqLen: Int,
        placeholderPos: Int,
        numVisionTokens: Int,
        gridThw: (Int, Int, Int),
        spatialMergeSize: Int
    ) -> MLXArray {
        let finalSeqLen = originalSeqLen - 1 + numVisionTokens
        var positions = Array(repeating: [Int](), count: 3)
        for d in 0..<3 {
            positions[d].reserveCapacity(finalSeqLen)
        }

        for i in 0..<placeholderPos {
            for d in 0..<3 {
                positions[d].append(i)
            }
        }

        let t = max(1, gridThw.0)
        let h = max(1, gridThw.1 / spatialMergeSize)
        let w = max(1, gridThw.2 / spatialMergeSize)
        let imageBase = placeholderPos
        for ti in 0..<t {
            for hi in 0..<h {
                for wi in 0..<w {
                    positions[0].append(imageBase + ti)
                    positions[1].append(imageBase + hi)
                    positions[2].append(imageBase + wi)
                }
            }
        }

        let maxVisionPos = max(positions[0].last ?? 0, positions[1].last ?? 0, positions[2].last ?? 0)
        let textContinueBase = maxVisionPos + 1
        let tokensAfterPlaceholder = originalSeqLen - placeholderPos - 1
        for i in 0..<tokensAfterPlaceholder {
            for d in 0..<3 {
                positions[d].append(textContinueBase + i)
            }
        }

        let flat = positions.flatMap { $0.map(Int32.init) }
        return MLXArray(flat, [3, 1, finalSeqLen])
    }

    /// GPU-side variant of `sampleToken`: greedy skips the softmax (argMax
    /// over logits picks the same token) and the sampled path uses GPU
    /// categorical sampling; both return a 0-d array with no host readback,
    /// so the decode loop can pipeline. The legacy sampled path drew from an
    /// unseeded host RNG, so sampled outputs were nondeterministic before
    /// and after this change.
    func sampleTokenArray(logits: MLXArray, temperature: Float) -> MLXArray {
        if temperature <= 0.1 {
            return MLX.argMax(logits, axis: -1).asType(.int32)
        }
        let scores = (logits.asType(.float32)) / temperature
        return categorical(scores).asType(.int32)
    }

    func sampleToken(logits: MLXArray, temperature: Float) -> Int {
        var adjustedLogits = logits
        if temperature > 0 {
            adjustedLogits = adjustedLogits / temperature
        }

        let probs = softmax(adjustedLogits)
        MLX.eval(probs)

        if temperature <= 0.1 {
            return Int(MLX.argMax(probs).item(Int32.self))
        }

        let probsArray = probs.asArray(Float.self)
        let cumProbs = probsArray.reduce(into: [Float]()) { result, prob in
            result.append((result.last ?? 0) + prob)
        }

        let rand = Float.random(in: 0..<1)
        for (i, cumProb) in cumProbs.enumerated() {
            if rand < cumProb {
                return i
            }
        }
        return probsArray.count - 1
    }
}
