import Foundation
import MediaIO
import MLX
import MLXNN
#if canImport(Darwin)
import Darwin
#endif

extension Q35Generator {
    func collectImageURLs(from messages: [ChatMessage]) -> [String] {
        messages.compactMap { message in
            guard message.role != .system else { return nil }
            guard let url = message.imageUrl?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !url.isEmpty else { return nil }
            return url
        }
    }

    func ensureVisionWeightsLoaded(
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) throws {
        guard let visionTower else { return }
        guard !visionTower.isLoaded else { return }
        guard let loadedVisionResources else { throw Q35Error.modelNotLoaded }

        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loading Qwen-family vision tower"))
        try visionTower.loadWeights(from: loadedVisionResources)
        progressHandler?(ChatProgress(stage: .loadingModel, message: "Loaded Qwen-family vision tower"))
    }

    func buildVisionReplacements(
        imageURLs: [String],
        visionTower: Q35VisionTower,
        maximumTokensPerImage: Int
    ) throws -> [Q35VisionReplacement] {
        var replacements: [Q35VisionReplacement] = []
        replacements.reserveCapacity(imageURLs.count)

        for imageURL in imageURLs {
            let prepared = try loadImageTensor(
                from: imageURL,
                patchSize: visionTower.patchSize,
                spatialMergeSize: visionTower.spatialMergeSize,
                maximumVisionTokens: maximumTokensPerImage
            )
            let embeds = try visionTower.encodeImage(
                pixelValues: prepared.tensor,
                gridTHW: prepared.gridTHW
            )
            replacements.append(Q35VisionReplacement(embeddings: embeds, gridTHW: prepared.gridTHW))
        }

        return replacements
    }

    func buildMRoPEPositionData(
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [Q35VisionReplacement],
        spatialMergeSize: Int
    ) throws -> Q35MRoPEPositionData? {
        guard !replacements.isEmpty else { return nil }

        let seqLen = inputIds.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var axes = Array(repeating: [Int](), count: 3)
        for axis in axes.indices {
            axes[axis].reserveCapacity(seqLen)
        }

        let mergeSize = max(1, spatialMergeSize)
        var cursor = 0
        var currentPosition = 0
        var replacementIndex = 0

        func appendText(count: Int) {
            guard count > 0 else { return }
            for offset in 0..<count {
                let position = currentPosition + offset
                for axis in axes.indices {
                    axes[axis].append(position)
                }
            }
            currentPosition += count
        }

        func appendVision(startPosition: Int, gridTHW: (Int, Int, Int)) -> Int {
            let gridT = max(1, gridTHW.0)
            let gridH = max(1, gridTHW.1 / mergeSize)
            let gridW = max(1, gridTHW.2 / mergeSize)
            for t in 0..<gridT {
                for h in 0..<gridH {
                    for w in 0..<gridW {
                        axes[0].append(startPosition + t)
                        axes[1].append(startPosition + h)
                        axes[2].append(startPosition + w)
                    }
                }
            }
            return max(gridH, gridW)
        }

        while cursor < seqLen {
            if tokenValues[cursor] != Int32(imageTokenId) {
                let textStart = cursor
                while cursor < seqLen, tokenValues[cursor] != Int32(imageTokenId) {
                    cursor += 1
                }
                appendText(count: cursor - textStart)
                continue
            }

            let runStart = cursor
            while cursor < seqLen, tokenValues[cursor] == Int32(imageTokenId) {
                cursor += 1
            }
            let runLength = cursor - runStart
            guard replacementIndex < replacements.count else {
                throw Q35Error.generationFailed("Qwen-family M-RoPE found more image-token runs than encoded images.")
            }

            let replacement = replacements[replacementIndex]
            guard runLength == replacement.embeddings.dim(0) else {
                throw Q35Error.generationFailed(
                    "Qwen-family image-token span mismatch: prompt has \(runLength) placeholders, vision tower produced \(replacement.embeddings.dim(0)) tokens."
                )
            }

            currentPosition += appendVision(
                startPosition: currentPosition,
                gridTHW: replacement.gridTHW
            )
            replacementIndex += 1
        }

        guard replacementIndex == replacements.count else {
            throw Q35Error.generationFailed("Qwen-family M-RoPE received encoded images without matching image-token runs.")
        }
        guard axes.allSatisfy({ $0.count == seqLen }) else {
            throw Q35Error.generationFailed("Qwen-family M-RoPE position length did not match prompt length.")
        }

        let maxPosition = axes.flatMap { $0 }.max() ?? (seqLen - 1)
        let ropeDelta = maxPosition + 1 - seqLen
        let values = axes.flatMap { $0.map(Int32.init) }
        return Q35MRoPEPositionData(
            positionIds: MLXArray(values, [3, 1, seqLen]),
            ropeDelta: ropeDelta
        )
    }

    func insertVisionEmbeddings(
        hiddenStates: MLXArray,
        inputIds: MLXArray,
        imageTokenId: Int,
        replacements: [Q35VisionReplacement]
    ) -> MLXArray {
        guard !replacements.isEmpty else { return hiddenStates }

        let seqLen = hiddenStates.dim(1)
        let tokenArray = inputIds.asType(.int32)
        MLX.eval(tokenArray)
        let tokenValues = tokenArray.asArray(Int32.self)

        var positions: [Int] = []
        positions.reserveCapacity(replacements.count)
        for index in 0..<seqLen where tokenValues[index] == Int32(imageTokenId) {
            positions.append(index)
        }

        guard !positions.isEmpty else { return hiddenStates }

        var runs: [(start: Int, end: Int)] = []
        for position in positions {
            if let last = runs.last, last.end == position {
                runs[runs.count - 1] = (start: last.start, end: position + 1)
            } else {
                runs.append((start: position, end: position + 1))
            }
        }

        let pairCount = min(runs.count, replacements.count)

        var parts: [MLXArray] = []
        parts.reserveCapacity(pairCount * 2 + 1)

        var cursor = 0
        for pairIndex in 0..<pairCount {
            let run = runs[pairIndex]
            if run.start > cursor {
                parts.append(hiddenStates[0..., cursor..<run.start, 0...])
            }

            var replacement = replacements[pairIndex].embeddings
            if replacement.dtype != hiddenStates.dtype {
                replacement = replacement.asType(hiddenStates.dtype)
            }
            parts.append(replacement.expandedDimensions(axis: 0))

            cursor = run.end
        }

        if cursor < seqLen {
            parts.append(hiddenStates[0..., cursor..., 0...])
        }

        if parts.isEmpty {
            return hiddenStates
        }
        if parts.count == 1 {
            return parts[0]
        }
        return MLX.concatenated(parts, axis: 1)
    }

    private func loadImageTensor(
        from imageRef: String,
        patchSize: Int,
        spatialMergeSize: Int,
        maximumVisionTokens: Int
    ) throws -> (tensor: MLXArray, gridTHW: (Int, Int, Int)) {
        let image = try loadImage(from: imageRef)
        let contextMaximumPixels = Self.visionPixelLimit(
            tokenLimit: maximumVisionTokens,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            configuredMaximum: visionMaxPixels
        )
        let target = try Self.qwen3VLTargetSize(
            originalWidth: image.width,
            originalHeight: image.height,
            patchSize: patchSize,
            spatialMergeSize: spatialMergeSize,
            minPixels: min(visionMinPixels, contextMaximumPixels),
            maxPixels: contextMaximumPixels
        )
        let resized = try MediaImageIO.resized(
            image,
            width: target.width,
            height: target.height
        )
        let floats = MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true)
        let pixels = MLXArray(
            floats,
            [1, 3, resized.height, resized.width]
        )
        let gridTHW = (
            1,
            resized.height / patchSize,
            resized.width / patchSize
        )
        return (pixels, gridTHW)
    }

    private func loadImage(from imageRef: String) throws -> MediaImage {
        if let remoteURL = URL(string: imageRef),
           let scheme = remoteURL.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            let data = try Data(contentsOf: remoteURL)
            do {
                return try MediaImageIO.decode(data: data)
            } catch {
                throw NSError(
                    domain: "Q35Generator",
                    code: 1002,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to decode image URL: \(imageRef)"]
                )
            }
        }

        let localURL: URL
        if imageRef.hasPrefix("file://"), let parsed = URL(string: imageRef) {
            localURL = parsed
        } else {
            localURL = URL(fileURLWithPath: imageRef)
        }
        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw NSError(
                domain: "Q35Generator",
                code: 1003,
                userInfo: [NSLocalizedDescriptionKey: "Image file not found: \(imageRef)"]
            )
        }
        do {
            return try MediaImageIO.decode(localURL)
        } catch {
            throw NSError(
                domain: "Q35Generator",
                code: 1004,
                userInfo: [NSLocalizedDescriptionKey: "Failed to decode image file: \(imageRef)"]
            )
        }
    }
}
