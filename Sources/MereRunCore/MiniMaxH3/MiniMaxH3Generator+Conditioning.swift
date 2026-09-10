import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3Generator {
    func encodeKeyframes(
        _ urls: [URL],
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> MLXArray? {
        guard !urls.isEmpty else { return nil }
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 0, totalSteps: urls.count))
        return try withMiniMaxH3AutoreleasePool {
            let vae = try loadVideoVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, url) in urls.enumerated() {
                let image: MediaImage
                do {
                    image = try MediaImageIO.resized(
                        try MediaImageIO.decode(url),
                        width: options.internalWidth,
                        height: options.internalHeight
                    )
                } catch {
                    throw MiniMaxH3GeneratorError.imageDecodeFailed(url)
                }
                let chw = MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
                    .reshaped(1, 3, options.internalHeight, options.internalWidth)
                let rgb = chw.transposed(0, 2, 3, 1)
                rows.append(MiniMaxH3Geometry.patchifyVideo(vae.encodeKeyframe(rgb)).asType(.float32))
                progressHandler?(.init(stage: .encodingKeyframes, stepIndex: index + 1, totalSteps: urls.count))
            }
            let result = MLX.concatenated(rows, axis: 1)
            MLX.eval(result)
            return result
        }
    }

    func encodeContinuation(
        _ continuation: MiniMaxH3ContinuationInput,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> ContinuationConditions {
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 0, totalSteps: 2))
        let videoVAE = try loadVideoVAE(resources: resources)
        let rgb = continuation.frames.asType(.float32) / 255
        var videoRows: [MLXArray] = []
        var videoAnchors: [MiniMaxH3KeyframeAnchor] = []
        if continuation.frameCount > 1 {
            let history = rgb[
                0...,
                0..<(continuation.frameCount - 1),
                0...,
                0...,
                0...
            ]
            let historyLatent = videoVAE.encodeReferenceVideo(history)
            videoRows.append(MiniMaxH3Geometry.patchifyVideo(historyLatent).asType(.float32))
            videoAnchors.append(.history(latentFrameCount: historyLatent.dim(2)))
        }
        let boundary = rgb[
            0...,
            (continuation.frameCount - 1)..<continuation.frameCount,
            0...,
            0...,
            0...
        ]
        let boundaryLatent = videoVAE.encodeKeyframe(boundary.squeezed(axis: 1))
        videoRows.append(MiniMaxH3Geometry.patchifyVideo(boundaryLatent).asType(.float32))
        videoAnchors.append(.first)
        let packedVideo = MLX.concatenated(videoRows, axis: 1)
        MLX.eval(packedVideo)
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 1, totalSteps: 2))

        let audioVAE = try loadAudioVAE(resources: resources)
        let audioLatent = audioVAE.encode(continuation.audio.asType(.float32))
        let boundaryLatentCount = min(
            audioLatent.dim(3),
            max(
                1,
                Int((Double(MiniMaxH3Geometry.audioLatentsPerSecond)
                    / Double(MiniMaxH3Geometry.framesPerSecond)).rounded())
            )
        )
        let historyLatentCount = audioLatent.dim(3) - boundaryLatentCount
        var audioRows: [MLXArray] = []
        var audioAnchors: [MiniMaxH3AudioConditionAnchor] = []
        if historyLatentCount > 0 {
            let history = audioLatent[0..., 0..., 0..., 0..<historyLatentCount]
            audioRows.append(MiniMaxH3Geometry.packAudio(history).expandedDimensions(axis: 0))
            audioAnchors.append(.history(latentFrameCount: historyLatentCount))
        }
        let first = audioLatent[
            0...,
            0...,
            0...,
            historyLatentCount..<audioLatent.dim(3)
        ]
        audioRows.append(MiniMaxH3Geometry.packAudio(first).expandedDimensions(axis: 0))
        audioAnchors.append(.first(latentFrameCount: boundaryLatentCount))
        let packedAudio = MLX.concatenated(audioRows, axis: 1)
        MLX.eval(packedAudio)
        progressHandler?(.init(stage: .encodingKeyframes, stepIndex: 2, totalSteps: 2))
        return ContinuationConditions(
            videoRows: packedVideo,
            audioRows: packedAudio,
            videoAnchors: videoAnchors,
            audioAnchors: audioAnchors
        )
    }

    static func concatenateRows(_ arrays: [MLXArray?]) -> MLXArray? {
        concatenateRows(arrays.compactMap { $0 })
    }

    static func concatenateRows(_ arrays: [MLXArray]) -> MLXArray? {
        let values = arrays
        guard !values.isEmpty else { return nil }
        return values.count == 1 ? values[0] : MLX.concatenated(values, axis: 1)
    }

    static func boundaryFrame(_ continuation: MiniMaxH3ContinuationInput) -> MLXArray {
        continuation.frames[
            0...,
            (continuation.frameCount - 1)..<continuation.frameCount,
            0...,
            0...,
            0...
        ]
    }

    func conditionerPresentation(
        tokenizer: QwenTokenizer,
        options: MiniMaxH3GenerationOptions,
        continuationFrame: MLXArray? = nil
    ) throws -> ConditionerPresentation {
        let keyframeURLs = Self.frameConditions(options: options).map(\.url)
        var tokenIDs: [Int] = []
        var tokenTags: [Int32] = []
        var images: [QwenVLEncoder.ConditioningImage] = []

        if continuationFrame != nil || !keyframeURLs.isEmpty {
            guard let imageTokenID = tokenizer.imageTokenId,
                  let visionStartTokenID = tokenizer.visionStartTokenId,
                  let visionEndTokenID = tokenizer.visionEndTokenId else {
                throw MiniMaxH3GeneratorError.invalidOptions("Qwen3-VL tokenizer is missing vision tokens")
            }
            var preparedImages: [MediaImage] = []
            if let continuationFrame {
                MLX.eval(continuationFrame)
                let frame = continuationFrame[0, 0]
                preparedImages.append(try MediaImageIO.imageFromRGBHWC(
                    frame.asArray(UInt8.self),
                    width: continuationFrame.dim(3),
                    height: continuationFrame.dim(2)
                ))
            }
            for url in keyframeURLs {
                do {
                    preparedImages.append(try MediaImageIO.resized(
                        try MediaImageIO.decode(url),
                        width: options.internalWidth,
                        height: options.internalHeight
                    ))
                } catch {
                    throw MiniMaxH3GeneratorError.imageDecodeFailed(url)
                }
            }
            for (index, prepared) in preparedImages.enumerated() {
                let pixelValues = try QwenVLImageLoader.pixelValues(
                    image: prepared,
                    patchSize: 16,
                    spatialMergeSize: 2
                )
                let patchHeight = pixelValues.dim(2) / 16
                let patchWidth = pixelValues.dim(3) / 16
                let imageTokenCount = QwenVLEncoder.imageTokenCount(
                    imageHeight: pixelValues.dim(2),
                    imageWidth: pixelValues.dim(3),
                    patchSize: 16,
                    spatialMergeSize: 2
                )
                let labelIDs = tokenizer.encodeText("<Picture \(index + 1)>: ")
                tokenIDs.append(contentsOf: labelIDs)
                tokenTags.append(contentsOf: repeatElement(
                    MiniMaxH3Modality.text.rawValue,
                    count: labelIDs.count
                ))
                tokenIDs.append(visionStartTokenID)
                tokenTags.append(MiniMaxH3Modality.video.rawValue)
                let tokenRange = tokenIDs.count..<(tokenIDs.count + imageTokenCount)
                tokenIDs.append(contentsOf: repeatElement(imageTokenID, count: imageTokenCount))
                tokenTags.append(contentsOf: repeatElement(
                    MiniMaxH3Modality.video.rawValue,
                    count: imageTokenCount
                ))
                tokenIDs.append(visionEndTokenID)
                tokenTags.append(MiniMaxH3Modality.video.rawValue)
                images.append(.init(
                    pixelValues: pixelValues,
                    tokenRange: tokenRange,
                    heightPatchCount: patchHeight,
                    widthPatchCount: patchWidth
                ))
            }
        }
        let promptIDs = tokenizer.encodeText(options.prompt)
        tokenIDs.append(contentsOf: promptIDs)
        tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.text.rawValue, count: promptIDs.count))
        return ConditionerPresentation(tokenIDs: tokenIDs, tokenTags: tokenTags, images: images)
    }

    static func frameConditions(options: MiniMaxH3GenerationOptions) -> [FrameCondition] {
        var conditions: [(index: Int, condition: FrameCondition)] = []
        if let firstFrameURL = options.firstFrameURL {
            conditions.append((0, .init(url: firstFrameURL, anchor: .first)))
        }
        conditions.append(contentsOf: options.frameInputs.map { input in
            let anchor: MiniMaxH3KeyframeAnchor
            if input.frameIndex == 0 {
                anchor = .first
            } else if input.frameIndex == options.numFrames - 1 {
                anchor = .last
            } else {
                anchor = .frame(input.frameIndex)
            }
            return (input.frameIndex, .init(url: input.url, anchor: anchor))
        })
        if let lastFrameURL = options.lastFrameURL {
            conditions.append((
                options.numFrames - 1,
                .init(url: lastFrameURL, anchor: .last)
            ))
        }
        return conditions.sorted { $0.index < $1.index }.map(\.condition)
    }

}
