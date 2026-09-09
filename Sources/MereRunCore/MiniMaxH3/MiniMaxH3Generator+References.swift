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
    func preparedReferences(
        options: MiniMaxH3GenerationOptions
    ) throws -> [PreparedReference] {
        let key = ReferenceCacheKey(
            references: options.references,
            maximumFrameCount: options.numFrames,
            targetWidth: options.internalWidth,
            targetHeight: options.internalHeight
        )
        if retainsRuntime,
           let retainedPreparedReferences,
           retainedPreparedReferences.key == key {
            return retainedPreparedReferences.values
        }
        let values = try prepareReferences(options: options)
        if retainsRuntime { retainedPreparedReferences = (key, values) }
        return values
    }

    func encodeReferences(
        _ prepared: [PreparedReference],
        options: MiniMaxH3GenerationOptions,
        resources: MiniMaxH3Resources,
        progressHandler: (@Sendable (MiniMaxH3GenerationProgress) -> Void)?
    ) throws -> PreparedReferenceRows {
        let key = ReferenceCacheKey(
            references: options.references,
            maximumFrameCount: options.numFrames,
            targetWidth: options.internalWidth,
            targetHeight: options.internalHeight
        )
        if retainsRuntime,
           let retainedPreparedReferenceRows,
           retainedPreparedReferenceRows.key == key {
            return retainedPreparedReferenceRows.values
        }
        progressHandler?(.init(stage: .encodingReferences, stepIndex: 0, totalSteps: prepared.count))
        let videoRows: [MLXArray] = try withMiniMaxH3AutoreleasePool {
            guard prepared.contains(where: { $0.visual != nil }) else { return [] }
            let vae = try loadVideoVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, reference) in prepared.enumerated() {
                guard let visual = reference.visual else { continue }
                let latent = reference.kind == .image
                    ? vae.encodeKeyframe(visual)
                    : vae.encodeReferenceVideo(visual)
                let packed = MiniMaxH3Geometry.patchifyVideo(latent).asType(.float32)
                MLX.eval(packed)
                rows.append(packed)
                progressHandler?(.init(
                    stage: .encodingReferences,
                    stepIndex: index + 1,
                    totalSteps: prepared.count
                ))
            }
            return rows
        }
        let audioRows: [MLXArray] = try withMiniMaxH3AutoreleasePool {
            guard prepared.contains(where: { $0.waveform != nil }) else { return [] }
            let vae = try loadAudioVAE(resources: resources)
            var rows: [MLXArray] = []
            for (index, reference) in prepared.enumerated() {
                guard let waveform = reference.waveform else { continue }
                let packed = MiniMaxH3Geometry.packAudio(
                    vae.encode(waveform)
                ).expandedDimensions(axis: 0)
                MLX.eval(packed)
                rows.append(packed)
                progressHandler?(.init(
                    stage: .encodingReferences,
                    stepIndex: index + 1,
                    totalSteps: prepared.count
                ))
            }
            return rows
        }
        let values = PreparedReferenceRows(video: videoRows, audio: audioRows)
        if retainsRuntime { retainedPreparedReferenceRows = (key, values) }
        return values
    }

    func prepareReferences(options: MiniMaxH3GenerationOptions) throws -> [PreparedReference] {
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("mererun-minimax-h3-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporaryRoot) }
        let maximumReferenceAudioSamples = MiniMaxH3AudioVAE.samplingRate * 15
        var totalReferenceAudioSamples = 0
        var prepared: [PreparedReference] = []
        for (referenceIndex, input) in options.references.enumerated() {
            switch input.kind {
            case .image:
                let source: MediaImage
                do { source = try MediaImageIO.decode(input.url) } catch {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, error.localizedDescription)
                }
                let size = try MiniMaxH3ServingContract.referenceImageCanvas(
                    width: source.width,
                    height: source.height,
                    targetWidth: options.internalWidth,
                    targetHeight: options.internalHeight
                )
                let image = try MediaImageIO.resized(source, width: size.width, height: size.height)
                let visual = Self.rgbTensor(image).expandedDimensions(axis: 0)
                let vision = try QwenVLImageLoader.pixelValues(image: image, patchSize: 16, spatialMergeSize: 2)
                prepared.append(.init(
                    kind: .image,
                    visual: visual,
                    visionBlocks: [vision],
                    blockTimestamps: [],
                    waveform: nil,
                    geometry: .init(
                        kind: .image,
                        videoLatentFrames: 1,
                        latentHeight: size.height / 16,
                        latentWidth: size.width / 16
                    )
                ))
            case .video:
                let framesDirectory = temporaryRoot.appendingPathComponent("video-\(referenceIndex)", isDirectory: true)
                try FileManager.default.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
                let sequence: VideoFrameSequence
                do { sequence = try MediaVideoIO.extractFrames(from: input.url, into: framesDirectory) } catch {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, error.localizedDescription)
                }
                guard !sequence.frameURLs.isEmpty else {
                    throw MiniMaxH3GeneratorError.mediaDecodeFailed(input.url, "video contains no frames")
                }
                let resampled = Self.resampledFrameIndices(
                    sourceCount: sequence.frameURLs.count,
                    sourceFPS: sequence.fps,
                    maximumCount: options.numFrames
                )
                let selected = resampled.isEmpty ? [0] : resampled
                let alignedCount = Self.trimReferenceFrameCount(selected.count)
                var indices = selected
                if indices.count < alignedCount {
                    indices.append(contentsOf: repeatElement(indices.last!, count: alignedCount - indices.count))
                } else {
                    indices = Array(indices.prefix(alignedCount))
                }
                let canvas = try resolveVideoCanvas(width: sequence.frameWidth, height: sequence.frameHeight)
                var images: [MediaImage] = []
                images.reserveCapacity(indices.count)
                for index in indices {
                    let image = try MediaImageIO.decode(sequence.frameURLs[index])
                    images.append(try MediaImageIO.resized(image, width: canvas.width, height: canvas.height))
                }
                let visual = MLX.stacked(images.map(Self.rgbTensor), axis: 0).expandedDimensions(axis: 0)
                let sampled = stride(from: 0, to: images.count, by: 12).map { images[$0] }
                var blocks: [MLXArray] = []
                var timestamps: [Double] = []
                for start in stride(from: 0, to: sampled.count, by: 2) {
                    let second = min(start + 1, sampled.count - 1)
                    let firstPixels = try QwenVLImageLoader.pixelValues(
                        image: sampled[start], patchSize: 16, spatialMergeSize: 2
                    )
                    let secondPixels = try QwenVLImageLoader.pixelValues(
                        image: sampled[second], patchSize: 16, spatialMergeSize: 2
                    )
                    blocks.append(MLX.concatenated([firstPixels, secondPixels], axis: 0))
                    timestamps.append((Double(start) / 2 + Double(second) / 2) / 2)
                }
                let waveform: MLXArray?
                if MediaVideoIO.hasAudioTrack(input.url) {
                    let maximumSamples = Int(
                        (Double(alignedCount) / Double(MiniMaxH3Geometry.framesPerSecond)
                            * Double(MiniMaxH3AudioVAE.samplingRate)).rounded()
                    )
                    let decoded = try decodeReferenceAudio(
                        input.url,
                        maximumSamples: maximumSamples,
                        truncatesAtLimit: true
                    )
                    guard decoded.dim(1) <= maximumReferenceAudioSamples - totalReferenceAudioSamples else {
                        throw MiniMaxH3GeneratorError.invalidOptions(
                            "ordered reference audio exceeds 15 seconds in total"
                        )
                    }
                    totalReferenceAudioSamples += decoded.dim(1)
                    waveform = decoded
                } else {
                    waveform = nil
                }
                prepared.append(.init(
                    kind: .video,
                    visual: visual,
                    visionBlocks: blocks,
                    blockTimestamps: timestamps,
                    waveform: waveform,
                    geometry: .init(
                        kind: .video,
                        videoLatentFrames: try MiniMaxH3Geometry.videoLatentFrameCount(for: alignedCount),
                        latentHeight: canvas.height / 16,
                        latentWidth: canvas.width / 16,
                        audioLatentFrames: waveform.map { ($0.dim(1) + 799) / 800 } ?? 0
                    )
                ))
            case .audio:
                let waveform = try decodeReferenceAudio(
                    input.url,
                    maximumSamples: maximumReferenceAudioSamples,
                    truncatesAtLimit: false
                )
                guard waveform.dim(1) <= maximumReferenceAudioSamples - totalReferenceAudioSamples else {
                    throw MiniMaxH3GeneratorError.invalidOptions(
                        "ordered reference audio exceeds 15 seconds in total"
                    )
                }
                totalReferenceAudioSamples += waveform.dim(1)
                prepared.append(.init(
                    kind: .audio,
                    visual: nil,
                    visionBlocks: [],
                    blockTimestamps: [],
                    waveform: waveform,
                    geometry: .init(
                        kind: .audio,
                        audioLatentFrames: (waveform.dim(1) + 799) / 800
                    )
                ))
            }
        }
        return prepared
    }

    func decodeReferenceAudio(
        _ url: URL,
        maximumSamples: Int,
        truncatesAtLimit: Bool
    ) throws -> MLXArray {
        guard maximumSamples >= MiniMaxH3AudioVAE.samplingRate * 2 else {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio requires at least 2 seconds at 32 kHz"
            )
        }
        let buffer: MediaAudioBuffer
        do {
            buffer = try MediaAudioIO.decode(
                url,
                targetSampleRate: MiniMaxH3AudioVAE.samplingRate,
                channels: 2
            )
        } catch {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(url, error.localizedDescription)
        }
        let availableSamples = buffer.samples.count / 2
        if !truncatesAtLimit, availableSamples > maximumSamples {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio exceeds the 15 second total limit"
            )
        }
        let frameCount = min(maximumSamples, availableSamples)
        guard frameCount >= MiniMaxH3AudioVAE.samplingRate * 2 else {
            throw MiniMaxH3GeneratorError.mediaDecodeFailed(
                url,
                "reference audio requires at least 2 seconds at 32 kHz"
            )
        }
        return MLXArray(Array(buffer.samples.prefix(frameCount * 2)))
            .reshaped(1, frameCount, 2)
    }

    func referenceConditionerPresentation(
        tokenizer: QwenTokenizer,
        prompt: String,
        references: [PreparedReference],
        continuationFrame: MLXArray? = nil
    ) throws -> ConditionerPresentation {
        guard let imageTokenID = tokenizer.imageTokenId,
              let videoTokenID = tokenizer.videoTokenId,
              let visionStartTokenID = tokenizer.visionStartTokenId,
              let visionEndTokenID = tokenizer.visionEndTokenId else {
            throw MiniMaxH3GeneratorError.invalidOptions("Qwen3-VL tokenizer is missing Ref2VA vision tokens")
        }
        var tokenIDs: [Int] = []
        var tokenTags: [Int32] = []
        var images: [QwenVLEncoder.ConditioningImage] = []
        var imageCount = 0
        var videoCount = 0
        var audioCount = 0
        func appendText(_ value: String) {
            let ids = tokenizer.encodeText(value)
            tokenIDs.append(contentsOf: ids)
            tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.text.rawValue, count: ids.count))
        }
        func appendVision(_ pixels: MLXArray, padToken: Int) {
            let patchHeight = pixels.dim(2) / 16
            let patchWidth = pixels.dim(3) / 16
            let temporal = max(1, (pixels.dim(0) + 1) / 2)
            let count = temporal * (patchHeight / 2) * (patchWidth / 2)
            tokenIDs.append(visionStartTokenID)
            tokenTags.append(MiniMaxH3Modality.video.rawValue)
            let range = tokenIDs.count..<(tokenIDs.count + count)
            tokenIDs.append(contentsOf: repeatElement(padToken, count: count))
            tokenTags.append(contentsOf: repeatElement(MiniMaxH3Modality.video.rawValue, count: count))
            tokenIDs.append(visionEndTokenID)
            tokenTags.append(MiniMaxH3Modality.video.rawValue)
            images.append(.init(
                pixelValues: pixels,
                tokenRange: range,
                temporalPatchCount: temporal,
                heightPatchCount: patchHeight,
                widthPatchCount: patchWidth
            ))
        }
        if let continuationFrame {
            MLX.eval(continuationFrame)
            let frame = continuationFrame[0, 0]
            let image = try MediaImageIO.imageFromRGBHWC(
                frame.asArray(UInt8.self),
                width: continuationFrame.dim(3),
                height: continuationFrame.dim(2)
            )
            let pixels = try QwenVLImageLoader.pixelValues(
                image: image,
                patchSize: 16,
                spatialMergeSize: 2
            )
            imageCount += 1
            appendText("<Picture \(imageCount)>: ")
            appendVision(pixels, padToken: imageTokenID)
        }
        for reference in references {
            if reference.waveform != nil {
                audioCount += 1
                appendText("<Audio \(audioCount)>: ")
            }
            switch reference.kind {
            case .image:
                imageCount += 1
                appendText("<Picture \(imageCount)>: ")
                appendVision(reference.visionBlocks[0], padToken: imageTokenID)
            case .video:
                videoCount += 1
                appendText("<Video \(videoCount)>: ")
                for (index, block) in reference.visionBlocks.enumerated() {
                    let rounded = (reference.blockTimestamps[index] * 10).rounded(.toNearestOrEven) / 10
                    appendText(String(format: "<%.1f seconds>", rounded))
                    appendVision(block, padToken: videoTokenID)
                }
            case .audio:
                break
            }
        }
        appendText(prompt)
        return ConditionerPresentation(tokenIDs: tokenIDs, tokenTags: tokenTags, images: images)
    }

    static func rgbTensor(_ image: MediaImage) -> MLXArray {
        MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
            .reshaped(3, image.height, image.width)
            .transposed(1, 2, 0)
    }

    func resolveVideoCanvas(width: Int, height: Int) throws -> (width: Int, height: Int) {
        guard width > 0, height > 0, width <= 4 * height, height <= 4 * width else {
            throw MiniMaxH3GeneratorError.invalidOptions("reference video aspect ratio must be between 1:4 and 4:1")
        }
        let ratio = Double(width) / Double(height)
        var resolvedWidth = ratio >= 1 ? 768 * ratio : 768
        var resolvedHeight = ratio >= 1 ? 768.0 : 768 / ratio
        let maximumArea = Double(768 * 1_344)
        if resolvedWidth * resolvedHeight > maximumArea {
            let scale = sqrt(maximumArea / (resolvedWidth * resolvedHeight))
            resolvedWidth *= scale
            resolvedHeight *= scale
        }
        return (
            max(32, Int((resolvedWidth / 32).rounded()) * 32),
            max(32, Int((resolvedHeight / 32).rounded()) * 32)
        )
    }

    static func resampledFrameIndices(
        sourceCount: Int,
        sourceFPS: Double,
        maximumCount: Int
    ) -> [Int] {
        guard sourceCount > 0, sourceFPS > 0 else { return [] }
        if sourceFPS == Double(MiniMaxH3Geometry.framesPerSecond) {
            return Array(0..<min(sourceCount, maximumCount))
        }
        let scale = Double(MiniMaxH3Geometry.framesPerSecond) / sourceFPS
        let slots = (0..<sourceCount).map { Int((Double($0) * scale).rounded()) }
        let end = Int((Double(sourceCount) * scale).rounded())
        var result: [Int] = []
        for index in 0..<sourceCount {
            let next = index + 1 < sourceCount ? slots[index + 1] : end
            if next > slots[index] {
                result.append(contentsOf: repeatElement(index, count: next - slots[index]))
            }
            if result.count >= maximumCount { return Array(result.prefix(maximumCount)) }
        }
        return result
    }

    static func trimReferenceFrameCount(_ count: Int) -> Int {
        max(1, (count - 5) / 17) * 17 + 5
    }

}
