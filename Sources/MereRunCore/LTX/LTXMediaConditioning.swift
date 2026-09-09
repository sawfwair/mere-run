import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func loadImageForEncoding(
    url: URL,
    width: Int,
    height: Int,
    dtype: DType,
    hdrColorSpace: LTXHDRColorSpace? = nil,
    crf: Int = 0
) throws -> MLXArray {
    if MediaHDRImageIO.isEXR(url) {
        guard let hdrColorSpace else {
            throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
        }
        do {
            let image = try MediaHDRImageIO.centerCropped(
                MediaHDRImageIO.decodeEXR(url),
                width: width,
                height: height
            )
            return LTXHDRColorPipeline.makeConditioningImage(
                image,
                colorSpace: hdrColorSpace,
                dtype: dtype
            )
        } catch {
            throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
        }
    }
    let image: MediaImage
    do {
        let decoded = try MediaImageIO.decode(url)
        image = try MediaImageIO.h264RoundTrip(decoded, crf: crf)
    } catch {
        throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
    }

    let channels: [Float]
    do {
        channels = try MediaImageIO.bilinearCenterCroppedRGBCHWFloat(
            image,
            width: width,
            height: height,
            normalizedToMinusOneToOne: true
        )
    } catch {
        throw LTXDistilledLatentGeneratorError.imageDecodeFailed(url)
    }

    let chw = MLXArray(channels).reshaped(1, 3, height, width).asType(dtype)
    return chw.reshaped(1, 3, 1, height, width)
}

func loadVideoForEncoding(
    url: URL,
    width: Int,
    height: Int,
    frameCap: Int,
    temporalScaleFactor: Int,
    dtype: DType,
    hdrColorSpace: LTXHDRColorSpace? = nil,
    duplicateEachFrame: Bool = false,
    hdrICLoRAReference: Bool = false
) throws -> MLXArray {
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw LTXUnifiedAVGeneratorError.referenceVideoNotFound(url)
    }
    if MediaHDRImageIO.isEXRDirectory(url) {
        guard let hdrColorSpace else {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
                url,
                "EXR input requires an explicit HDR color space"
            )
        }
        do {
            let sourceFrameCap = duplicateEachFrame ? (frameCap + 1) / 2 : frameCap
            let urls = Array(try MediaHDRImageIO.exrFrameURLs(in: url).prefix(sourceFrameCap))
            guard !urls.isEmpty else {
                throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, "no EXR frames")
            }
            var indices = [0]
            if urls.count > 1 {
                indices.append(contentsOf: stride(from: 1, to: urls.count, by: temporalScaleFactor))
            }
            var frames = try indices.map { index in
                let image = try MediaHDRImageIO.reflectPadded(
                    MediaHDRImageIO.decodeEXR(urls[index]),
                    width: width,
                    height: height
                )
                return LTXHDRColorPipeline.makeConditioningImage(
                    image,
                    colorSpace: hdrColorSpace,
                    dtype: dtype
                )
            }
            if duplicateEachFrame {
                frames = Array(frames.flatMap { [$0, $0] }.prefix(frameCap))
            }
            return MLX.concatenated(frames, axis: 2)
        } catch let error as LTXUnifiedAVGeneratorError {
            throw error
        } catch {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, error.localizedDescription)
        }
    }
    let temporaryDirectory = FileManager.default.temporaryDirectory
        .appendingPathComponent("mererun-ltx-reference-\(UUID().uuidString)", isDirectory: true)
    do {
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: temporaryDirectory) }
        let sourceFrameCap = duplicateEachFrame ? (frameCap + 1) / 2 : frameCap
        let sequence = try MediaVideoIO.extractFrames(
            from: url,
            into: temporaryDirectory,
            endFrame: max(0, sourceFrameCap - 1)
        )
        guard !sequence.frameURLs.isEmpty else {
            throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(url, "no frames")
        }
        var indices = [0]
        if sequence.frameURLs.count > 1 {
            indices.append(contentsOf: stride(
                from: 1,
                to: sequence.frameURLs.count,
                by: temporalScaleFactor
            ))
        }
        var frames = try indices.map { index in
            if hdrICLoRAReference {
                let image = try MediaImageIO.decode(sequence.frameURLs[index])
                var rgb = [Float](repeating: 0, count: image.width * image.height * 3)
                for pixel in 0..<(image.width * image.height) {
                    rgb[pixel * 3] = Float(image.rgba8[pixel * 4]) / 255
                    rgb[pixel * 3 + 1] = Float(image.rgba8[pixel * 4 + 1]) / 255
                    rgb[pixel * 3 + 2] = Float(image.rgba8[pixel * 4 + 2]) / 255
                }
                let padded = try MediaHDRImageIO.reflectPadded(
                    MediaFloatImage(width: image.width, height: image.height, rgb: rgb),
                    width: width,
                    height: height
                )
                return (MLXArray(padded.rgb).reshaped(1, height, width, 3) * MLXArray(Float(2))
                    - MLXArray(Float(1)))
                    .transposed(0, 3, 1, 2)
                    .reshaped(1, 3, 1, height, width)
                    .asType(dtype)
            }
            return try loadImageForEncoding(
                url: sequence.frameURLs[index],
                width: width,
                height: height,
                dtype: dtype
            )
        }
        if duplicateEachFrame {
            frames = Array(frames.flatMap { [$0, $0] }.prefix(frameCap))
        }
        return MLX.concatenated(frames, axis: 2)
    } catch let error as LTXUnifiedAVGeneratorError {
        throw error
    } catch {
        throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
            url,
            error.localizedDescription
        )
    }
}

func loadLTXReferenceAttentionWeights(
    reference: LTXReferenceVideoConditioningInput,
    width: Int,
    height: Int,
    frameCap: Int,
    targetLatent: MLXArray,
    dtype: DType
) throws -> MLXArray? {
    guard let maskURL = reference.attentionMaskVideoURL else { return nil }
    let pixelMask = try loadVideoForEncoding(
        url: maskURL,
        width: width,
        height: height,
        frameCap: frameCap,
        temporalScaleFactor: 1,
        dtype: dtype
    )
    let targetShape = LTXVideoLatentShape(
        batch: targetLatent.dim(0),
        channels: targetLatent.dim(1),
        frames: targetLatent.dim(2),
        height: targetLatent.dim(3),
        width: targetLatent.dim(4)
    )
    guard pixelMask.dim(0) == targetShape.batch,
          pixelMask.dim(3).isMultiple(of: targetShape.height),
          pixelMask.dim(4).isMultiple(of: targetShape.width),
          targetShape.frames == 1
            || (pixelMask.dim(2) - 1).isMultiple(of: targetShape.frames - 1) else {
        throw LTXUnifiedAVGeneratorError.referenceVideoDecodeFailed(
            maskURL,
            "mask video frames or dimensions are incompatible with the encoded reference"
        )
    }
    let grayscale = MLX.clip(
        (MLX.mean(pixelMask, axis: 1, keepDims: true) + MLXArray(Float(1)))
            / MLXArray(Float(2)),
        min: MLXArray(Float(0)),
        max: MLXArray(Float(1))
    )
    let weights = downsampleLTXReferenceAttentionMask(
        grayscale,
        targetLatentShape: targetShape
    )
    MLX.eval(weights)
    return weights
}
