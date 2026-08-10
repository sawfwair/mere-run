import MLX
import MLXNN
#if canImport(Accelerate)
import Accelerate
#endif

enum MiniMaxH3FrameScaler {
    static func scaled(
        _ frames: MLXArray,
        width: Int,
        height: Int
    ) throws -> MLXArray {
        guard frames.ndim == 5,
              frames.dim(4) == 3,
              frames.dtype == .uint8,
              width > 0,
              height > 0 else {
            throw MiniMaxH3GeneratorError.invalidOptions(
                "H3 output scaling requires uint8 [batch, frames, height, width, 3] pixels"
            )
        }
        let sourceHeight = frames.dim(2)
        let sourceWidth = frames.dim(3)
        guard sourceWidth != width || sourceHeight != height else { return frames }

        #if canImport(Accelerate)
        return try scaledWithVImage(
            frames,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            outputWidth: width,
            outputHeight: height
        )
        #else
        let batch = frames.dim(0)
        let frameCount = frames.dim(1)
        let resized = Upsample(
            scaleFactor: .array([
                Float(height) / Float(sourceHeight) + 1e-6,
                Float(width) / Float(sourceWidth) + 1e-6,
            ]),
            mode: .cubic(alignCorners: false)
        )(frames.asType(.float32).reshaped(batch * frameCount, sourceHeight, sourceWidth, 3))
        guard resized.dim(1) == height, resized.dim(2) == width else {
            throw MiniMaxH3GeneratorError.invalidOptions("H3 output scaler produced an unexpected canvas")
        }
        return MLX.clip(resized, min: 0, max: 255)
            .asType(.uint8)
            .reshaped(batch, frameCount, height, width, 3)
        #endif
    }

    #if canImport(Accelerate)
    private static func scaledWithVImage(
        _ frames: MLXArray,
        sourceWidth: Int,
        sourceHeight: Int,
        outputWidth: Int,
        outputHeight: Int
    ) throws -> MLXArray {
        MLX.eval(frames)
        let input = frames.asArray(UInt8.self)
        let frameCount = frames.dim(0) * frames.dim(1)
        let sourcePixelCount = sourceWidth * sourceHeight
        let outputPixelCount = outputWidth * outputHeight
        let sourceFrameBytes = sourcePixelCount * 3
        let outputFrameBytes = outputPixelCount * 3
        var output = [UInt8](repeating: 0, count: frameCount * outputFrameBytes)
        var sourceARGB = [UInt8](repeating: 255, count: sourcePixelCount * 4)
        var outputARGB = [UInt8](repeating: 255, count: outputPixelCount * 4)

        try sourceARGB.withUnsafeMutableBytes { sourceBytes in
            try outputARGB.withUnsafeMutableBytes { outputBytes in
                guard let sourceBase = sourceBytes.baseAddress,
                      let outputBase = outputBytes.baseAddress else {
                    throw MiniMaxH3GeneratorError.invalidOptions("H3 output scaler could not allocate buffers")
                }
                let sourcePixels = sourceBytes.bindMemory(to: UInt8.self)
                let outputPixels = outputBytes.bindMemory(to: UInt8.self)
                var sourceBuffer = vImage_Buffer(
                    data: sourceBase,
                    height: vImagePixelCount(sourceHeight),
                    width: vImagePixelCount(sourceWidth),
                    rowBytes: sourceWidth * 4
                )
                var outputBuffer = vImage_Buffer(
                    data: outputBase,
                    height: vImagePixelCount(outputHeight),
                    width: vImagePixelCount(outputWidth),
                    rowBytes: outputWidth * 4
                )
                let flags = vImage_Flags(kvImageHighQualityResampling | kvImageEdgeExtend)
                for frameIndex in 0..<frameCount {
                    let inputOffset = frameIndex * sourceFrameBytes
                    for pixel in 0..<sourcePixelCount {
                        sourcePixels[4 * pixel] = 255
                        sourcePixels[4 * pixel + 1] = input[inputOffset + 3 * pixel]
                        sourcePixels[4 * pixel + 2] = input[inputOffset + 3 * pixel + 1]
                        sourcePixels[4 * pixel + 3] = input[inputOffset + 3 * pixel + 2]
                    }
                    let error = vImageScale_ARGB8888(
                        &sourceBuffer,
                        &outputBuffer,
                        nil,
                        flags
                    )
                    guard error == kvImageNoError else {
                        throw MiniMaxH3GeneratorError.invalidOptions(
                            "H3 high-quality output scaling failed with vImage error \(error)"
                        )
                    }
                    let outputOffset = frameIndex * outputFrameBytes
                    for pixel in 0..<outputPixelCount {
                        output[outputOffset + 3 * pixel] = outputPixels[4 * pixel + 1]
                        output[outputOffset + 3 * pixel + 1] = outputPixels[4 * pixel + 2]
                        output[outputOffset + 3 * pixel + 2] = outputPixels[4 * pixel + 3]
                    }
                }
            }
        }
        return MLXArray(output).reshaped(
            frames.dim(0),
            frames.dim(1),
            outputHeight,
            outputWidth,
            3
        )
    }
    #endif
}
