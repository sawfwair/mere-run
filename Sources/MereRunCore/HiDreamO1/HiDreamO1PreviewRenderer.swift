import Foundation
import MLX

enum HiDreamO1PreviewRenderer {
    static func promptPreview(
        prompt: String,
        seed: UInt64,
        resolution: HiDreamO1SampleBuilder.Resolution
    ) -> MLXArray {
        let width = resolution.width
        let height = resolution.height
        let digest = fnv1a64("\(prompt)\u{0}\(seed)")
        let hueA = Float(digest & 0xFF) / 255.0
        let hueB = Float((digest >> 8) & 0xFF) / 255.0
        let hueC = Float((digest >> 16) & 0xFF) / 255.0
        var values = [Float](repeating: 0, count: width * height * 3)

        for yIndex in 0..<height {
            let y = Float(yIndex) / Float(max(1, height - 1))
            for xIndex in 0..<width {
                let x = Float(xIndex) / Float(max(1, width - 1))
                let wave = sin((x * Float.pi * 6.0) + hueA * Float.pi * 2.0)
                    * cos((y * Float.pi * 5.0) + hueB * Float.pi * 2.0)
                let vignette = 1.0 - min(1.0, hypot(x - 0.5, y - 0.5) * 1.65)
                let pixel = yIndex * width + xIndex
                values[pixel] = clamp01(0.35 + 0.45 * x + 0.20 * wave + 0.15 * vignette) * 2.0 - 1.0
                values[pixel + width * height] = clamp01(0.28 + 0.42 * y + 0.18 * (1.0 - wave) + 0.12 * hueC) * 2.0 - 1.0
                values[pixel + 2 * width * height] = clamp01(0.25 + 0.30 * (1.0 - x) + 0.30 * vignette + 0.15 * hueB) * 2.0 - 1.0
            }
        }

        return MLXArray(values, [3, height, width])
    }

    static func referencePreview(
        referenceTensors: [HiDreamO1ImagePreprocessor.PatchTensor],
        targetResolution: HiDreamO1SampleBuilder.Resolution,
        seed: UInt64
    ) -> MLXArray {
        guard !referenceTensors.isEmpty else {
            return promptPreview(prompt: "", seed: seed, resolution: targetResolution)
        }
        if referenceTensors.count == 1,
           referenceTensors[0].resolution == targetResolution {
            return referenceTensors[0].imageCHW
        }

        let width = targetResolution.width
        let height = targetResolution.height
        var accum = [Float](repeating: 0, count: width * height * 3)
        for (index, tensor) in referenceTensors.enumerated() {
            let source = tensor.imageCHW.asType(.float32)
            MLX.eval(source)
            let sourceValues = source.asArray(Float.self)
            blend(
                sourceValues: sourceValues,
                sourceWidth: tensor.resolution.width,
                sourceHeight: tensor.resolution.height,
                into: &accum,
                targetWidth: width,
                targetHeight: height,
                weight: 1.0 / Float(index + 1)
            )
        }
        let scale = Float(referenceTensors.count)
        return MLXArray(accum.map { max(-1.0, min(1.0, $0 / scale)) }, [3, height, width])
    }

    private static func blend(
        sourceValues: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        into target: inout [Float],
        targetWidth: Int,
        targetHeight: Int,
        weight: Float
    ) {
        let targetPlane = targetWidth * targetHeight
        let sourcePlane = sourceWidth * sourceHeight
        for yIndex in 0..<targetHeight {
            let sourceY = min(sourceHeight - 1, yIndex * sourceHeight / targetHeight)
            for xIndex in 0..<targetWidth {
                let sourceX = min(sourceWidth - 1, xIndex * sourceWidth / targetWidth)
                let targetPixel = yIndex * targetWidth + xIndex
                let sourcePixel = sourceY * sourceWidth + sourceX
                for channel in 0..<3 {
                    target[targetPixel + channel * targetPlane] += sourceValues[sourcePixel + channel * sourcePlane] * weight
                }
            }
        }
    }

    private static func fnv1a64(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return hash
    }

    private static func clamp01(_ value: Float) -> Float {
        max(0.0, min(1.0, value))
    }
}
