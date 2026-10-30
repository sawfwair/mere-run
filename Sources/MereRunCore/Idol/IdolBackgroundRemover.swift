import Foundation
import MediaIO
import MLX

/// The BiRefNet graph stays in the image runtime; this boundary owns image bytes.
enum IdolBackgroundRemover {
    static func remove(_ image: MediaImage, root: URL, progress: ((String) -> Void)? = nil) throws -> MediaImage {
        let model = try BiRefNet(weights: MLX.loadArrays(url: root.appending(path: "birefnet.safetensors")))
        let imageTensor = MLXArray(MediaImageIO.rgbCHWFloat(image, normalizedToMinusOneToOne: false))
            .reshaped(1, 3, image.height, image.width).transposed(0, 2, 3, 1)
        let resized = BiRefNetOps.resize(imageTensor, height: 1024, width: 1024)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]), std = MLXArray([Float(0.229), 0.224, 0.225])
        let alpha = try model((resized - mean) / std, progress: progress)
        let restored = BiRefNetOps.resize(alpha.asType(.float32), height: image.height, width: image.width)
        let mask = (clip(restored, min: 0, max: 1) * 255).asType(.uint8).asArray(UInt8.self)
        var bytes = image.rgba8
        for pixel in 0..<(image.width * image.height) { bytes[pixel * 4 + 3] = mask[pixel] }
        return try MediaImage(width: image.width, height: image.height, rgba8: bytes)
    }
}
