import MLX

public enum SCAIL2MaskEncoder {
    public static let colorNames = [
        "white", "red", "green", "blue", "yellow", "magenta", "cyan",
    ]
    public static let colorChannels = 7
    public static let temporalCompression = 4
    public static let outputChannels = colorChannels * temporalCompression

    /// Converts an RGB mask in `[T, H, W, 3]` layout and `[-1, 1]` range to the
    /// exact 28-channel latent mask consumed by SCAIL-2.
    public static func encode(_ mask: MLXArray, spatialCompression: Int = 8) -> MLXArray {
        precondition(mask.ndim == 4 && mask.dim(3) == 3)
        precondition(mask.dim(0) >= 1 && (mask.dim(0) - 1).isMultiple(of: temporalCompression))
        precondition(spatialCompression > 0)
        precondition(mask.dim(1).isMultiple(of: spatialCompression))
        precondition(mask.dim(2).isMultiple(of: spatialCompression))

        let frames = mask.dim(0)
        let height = mask.dim(1)
        let width = mask.dim(2)
        let latentFrames = (frames - 1) / temporalCompression + 1
        let latentHeight = height / spatialCompression
        let latentWidth = width / spatialCompression
        let threshold = MLXArray(Float((225.0 - 127.5) / 127.5))
        let on = (mask.asType(.float32) .> threshold).asType(.float32)
        let red = on[0..., 0..., 0..., 0..<1]
        let green = on[0..., 0..., 0..., 1..<2]
        let blue = on[0..., 0..., 0..., 2..<3]
        let notRed = 1 - red
        let notGreen = 1 - green
        let notBlue = 1 - blue
        let colors = MLX.concatenated([
            red * green * blue,
            red * notGreen * notBlue,
            notRed * green * notBlue,
            notRed * notGreen * blue,
            red * green * notBlue,
            red * notGreen * blue,
            notRed * green * blue,
        ], axis: 3)
        let downsampled = MLX.mean(
            colors.reshaped(
                frames,
                latentHeight, spatialCompression,
                latentWidth, spatialCompression,
                colorChannels
            ),
            axes: [2, 4]
        )
        let colorFirst = downsampled.transposed(0, 3, 1, 2)
        let first = MLX.repeated(colorFirst[0..<1], count: temporalCompression, axis: 0)
        let padded = frames == 1
            ? first
            : MLX.concatenated([first, colorFirst[1...]], axis: 0)
        return padded
            .reshaped(latentFrames, outputChannels, latentHeight, latentWidth)
            .transposed(1, 0, 2, 3)
    }
}
