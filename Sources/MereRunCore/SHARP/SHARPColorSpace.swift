import MLX

public enum SharpColorSpaceOps {
    public static func sRGBToLinearRGB(_ sRGB: MLXArray) -> MLXArray {
        let threshold = MLXArray(0.04045).asType(sRGB.dtype)
        let condition = sRGB .<= threshold

        let trueBranch = sRGB / 12.92
        let safeFalseInput = which(condition, threshold, sRGB)
        let falseBranch = ((safeFalseInput + 0.055) / 1.055).pow(2.4)

        return which(condition, trueBranch, falseBranch)
    }

    public static func linearRGBToSRGB(_ linearRGB: MLXArray) -> MLXArray {
        let threshold = MLXArray(0.0031308).asType(linearRGB.dtype)
        let condition = linearRGB .<= threshold

        let trueBranch = linearRGB * 12.92
        let safeFalseInput = which(condition, threshold, linearRGB)
        let falseBranch = (1.055 * safeFalseInput.pow(1.0 / 2.4)) - 0.055

        return which(condition, trueBranch, falseBranch)
    }
}
