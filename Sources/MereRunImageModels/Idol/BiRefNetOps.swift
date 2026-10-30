import MLX

public enum BiRefNetOps {
    /// NHWC bilinear resize with PyTorch align_corners=true arithmetic.
    public static func resize(_ input: MLXArray, height: Int, width: Int) -> MLXArray {
        let oldHeight = input.dim(1), oldWidth = input.dim(2)
        if height == oldHeight && width == oldWidth { return input }
        let y = MLXArray(0..<height).asType(.float32) * (height == 1 ? 0 : Float(oldHeight - 1) / Float(height - 1))
        let x = MLXArray(0..<width).asType(.float32) * (width == 1 ? 0 : Float(oldWidth - 1) / Float(width - 1))
        let y0 = floor(y).asType(.int32), x0 = floor(x).asType(.int32)
        let dy = (y - y0.asType(.float32)).reshaped(1, height, 1, 1)
        let dx = (x - x0.asType(.float32)).reshaped(1, 1, width, 1)
        let source = input.asType(.float32)
        let top = source.take(y0, axis: 1), bottom = source.take(minimum(y0 + 1, MLXArray(oldHeight - 1)), axis: 1)
        let a = top.take(x0, axis: 2), b = top.take(minimum(x0 + 1, MLXArray(oldWidth - 1)), axis: 2)
        let c = bottom.take(x0, axis: 2), d = bottom.take(minimum(x0 + 1, MLXArray(oldWidth - 1)), axis: 2)
        return ((1 - dy) * ((1 - dx) * a + dx * b) + dy * ((1 - dx) * c + dx * d)).asType(input.dtype)
    }

    /// Exact c,hg,wg channel order from einops b c (hg h) (wg w) -> b (c hg wg) h w.
    static func imagePatches(_ image: MLXArray, height: Int, width: Int) -> MLXArray {
        let batch = image.dim(0), channels = image.dim(3)
        let rows = image.dim(1) / height, columns = image.dim(2) / width
        return image.reshaped(batch, rows, height, columns, width, channels)
            .transposed(0, 2, 4, 5, 1, 3).reshaped(batch, height, width, channels * rows * columns)
    }

    /// Torchvision modulated deform_conv2d, offset groups=1, convolution groups=1.
    /// Offsets have interleaved (dy,dx) channels for each row-major kernel point.
    /// Query chunks bound the bilinear im2col allocation for the 7x7 ASPP branch.
    public static func deformConv(_ input: MLXArray, offsets: MLXArray, mask: MLXArray,
                                  weight: MLXArray, stride: Int = 1, padding: Int = 0,
                                  chunkSize: Int = 2048) throws -> MLXArray {
        guard input.ndim == 4, input.dim(0) == 1, weight.ndim == 4, stride > 0, padding >= 0, chunkSize > 0,
              weight.dim(1) == weight.dim(2), weight.dim(3) == input.dim(3) else {
            throw IdolError.invalidInput("Invalid BiRefNet deformable convolution tensors.")
        }
        let kernel = weight.dim(1), height = input.dim(1), width = input.dim(2), channels = input.dim(3)
        let outHeight = (height + 2 * padding - kernel) / stride + 1
        let outWidth = (width + 2 * padding - kernel) / stride + 1
        let points = kernel * kernel, outputChannels = weight.dim(0)
        guard outHeight > 0, outWidth > 0, offsets.shape == [1, outHeight, outWidth, 2 * points],
              mask.shape == [1, outHeight, outWidth, points] else {
            throw IdolError.invalidInput("BiRefNet offset/mask grids do not match the convolution.")
        }
        let source = input.reshaped(height * width, channels).asType(.float32)
        let allOffsets = offsets.reshaped(outHeight * outWidth, points, 2).asType(.float32)
        let allMasks = mask.reshaped(outHeight * outWidth, points, 1).asType(.float32)
        let kernelY = MLXArray((0..<points).map { Float($0 / kernel - padding) }).reshaped(1, points)
        let kernelX = MLXArray((0..<points).map { Float($0 % kernel - padding) }).reshaped(1, points)
        let projection = weight.reshaped(outputChannels, points * channels).T
        var outputs: [MLXArray] = []
        for start in Swift.stride(from: 0, to: outHeight * outWidth, by: chunkSize) {
            try Task.checkCancellation()
            let end = min(start + chunkSize, outHeight * outWidth), rows = end - start
            let baseY = MLXArray((start..<end).map { Float(($0 / outWidth) * stride) }).reshaped(rows, 1)
            let baseX = MLXArray((start..<end).map { Float(($0 % outWidth) * stride) }).reshaped(rows, 1)
            let y = baseY + kernelY + allOffsets[start..<end, 0..., 0]
            let x = baseX + kernelX + allOffsets[start..<end, 0..., 1]
            let y0 = floor(y), x0 = floor(x), dy = (y - y0).expandedDimensions(axis: -1), dx = (x - x0).expandedDimensions(axis: -1)
            func gather(_ ys: MLXArray, _ xs: MLXArray) -> MLXArray {
                let valid = (ys .>= 0) * (ys .< height) * (xs .>= 0) * (xs .< width)
                let index = clip(ys, min: 0, max: Float(height - 1)).asType(.int32) * width
                    + clip(xs, min: 0, max: Float(width - 1)).asType(.int32)
                return source.take(index, axis: 0) * valid.expandedDimensions(axis: -1).asType(.float32)
            }
            let samples = ((1 - dy) * ((1 - dx) * gather(y0, x0) + dx * gather(y0, x0 + 1))
                + dy * ((1 - dx) * gather(y0 + 1, x0) + dx * gather(y0 + 1, x0 + 1))) * allMasks[start..<end]
            let output = matmul(samples.asType(input.dtype).reshaped(rows, points * channels), projection)
            eval(output)
            outputs.append(output)
        }
        return concatenated(outputs, axis: 0).reshaped(1, outHeight, outWidth, outputChannels)
    }
}
