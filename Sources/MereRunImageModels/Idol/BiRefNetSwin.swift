import Foundation
import MLX
import MLXNN

struct BiRefNetSwin {
    let weights: BiRefNetWeights
    let configuration: BiRefNetConfiguration

    static func partition(_ x: MLXArray, window: Int) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1) / window, window, x.dim(2) / window, window, x.dim(3))
            .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, window * window, x.dim(3))
    }
    static func reverse(_ x: MLXArray, height: Int, width: Int, window: Int) -> MLXArray {
        x.reshaped(-1, height / window, width / window, window, window, x.dim(-1))
            .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, height, width, x.dim(-1))
    }
    static func shiftMask(height: Int, width: Int, window: Int, dtype: DType) -> MLXArray {
        let shift = window / 2
        let values = (0..<(height * width)).map { index -> Float in
            let y = index / width, x = index % width
            let row = y < height - window ? 0 : (y < height - shift ? 1 : 2)
            let column = x < width - window ? 0 : (x < width - shift ? 1 : 2)
            return Float(row * 3 + column)
        }
        let labels = partition(MLXArray(values).reshaped(1, height, width, 1), window: window).squeezed(axis: -1)
        return which(labels.expandedDimensions(axis: 1) .!= labels.expandedDimensions(axis: 2),
                     MLXArray(Float(-100)), MLXArray(Float(0))).asType(dtype)
    }

    func windowAttention(_ x: MLXArray, prefix: String, heads: Int, mask: MLXArray?) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1), channels = x.dim(2), dimension = channels / heads
        let qkv = weights.linear(x, prefix + ".qkv").reshaped(batch, length, 3, heads, dimension)
        let q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3) * (1 / sqrt(Float(dimension)))
        let k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 3, 1)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        let indices = weights[prefix + ".relative_position_index"].reshaped(-1)
        let bias = weights[prefix + ".relative_position_bias_table"].take(indices, axis: 0)
            .reshaped(length, length, heads).transposed(2, 0, 1).expandedDimensions(axis: 0)
        var scores = matmul(q, k) + bias
        if let mask {
            let windows = mask.dim(0)
            scores = (scores.reshaped(batch / windows, windows, heads, length, length)
                + mask.reshaped(1, windows, 1, length, length)).reshaped(batch, heads, length, length)
        }
        let output = matmul(softmax(scores, axis: -1), v).transposed(0, 2, 1, 3).reshaped(batch, length, channels)
        return weights.linear(output, prefix + ".proj")
    }

    func block(_ input: MLXArray, prefix: String, heads: Int, shifted: Bool, mask: MLXArray) -> MLXArray {
        let height = input.dim(1), width = input.dim(2), window = configuration.windowSize
        let paddedHeight = ((height + window - 1) / window) * window
        let paddedWidth = ((width + window - 1) / window) * window
        var normalized = weights.norm(input, prefix + ".norm1")
        normalized = padded(normalized, widths: [.init(0), .init((0, paddedHeight - height)),
                                                 .init((0, paddedWidth - width)), .init(0)])
        let shift = shifted ? window / 2 : 0
        if shift > 0 { normalized = roll(roll(normalized, shift: -shift, axis: 1), shift: -shift, axis: 2) }
        let windows = Self.partition(normalized, window: window)
        let attended = windowAttention(windows, prefix: prefix + ".attn", heads: heads, mask: shifted ? mask : nil)
        var merged = Self.reverse(attended, height: paddedHeight, width: paddedWidth, window: window)
        if shift > 0 { merged = roll(roll(merged, shift: shift, axis: 1), shift: shift, axis: 2) }
        let residual = input + merged[0..., 0..<height, 0..<width, 0...]
        return residual + weights.linear(gelu(weights.linear(weights.norm(residual, prefix + ".norm2"),
                                                              prefix + ".mlp.fc1")), prefix + ".mlp.fc2")
    }

    func callAsFunction(_ input: MLXArray) throws -> [MLXArray] {
        var hidden = weights.norm(weights.conv(input, "bb.patch_embed.proj", stride: 4), "bb.patch_embed.norm")
        var features: [MLXArray] = []
        for stage in 0..<4 {
            let prefix = "bb.layers.\(stage)", window = configuration.windowSize
            let height = hidden.dim(1), width = hidden.dim(2)
            let mask = Self.shiftMask(height: ((height + window - 1) / window) * window,
                width: ((width + window - 1) / window) * window, window: window, dtype: hidden.dtype)
            for depth in 0..<configuration.depths[stage] {
                try Task.checkCancellation()
                hidden = block(hidden, prefix: prefix + ".blocks.\(depth)", heads: configuration.heads[stage],
                               shifted: depth % 2 == 1, mask: mask)
                eval(hidden)
            }
            let feature = weights.norm(hidden, "bb.norm\(stage)")
            eval(feature)
            features.append(feature)
            if stage < 3 {
                let evenY = MLXArray(Array(stride(from: 0, to: height, by: 2)))
                let evenX = MLXArray(Array(stride(from: 0, to: width, by: 2)))
                let a = hidden.take(evenY, axis: 1).take(evenX, axis: 2)
                let b = hidden.take(evenY + 1, axis: 1).take(evenX, axis: 2)
                let c = hidden.take(evenY, axis: 1).take(evenX + 1, axis: 2)
                let d = hidden.take(evenY + 1, axis: 1).take(evenX + 1, axis: 2)
                hidden = weights.linear(weights.norm(concatenated([a, b, c, d], axis: -1), prefix + ".downsample.norm"),
                                        prefix + ".downsample.reduction")
            }
        }
        return features
    }
}
