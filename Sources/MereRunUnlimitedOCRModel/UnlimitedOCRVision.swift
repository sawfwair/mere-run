import Foundation
import MLX
import MLXFast

package struct UnlimitedOCRVision {
    package let weights: UnlimitedOCRWeights

    package init(weights: UnlimitedOCRWeights) { self.weights = weights }

    /// Pixels use NHWC, RGB in [-1, 1]. SAM supplies CLIP's patch embeddings.
    package func encode(_ pixels: MLXArray, trace: ((String, MLXArray) -> Void)? = nil,
                        transform: ((String, MLXArray) -> MLXArray)? = nil) -> MLXArray {
        let sam = sam(pixels, trace: trace, transform: transform)
        let grid = sam.dim(1)
        let channels = sam.dim(-1)
        let clip = clip(sam, trace: trace)
        let features = concatenated([clip[0, 1..., 0...], sam.reshaped(grid * grid, channels)], axis: -1)
        let projected = weights.linear(features, "model.projector.layers")
        trace?("projector", projected)
        return projected
    }

    package func page(global: MLXArray, tiles: [MLXArray], columns: Int, rows: Int) -> MLXArray {
        func withNewlines(_ features: MLXArray, height: Int, width: Int) -> MLXArray {
            let hidden = features.dim(-1)
            let newline = broadcast(weights.tensor("model.image_newline"), to: [height, 1, hidden])
            return concatenated([features.reshaped(height, width, hidden), newline], axis: 1).reshaped(-1, hidden)
        }
        let globalFeatures = withNewlines(encode(global), height: 16, width: 16)
        var pieces: [MLXArray] = []
        if !tiles.isEmpty {
            let tileFeatures = stacked(tiles.map { encode($0) })
            let arranged = tileFeatures.reshaped(rows, columns, 10, 10, -1)
                .transposed(0, 2, 1, 3, 4).reshaped(rows * 10, columns * 10, -1)
            pieces.append(withNewlines(arranged, height: rows * 10, width: columns * 10))
        }
        pieces.append(globalFeatures)
        pieces.append(weights.tensor("model.view_seperator").reshaped(1, -1))
        let output = concatenated(pieces, axis: 0)
        eval(output)
        return output
    }

    private func sam(_ pixels: MLXArray, trace: ((String, MLXArray) -> Void)?,
                     transform: ((String, MLXArray) -> MLXArray)?) -> MLXArray {
        let prefix = "model.sam_model"
        var x = weights.conv(pixels, prefix + ".patch_embed.proj", stride: 16)
        x = transform?("sam.patch", x) ?? x
        trace?("sam.patch", x)
        x = x + UnlimitedOCRMath.resizeGrid(weights.tensor(prefix + ".pos_embed"), height: x.dim(1), width: x.dim(2))
        for layer in 0..<12 {
            x = samBlock(x, layer: layer, trace: trace)
            trace?("sam.block.\(layer)", x)
        }
        x = weights.conv(x, prefix + ".neck.0")
        trace?("sam.neck.0", x)
        x = weights.channelNorm(x, prefix + ".neck.1")
        trace?("sam.neck.1", x)
        x = weights.conv(x, prefix + ".neck.2", padding: 1)
        trace?("sam.neck.2", x)
        x = weights.channelNorm(x, prefix + ".neck.3")
        trace?("sam.neck.3", x)
        x = weights.conv(x, prefix + ".net_2", stride: 2, padding: 1)
        trace?("sam.net2", x)
        x = weights.conv(x, prefix + ".net_3", stride: 2, padding: 1)
        trace?("sam.net3", x)
        return x
    }

    package func samBlock(_ input: MLXArray, layer: Int, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var x = input
        let path = "model.sam_model.blocks.\(layer)"
        let normalized = weights.norm(x, path + ".norm1", eps: 1e-6)
        if layer == 0 { trace?("sam.first.norm1", normalized) }
        let attended: MLXArray
        if [2, 5, 8, 11].contains(layer) {
            attended = samAttention(normalized, path: path + ".attn")
        } else {
            let height = normalized.dim(1)
            let width = normalized.dim(2)
            let paddedH = (height + 13) / 14 * 14
            let paddedW = (width + 13) / 14 * 14
            let padded = padded(normalized, widths: [.init(0), .init((0, paddedH - height)), .init((0, paddedW - width)), .init(0)])
            let windows = padded.reshaped(1, paddedH / 14, 14, paddedW / 14, 14, 768)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, 14, 14, 768)
            let output = samAttention(windows, path: path + ".attn", trace: layer == 0 ? trace : nil)
            attended = output.reshaped(1, paddedH / 14, paddedW / 14, 14, 14, 768)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(1, paddedH, paddedW, 768)[0..., ..<height, ..<width, 0...]
        }
        x = x + attended
        let norm2 = weights.norm(x, path + ".norm2", eps: 1e-6)
        let up = weights.linear(norm2, path + ".mlp.lin1")
        let activated = UnlimitedOCRMath.gelu(up)
        let down = weights.linear(activated, path + ".mlp.lin2")
        if layer == 0 {
            trace?("sam.first.norm2", norm2)
            trace?("sam.first.up", up)
            trace?("sam.first.gelu", activated)
            trace?("sam.first.down", down)
        }
        x = x + down
        eval(x)
        return x
    }

    package func samAttention(_ x: MLXArray, path: String, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        let seq = height * width
        let projected = weights.linear(x, path + ".qkv")
        trace?("sam.first.qkv", projected)
        let qkv = projected.reshaped(batch, seq, 3, 12, 64).transposed(2, 0, 3, 1, 4)
        let q = qkv[0]
        let k = qkv[1]
        let v = qkv[2]
        let relH = Self.relativePositions(weights.tensor(path + ".rel_pos_h"), size: height)
        let relW = Self.relativePositions(weights.tensor(path + ".rel_pos_w"), size: width)
        // Global SAM attention has 4096 rows. Chunking bounds relative-bias memory.
        var pieces: [MLXArray] = []
        for start in stride(from: 0, to: seq, by: 64) {
            let end = min(start + 64, seq)
            let positions = MLXArray(start..<end)
            let rows = positions.floorDivide(width)
            let cols = positions % width
            let queries = q[0..., 0..., start..<end, 0...]
            let rh = einsum("bhqd,qkd->bhqk", queries, relH.take(rows, axis: 0))
            let rw = einsum("bhqd,qkd->bhqk", queries, relW.take(cols, axis: 0))
            let bias = (rh.expandedDimensions(axis: -1) + rw.expandedDimensions(axis: -2))
                .reshaped(batch, 12, end - start, seq)
            let out = MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v, scale: 0.125, mask: bias)
            eval(out)
            pieces.append(out)
        }
        let output = concatenated(pieces, axis: 2).transposed(0, 2, 1, 3).reshaped(batch, height, width, 768)
        let projectedOutput = weights.linear(output, path + ".proj")
        trace?("sam.first.proj", projectedOutput)
        return projectedOutput
    }

    package static func relativePositions(_ table: MLXArray, size: Int) -> MLXArray {
        let length = 2 * size - 1
        var resized = table
        if table.dim(0) != length {
            let position = maximum((MLXArray(0..<length).asType(.float32) + 0.5) * Float(table.dim(0)) / Float(length) - 0.5, 0)
            let lo = floor(position).asType(.int32)
            let hi = minimum(lo + 1, table.dim(0) - 1)
            let fraction = (position - lo.asType(.float32)).expandedDimensions(axis: -1)
            resized = (table.take(lo, axis: 0).asType(.float32) * (1 - fraction)
                + table.take(hi, axis: 0).asType(.float32) * fraction).asType(table.dtype)
        }
        let positions = MLXArray(0..<size)
        return resized.take(positions.expandedDimensions(axis: -1) - positions + size - 1, axis: 0)
    }

    private func clip(_ patches: MLXArray, trace: ((String, MLXArray) -> Void)?) -> MLXArray {
        let prefix = "model.vision_model"
        let seq = patches.dim(1) * patches.dim(2)
        let cls = weights.tensor(prefix + ".embeddings.class_embedding").reshaped(1, 1, 1024)
        let table = weights.tensor(prefix + ".embeddings.position_embedding.weight")
        let grid = UnlimitedOCRMath.resizeGrid(table[1...].reshaped(1, 16, 16, 1024), height: patches.dim(1), width: patches.dim(2))
        let position = concatenated([table[0..<1].reshaped(1, 1, 1024), grid.reshaped(1, seq, 1024)], axis: 1)
        var x = weights.norm(concatenated([cls, patches.reshaped(1, seq, 1024)], axis: 1) + position, prefix + ".pre_layrnorm")
        trace?("clip.pre", x)
        for layer in 0..<24 {
            x = clipBlock(x, layer: layer)
            trace?("clip.block.\(layer)", x)
        }
        return x
    }
    package func clipBlock(_ input: MLXArray, layer: Int) -> MLXArray {
        var x = input
        let seq = x.dim(1) - 1
        let path = "model.vision_model.transformer.layers.\(layer)"
        let qkv = weights.linear(weights.norm(x, path + ".layer_norm1"), path + ".self_attn.qkv_proj")
            .reshaped(1, seq + 1, 3, 16, 64).transposed(2, 0, 3, 1, 4)
        let attention = MLXFast.scaledDotProductAttention(queries: qkv[0], keys: qkv[1], values: qkv[2], scale: 0.125, mask: .none)
            .transposed(0, 2, 1, 3).reshaped(1, seq + 1, 1024)
        x = x + weights.linear(attention, path + ".self_attn.out_proj")
        let up = weights.linear(weights.norm(x, path + ".layer_norm2"), path + ".mlp.fc1")
        x = x + weights.linear(UnlimitedOCRMath.quickGelu(up), path + ".mlp.fc2")
        eval(x)
        return x
    }

}
