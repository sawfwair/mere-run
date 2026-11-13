import Foundation
import MLX

extension Inkling2Model {
    /// Input is one normalized 40x40 HWC patch per language image token.
    /// Each call encodes one image, so attention never crosses image boundaries.
    public func imageFeatures(pixels: MLXArray, rows: Int, columns: Int) throws -> MLXArray {
        guard let v = config.visionConfig else { throw Inkling2Error.invalid("checkpoint has no vision tower") }
        let merge = v.adapterMergeSize, patch = v.nativePatchSize, d = v.hiddenSize
        guard rows > 0, columns > 0, pixels.shape == [rows * columns, merge * patch, merge * patch, 3] else {
            throw Inkling2Error.invalid("image patch shape mismatch")
        }
        let p = "model.visual.backbone.", e = p + "embeddings."
        let patchWeight = try weights.read(e + "patch_embedding.weight")
        let patches = pixels.reshaped(rows * columns, merge, patch, merge, patch, 3)
            .transposed(0, 1, 3, 2, 4, 5).reshaped(-1, patch * patch * 3).asType(patchWeight.dtype)
        var embedded = matmul(patches, patchWeight.T)
        let gridH = rows * merge, gridW = columns * merge
        var coords: [(Int, Int)] = []
        for row in 0..<rows {
            for col in 0..<columns {
                for y in 0..<merge { for x in 0..<merge { coords.append((row * merge + y, col * merge + x)) } }
            }
        }
        var indices: [Int32] = [], interpolation: [Float] = []
        for (y, x) in coords {
            let fy = max(0, (Float(y) + 0.5) * Float(v.posembGridSize) / Float(gridH) - 0.5)
            let fx = max(0, (Float(x) + 0.5) * Float(v.posembGridSize) / Float(gridW) - 0.5)
            let y0 = min(Int(fy), v.posembGridSize - 1), x0 = min(Int(fx), v.posembGridSize - 1)
            let y1 = min(y0 + 1, v.posembGridSize - 1), x1 = min(x0 + 1, v.posembGridSize - 1)
            let dy = fy - Float(y0), dx = fx - Float(x0)
            indices += [Int32(y0 * v.posembGridSize + x0), Int32(y0 * v.posembGridSize + x1),
                        Int32(y1 * v.posembGridSize + x0), Int32(y1 * v.posembGridSize + x1)]
            interpolation += [(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx]
        }
        let positions = try weights.read(e + "position_embedding")[MLXArray(indices).reshaped(-1, 4)]
        let position = (positions * MLXArray(interpolation).reshaped(-1, 4, 1)).sum(axis: 1)
        embedded = embedded + position.asType(embedded.dtype)
        let cls = try (weights.read(e + "class_embedding") + weights.read(e + "cls_position_embedding"))
            .reshaped(1, d).asType(embedded.dtype)
        var hidden = try visionNorm(concatenated([cls, embedded], axis: 0), p + "ln_pre", epsilon: v.layerNormEps)
        let headDim = d / v.numAttentionHeads
        let rope = Self.visionRoPE(coordinates: [(0, 0)] + coords.map { ($0.0 + 1, $0.1 + 1) }, headDim: headDim)
        for i in 0..<v.numHiddenLayers {
            try Task.checkCancellation()
            let l = p + "layers.\(i)."
            let normalized = try visionNorm(hidden, l + "ln_1", epsilon: v.layerNormEps)
            let q = try linear(normalized, l + "self_attn.q_proj", bias: true).reshaped(-1, v.numAttentionHeads, headDim)
            let k = try linear(normalized, l + "self_attn.k_proj", bias: true).reshaped(-1, v.numAttentionHeads, headDim)
            let value = try linear(normalized, l + "self_attn.v_proj", bias: true).reshaped(-1, v.numAttentionHeads, headDim)
            let rotatedQ = Self.rotateVision(q, cosine: rope.0, sine: rope.1).transposed(1, 0, 2)
            let rotatedK = Self.rotateVision(k, cosine: rope.0, sine: rope.1).transposed(1, 0, 2)
            let scores = matmul(rotatedQ.asType(.float32), rotatedK.asType(.float32).swappedAxes(-1, -2)) / sqrt(Float(headDim))
            let attended = matmul(softmax(scores, axis: -1).asType(value.dtype), value.transposed(1, 0, 2))
                .transposed(1, 0, 2).reshaped(-1, d)
            hidden = try hidden + weights.read(l + "ls_1.gamma") * linear(attended, l + "self_attn.out_proj", bias: true)
            let mlpInput = try visionNorm(hidden, l + "ln_2", epsilon: v.layerNormEps)
            let up = try linear(mlpInput, l + "mlp.c_fc", bias: true)
            hidden = try hidden + weights.read(l + "ls_2.gamma") * linear(Self.gelu(up), l + "mlp.c_proj", bias: true)
            eval(hidden)
        }
        let merged = hidden[1..., 0...].reshaped(rows * columns, merge * merge * d)
        return try linear(Self.gelu(linear(merged, "model.visual.proj_in")), "model.visual.proj_out")
    }

    private func visionNorm(_ x: MLXArray, _ key: String, epsilon: Float) throws -> MLXArray {
        let f = x.asType(.float32)
        let centered = f - f.mean(axis: -1, keepDims: true)
        return try (centered * rsqrt(square(centered).mean(axis: -1, keepDims: true) + epsilon)
                    * weights.read(key + ".weight").asType(.float32)
                    + weights.read(key + ".bias").asType(.float32)).asType(x.dtype)
    }

    static func gelu(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * 0.5 * (1 + erf(f / sqrt(Float(2))))).asType(x.dtype)
    }

    static func visionRoPE(coordinates: [(Int, Int)], headDim: Int) -> (MLXArray, MLXArray) {
        var angles: [Float] = []
        for (y, x) in coordinates {
            for coordinate in [x, y] {
                for pair in 0..<(headDim / 4) {
                    let angle = Float(coordinate) / pow(10_000, Float(2 * pair) / Float(headDim / 2))
                    angles += [angle, angle]
                }
            }
        }
        let a = MLXArray(angles).reshaped(coordinates.count, 1, headDim)
        return (cos(a), sin(a))
    }

    static func rotateVision(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), x.dim(1), -1, 2)
        let rotated = stacked([-pairs[0..., 0..., 0..., 1], pairs[0..., 0..., 0..., 0]], axis: -1).reshaped(x.shape)
        return (x.asType(.float32) * cosine + rotated * sine).asType(x.dtype)
    }
}

public enum Inkling2ImagePatches {
    /// No resize: pad the bottom/right with -1 in the original 0...255 scale.
    /// Composite transparent pixels over the reference 8-pixel checkerboard.
    public static func make(rgba: [UInt8], width: Int, height: Int, tokenSize: Int = 40) throws -> (pixels: MLXArray, rows: Int, columns: Int) {
        guard width > 0, height > 0, tokenSize > 0, rgba.count == width * height * 4 else {
            throw Inkling2Error.invalid("invalid image dimensions or RGBA data")
        }
        let rows = (height + tokenSize - 1) / tokenSize, columns = (width + tokenSize - 1) / tokenSize
        var pixels: [Float] = []
        pixels.reserveCapacity(rows * columns * tokenSize * tokenSize * 3)
        for row in 0..<rows { for col in 0..<columns {
            for y in 0..<tokenSize { for x in 0..<tokenSize {
                let yy = row * tokenSize + y, xx = col * tokenSize + x
                for channel in 0..<3 {
                    var value = -1
                    if yy < height, xx < width {
                        let i = (yy * width + xx) * 4, alpha = Int(rgba[i + 3])
                        let background = ((yy / 8 + xx / 8) % 2 == 0) ? 204 : 153
                        value = (Int(rgba[i + channel]) * alpha + background * (255 - alpha) + 127) / 255
                    }
                    pixels.append(Float(value) / 127.5 - 1)
                }
            } }
        } }
        return (MLXArray(pixels).reshaped(rows * columns, tokenSize, tokenSize, 3), rows, columns)
    }
}
