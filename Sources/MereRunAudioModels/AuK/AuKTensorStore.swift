// Native adaptation of Tencent-Hunyuan/AuK (MIT); see THIRD_PARTY_NOTICES.md.
import Foundation
import MLX
import MLXFast

public enum AuKError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

/// Checkpoint tensors in MLX layout. Missing parameters are errors, never random initialization.
public struct AuKTensorStore {
    public let tensors: [String: MLXArray]
    public init(_ tensors: [String: MLXArray]) { self.tensors = tensors }

    public func tensor(_ name: String) throws -> MLXArray {
        guard let value = tensors[name] else { throw AuKError.invalid("Missing AuK tensor: \(name)") }
        return value
    }

    func linear(_ x: MLXArray, _ key: String) throws -> MLXArray {
        let weight = try tensor(key + ".weight")
        guard weight.ndim == 2, weight.dim(1) == x.dim(-1) else {
            throw AuKError.invalid("Invalid AuK linear shape: \(key)")
        }
        let y = matmul(x, weight.T)
        return tensors[key + ".bias"].map { y + $0 } ?? y
    }

    func rms(_ x: MLXArray, _ key: String, eps: Float = Float.ulpOfOne) throws -> MLXArray {
        MLXFast.rmsNorm(x, weight: try tensor(key + ".weight"), eps: eps)
    }

    func norm(_ x: MLXArray, _ key: String) throws -> MLXArray {
        MLXFast.layerNorm(x, weight: try tensor(key + ".weight"), bias: try tensor(key + ".bias"), eps: 1e-5)
    }

    func conv(_ x: MLXArray, _ key: String, stride: Int = 1, dilation: Int = 1,
              groups: Int = 1, causal: Bool = false, padding: Int? = nil) throws -> MLXArray {
        let w = try tensor(key + ".weight")
        let width = dilation * (w.dim(1) - 1)
        let input = causal ? padded(x, widths: [.init(0), .init((width, 0)), .init(0)]) : x
        let y = MLX.conv1d(input, w, stride: stride, padding: causal ? 0 : (padding ?? width / 2),
                           dilation: dilation, groups: groups)
        return tensors[key + ".bias"].map { y + $0 } ?? y
    }
}

func aukNorm(_ x: MLXArray) -> MLXArray {
    MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
}
func aukSiLU(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
func aukHeads(_ x: MLXArray, _ heads: Int) -> MLXArray {
    x.reshaped(x.dim(0), x.dim(1), heads, -1).transposed(0, 2, 1, 3)
}
func aukUnheads(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(2), -1)
}
func aukAttention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
        scale: 1 / sqrt(Float(q.dim(-1))), mask: mask)
}

/// Builds angles in Double before casting, preserving the checkpoint's rounded frequencies.
public struct AuKRotary {
    let frequencies: [Float]
    public init(frequencies: [Float]) { self.frequencies = frequencies }
    public func apply(_ x: MLXArray, interleaved: Bool = true) -> MLXArray {
        let n = x.dim(-2), d = x.dim(-1), half = d / 2
        var cosines = [Float](), sines = [Float]()
        for position in 0..<n {
            let angles = frequencies.map { Double(position) * Double($0) }
            for column in 0..<d {
                let index = interleaved ? column / 2 : column % half
                cosines.append(Float(cos(angles[index])))
                sines.append(Float(sin(angles[index])))
            }
        }
        let rotated: MLXArray
        if interleaved {
            let pairs = x.reshaped(x.dim(0), x.dim(1), n, half, 2)
            rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(x.shape)
        } else {
            rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        }
        return x * MLXArray(cosines, [n, d]) + rotated * MLXArray(sines, [n, d])
    }
}
