import Foundation
import MLX

/// IDOL's Swin-L BiRefNet. Smaller dimensions support independent numerical fixtures.
public struct BiRefNetConfiguration: Sendable {
    public let embedDim: Int
    public let depths: [Int]
    public let heads: [Int]
    public let windowSize: Int
    public let decoderIntermediate: Int
    public let asppChannels: Int
    public init(embedDim: Int = 192, depths: [Int] = [2, 2, 18, 2], heads: [Int] = [6, 12, 24, 48],
                windowSize: Int = 12, decoderIntermediate: Int = 64, asppChannels: Int = 256) {
        self.embedDim = embedDim; self.depths = depths; self.heads = heads; self.windowSize = windowSize
        self.decoderIntermediate = decoderIntermediate; self.asppChannels = asppChannels
    }
    var channels: [Int] { [16, 8, 4, 2].map { $0 * embedDim } }
    func validate() throws {
        guard embedDim >= 8, embedDim <= 192, depths.count == 4, heads.count == 4,
              depths.allSatisfy({ (1...18).contains($0) }), (2...12).contains(windowSize),
              (1...64).contains(decoderIntermediate), (1...256).contains(asppChannels) else {
            throw IdolError.invalidConfiguration("Invalid BiRefNet architecture.")
        }
        for index in 0..<4 {
            guard heads[index] > 0, (embedDim << index) % heads[index] == 0 else {
                throw IdolError.invalidConfiguration("BiRefNet attention width must divide evenly into heads.")
            }
        }
    }
}

struct BiRefNetWeights {
    let arrays: [String: MLXArray]
    let dtype: DType
    init(_ arrays: [String: MLXArray], configuration: BiRefNetConfiguration, dtype: DType) throws {
        let expected = Self.shapes(configuration)
        guard Set(arrays.keys) == Set(expected.keys) else {
            throw IdolError.invalidWeights("BiRefNet tensor inventory mismatch: missing \(Set(expected.keys).subtracting(arrays.keys).sorted()), unexpected \(Set(arrays.keys).subtracting(expected.keys).sorted()).")
        }
        for (key, shape) in expected {
            guard let array = arrays[key], array.shape == shape,
                  (key.hasSuffix("relative_position_index") || key.hasSuffix("num_batches_tracked"))
                    ? (array.dtype == .int64 || array.dtype == .int32)
                    : (array.dtype == .float32 || array.dtype == .float16) else {
                throw IdolError.invalidWeights("Invalid BiRefNet tensor \(key); expected \(shape).")
            }
        }
        self.dtype = dtype
        self.arrays = Dictionary(uniqueKeysWithValues: arrays.map { key, value in
            if key.hasSuffix("relative_position_index") || key.hasSuffix("num_batches_tracked") { return (key, value) }
            return (key, (value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value).asType(dtype))
        })
    }
    subscript(_ name: String) -> MLXArray { arrays[name]! }
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let result = matmul(x, self[name + ".weight"].T)
        return arrays[name + ".bias"].map { result + $0 } ?? result
    }
    func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 0) -> MLXArray {
        let result = conv2d(x, self[name + ".weight"], stride: IntOrPair(stride), padding: IntOrPair(padding))
        return arrays[name + ".bias"].map { result + $0 } ?? result
    }
    func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: self[name + ".weight"].asType(.float32),
            bias: self[name + ".bias"].asType(.float32), eps: 1e-5).asType(x.dtype)
    }
    func batchNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        let mean = self[name + ".running_mean"].asType(.float32)
        let variance = self[name + ".running_var"].asType(.float32)
        return ((x.asType(.float32) - mean) / sqrt(variance + 1e-5) * self[name + ".weight"].asType(.float32)
                + self[name + ".bias"].asType(.float32)).asType(x.dtype)
    }
}
