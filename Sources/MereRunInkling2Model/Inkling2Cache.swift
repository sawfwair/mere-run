import MLX

public final class Inkling2Cache {
    let layers: [Inkling2LayerCache]
    public var offset: Int { layers.first?.offset ?? 0 }
    public init(layerCount: Int) { layers = (0..<layerCount).map { _ in Inkling2LayerCache() } }
}

final class Inkling2LayerCache {
    var offset = 0
    var keys: MLXArray?
    var values: MLXArray?
    var convolutions: [Int: MLXArray] = [:]

    func context(_ x: MLXArray, index: Int, kernel: Int) -> MLXArray {
        let prior = convolutions[index] ?? MLXArray.zeros([x.dim(0), kernel - 1, x.dim(2)], dtype: .float32)
        let context = concatenated([prior, x.asType(.float32)], axis: 1)
        convolutions[index] = context[0..., (context.dim(1) - kernel + 1)..., 0...]
        return context
    }

    func update(keys k: MLXArray, values v: MLXArray, window: Int?) -> (MLXArray, MLXArray, Int) {
        let previous = keys?.dim(2) ?? 0
        let keyStart = offset - previous
        let allK = keys.map { concatenated([$0, k], axis: 2) } ?? k
        let allV = values.map { concatenated([$0, v], axis: 2) } ?? v
        offset += k.dim(2)
        let keep = min(window.map { $0 - 1 } ?? allK.dim(2), allK.dim(2))
        keys = allK[0..., 0..., (allK.dim(2) - keep)..., 0...]
        values = allV[0..., 0..., (allV.dim(2) - keep)..., 0...]
        return (allK, allV, keyStart)
    }
}
