import MLX

struct Inkling2VerificationPosition {
    let committed: MLXArray
    let queryScale: MLXArray
}

public final class Inkling2Cache {
    let layers: [Inkling2LayerCache]
    var verificationPosition: Inkling2VerificationPosition?
    private var masks: [Inkling2AttentionGeometry: Inkling2AttentionMask] = [:]
    public var offset: Int { layers.first?.offset ?? 0 }
    var isVerifying: Bool { layers.first?.verification != nil }
    public init(layerCount: Int) { layers = (0..<layerCount).map { _ in Inkling2LayerCache() } }

    func beginForward() { masks.removeAll(keepingCapacity: true) }

    func beginVerification() {
        for layer in layers { layer.verification = Inkling2LayerCache.Verification(offset: layer.offset) }
    }

    func commitVerification(_ count: Int) {
        for layer in layers { layer.commitVerification(count) }
    }

    func attentionMask(for geometry: Inkling2AttentionGeometry) -> Inkling2AttentionMask {
        if let mask = masks[geometry] { return mask }
        let mask = Inkling2AttentionMask(geometry)
        masks[geometry] = mask
        return mask
    }
}

struct Inkling2AttentionGeometry: Hashable {
    let offset: Int
    let queryCount: Int
    let keyStart: Int
    let keyCount: Int
    let extent: Int
    let local: Bool
}

struct Inkling2AttentionMask {
    let relativeIndices: MLXArray
    let withinExtent: MLXArray
    let allowed: MLXArray

    init(_ geometry: Inkling2AttentionGeometry) {
        let offset = geometry.offset, keyStart = geometry.keyStart
        let queries = MLXArray(Int32(offset)..<Int32(offset + geometry.queryCount)).reshaped(-1, 1)
        let keys = MLXArray(Int32(keyStart)..<Int32(keyStart + geometry.keyCount)).reshaped(1, -1)
        let distance = queries - keys
        relativeIndices = clip(distance, min: 0, max: geometry.extent - 1).asType(.int32)
        let causal = distance .>= 0
        withinExtent = causal .&& (distance .< geometry.extent)
        allowed = geometry.local ? withinExtent : causal
    }
}

final class Inkling2LayerCache {
    struct Verification {
        let offset: Int
        var convolutionContexts: [Int: (values: MLXArray, history: Int)] = [:]
        var attention: (keys: MLXArray, values: MLXArray, start: Int, window: Int?)?
        var pendingKeysAndValues: (MLXArray, MLXArray)?
    }

    var offset = 0
    var keys: MLXArray?
    var values: MLXArray?
    var convolutions: [Int: MLXArray] = [:]
    var verification: Verification?
    var arrays: [MLXArray] { Array(convolutions.values) + [keys, values].compactMap { $0 } }

    func context(_ x: MLXArray, index: Int, kernel: Int) -> MLXArray {
        let prior = convolutions[index] ?? MLXArray.zeros([x.dim(0), kernel - 1, x.dim(2)], dtype: .float32)
        let context = concatenated([prior, x.asType(.float32)], axis: 1)
        storeConvolutionContext(context, index: index, history: kernel - 1)
        return context
    }

    func storeConvolutionContext(_ context: MLXArray, index: Int, history: Int) {
        verification?.convolutionContexts[index] = (context, history)
        convolutions[index] = context[0..., (context.dim(1) - history)..., 0...]
    }

    func update(keys k: MLXArray, values v: MLXArray, window: Int?) -> (MLXArray, MLXArray, Int) {
        let previous = keys?.dim(2) ?? 0
        let keyStart = offset - previous
        let allK = keys.map { concatenated([$0, k], axis: 2) } ?? k
        let allV = values.map { concatenated([$0, v], axis: 2) } ?? v
        verification?.attention = (allK, allV, keyStart, window)
        offset += k.dim(2)
        let keep = min(window.map { $0 - 1 } ?? allK.dim(2), allK.dim(2))
        keys = allK[0..., 0..., (allK.dim(2) - keep)..., 0...]
        values = allV[0..., 0..., (allV.dim(2) - keep)..., 0...]
        return (allK, allV, keyStart)
    }

    func commitVerification(_ count: Int) {
        guard let verification else { return }
        self.verification = nil
        let committed = verification.offset + count
        guard committed < offset, let attention = verification.attention else { return }
        offset = committed
        let end = committed - attention.start
        let start = attention.window.map { max(0, end - ($0 - 1)) } ?? 0
        keys = attention.keys[0..., 0..., start..<end, 0...]
        values = attention.values[0..., 0..., start..<end, 0...]
        for (index, context) in verification.convolutionContexts {
            convolutions[index] = context.values[0..., count..<(count + context.history), 0...]
        }
    }
}
