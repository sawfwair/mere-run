import Foundation
import MLX

#if os(macOS)
/// Request-local verification graphs with a fixed-shape global attention cache.
final class Inkling2CompiledVerifier {
    static let capacityLimit = 16_384
    static let supportedWidths = 4...8
    private static let convolutionCount = 4
    private static let arraysPerLayer = 2 + convolutionCount

    struct Panel {
        fileprivate let arrays: [MLXArray]
        var logits: MLXArray { arrays[0] }
        var features: MLXArray { arrays[1] }
    }

    private struct Layer {
        let local: Bool
        var keys: MLXArray
        var values: MLXArray
        var convolutions: [MLXArray]
    }

    private let model: Inkling2Model
    private let cache: Inkling2Cache
    private let featureIDs: [Int]
    private var layers: [Layer]
    private var programs: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    private var traceError: Error?
    private(set) var offset: Int

    init?(draft: Inkling2DSpark, cache: Inkling2Cache, budget: Int) {
        let model = draft.target, config = model.config.textConfig
        let offset = cache.offset
        let head = model.prefix + "unembed.weight"
        let supportedHead = model.q8OutputHead != nil || model.weights.dtype(head) == "BF16" ||
            model.weights.referenceAffine[head] != nil
        // Fixed local windows and the qualified BF16 split attention geometry.
        guard model.usesSpeedPack, model.usePackedExperts, supportedHead, let loop = config.loopConfig,
              config.numAttentionHeads == 24, config.numKeyValueHeads == 8, config.headDim == 128,
              config.swaNumAttentionHeads == 24, config.swaNumKeyValueHeads == 8, config.swaHeadDim == 128,
              config.slidingWindowSize == 1024, offset >= config.slidingWindowSize,
              budget >= 128, budget <= Self.capacityLimit - offset else { return nil }
        let projections = ["wq_du", "wk_dv", "wv_dv", "wr_du"]
        let bf16Projections = (0..<config.numHiddenLayers).allSatisfy { layer in
            let prefix = model.prefix + "layers.\(layer).attn."
            return projections.allSatisfy {
                let key = prefix + $0 + ".weight"
                return model.weights.dtype(key) == "BF16" || model.weights.referenceAffine[key] != nil
            }
        }
        let embedding = model.prefix + "embed.weight"
        guard model.weights.dtype(embedding) == "BF16" || model.weights.referenceAffine[embedding] != nil,
              bf16Projections else { return nil }
        let capacity = ((offset + budget + 511) / 512) * 512
        var physicalLayers = Array(0..<config.numHiddenLayers)
        for _ in 1..<loop.numPasses { physicalLayers += loop.layers }
        var layers: [Layer] = []
        for (physical, state) in zip(physicalLayers, cache.layers) {
            let local = config.isLocal(physical)
            let expectedRows = local ? config.slidingWindowSize - 1 : offset
            let history = (0..<Self.convolutionCount).compactMap { state.convolutions[$0] }
            guard let keys = state.keys, let values = state.values, keys.dtype == .bfloat16,
                  values.dtype == keys.dtype, keys.shape == [1, 8, expectedRows, 128],
                  values.shape == keys.shape, history.count == Self.convolutionCount else { return nil }
            if local {
                layers.append(Layer(local: true, keys: keys, values: values, convolutions: history))
            } else {
                let padding = MLXArray.zeros([1, 8, capacity - offset, 128], dtype: keys.dtype)
                layers.append(Layer(local: false, keys: concatenated([keys, padding], axis: 2),
                                    values: concatenated([values, padding], axis: 2), convolutions: history))
            }
        }
        self.model = model
        self.cache = cache
        self.offset = offset
        self.layers = layers
        featureIDs = draft.config.targetLayerIds
    }

    func verify(_ tokens: MLXArray) throws -> Panel? {
        let width = tokens.dim(1)
        guard Self.supportedWidths.contains(width) else { return nil }
        let scale = model.attentionQueryScale(offset: offset, count: width)
        var inputs = [tokens, MLXArray(Int32(offset)), scale]
        inputs.reserveCapacity(3 + layers.count * Self.arraysPerLayer)
        for layer in layers { inputs += [layer.keys, layer.values] + layer.convolutions }
        let evaluate = program(for: width)
        let result = try? MLX.withError { evaluate(inputs) }
        if let traceError { throw traceError }
        guard let result, result.count == 2 + layers.count * Self.arraysPerLayer else { return nil }
        return Panel(arrays: result)
    }

    private func program(for width: Int) -> ([MLXArray]) -> [MLXArray] {
        if let cached = programs[width] { return cached }
        let compiled = MLX.compile { [weak self] inputs in
            guard let self else { return [] }
            do {
                return try self.trace(inputs)
            } catch {
                self.traceError = error
                return []
            }
        }
        programs[width] = compiled
        return compiled
    }

    private func trace(_ inputs: [MLXArray]) throws -> [MLXArray] {
        let state = model.makeCache()
        state.verificationPosition = .init(committed: inputs[1], queryScale: inputs[2])
        for (index, layer) in state.layers.enumerated() {
            let base = 3 + index * Self.arraysPerLayer
            layer.keys = inputs[base]
            layer.values = inputs[base + 1]
            for tap in 0..<Self.convolutionCount { layer.convolutions[tap] = inputs[base + 2 + tap] }
        }
        state.beginVerification()
        let capture = Inkling2DSparkFeatures(featureIDs)
        let embeddings = try model.embeddings(inputs[0])
        let logits = try model.forward(embeddings: embeddings, cache: state, lastPositionOnly: false, capture: capture)
        var outputs = [logits, try capture.joined()]
        for (layout, layer) in zip(layers, state.layers) {
            guard let verification = layer.verification else { return [] }
            if layout.local, let attention = verification.attention {
                outputs += [attention.keys, attention.values]
            } else if let (keys, values) = verification.pendingKeysAndValues {
                outputs += [keys, values]
            } else {
                return []
            }
            outputs += (0..<Self.convolutionCount).compactMap { verification.convolutionContexts[$0]?.values }
        }
        return outputs
    }

    func commit(_ panel: Panel, count: Int) {
        let config = model.config.textConfig
        let history = config.sconvKernelSize - 1
        let position = offset..<(offset + count)
        for index in layers.indices {
            let base = 2 + index * Self.arraysPerLayer
            let keys = panel.arrays[base], values = panel.arrays[base + 1]
            if layers[index].local {
                let window = count..<(count + config.slidingWindowSize - 1)
                layers[index].keys = keys[0..., 0..., window, 0...]
                layers[index].values = values[0..., 0..., window, 0...]
            } else {
                let bankK = layers[index].keys.reshaped(layers[index].keys.shape)
                let bankV = layers[index].values.reshaped(layers[index].values.shape)
                bankK[0..., 0..., position, 0...] = keys[0..., 0..., ..<count, 0...]
                bankV[0..., 0..., position, 0...] = values[0..., 0..., ..<count, 0...]
                layers[index].keys = bankK
                layers[index].values = bankV
            }
            for tap in 0..<Self.convolutionCount {
                layers[index].convolutions[tap] = panel.arrays[base + 2 + tap][0..., count..<(count + history), 0...]
            }
        }
        offset += count
    }

    func resumeAfterDirectCommit() -> Bool {
        let committed = cache.offset
        guard committed > offset else { return false }
        let suffix = offset..<committed
        var updated = layers
        for index in updated.indices {
            let direct = cache.layers[index]
            let history = (0..<Self.convolutionCount).compactMap { direct.convolutions[$0] }
            guard direct.offset == committed, let keys = direct.keys, let values = direct.values,
                  keys.dtype == updated[index].keys.dtype, keys.shape == values.shape,
                  history.count == Self.convolutionCount else { return false }
            if updated[index].local {
                updated[index].keys = keys
                updated[index].values = values
            } else {
                guard keys.dim(2) == committed, committed <= updated[index].keys.dim(2) else { return false }
                let bankK = updated[index].keys.reshaped(updated[index].keys.shape)
                let bankV = updated[index].values.reshaped(updated[index].values.shape)
                bankK[0..., 0..., suffix, 0...] = keys[0..., 0..., suffix, 0...]
                bankV[0..., 0..., suffix, 0...] = values[0..., 0..., suffix, 0...]
                updated[index].keys = bankK
                updated[index].values = bankV
            }
            updated[index].convolutions = history
        }
        layers = updated
        offset = committed
        return true
    }

    func restore() {
        for (source, destination) in zip(layers, cache.layers) {
            destination.offset = offset
            destination.keys = source.local ? source.keys : source.keys[0..., 0..., ..<offset, 0...]
            destination.values = source.local ? source.values : source.values[0..., 0..., ..<offset, 0...]
            for (tap, history) in source.convolutions.enumerated() { destination.convolutions[tap] = history }
        }
    }
}
#endif
