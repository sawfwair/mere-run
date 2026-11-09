import Foundation
import MLX
import MLXNN
import MereRunTensor

#if os(macOS)
/// Keeps lossless packed views outside the parameter tree and retains ordinary prefill.
final class MuseGlimmerTiledQ8Linear: QuantizedLinear {
    private(set) var tiled: M5TiledAffineQ8.Weights?
    private let splits: Int

    init(_ source: QuantizedLinear, tiled: M5TiledAffineQ8.Weights, splits: Int) {
        self.tiled = tiled
        self.splits = splits
        super.init(weight: source.weight, bias: source.bias, scales: source.scales, biases: source.biases,
                   groupSize: source.groupSize, bits: source.bits, mode: source.mode, globalScale: source.globalScale)
    }

    var hasCurrentTiles: Bool {
        guard let biases, let tiled else { return false }
        return tiled.matches(weight: weight, scales: scales, biases: biases)
    }

    @discardableResult
    override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate,
                         path: [String] = [], modulePath: [String] = []) throws -> Self {
        // Module.update changes the backing array while retaining its Swift identity.
        // Drop packed views before an update, including an update that throws partway.
        if parameters.flattened().contains(where: { ["weight", "scales", "biases"].contains($0.0) }) {
            tiled = nil
        }
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if hasCurrentTiles, let tiled, let output = M5TiledAffineQ8.matmul(x, weights: tiled, splits: splits) {
            return bias.map { output + $0 } ?? output
        }
        let output: MLXArray
        if x.shape.dropLast().reduce(1, *) >= 128 {
            let dense = dequantized(weight, scales: scales, biases: biases, groupSize: groupSize,
                bits: bits, mode: mode, dtype: x.dtype)
            output = matmul(x, dense.T)
        } else if weight.dim(0) >= 65_536, let biases,
                  let shared = SmallBatchAffineQ8.matmul(x, weight: weight, scales: scales, biases: biases,
                      groupSize: groupSize, bits: bits, mode: mode, rowsPerSIMD: 4) {
            output = shared
        } else {
            return super.callAsFunction(x)
        }
        return bias.map { output + $0 } ?? output
    }
}
#endif

extension MuseGlimmerModel {
    /// Bound first-use compilation and weight materialization before submitting
    /// the complete decoder graph. These disposable caches never enter requests.
    func materializeTiledWarmup(checkCancellation: (() throws -> Void)?) throws {
        #if os(macOS)
        guard model.languageModel.layers.contains(where: { $0.mlp.gateProj is MuseGlimmerTiledQ8Linear }) else { return }
        for rows in [1, 4] {
            let cache = makeCache()
            var hidden = inputEmbeddings(for: MLXArray.zeros([1, rows], dtype: .int32))
            MLX.eval(hidden)
            for (index, layer) in model.languageModel.layers.enumerated() {
                try checkCancellation?()
                hidden = layer(hidden, cache: cache[index])
                MLX.eval(hidden)
                evaluateGemma4CacheStorage(cache)
            }
            MLX.eval(outputLogits(from: model.languageModel.norm(hidden)))
        }
        #endif
    }

    func installTiledQ8(mode: String, environment: [String: String]) throws {
        #if os(macOS)
        guard M5AffineQ8Head.isSupported else {
            throw MuseGlimmerError.unsupportedConfiguration("Tiled Q8 requires an M5 GPU and macOS 26.2 or later.")
        }
        let headSplits = try Self.tiledQ8HeadSplits(environment: environment)
        let modules = namedModules()
        var parents = Dictionary(uniqueKeysWithValues: modules)
        parents[""] = self
        let selected = modules.compactMap { path, module -> (String, QuantizedLinear)? in
            let decoder = path.hasPrefix("model.language_model.layers.")
            let wanted = mode == "tiled-ffn" ? decoder && path.contains(".mlp.")
                : path == "lm_head" || (mode != "tiled-head" && decoder)
            guard wanted,
                  let linear = module as? QuantizedLinear,
                  linear.weight.dim(0).isMultiple(of: 64) else { return nil }
            return (path, linear)
        }
        guard !selected.isEmpty, mode == "tiled-ffn" || selected.contains(where: { $0.0 == "lm_head" }) else {
            throw MuseGlimmerError.unsupportedConfiguration("Tiled Q8 requires eligible quantized projections for the selected scope.")
        }
        // Validate every selected projection before allocating its reordered buffers.
        for (path, linear) in selected {
            guard linear.bits == 8, linear.mode == .affine, linear.groupSize == 64,
                  linear.scales.dtype == .bfloat16, linear.biases?.dtype == .bfloat16,
                  linear.weight.dim(1).isMultiple(of: 16) else {
                throw MuseGlimmerError.unsupportedConfiguration("Tiled Q8 requires affine Q8/group-64 weights with BF16 metadata.")
            }
            if path == "lm_head", !(linear.weight.dim(1) * 4).isMultiple(of: 64 * headSplits) {
                throw MuseGlimmerError.unsupportedConfiguration("Tiled Q8 head dimensions do not support the requested split count.")
            }
        }
        for (path, linear) in selected {
            try Task.checkCancellation()
            let tiles = M5TiledAffineQ8.Weights(weight: linear.weight, scales: linear.scales, biases: linear.biases!)!
            let splits = path == "lm_head" ? headSplits : M5TiledAffineQ8.partitionCount(
                inputSize: linear.weight.dim(1) * 4, outputSize: linear.weight.dim(0))
            let parts = path.split(separator: ".").map(String.init)
            parents[parts.dropLast().joined(separator: ".")]!.update(modules: ModuleChildren.unflattened([
                (parts.last!, MuseGlimmerTiledQ8Linear(linear, tiled: tiles, splits: splits))
            ]))
        }
        if mode == "tiled-all" || mode == "tiled-ffn" {
            for (path, module) in modules where path.hasPrefix("model.language_model.layers.") {
                (module as? MuseGlimmerTextMLP)?.useTiledQ8 = true
            }
        }
        #else
        throw MuseGlimmerError.unsupportedConfiguration("Tiled Q8 requires an M5 GPU on macOS.")
        #endif
    }

    static func tiledQ8HeadSplits(environment: [String: String]) throws -> Int {
        let raw = environment["MERERUN_MUSE_GLIMMER_Q8_TILED_HEAD_SPLITS"] ?? "1"
        guard let splits = Int(raw), [1, 2, 4, 8].contains(splits) else {
            throw MuseGlimmerError.unsupportedConfiguration("MERERUN_MUSE_GLIMMER_Q8_TILED_HEAD_SPLITS must be 1, 2, 4, or 8.")
        }
        return splits
    }
}
