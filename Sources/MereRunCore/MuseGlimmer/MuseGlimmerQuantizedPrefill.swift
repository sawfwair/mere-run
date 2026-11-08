import Foundation
import MLX
import MLXNN

/// The pinned backend disables quantized NAX on generation 17 for correctness.
/// Decode keeps its packed matmul. Large prefill matrices use a temporary activation-dtype
/// reconstruction and the supported dense NAX path, without retaining a second
/// copy of the model or removing the backend's safety guard.
final class MuseGlimmerPrefillLinear: QuantizedLinear {
    private let minimumRows: Int
    private let packedForward: (MLXArray) -> MLXArray

    init(_ source: QuantizedLinear, minimumRows: Int = 128) {
        self.minimumRows = minimumRows
        self.packedForward = { source($0) }
        super.init(weight: source.weight, bias: source.bias, scales: source.scales, biases: source.biases,
                   groupSize: source.groupSize, bits: source.bits, mode: source.mode, globalScale: source.globalScale)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.shape.dropLast().reduce(1, *) >= minimumRows else { return packedForward(x) }
        let dense = dequantized(weight, scales: scales, biases: biases, groupSize: groupSize,
                                bits: bits, mode: mode, dtype: x.dtype)
        let output = matmul(x, dense.T)
        return bias.map { output + $0 } ?? output
    }
}

extension MuseGlimmerModel {
    @discardableResult
    func installQuantizedPrefill(environment: [String: String] = ProcessInfo.processInfo.environment) throws -> Int {
        let raw = environment["MERERUN_MUSE_GLIMMER_QUANTIZED_PREFILL"] ?? "auto"
        guard ["auto", "native", "dense"].contains(raw) else {
            throw MuseGlimmerError.unsupportedConfiguration("MERERUN_MUSE_GLIMMER_QUANTIZED_PREFILL must be auto, native, or dense.")
        }
        let enabled: Bool
        #if os(macOS)
        if #available(macOS 26.2, *) {
            enabled = raw == "dense" || (raw == "auto" && GPU.deviceInfo().architecture.hasPrefix("applegpu_g17"))
        } else {
            enabled = raw == "dense"
        }
        #else
        enabled = raw == "dense"
        #endif
        guard enabled else { return 0 }
        let modules = namedModules()
        var parents = Dictionary(uniqueKeysWithValues: modules)
        parents[""] = self
        var count = 0
        for (path, module) in modules {
            guard let linear = module as? QuantizedLinear, !(linear is MuseGlimmerPrefillLinear),
                  linear.mode == .affine, [4, 8].contains(linear.bits) else { continue }
            let parts = path.split(separator: ".").map(String.init)
            let parent = parents[parts.dropLast().joined(separator: ".")]!
            parent.update(modules: ModuleChildren.unflattened([(parts.last!, MuseGlimmerPrefillLinear(linear))]))
            count += 1
        }
        return count
    }
}
