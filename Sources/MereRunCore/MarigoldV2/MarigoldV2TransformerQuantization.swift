import Foundation
import MLXNN

/// Keep the reference checkpoint's first image-modulation projection dense.
/// Quantizing it destroys the depth structure even when the other projections
/// use the same weights and precision. This is the native path for the reference
/// exclusion `transformer_blocks.0.img_mod`.
enum MarigoldV2TransformerQuantization {
    static let firstImageModulationPath = "transformer_blocks.0.adaLN_modulation.linear"

    enum QuantizationError: LocalizedError {
        case quantizedFirstImageModulation

        var errorDescription: String? {
            "Marigold V2 requires the first image-modulation projection to remain unquantized. "
                + "Use the original base weights or a quantized checkpoint that preserves this layer."
        }
    }

    static func validate(factory: DenseLayerFactory) throws {
        guard !factory.isQuantized(path: firstImageModulationPath) else {
            throw QuantizationError.quantizedFirstImageModulation
        }
    }

    static func apply(to transformer: MMDiT) {
        MLXNN.quantize(model: transformer, groupSize: 64, bits: 4) { path, module in
            guard path != firstImageModulationPath else { return false }
            if let linear = module as? Linear {
                return linear.shape.1 % 64 == 0
            }
            return true
        }
    }
}
