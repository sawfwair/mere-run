import Foundation
import MLX
import MLXNN

public enum QuantizationIO {
    /// Saves a quantized MLX `Module` to a single safetensors file, including required `.scales` (+ optional `.biases`)
    /// arrays for `QuantizedLinear` / `QuantizedEmbedding` leaf modules.
    public static func saveQuantizedModule(
        _ module: Module,
        to url: URL,
        extraArrays: [String: MLXArray] = [:]
    ) throws {
        var arrays: [String: MLXArray] = [:]

        for (key, value) in module.parameters().flattened() {
            arrays[key] = value
        }

        for (path, m) in module.leafModules().flattened() {
            if let qlinear = m as? QuantizedLinear {
                arrays["\(path).scales"] = qlinear.scales
                if let qbiases = qlinear.biases {
                    arrays["\(path).biases"] = qbiases
                }
            } else if let qembed = m as? QuantizedEmbedding {
                arrays["\(path).scales"] = qembed.scales
                if let qbiases = qembed.biases {
                    arrays["\(path).biases"] = qbiases
                }
            } else if let qlinear = m as? ResidualQuantizedLinear {
                arrays["\(path).scales"] = qlinear.scales
                if let qbiases = qlinear.biases {
                    arrays["\(path).biases"] = qbiases
                }
                // Residual matrices are stored explicitly.
                arrays["\(path).svd_up"] = qlinear.residualUp
                arrays["\(path).svd_down"] = qlinear.residualDown
            }
        }

        for (key, value) in extraArrays {
            arrays[key] = value
        }

        try MLX.save(arrays: arrays, url: url)
    }

    public static func captureLinearWeights(_ module: Module) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for (path, m) in module.leafModules().flattened() {
            if let linear = m as? Linear {
                weights[path] = linear.weight
            }
        }
        return weights
    }

    public static func computeSVDResiduals(
        in module: Module,
        originalWeights: [String: MLXArray],
        rank: Int,
        targetSuffixes: [String],
        maxLayers: Int,
        label: String
    ) -> [String: MLXArray] {
        guard rank > 0 else { return [:] }

        let shouldTarget: (String) -> Bool = { path in
            if targetSuffixes.isEmpty { return true }
            return targetSuffixes.contains { path.hasSuffix($0) }
        }

        var residuals: [String: MLXArray] = [:]
        var processed = 0
        var skipped = 0

        for (path, m) in module.leafModules().flattened() {
            if maxLayers > 0, processed >= maxLayers { break }
            guard let qlinear = m as? QuantizedLinear else { continue }
            guard shouldTarget(path) else { continue }
            guard let original = originalWeights[path] else { continue }
            guard original.ndim == 2 else { continue }

            let originalF = original.asType(.float32, stream: .cpu)
            let dequantized = MLX.dequantized(
                qlinear.weight,
                scales: qlinear.scales,
                biases: qlinear.biases,
                groupSize: qlinear.groupSize,
                bits: qlinear.bits,
                dtype: .float32,
                stream: .cpu
            )

            let residual = subtract(originalF, dequantized, stream: .cpu)
            let (u, s, vt) = MLX.svd(residual, stream: .cpu)
            let maxRank = min(s.dim(0), min(u.dim(1), vt.dim(0)))
            let r = min(rank, maxRank)
            if r <= 0 {
                skipped += 1
                continue
            }

            let uR = u[0..., 0..<r]
            let sR = s[0..<r].reshaped([1, r])
            let vtR = vt[0..<r, 0...]
            let up = multiply(uR, sR, stream: .cpu)
            let down = vtR

            residuals["\(path).svd_up"] = up.asType(.float32)
            residuals["\(path).svd_down"] = down.asType(.float32)
            processed += 1

            if processed % 10 == 0 {
                print("  [\(label)] SVD residuals: \(processed) layers...")
            }
        }

        if processed > 0 {
            print("  [\(label)] SVD residuals: \(processed) layers written (\(skipped) skipped)")
        } else if rank > 0 {
            print("  [\(label)] SVD residuals: none written")
        }

        return residuals
    }
}

