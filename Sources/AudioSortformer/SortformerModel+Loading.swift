// Adapted from mlx-audio-swift at commit 4266f988d170a83017d1e82e2e4654602f277f1d.
// Copyright (c) 2025 Prince Canuma. Licensed under the MIT License.
import Foundation
import MLX
import MLXNN
import MereRunModelKit

// MARK: - FastConformer Encoder Components

extension SortformerModel {
    // MARK: - Weight Sanitization & Loading

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        let skipKeys: Set<String> = ["num_batches_tracked"]

        let alreadyConverted = weights.keys.contains { $0.contains("subsampling.layers_") }

        for (k, var v) in weights {
            if skipKeys.contains(where: { k.contains($0) }) { continue }

            var newK = k

            if !alreadyConverted {
                if newK.contains("fc_encoder.subsampling.layers.") {
                    newK = newK.replacingOccurrences(of: "subsampling.layers.", with: "subsampling.layers_")
                }

                // Conv2d: PyTorch (O,I,H,W) → MLX (O,H,W,I)
                if newK.contains("subsampling") && newK.contains("weight") && !newK.contains("linear") {
                    if v.ndim == 4 {
                        v = v.transposed(0, 2, 3, 1)
                    }
                }

                // Conv1d: PyTorch (O,I,K) → MLX (O,K,I)
                if (newK.contains("pointwise_conv1") || newK.contains("pointwise_conv2") || newK.contains("depthwise_conv"))
                    && newK.contains("weight") {
                    if v.ndim == 3 {
                        v = v.transposed(0, 2, 1)
                    }
                }
            }

            sanitized[newK] = v
        }

        return sanitized
    }

    static func runtimeCompatibleWeights(
        _ weights: [String: MLXArray],
        promoteFloat16: Bool
    ) -> [String: MLXArray] {
        guard promoteFloat16 else { return weights }
        return weights.mapValues { weight in
            weight.dtype == .float16 ? weight.asType(.float32) : weight
        }
    }

    public static func fromModelDirectory(_ modelURL: URL) throws -> SortformerModel {
        // Load config
        let configURL = modelURL.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(SortformerConfig.self, from: configData)

        let model = SortformerModel(config)

        // Load weights
        let weightFiles = try FileManager.default.contentsOfDirectoryResolvingSymlinks(
            at: modelURL, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }

        var allWeights = [String: MLXArray]()
        for file in weightFiles {
            let weights = try loadArrays(url: file)
            for (k, v) in weights {
                allWeights[k] = v
            }
        }

        let sanitized = sanitize(allWeights)
        #if os(Linux) && arch(x86_64)
        // mlx-swift cannot construct Float16 host scalars on x86_64. Promoting
        // the compact Sortformer checkpoint keeps scalar model operations on
        // the native CUDA path without triggering that host-only initializer.
        let runtimeWeights = runtimeCompatibleWeights(sanitized, promoteFloat16: true)
        #else
        let runtimeWeights = sanitized
        #endif
        try model.update(parameters: ModuleParameters.unflattened(runtimeWeights), verify: .noUnusedKeys)
        eval(model.parameters())

        return model
    }
}
