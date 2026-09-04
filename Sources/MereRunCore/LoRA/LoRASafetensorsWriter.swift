import Foundation
import MLX

public enum LoRASafetensorsWriter {
    /// Save LoRA weights to safetensors file
    /// - Parameters:
    ///   - loraLayers: The LoRA layers to save
    ///   - url: Destination URL
    ///   - dtype: Data type for weights (default float16)
    ///   - includeOptimizerState: If true, also saves Adam m/v state for resumable training (~3x file size)
    ///   - metadata: Additional metadata to include
    public static func save(
        loraLayers: [String: TrainableLoRALayer],
        to url: URL,
        dtype: DType = .float16,
        includeOptimizerState: Bool = false,
        metadata: [String: String] = [:]
    ) throws {
        var arrays: [String: MLXArray] = [:]
        let capacity = includeOptimizerState ? loraLayers.count * 6 : loraLayers.count * 2
        arrays.reserveCapacity(capacity)

        for (path, layer) in loraLayers.sorted(by: { $0.key < $1.key }) {
            // Always save weights
            arrays["\(path).lora_down.weight"] = layer.loraDown.asType(dtype)
            arrays["\(path).lora_up.weight"] = layer.loraUp.asType(dtype)

            // Optionally save optimizer state (for resumable training)
            if includeOptimizerState {
                if let m = layer.loraDownM {
                    arrays["\(path).lora_down.m"] = m.asType(.float32)
                }
                if let v = layer.loraDownV {
                    arrays["\(path).lora_down.v"] = v.asType(.float32)
                }
                if let m = layer.loraUpM {
                    arrays["\(path).lora_up.m"] = m.asType(.float32)
                }
                if let v = layer.loraUpV {
                    arrays["\(path).lora_up.v"] = v.asType(.float32)
                }
            }
        }

        var mergedMetadata = metadata
        if let first = loraLayers.values.first {
            mergedMetadata["lora_rank"] = "\(first.loraRank)"
            mergedMetadata["lora_alpha"] = "\(first.loraAlpha)"
        }
        if includeOptimizerState {
            mergedMetadata["has_optimizer_state"] = "true"
        }

        // Evaluate all arrays before saving (MLX lazy evaluation)
        eval(Array(arrays.values))

        try MLX.save(arrays: arrays, metadata: mergedMetadata, url: url)
    }

    /// Saves a safetensors file using EMA weights without permanently mutating the provided layers.
    public static func saveEMASnapshot(
        emaState: LoRAEMAState,
        loraLayers: [String: TrainableLoRALayer],
        to url: URL,
        dtype: DType = .float16,
        metadata: [String: String] = [:]
    ) throws {
        var originalWeights: [String: (down: MLXArray, up: MLXArray)] = [:]
        originalWeights.reserveCapacity(loraLayers.count)
        for (path, layer) in loraLayers {
            originalWeights[path] = (down: layer.loraDown, up: layer.loraUp)
        }

        emaState.apply(to: loraLayers)
        defer {
            for (path, weights) in originalWeights {
                if let layer = loraLayers[path] {
                    layer.loraDown = weights.down
                    layer.loraUp = weights.up
                }
            }
        }

        try save(
            loraLayers: loraLayers,
            to: url,
            dtype: dtype,
            includeOptimizerState: false,
            metadata: metadata
        )
    }
}
