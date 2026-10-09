import Foundation
import MereRunTensor
import MLX
import MLXNN

/// Explicit per-module affine packing. Recurrent gates, norms, convolution,
/// vision, and embedding output heads retain their original FP32 tensors.
public struct PPLXEmbedV2QuantizationConfig: Codable, Sendable {
    public struct Parameters: Codable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String
        enum CodingKeys: String, CodingKey { case bits, groupSize = "group_size", mode }
    }
    public let bits: Int
    public let groupSize: Int
    public let mode: String
    public let modules: [String: Parameters]
    enum CodingKeys: String, CodingKey { case bits, groupSize = "group_size", mode, modules }

    public func validate() throws {
        guard [4, 8].contains(bits), [32, 64, 128].contains(groupSize), mode == "affine", !modules.isEmpty,
              modules.allSatisfy({ path, value in
                  Self.permits(path) && [4, 8].contains(value.bits)
                      && [32, 64, 128].contains(value.groupSize) && value.mode == "affine"
              }) else {
            throw PPLXEmbedV2QuantizationError.invalid("Unsupported PPLX affine quantization contract.")
        }
    }

    private static func permits(_ path: String) -> Bool {
        if path == "embed_tokens" { return true }
        let parts = path.split(separator: ".")
        guard parts.count == 4, parts[0] == "layers", let layer = Int(parts[1]), layer >= 0 else { return false }
        switch parts[2] {
        case "self_attn": return ["q_proj", "k_proj", "v_proj", "o_proj"].contains(parts[3])
        case "linear_attn": return ["in_proj_qkv", "in_proj_z", "out_proj"].contains(parts[3])
        case "mlp": return ["gate_proj", "up_proj", "down_proj"].contains(parts[3])
        default: return false
        }
    }
}

public enum PPLXEmbedV2QuantizationError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

extension PPLXEmbedV2Encoder {
    public func installQuantizedWeights(_ arrays: [String: MLXArray], config: PPLXEmbedV2QuantizationConfig) throws {
        try config.validate()
        let expected = checkpointParameterNames
        let extras = Set(config.modules.keys.flatMap { [$0 + ".scales", $0 + ".biases"] })
        guard Set(arrays.keys) == expected.union(extras) else {
            let missing = expected.union(extras).subtracting(arrays.keys).sorted()
            let unexpected = Set(arrays.keys).subtracting(expected.union(extras)).sorted()
            throw PPLXEmbedV2QuantizationError.invalid("PPLX packed parameter mismatch. Missing: \(missing); unexpected: \(unexpected).")
        }
        let leaves = Dictionary(uniqueKeysWithValues: leafModules().flattened())
        var replacements: [(String, Module)] = []
        var packed: Set<String> = []
        for (path, parameters) in config.modules.sorted(by: { $0.key < $1.key }) {
            guard let module = leaves[path], module is Linear || module is Embedding,
                  let original = module.parameters().flattened().first(where: { $0.0 == "weight" })?.1,
                  let weight = arrays[path + ".weight"], let scales = arrays[path + ".scales"],
                  let biases = arrays[path + ".biases"], original.ndim == 2,
                  original.dim(1).isMultiple(of: parameters.groupSize),
                  weight.dtype == .uint32,
                  weight.shape == [original.dim(0), original.dim(1) * parameters.bits / 32],
                  scales.shape == [original.dim(0), original.dim(1) / parameters.groupSize],
                  biases.shape == scales.shape, scales.dtype == .float32, biases.dtype == .float32 else {
                throw PPLXEmbedV2QuantizationError.invalid("Invalid PPLX packed tensor geometry or dtype: \(path).")
            }
            let replacement: Module
            if let linear = module as? Linear {
                replacement = PortableQuantizedLinear(weight: weight, bias: linear.bias, scales: scales, biases: biases,
                    groupSize: parameters.groupSize, bits: parameters.bits)
            } else {
                replacement = PreQuantizedEmbedding(weight: weight, scales: scales, biases: biases,
                    groupSize: parameters.groupSize, bits: parameters.bits)
            }
            replacements.append((path, replacement))
            packed.formUnion([path + ".weight", path + ".scales", path + ".biases"])
        }
        let ordinary = arrays.filter { !packed.contains($0.key) }
        guard ordinary.values.allSatisfy({ $0.dtype == .float32 }) else {
            throw PPLXEmbedV2QuantizationError.invalid("Unpacked PPLX text tensors must retain FP32 precision.")
        }
        update(modules: ModuleChildren.unflattened(replacements))
        try update(parameters: ModuleParameters.unflattened(ordinary), verify: [.noUnusedKeys, .shapeMismatch])
    }
}
