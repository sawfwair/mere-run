import Foundation
import MLX
import MLXNN

/// Installs the Marigold V2 depth adapters as runtime LoRA layers on a frozen
/// Qwen-Image-Edit-2509 transformer.
///
/// The checkpoint stores PEFT-style pairs (`lora_A.default.weight` /
/// `lora_B.default.weight`) under a `Diffuser.` prefix. Pairs are streamed one at
/// a time so the 1.8 GB payload is never duplicated into a temporary weight
/// dictionary, matching how the Lightning adapter loads.
public enum MarigoldV2LoRAAdapter {
    /// 60 transformer blocks with 12 adapted projections each, plus `img_in`,
    /// `txt_in`, and the final AdaLN projection.
    public static let expectedPairCount = 723
    public static let rank = 128
    /// Marigold trains with `lora_alpha == rank`, so the applied scale is 1.0.
    public static let alpha: Float = 128

    public static let downSuffix = ".lora_A.default.weight"
    public static let upSuffix = ".lora_B.default.weight"
    public static let sourcePrefix = "Diffuser."

    public enum AdapterError: LocalizedError {
        case unexpectedPairCount(expected: Int, actual: Int)
        case missingTarget(String)
        case targetIsNotLinear(String)
        case invalidPairShape(String, down: [Int], up: [Int], target: [Int])

        public var errorDescription: String? {
            switch self {
            case .unexpectedPairCount(let expected, let actual):
                return "Marigold V2 tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .missingTarget(let path):
                return "Marigold V2 target is missing: \(path)."
            case .targetIsNotLinear(let path):
                return "Marigold V2 target is not a Linear layer: \(path)."
            case .invalidPairShape(let path, let down, let up, let target):
                return "Marigold V2 target \(path) has incompatible shapes "
                    + "down=\(down), up=\(up), target=\(target)."
            }
        }
    }

    /// Installs the adapters in `url` onto `transformer`.
    ///
    /// Artifact pinning is verified once against the install root before load,
    /// so this entry point takes the checkpoint path as given.
    @discardableResult
    public static func install(
        url: URL,
        into transformer: MMDiT
    ) throws -> Int {
        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: downSuffix,
            secondSuffix: upSuffix
        ) { sourcePath, down, up in
            let path = mappedTargetPath(sourcePath)
            guard replacements[path] == nil else {
                throw AdapterError.missingTarget("duplicate \(path)")
            }
            guard let module = modulesByPath[path] else {
                throw AdapterError.missingTarget(path)
            }
            guard let linear = module as? Linear else {
                throw AdapterError.targetIsNotLinear(path)
            }
            guard down.shape == [rank, linear.shape.1],
                  up.shape == [linear.shape.0, rank] else {
                throw AdapterError.invalidPairShape(
                    path,
                    down: down.shape,
                    up: up.shape,
                    target: [linear.shape.0, linear.shape.1]
                )
            }

            let layer: TrainableLoRALayer & Module
            if let quantized = linear as? QuantizedLinear {
                layer = LoRAQuantizedLinear(
                    base: quantized,
                    rank: rank,
                    alpha: alpha,
                    zeroInitUp: true
                )
            } else {
                layer = LoRALinear(
                    base: linear,
                    rank: rank,
                    alpha: alpha,
                    zeroInitUp: true
                )
            }
            layer.loraDown = down.asType(.float32)
            layer.loraUp = up.asType(.float32)
            layer.role = .assistant
            MLX.eval(layer.loraDown, layer.loraUp)
            replacements[path] = layer
        }

        guard pairCount == expectedPairCount else {
            throw AdapterError.unexpectedPairCount(expected: expectedPairCount, actual: pairCount)
        }
        Krea2LoRAInjector.applyModuleReplacements(
            replacements,
            leafModules: leafModules,
            to: transformer
        )
        Memory.clearCache()
        return pairCount
    }

    /// Maps a checkpoint tensor path onto its `MMDiT` module path.
    ///
    /// Only `Diffuser.*` tensors carry the LoRA suffixes this loader selects, so
    /// anything else reaching here is an unrecognized checkpoint layout and is
    /// left unmapped to fail loudly against the module tree.
    public static func mappedTargetPath(_ sourcePath: String) -> String {
        let path = sourcePath.hasPrefix(sourcePrefix)
            ? String(sourcePath.dropFirst(sourcePrefix.count))
            : sourcePath

        // The diffusers patch/context embedders and the final AdaLN projection
        // carry names the native transformer spells differently.
        if path == "img_in" {
            return "x_embedder"
        }
        if path == "txt_in" {
            return "context_embedder"
        }

        return path
            .replacingOccurrences(of: ".attn.to_out.0", with: ".attn.to_out")
            .replacingOccurrences(of: ".img_mlp.net.0.proj", with: ".ff.linear1")
            .replacingOccurrences(of: ".img_mlp.net.2", with: ".ff.linear2")
            .replacingOccurrences(of: ".txt_mlp.net.0.proj", with: ".ff_context.linear1")
            .replacingOccurrences(of: ".txt_mlp.net.2", with: ".ff_context.linear2")
    }
}
