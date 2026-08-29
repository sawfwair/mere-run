import Foundation
import MLX
import MLXNN

/// Installs the exact LightX2V four-step 2511 adapter as runtime LoRA layers.
/// The checkpoint is streamed one pair at a time so the 850 MB adapter is not
/// duplicated in a temporary in-memory weight dictionary.
public enum QwenImageEditLightningAdapter {
    public static let expectedPairCount = 720
    public static let rank = 64
    public static let alpha: Float = 8

    public enum AdapterError: LocalizedError {
        case unexpectedPairCount(expected: Int, actual: Int)
        case missingTarget(String)
        case targetIsNotLinear(String)
        case invalidPairShape(String, down: [Int], up: [Int], target: [Int])

        public var errorDescription: String? {
            switch self {
            case .unexpectedPairCount(let expected, let actual):
                return "Qwen Edit Lightning tensor-pair count mismatch: expected \(expected), found \(actual)."
            case .missingTarget(let path):
                return "Qwen Edit Lightning target is missing: \(path)."
            case .targetIsNotLinear(let path):
                return "Qwen Edit Lightning target is not a Linear layer: \(path)."
            case .invalidPairShape(let path, let down, let up, let target):
                return "Qwen Edit Lightning target \(path) has incompatible shapes "
                    + "down=\(down), up=\(up), target=\(target)."
            }
        }
    }

    @discardableResult
    public static func install(
        url: URL,
        into transformer: MMDiT,
        verifyArtifact: Bool = true
    ) throws -> Int {
        if verifyArtifact {
            _ = try QwenImageEditRepository.lightningPin.verify(
                in: url.deletingLastPathComponent().deletingLastPathComponent()
            )
        }

        let leafModules = transformer.leafModules().flattened()
        let modulesByPath = Dictionary(uniqueKeysWithValues: leafModules)
        var replacements: [String: Module] = [:]

        let pairCount = try SafetensorsStreamingLoader.forEachTensorPair(
            url: url,
            firstSuffix: ".lora_down.weight",
            secondSuffix: ".lora_up.weight"
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

    public static func mappedTargetPath(_ sourcePath: String) -> String {
        sourcePath
            .replacingOccurrences(of: ".attn.to_out.0", with: ".attn.to_out")
            .replacingOccurrences(of: ".img_mlp.net.0.proj", with: ".ff.linear1")
            .replacingOccurrences(of: ".img_mlp.net.2", with: ".ff.linear2")
            .replacingOccurrences(of: ".txt_mlp.net.0.proj", with: ".ff_context.linear1")
            .replacingOccurrences(of: ".txt_mlp.net.2", with: ".ff_context.linear2")
    }
}
