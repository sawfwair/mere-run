import Foundation
import MLX
import MLXNN

extension MiniMaxH3TurboAdapter {
    static func isAdaLNTarget(_ path: String) -> Bool {
        path == "final_layer.adaln_proj.linear"
            || (path.hasPrefix("blocks.") && path.hasSuffix(".adaln_proj.linear"))
    }

    static func augmentedAdaLNCache(
        _ cache: MiniMaxH3AdaLNCache,
        pairs: [String: LoRAPair],
        configuration: MiniMaxH3TransformerConfiguration,
        strength: Float
    ) throws -> MiniMaxH3AdaLNCache {
        let activated = MLXNN.silu(cache.timeEmbeddings)
            .reshaped(cache.stepCount * 3, configuration.timeEmbeddingDimension)
        var blocks = cache.blockModulations
        var final = cache.finalModulations
        for (path, pair) in pairs {
            guard pair.down.dim(1) == configuration.timeEmbeddingDimension else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [pair.up.dim(0), configuration.timeEmbeddingDimension],
                    actual: [pair.up.dim(0), pair.down.dim(1)]
                )
            }
            let delta = MLX.matmul(
                MLX.matmul(activated.asType(pair.down.dtype), pair.down.T),
                pair.up.T
            ) * MLXArray(strength).asType(pair.down.dtype)
            if path == "final_layer.adaln_proj.linear" {
                guard pair.up.dim(0) == 2 * configuration.hiddenSize else {
                    throw AdapterError.targetShapeMismatch(
                        path,
                        expected: [2 * configuration.hiddenSize, configuration.timeEmbeddingDimension],
                        actual: [pair.up.dim(0), pair.down.dim(1)]
                    )
                }
                final = final + delta.reshaped(
                    cache.stepCount,
                    3,
                    2 * configuration.hiddenSize
                ).asType(final.dtype)
                continue
            }
            let components = path.split(separator: ".")
            guard components.count == 4,
                  components[0] == "blocks",
                  let index = Int(components[1]),
                  blocks.indices.contains(index),
                  pair.up.dim(0) == 18 * configuration.hiddenSize else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [18 * configuration.hiddenSize, configuration.timeEmbeddingDimension],
                    actual: [pair.up.dim(0), pair.down.dim(1)]
                )
            }
            blocks[index] = blocks[index] + delta.reshaped(
                cache.stepCount,
                9,
                6 * configuration.hiddenSize
            ).asType(blocks[index].dtype)
        }
        MLX.eval([final] + blocks)
        return MiniMaxH3AdaLNCache(
            timeEmbeddings: cache.timeEmbeddings,
            blockModulations: blocks,
            finalModulations: final,
            videoSigmas: cache.videoSigmas,
            audioSigmas: cache.audioSigmas,
            sourceIdentity: cache.sourceIdentity
        )
    }

    static func applyModuleReplacements(
        _ replacements: [String: Module],
        leafModules: [(String, Module)],
        to model: Module
    ) {
        var arrayUpdates: [String: [(Int, Module)]] = [:]
        var directUpdates: [(String, Module)] = []

        for (path, replacement) in replacements {
            let components = path.split(separator: ".")
            if let last = components.last, let index = Int(last) {
                let parentPath = components.dropLast().joined(separator: ".")
                arrayUpdates[parentPath, default: []].append((index, replacement))
            } else {
                directUpdates.append((path, replacement))
            }
        }

        var moduleUpdates = directUpdates
        for (parentPath, indexedReplacements) in arrayUpdates {
            let currentModules = leafModules.filter { path, _ in
                let parts = path.split(separator: ".")
                guard parts.count >= 2, Int(parts.last!) != nil else { return false }
                return parts.dropLast().joined(separator: ".") == parentPath
            }
            let replacementMap = Dictionary(uniqueKeysWithValues: indexedReplacements)
            for (modulePath, originalModule) in currentModules {
                let index = Int(modulePath.split(separator: ".").last!)!
                moduleUpdates.append((modulePath, replacementMap[index] ?? originalModule))
            }
        }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }

}
