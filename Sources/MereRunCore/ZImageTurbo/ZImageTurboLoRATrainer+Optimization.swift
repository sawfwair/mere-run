import Foundation
import MLX
import MLXNN
import MLXRandom

extension ZImageTurboLoRATrainer {
    // MARK: - LoRA Optimization

    /// Minimal state object for MLX compile/eval: only LoRA weights + optimizer state.
    final class LoRAState: Updatable, Evaluatable {
        let layers: [(path: String, layer: TrainableLoRALayer)]

        init(loraLayers: [String: TrainableLoRALayer]) {
            self.layers = loraLayers
                .map { (path: $0.key, layer: $0.value) }
                .sorted { $0.path < $1.path }
        }

        func innerState() -> [MLXArray] {
            var arrays: [MLXArray] = []
            arrays.reserveCapacity(layers.count * 6)
            for (_, layer) in layers {
                arrays.append(layer.loraDown)
                arrays.append(layer.loraUp)
                arrays.append(layer.loraDownM!)
                arrays.append(layer.loraDownV!)
                arrays.append(layer.loraUpM!)
                arrays.append(layer.loraUpV!)
            }
            return arrays
        }
    }

    static func initializeAdamStateIfNeeded(for loraLayers: [String: TrainableLoRALayer]) {
        for (_, layer) in loraLayers {
            if layer.loraDownM == nil { layer.loraDownM = MLXArray.zeros(like: layer.loraDown) }
            if layer.loraDownV == nil { layer.loraDownV = MLXArray.zeros(like: layer.loraDown) }
            if layer.loraUpM == nil { layer.loraUpM = MLXArray.zeros(like: layer.loraUp) }
            if layer.loraUpV == nil { layer.loraUpV = MLXArray.zeros(like: layer.loraUp) }
        }
    }

    /// Manual AdamW update for LoRA weights only (compiled path).
    static func applyAdamW(
        loraLayers: [(path: String, layer: TrainableLoRALayer)],
        gradMap: [String: MLXArray],
        lr: MLXArray,
        beta1: MLXArray,
        beta2: MLXArray,
        oneMinusBeta1: MLXArray,
        oneMinusBeta2: MLXArray,
        eps: MLXArray,
        oneMinusLrWd: MLXArray,
        useWeightDecay: Bool
    ) {
        // Gradient clipping (max_grad_norm = 1.0, matches ai-toolkit)
        let maxGradNorm = MLXArray(Float(1.0))
        var gradNormSquared = MLXArray(Float(0.0))
        for (path, _) in loraLayers {
            if let downGrad = gradMap["\(path).loraDown"] {
                gradNormSquared = gradNormSquared + downGrad.square().sum()
            }
            if let upGrad = gradMap["\(path).loraUp"] {
                gradNormSquared = gradNormSquared + upGrad.square().sum()
            }
        }
        let gradNorm = gradNormSquared.sqrt()
        let clipCoef = maxGradNorm / (gradNorm + MLXArray(1e-6))
        let shouldClip = gradNorm .> maxGradNorm

        for (path, layer) in loraLayers {
            guard let downGradRaw = gradMap["\(path).loraDown"],
                  let upGradRaw = gradMap["\(path).loraUp"] else {
                continue
            }

            // Apply gradient clipping
            let downGrad = MLX.where(shouldClip, downGradRaw * clipCoef, downGradRaw)
            let upGrad = MLX.where(shouldClip, upGradRaw * clipCoef, upGradRaw)

            let mDown = beta1 * layer.loraDownM! + oneMinusBeta1 * downGrad
            let vDown = beta2 * layer.loraDownV! + oneMinusBeta2 * downGrad.square()
            let wDown = useWeightDecay ? layer.loraDown * oneMinusLrWd : layer.loraDown
            let newDown = wDown - lr * mDown / (vDown.sqrt() + eps)
            layer.loraDown._updateInternal(newDown)
            layer.loraDownM!._updateInternal(mDown)
            layer.loraDownV!._updateInternal(vDown)

            let mUp = beta1 * layer.loraUpM! + oneMinusBeta1 * upGrad
            let vUp = beta2 * layer.loraUpV! + oneMinusBeta2 * upGrad.square()
            let wUp = useWeightDecay ? layer.loraUp * oneMinusLrWd : layer.loraUp
            let newUp = wUp - lr * mUp / (vUp.sqrt() + eps)
            layer.loraUp._updateInternal(newUp)
            layer.loraUpM!._updateInternal(mUp)
            layer.loraUpV!._updateInternal(vUp)
        }
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

        var moduleUpdates: [(String, Module)] = directUpdates

        for (parentPath, indexedReplacements) in arrayUpdates {
            let currentModules = leafModules.filter { (path, _) in
                let parts = path.split(separator: ".")
                guard parts.count >= 2 else { return false }
                guard Int(parts.last!) != nil else { return false }
                let parent = parts.dropLast().joined(separator: ".")
                return parent == parentPath
            }

            var replacementMap: [Int: Module] = [:]
            for (index, replacement) in indexedReplacements {
                replacementMap[index] = replacement
            }

            for (modulePath, originalModule) in currentModules {
                let index = Int(modulePath.split(separator: ".").last!)!
                let resolved = replacementMap[index] ?? originalModule
                moduleUpdates.append((modulePath, resolved))
            }
        }

        guard !moduleUpdates.isEmpty else { return }
        model.update(modules: ModuleChildren.unflattened(moduleUpdates))
    }
}

