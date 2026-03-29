import Foundation
import MLX

/// Owns LoRA application for image and chat paths.
/// This keeps cartridge-specific state transitions separate from model loading
/// and request execution.
extension ZImageTurboGenerator {
    static let noMatchingTransformerLayersMessage = "No matching transformer layers found for this LoRA."

    func applyTransformerLoRAIfNeeded(
        _ lora: LoRA?,
        model: LoadedModel,
        progressHandler: (@Sendable (GenerationProgress) -> Void)?
    ) async throws -> LoadedModel {
        guard lora != currentTransformerLoRA else { return model }

        if lora == nil {
            if let layers = transformerLoRALayers {
                for layer in layers.values {
                    layer.isActive = false
                }
            }
            currentTransformerLoRA = nil
            return model
        }

        let lora = lora!
        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 0, totalSteps: 2))

        let loraURL = try await LoRAWeightLoader.resolveURL(for: lora)
        let loraWeights = try LoRAWeightLoader.load(from: loraURL)
        let targetRank = loraWeights.rank
        let quantization = model.transformerQuantization

        var model = model
        if let existingRank = transformerLoRARank, existingRank != targetRank {
            let transformer = ZImageTransformer2DModel(configuration: model.configs.transformer)
            try loadTransformerWeights(
                resources: model.resources,
                into: transformer,
                quantization: quantization,
                progressHandler: progressHandler
            )

            let rebuilt = LoadedModel(
                modelSpec: model.modelSpec,
                rootURL: model.rootURL,
                manifest: model.manifest,
                resources: model.resources,
                configs: model.configs,
                textEncoderQuantization: model.textEncoderQuantization,
                transformerQuantization: model.transformerQuantization,
                tokenizer: model.tokenizer,
                textEncoder: model.textEncoder,
                transformer: transformer,
                vae: model.vae
            )

            loaded = rebuilt
            transformerLoRALayers = nil
            transformerLoRARank = nil
            currentTransformerLoRA = nil
            model = rebuilt
        }

        if transformerLoRALayers == nil {
            if Self.loraDebugEnabled {
                FileHandle.standardError.write(
                    Data("[LoRA Debug] Injecting with rank=\(targetRank), alpha=\(loraWeights.alpha)\n".utf8)
                )
            }
            transformerLoRALayers = try ZImageLoRAInjector.inject(
                into: model.transformer,
                rank: targetRank,
                alpha: loraWeights.alpha,
                targetSuffixes: ZImageLoRAInjector.defaultTargetSuffixes,
                zeroInitUp: true
            )
            if let layers = transformerLoRALayers {
                for layer in layers.values {
                    layer.isActive = false
                }
            }
            transformerLoRARank = targetRank
        }

        guard let layers = transformerLoRALayers else {
            throw LoRAError.invalidFormat("Failed to inject Z-Image Turbo LoRA layers.")
        }

        let updatedCount = ZImageLoRAInjector.applyWeights(
            loraWeights,
            to: layers,
            debug: Self.loraDebugEnabled
        )
        for (path, layer) in layers {
            layer.isActive = loraWeights.weights[path] != nil
        }
        guard updatedCount > 0 else {
            throw LoRAError.invalidFormat(Self.noMatchingTransformerLayersMessage)
        }

        let userScale = Self.loraScale(for: lora)
        if userScale != 1 {
            let scale = MLXArray(userScale)
            for layer in layers.values where layer.isActive {
                layer.loraDown = layer.loraDown * scale
            }
        }

        currentTransformerLoRA = lora
        progressHandler?(GenerationProgress(stage: .loadingLoRA, stepIndex: 2, totalSteps: 2))
        return model
    }

    func applyTextLoRAIfNeeded(
        _ lora: LoRA?,
        model: LoadedModel,
        progressHandler: (@Sendable (ChatProgress) -> Void)?
    ) async throws {
        guard lora != currentTextLoRA else { return }

        if currentTextLoRA != nil {
            try loadTextEncoderWeights(
                resources: model.resources,
                into: model.textEncoder,
                quantization: model.textEncoderQuantization,
                progressHandler: nil
            )
        }

        guard let lora else {
            currentTextLoRA = nil
            return
        }

        progressHandler?(ChatProgress(stage: .encoding, message: "Loading cartridge"))
        let weights = try await QwenTextLoRAWeightLoader.load(from: lora)
        let applied = QwenTextLoRAApplicator.mergeIntoTextEncoder(
            model.textEncoder,
            loraWeights: weights,
            scale: 1.0
        )

        if applied == 0 {
            throw LoRAError.invalidFormat("No matching Qwen text encoder layers found for this cartridge.")
        }

        currentTextLoRA = lora
    }

    static func loraScale(for lora: LoRA) -> Float {
        switch lora {
        case .local(_, let scale):
            return Float(scale)
        case .remote(_, let scale):
            return Float(scale)
        }
    }
}
