import Foundation

/// The base configuration Marigold V2 actually needs.
///
/// This is deliberately narrower than `QwenImageEditModelConfigs`: the text
/// encoder and its tokenizer are never instantiated, so their configs are not
/// downloaded and cannot be decoded here. The scheduler is unused too, because
/// inference is a single step at a fixed timestep rather than a sampled
/// trajectory.
public struct MarigoldV2ModelConfigs: Sendable, Hashable {
    public let transformer: QwenImageEditTransformerConfig
    public let vae: QwenImageEditVAEConfig

    public init(transformer: QwenImageEditTransformerConfig, vae: QwenImageEditVAEConfig) {
        self.transformer = transformer
        self.vae = vae
    }

    public static func load(
        from resources: MarigoldV2Resources,
        decoder: JSONDecoder = JSONDecoder()
    ) throws -> MarigoldV2ModelConfigs {
        MarigoldV2ModelConfigs(
            transformer: try decoder.decode(
                QwenImageEditTransformerConfig.self,
                from: try Data(contentsOf: resources.transformerConfigURL)
            ),
            vae: try decoder.decode(
                QwenImageEditVAEConfig.self,
                from: try Data(contentsOf: resources.vaeConfigURL)
            )
        )
    }
}
