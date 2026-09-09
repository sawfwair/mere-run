import Foundation

public struct ZImageTurboModelConfigs: Sendable, Hashable {
    public let modelIndex: ZImageTurboModelIndex?
    public let transformer: ZImageTurboTransformerConfig
    public let vae: ZImageTurboVAEConfig
    public let scheduler: ZImageTurboSchedulerConfig
    public let textEncoder: ZImageTurboTextEncoderConfig

    public static func load(
        from resources: ZImageTurboResources,
        decoder: JSONDecoder = JSONDecoder(),
        fileManager: FileManager = .default
    ) throws -> ZImageTurboModelConfigs {
        let useMFluxDefaults = resources.hasMFluxWeights(fileManager: fileManager)

        func decode<T: Decodable>(_ type: T.Type, url: URL) throws -> T {
            try decoder.decode(T.self, from: Data(contentsOf: url))
        }
        func decodeOrMFluxDefault<T: Decodable>(_ type: T.Type, url: URL, fallback: T) throws -> T {
            if fileManager.fileExists(atPath: url.path) {
                return try decode(type, url: url)
            }
            guard useMFluxDefaults else {
                return try decode(type, url: url)
            }
            return fallback
        }

        return ZImageTurboModelConfigs(
            modelIndex: fileManager.fileExists(atPath: resources.modelIndexURL.path)
                ? try decode(ZImageTurboModelIndex.self, url: resources.modelIndexURL)
                : nil,
            transformer: try decodeOrMFluxDefault(
                ZImageTurboTransformerConfig.self,
                url: resources.transformerConfigURL,
                fallback: ZImageTurboTransformerConfig.mfluxZImageTurbo
            ),
            vae: try decodeOrMFluxDefault(
                ZImageTurboVAEConfig.self,
                url: resources.vaeConfigURL,
                fallback: ZImageTurboVAEConfig.mfluxZImageTurbo
            ),
            scheduler: try decodeOrMFluxDefault(
                ZImageTurboSchedulerConfig.self,
                url: resources.schedulerConfigURL,
                fallback: ZImageTurboSchedulerConfig.mfluxZImageTurbo
            ),
            textEncoder: try decodeOrMFluxDefault(
                ZImageTurboTextEncoderConfig.self,
                url: resources.textEncoderConfigURL,
                fallback: ZImageTurboTextEncoderConfig.mfluxZImageTurbo
            )
        )
    }
}
