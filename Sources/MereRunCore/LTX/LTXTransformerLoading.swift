import AudioCodecs
import Foundation
import MediaIO
import MLX
import MLXFast
import MLXNN

func mapUnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard let mapped = mapUnifiedTransformerKey(key) else { return [] }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapUnifiedTransformerKey(_ key: String) -> String? {
    guard key.hasPrefix("model.diffusion_model.") else { return nil }
    var mapped = String(key.dropFirst("model.diffusion_model.".count))
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    if mapped.hasPrefix("video_embeddings_connector")
        || mapped.hasPrefix("audio_embeddings_connector")
        || mapped.hasPrefix("text_embedding_projection")
    {
        return nil
    }
    return mapped
}

func resolvedLTX25TransformerURL(
    resources: LTX25Resources,
    kind: LTX25NativeModelPackKind
) -> URL {
    LTX25NativeModelPack.optimizedURLIfValid(resources: resources, kind: kind)
        ?? (kind == .dev ? resources.devTransformerURL : resources.distilledTransformerURL)
}

func loadLTX25TransformerWeights(
    url: URL,
    model: Module,
    dtype: DType,
    sourceInclude: (String) -> Bool = { $0.hasPrefix("model.diffusion_model.") },
    nativeInclude: (String) -> Bool = { !isLTX25ConnectorTensorKey($0) }
) throws {
    let isNative = LTX25NativeModelPack.isNativePack(url)
    if isNative {
        try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
            url: url,
            to: model,
            verify: .none,
            include: nativeInclude,
            mapper: { key, value in
                let casted = value.dtype.isFloatingPoint && value.dtype != dtype
                    ? value.asType(dtype)
                    : value
                return [(key, casted)]
            },
            batchSize: 24
        )
        return
    }
    try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(
        url: url,
        to: model,
        verify: .none,
        include: sourceInclude,
        mapper: { key, value in
            mapUnifiedTransformerWeight(key: key, value: value, dtype: dtype)
        },
        batchSize: 24
    )
}

func ltx25UnifiedTransformerParameterShapes() -> [String: [Int]] {
    Dictionary(
        uniqueKeysWithValues: LTXUnifiedAVTransformerV2().parameters().flattened().map {
            ($0.0, $0.1.shape)
        }
    )
}

func loadLTX25UnifiedTransformerParametersForValidation(
    url: URL,
    dtype: DType
) throws -> [(String, MLXArray)] {
    let model = LTXUnifiedAVTransformerV2()
    try loadLTX25TransformerWeights(url: url, model: model, dtype: dtype)
    return model.parameters().flattened()
}

func mapLTX23UnifiedTransformerWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    guard key.hasPrefix("transformer."),
          let mapped = mapLTX23UnifiedTransformerKey(key) else {
        return []
    }

    var casted = value
    if casted.dtype.isFloatingPoint && casted.dtype != dtype {
        casted = casted.asType(dtype)
    }
    return [(mapped, casted)]
}

func mapLTX23UnifiedTransformerKey(_ key: String) -> String? {
    var mapped: String
    if key.hasPrefix("transformer.") {
        mapped = String(key.dropFirst("transformer.".count))
    } else if key.hasPrefix("diffusion_model.") {
        mapped = String(key.dropFirst("diffusion_model.".count))
    } else {
        return nil
    }
    mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
    mapped = mapped.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
    mapped = mapped.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
    mapped = mapped.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

    let ignoredPrefixes = [
        "text_embedding_projection",
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "caption_projection",
        "audio_caption_projection",
    ]
    if ignoredPrefixes.contains(where: { mapped.hasPrefix($0) }) {
        return nil
    }
    return mapped
}
