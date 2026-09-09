import Foundation
import MLX

func mapLTXDiffusionVideoDecoderWeight(
    key: String,
    value: MLXArray,
    dtype: DType
) -> [(String, MLXArray)] {
    let targets = ltxDiffusionVideoDecoderWeightTargets(key: key, shape: value.shape)
    guard !targets.isEmpty else { return [] }
    let casted = value.dtype.isFloatingPoint && value.dtype != dtype
        ? value.asType(dtype)
        : value
    if targets.count == 3 {
        let dimension = casted.dim(0) / 3
        return [
            (targets[0].name, casted[0..<dimension]),
            (targets[1].name, casted[dimension..<(dimension * 2)]),
            (targets[2].name, casted[(dimension * 2)...]),
        ]
    }
    return [(targets[0].name, casted)]
}

func ltxDiffusionVideoDecoderWeightTargets(
    key: String,
    shape: [Int]
) -> [(name: String, shape: [Int])] {
    guard key.hasPrefix("decoder."), key != "decoder.type_emb" else { return [] }
    var mapped = String(key.dropFirst("decoder.".count))
    if mapped.hasPrefix("t_embedder.mlp.0.") {
        mapped = mapped.replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.linear_1.")
    } else if mapped.hasPrefix("t_embedder.mlp.2.") {
        mapped = mapped.replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.linear_2.")
    }
    for stage in 0..<4 {
        mapped = mapped.replacingOccurrences(
            of: "det_stages.\(stage).",
            with: "det_stages.\(stage).blocks."
        )
    }
    if mapped.contains(".attn.qkv.weight") || mapped.contains(".attn.qkv.bias") {
        guard let firstDimension = shape.first, firstDimension % 3 == 0 else { return [] }
        let dimension = firstDimension / 3
        let qkvSuffix = mapped.hasSuffix("weight") ? "qkv.weight" : "qkv.bias"
        let prefix = String(mapped.dropLast(qkvSuffix.count))
        let suffix = mapped.hasSuffix("weight") ? "weight" : "bias"
        var splitShape = shape
        splitShape[0] = dimension
        return [
            ("\(prefix)to_q.\(suffix)", splitShape),
            ("\(prefix)to_k.\(suffix)", splitShape),
            ("\(prefix)to_v.\(suffix)", splitShape),
        ]
    }
    return [(mapped, shape)]
}
