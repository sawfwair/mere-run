import Foundation
import MLX
import MLXNN

extension MiniMaxH3TurboAdapter {
    static func installFastVideoCompressionGates(
        url: URL,
        into transformer: MiniMaxH3Transformer
    ) throws -> Int {
        var blockIndices: Set<Int> = []
        return try SafetensorsStreamingLoader.forEachTensor(
            url: url,
            where: { $0.hasSuffix(".attn.to_gate_compress.set_weight") }
        ) { sourceKey, weight in
            let components = sourceKey.split(separator: ".")
            guard components.count == 5,
                  components[0] == "transformer_blocks",
                  let blockIndex = Int(components[1]),
                  components[2] == "attn",
                  components[3] == "to_gate_compress",
                  components[4] == "set_weight",
                  (0..<transformer.configuration.layerCount).contains(blockIndex) else {
                throw AdapterError.unsupportedSourceModule(sourceKey)
            }
            guard blockIndices.insert(blockIndex).inserted else {
                throw AdapterError.duplicateTarget(sourceKey)
            }
            let expected = [
                transformer.configuration.attentionHeadCount
                    * transformer.configuration.attentionHeadDimension,
                transformer.configuration.hiddenSize,
            ]
            guard weight.shape == expected else {
                throw AdapterError.targetShapeMismatch(
                    sourceKey,
                    expected: expected,
                    actual: weight.shape
                )
            }
            transformer.installFastH3CompressionGate(weight, blockIndex: blockIndex)
        }
    }

    static func installFastH3QuantizedCompressionGates(
        url: URL,
        into transformer: MiniMaxH3Transformer
    ) throws -> Int {
        let metadata = try SafetensorsStreamingLoader.fileMetadata(url: url)
        guard metadata["gate_quantization"] == "affine 8-bit g64" else {
            throw AdapterError.unrecognizedFormat(url)
        }
        let arrays = try SafetensorsStreamingLoader.loadArrays(
            url: url,
            where: { $0.hasPrefix("transformer_blocks.") }
        )
        var installed = 0
        for blockIndex in 0..<transformer.configuration.layerCount {
            let prefix = "transformer_blocks.\(blockIndex).attn.to_gate_compress"
            guard let codes = arrays["\(prefix).weight"],
                  let scales = arrays["\(prefix).scales"],
                  let biases = arrays["\(prefix).biases"] else {
                throw AdapterError.missingTargetParameter(prefix)
            }
            transformer.installFastH3QuantizedCompressionGate(
                codes: codes,
                scales: scales,
                biases: biases,
                groupSize: 64,
                bits: 8,
                blockIndex: blockIndex
            )
            installed += 1
        }
        guard arrays.count == installed * 3 else {
            throw AdapterError.unexpectedAuxiliaryTensorCount(
                kind: "quantized-compression-gate tensor",
                expected: installed * 3,
                actual: arrays.count
            )
        }
        return installed
    }

}
