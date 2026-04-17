import Foundation
import MLX
import MLXFast
import MLXNN
import AudioCodecs
import MereRunCore

// Owns the public speech tokenizer surface and weight sanitization helpers.
// Decoder and encoder architecture blocks live in companion files so readers
// can start with the encode/decode API before diving into the internals.

// MARK: - Speech Tokenizer

final class Qwen3TTSSpeechTokenizer: Module {
    let config: Qwen3TTSTokenizerConfig
    let decodeUpsampleRate: Int

    @ModuleInfo(key: "decoder") var decoder: Qwen3TTSSpeechTokenizerDecoder
    @ModuleInfo(key: "encoder_model") var encoderModel: Qwen3TTSSpeechTokenizerEncoder?

    init(config: Qwen3TTSTokenizerConfig) {
        self.config = config
        self.decodeUpsampleRate = config.decodeUpsampleRate
        self._decoder.wrappedValue = Qwen3TTSSpeechTokenizerDecoder(config: config.decoderConfig)
        if let encoderConfig = config.encoderConfig {
            self._encoderModel.wrappedValue = Qwen3TTSSpeechTokenizerEncoder(
                config: encoderConfig,
                validNumQuantizers: config.encoderValidNumQuantizers
            )
        } else {
            self._encoderModel.wrappedValue = nil
        }
    }

    var hasEncoder: Bool { encoderModel != nil }

    func decode(_ audioCodes: MLXArray) -> (audio: MLXArray, lengths: MLXArray) {
        let codes = audioCodes.transposed(0, 2, 1)
        let wav = decoder.chunkedDecode(codes: codes).squeezed(axis: 1)
        let valid = (audioCodes[.ellipsis, 0] .> MLXArray(Int32(0))).asType(.int32)
        let lengths = valid.sum(axis: 1) * MLXArray(Int32(decodeUpsampleRate))
        return (wav, lengths)
    }

    /// Native reference-audio encoder path using Seanet -> Transformer -> Downsample -> Split RVQ.
    func encode(samples: [Float], sampleRate: Int) -> MLXArray {
        guard let encoderModel else {
            return MLXArray.zeros([1, 1, max(1, config.encoderValidNumQuantizers)], dtype: .int32)
        }

        let targetSampleRate = max(1, config.inputSampleRate)
        let prepared: [Float]
        if sampleRate == targetSampleRate {
            prepared = samples
        } else {
            prepared = PCMStreamConverter.resampleLinear(
                samples,
                from: sampleRate,
                to: targetSampleRate
            )
        }

        guard !prepared.isEmpty else {
            return MLXArray.zeros([1, 1, max(1, config.encoderValidNumQuantizers)], dtype: .int32)
        }

        let audio = MLXArray(prepared).reshaped(1, 1, prepared.count).asType(.float32)
        let codes = encoderModel.encode(audio) // [B, Q, T]
        let transposed = codes.transposed(0, 2, 1).asType(.int32) // [B, T, Q]
        MLX.eval(transposed)
        return transposed
    }

    static func sanitize(_ weights: [String: MLXArray], config: Qwen3TTSTokenizerConfig) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var decoderCodebookData: [String: (cluster: MLXArray?, sum: MLXArray?)] = [:]
        var encoderTransformerQKV: [Int: (q: MLXArray?, k: MLXArray?, v: MLXArray?)] = [:]
        var encoderCodebookData: [String: (cluster: MLXArray?, sum: MLXArray?)] = [:]

        let hasEncoder = config.encoderConfig != nil
        let seanetConvMap: [Int: String] = [
            0: "encoder_model.encoder.init_conv1d",
            3: "encoder_model.encoder.layers.0.downsample",
            6: "encoder_model.encoder.layers.1.downsample",
            9: "encoder_model.encoder.layers.2.downsample",
            12: "encoder_model.encoder.layers.3.downsample",
            14: "encoder_model.encoder.final_conv1d"
        ]
        let seanetResidualMap: [Int: Int] = [1: 0, 4: 1, 7: 2, 10: 3]
        let seanetBlockMap: [Int: Int] = [1: 0, 3: 1]

        for (key, value) in weights {
            if key.hasPrefix("encoder_model.") {
                sanitized[key] = value
                continue
            }

            if !hasEncoder, key.hasPrefix("encoder.") {
                continue
            }

            if hasEncoder, key.hasPrefix("encoder.") {
                if key.hasPrefix("encoder.encoder.layers.") {
                    let parts = key.split(separator: ".").map(String.init)
                    guard parts.count >= 5, let n = Int(parts[3]) else { continue }

                    let mappedBase: String
                    let suffix: String
                    if key.contains(".block.") {
                        guard let layerIdx = seanetResidualMap[n], parts.count >= 7, let blockIdx = Int(parts[5]), let convIdx = seanetBlockMap[blockIdx] else {
                            continue
                        }
                        mappedBase = "encoder_model.encoder.layers.\(layerIdx).residuals.0.block.\(convIdx)"
                        suffix = parts.dropFirst(6).joined(separator: ".")
                    } else {
                        guard let base = seanetConvMap[n] else { continue }
                        mappedBase = base
                        suffix = parts.dropFirst(4).joined(separator: ".")
                    }

                    var mappedValue = value
                    if suffix.contains("weight"), mappedValue.ndim == 3 {
                        mappedValue = mappedValue.transposed(0, 2, 1)
                    }
                    sanitized["\(mappedBase).conv.\(suffix)"] = mappedValue
                    continue
                }

                if key.hasPrefix("encoder.encoder_transformer.layers.") {
                    let parts = key.split(separator: ".").map(String.init)
                    guard parts.count >= 5, let layerIdx = Int(parts[3]) else { continue }
                    let rest = parts.dropFirst(4).joined(separator: ".")

                    if rest == "self_attn.q_proj.weight" {
                        var qkv = encoderTransformerQKV[layerIdx] ?? (q: nil, k: nil, v: nil)
                        qkv.q = value
                        encoderTransformerQKV[layerIdx] = qkv
                    } else if rest == "self_attn.k_proj.weight" {
                        var qkv = encoderTransformerQKV[layerIdx] ?? (q: nil, k: nil, v: nil)
                        qkv.k = value
                        encoderTransformerQKV[layerIdx] = qkv
                    } else if rest == "self_attn.v_proj.weight" {
                        var qkv = encoderTransformerQKV[layerIdx] ?? (q: nil, k: nil, v: nil)
                        qkv.v = value
                        encoderTransformerQKV[layerIdx] = qkv
                    } else if rest == "self_attn.o_proj.weight" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.out_proj.weight"] = value
                    } else if rest == "mlp.fc1.weight" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear1.weight"] = value
                    } else if rest == "mlp.fc2.weight" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).gating.linear2.weight"] = value
                    } else if rest == "input_layernorm.weight" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.weight"] = value
                    } else if rest == "input_layernorm.bias" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm1.bias"] = value
                    } else if rest == "post_attention_layernorm.weight" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.weight"] = value
                    } else if rest == "post_attention_layernorm.bias" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).norm2.bias"] = value
                    } else if rest == "self_attn_layer_scale.scale" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_1.scale"] = value
                    } else if rest == "mlp_layer_scale.scale" {
                        sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).layer_scale_2.scale"] = value
                    }
                    continue
                }

                if key.hasPrefix("encoder.downsample.") {
                    var mappedValue = value
                    let suffix = String(key.dropFirst("encoder.downsample.".count))
                    if suffix.contains("weight"), mappedValue.ndim == 3 {
                        mappedValue = mappedValue.transposed(0, 2, 1)
                    }
                    sanitized["encoder_model.downsample.conv.conv.\(suffix)"] = mappedValue
                    continue
                }

                if key.hasPrefix("encoder.quantizer.") {
                    let rest = String(key.dropFirst("encoder.quantizer.".count))
                    let clusterSuffixes = [".codebook.cluster_usage", ".codebook.embedding_sum", ".codebook.embed_sum"]
                    if let matchedSuffix = clusterSuffixes.first(where: { rest.hasSuffix($0) }) {
                        let base = String(rest.dropLast(matchedSuffix.count))
                        var entry = encoderCodebookData[base] ?? (nil, nil)
                        if matchedSuffix.contains("cluster_usage") {
                            entry.cluster = value
                        } else {
                            entry.sum = value
                        }
                        encoderCodebookData[base] = entry
                        continue
                    }

                    if rest.contains(".codebook.initialized") {
                        continue
                    }

                    if rest.contains("input_proj.weight") || rest.contains("output_proj.weight") {
                        let projType = rest.contains("input_proj.weight") ? "input_proj" : "output_proj"
                        let isSemantic = rest.contains("semantic_residual_vector_quantizer")
                        let rvqPath = isSemantic ? "rvq_first" : "rvq_rest"
                        var mappedValue = value
                        if mappedValue.ndim == 3 {
                            mappedValue = mappedValue.transposed(0, 2, 1)
                        }
                        sanitized["encoder_model.quantizer.\(rvqPath).\(projType).weight"] = mappedValue
                    }
                    continue
                }

                continue
            }

            if key.contains("_codebook.cluster_usage") || key.contains("_codebook.embedding_sum") || key.contains("_codebook.embed_sum") {
                let base = key
                    .replacingOccurrences(of: "._codebook.cluster_usage", with: "")
                    .replacingOccurrences(of: "._codebook.embedding_sum", with: "")
                    .replacingOccurrences(of: "._codebook.embed_sum", with: "")
                var entry = decoderCodebookData[base] ?? (nil, nil)
                if key.contains("cluster_usage") {
                    entry.cluster = value
                } else {
                    entry.sum = value
                }
                decoderCodebookData[base] = entry
                continue
            }

            var newValue = value
            let isTransposeConv = (key.contains("upsample") && key.contains(".0.conv.weight")) ||
                (key.contains("decoder.decoder") && key.contains("block.1.conv.weight"))

            if isTransposeConv && value.ndim == 3 {
                if !checkArrayShapeQwen3(value) {
                    newValue = value.transposed(1, 2, 0)
                }
            } else if key.contains("conv.weight") && value.ndim == 3 {
                if !checkArrayShapeQwen3(value) {
                    newValue = value.transposed(0, 2, 1)
                }
            } else if key.contains("_proj.weight") && value.ndim == 3 {
                if !checkArrayShapeQwen3(value) {
                    newValue = value.transposed(0, 2, 1)
                }
            }

            sanitized[key] = newValue
        }

        for (layerIdx, qkv) in encoderTransformerQKV {
            guard let q = qkv.q, let k = qkv.k, let v = qkv.v else { continue }
            sanitized["encoder_model.encoder_transformer.transformer.layers.\(layerIdx).self_attn.in_proj.weight"] =
                MLX.concatenated([q, k, v], axis: 0)
        }

        let eps = MLXArray(Float(1e-5))
        for (base, data) in encoderCodebookData {
            guard let cluster = data.cluster, let sum = data.sum else { continue }

            let pathParts = base.split(separator: ".").map(String.init)
            guard let layerMarker = pathParts.firstIndex(of: "layers"), layerMarker + 1 < pathParts.count, let layerIndex = Int(pathParts[layerMarker + 1]) else {
                continue
            }

            let rvqPath: String
            if base.contains("semantic_residual_vector_quantizer") {
                rvqPath = "rvq_first"
            } else if base.contains("acoustic_residual_vector_quantizer") {
                rvqPath = "rvq_rest"
            } else {
                continue
            }

            let denom = MLX.maximum(cluster.reshaped(-1, 1), eps)
            let embedding = sum / denom
            sanitized["encoder_model.quantizer.\(rvqPath).vq.layers.\(layerIndex).codebook.embed.weight"] = embedding
        }

        for (base, data) in decoderCodebookData {
            guard let cluster = data.cluster, let sum = data.sum else { continue }
            let denom = MLX.maximum(cluster.reshaped(-1, 1), eps)
            let embedding = sum / denom
            sanitized["\(base).codebook.embed.weight"] = embedding
        }

        return sanitized
    }

}

private func checkArrayShapeQwen3(_ array: MLXArray) -> Bool {
    guard array.ndim == 3 else { return false }
    let dim2 = array.dim(1)
    let dim3 = array.dim(2)

    if dim2 == 1 {
        return dim3 > 64
    } else if dim3 == 1 {
        return dim2 <= 64
    }

    return dim2 < dim3
}
