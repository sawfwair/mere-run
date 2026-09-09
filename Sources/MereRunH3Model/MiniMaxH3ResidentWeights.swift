import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTensor
#if DEBUG
import MLXRandom
#endif

package struct MiniMaxH3ResidentBF16Materialization: Sendable, Equatable {
    package let linearCount: Int
    package let byteCount: UInt64
}

func miniMaxH3ResidentBF16Linear(
    _ linear: Linear
) -> (linear: Linear, byteCount: UInt64)? {
    guard let quantized = linear as? QuantizedLinear else { return nil }
    var weight = MLX.dequantized(
        quantized.weight,
        scales: quantized.scales,
        biases: quantized.biases,
        groupSize: quantized.groupSize,
        bits: quantized.bits,
        mode: quantized.mode,
        dtype: .bfloat16
    )
    if let globalScale = quantized.globalScale {
        weight = (weight * (globalScale / (448 * 6))).asType(.bfloat16)
    }
    let bias = quantized.bias?.asType(.bfloat16)
    if let bias {
        MLX.eval(weight, bias)
    } else {
        MLX.eval(weight)
    }
    let shape = quantized.shape
    let weightBytes = UInt64(shape.0) * UInt64(shape.1) * 2
    let biasBytes = UInt64(quantized.bias?.size ?? 0) * 2
    let base = Linear(weight: weight, bias: bias)
    if let lora = quantized as? MiniMaxH3RuntimeQuantizedLoRALinear {
        let adapterBytes = UInt64(lora.loraDown.size + lora.loraUp.size) * 2
        return (
            MiniMaxH3RuntimeLoRALinear(
                base: base,
                loraDown: lora.loraDown,
                loraUp: lora.loraUp,
                strength: lora.strength
            ),
            weightBytes + biasBytes + adapterBytes
        )
    }
    if let lora = quantized as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
        let adapterCount = lora.queryDown.size + lora.queryUp.size
            + lora.keyDown.size + lora.keyUp.size
            + lora.valueDown.size + lora.valueUp.size
        return (
            MiniMaxH3RuntimeQKVLoRALinear(
                base: base,
                queryDown: lora.queryDown,
                queryUp: lora.queryUp,
                keyDown: lora.keyDown,
                keyUp: lora.keyUp,
                valueDown: lora.valueDown,
                valueUp: lora.valueUp,
                strength: lora.strength
            ),
            weightBytes + biasBytes + UInt64(adapterCount) * 2
        )
    }
    return (base, weightBytes + biasBytes)
}

func miniMaxH3MaterializeResidentBF16(
    in module: Module
) -> MiniMaxH3ResidentBF16Materialization {
    var replacements: [(String, Module)] = []
    var byteCount: UInt64 = 0
    for (path, leaf) in module.leafModules().flattened() {
        guard let linear = leaf as? Linear,
              let resident = miniMaxH3ResidentBF16Linear(linear) else { continue }
        replacements.append((path, resident.linear))
        byteCount += resident.byteCount
    }
    if !replacements.isEmpty {
        module.update(modules: ModuleChildren.unflattened(replacements))
    }
    return .init(linearCount: replacements.count, byteCount: byteCount)
}

func miniMaxH3ResidentBF16ByteCount(_ linear: Linear) -> UInt64 {
    if let quantized = linear as? QuantizedLinear {
        let shape = quantized.shape
        var byteCount = UInt64(shape.0) * UInt64(shape.1) * 2
            + UInt64(quantized.bias?.size ?? 0) * 2
        if let lora = quantized as? MiniMaxH3RuntimeQuantizedLoRALinear {
            byteCount += UInt64(lora.loraDown.size + lora.loraUp.size) * 2
        } else if let lora = quantized as? MiniMaxH3RuntimeQuantizedQKVLoRALinear {
            byteCount += UInt64(
                lora.queryDown.size + lora.queryUp.size
                    + lora.keyDown.size + lora.keyUp.size
                    + lora.valueDown.size + lora.valueUp.size
            ) * 2
        }
        return byteCount
    }
    return linear.parameters().flattened().reduce(into: UInt64(0)) { total, entry in
        total += UInt64(entry.1.size) * 2
    }
}

func miniMaxH3EvaluateParameters(in module: Module) {
    let parameters = module.parameters().flattened().map(\.1)
    guard !parameters.isEmpty else { return }
    MLX.eval(parameters)
}

package struct MiniMaxH3RotaryEmbedding {
    package let cosine: MLXArray
    package let sine: MLXArray

    package func apply(_ value: MLXArray) -> MLXArray {
        let rotaryDimension = cosine.dim(-1)
        let rotary = value[0..., 0..., 0..., 0..<rotaryDimension]
        let passthrough = value[0..., 0..., 0..., rotaryDimension...]
        let halves = MLX.split(rotary, parts: 2, axis: -1)
        let rotated = MLX.concatenated([-halves[1], halves[0]], axis: -1)
        return MLX.concatenated(
            [rotary * cosine + rotated * sine, passthrough],
            axis: -1
        )
    }
}
