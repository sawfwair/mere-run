import Foundation
import MLX
import MLXNN

extension MiniMaxH3TurboAdapter {
    static func targetLinear(
        at path: String,
        in modulesByPath: [String: Module]
    ) throws -> Linear {
        guard let module = modulesByPath[path] else {
            throw AdapterError.missingTargetModule(path)
        }
        guard let linear = module as? Linear else {
            throw AdapterError.targetIsNotLinear(path)
        }
        return linear
    }

    static func validate(
        _ path: String,
        down: MLXArray,
        up: MLXArray,
        base: Linear
    ) throws {
        guard down.dim(1) == base.shape.1,
              up.dim(0) == base.shape.0 else {
            throw AdapterError.targetShapeMismatch(
                path,
                expected: [base.shape.0, base.shape.1],
                actual: [up.dim(0), down.dim(1)]
            )
        }
    }

    static func validateQKV(
        _ path: String,
        query: LoRAPair,
        key: LoRAPair,
        value: LoRAPair,
        base: Linear
    ) throws {
        guard base.shape.0.isMultiple(of: 3) else {
            throw AdapterError.targetShapeMismatch(
                path,
                expected: [base.shape.0, base.shape.1],
                actual: [base.shape.0, base.shape.1]
            )
        }
        let branchOutputSize = base.shape.0 / 3
        for pair in [query, key, value] {
            guard pair.down.dim(1) == base.shape.1,
                  pair.up.dim(0) == branchOutputSize else {
                throw AdapterError.targetShapeMismatch(
                    path,
                    expected: [base.shape.0, base.shape.1],
                    actual: [3 * pair.up.dim(0), pair.down.dim(1)]
                )
            }
        }
    }

    static func runtimeLayer(
        base: Linear,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) -> Module {
        if let quantized = base as? QuantizedLinear {
            let layer = MiniMaxH3RuntimeQuantizedLoRALinear(
                base: quantized,
                loraDown: down,
                loraUp: up,
                strength: strength
            )
            MLX.eval(layer.loraDown, layer.loraUp)
            return layer
        }
        let layer = MiniMaxH3RuntimeLoRALinear(
            base: base,
            loraDown: down,
            loraUp: up,
            strength: strength
        )
        MLX.eval(layer.loraDown, layer.loraUp)
        return layer
    }

    static func runtimeQKVLayer(
        base: Linear,
        query: LoRAPair,
        key: LoRAPair,
        value: LoRAPair,
        strength: Float
    ) -> Module {
        if let quantized = base as? QuantizedLinear {
            let layer = MiniMaxH3RuntimeQuantizedQKVLoRALinear(
                base: quantized,
                queryDown: query.down,
                queryUp: query.up,
                keyDown: key.down,
                keyUp: key.up,
                valueDown: value.down,
                valueUp: value.up,
                strength: strength
            )
            layer.evaluateAdapterParameters()
            return layer
        }
        let layer = MiniMaxH3RuntimeQKVLoRALinear(
            base: base,
            queryDown: query.down,
            queryUp: query.up,
            keyDown: key.down,
            keyUp: key.up,
            valueDown: value.down,
            valueUp: value.up,
            strength: strength
        )
        layer.evaluateAdapterParameters()
        return layer
    }

}
