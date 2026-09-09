import MLX
import MLXNN

package final class MiniMaxH3RuntimeLoRALinear: Linear {
    @ParameterInfo(key: "lora_down") package var loraDown: MLXArray
    @ParameterInfo(key: "lora_up") package var loraUp: MLXArray
    package let strength: Float

    package init(base: Linear, loraDown: MLXArray, loraUp: MLXArray, strength: Float) {
        self._loraDown.wrappedValue = loraDown.asType(base.weight.dtype)
        self._loraUp.wrappedValue = loraUp.asType(base.weight.dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MLX.matmul(
            MLX.matmul(input.asType(loraDown.dtype), loraDown.T),
            loraUp.T
        ) * MLXArray(strength).asType(loraDown.dtype)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}

package final class MiniMaxH3RuntimeQuantizedLoRALinear: QuantizedLinear {
    @ParameterInfo(key: "lora_down") package var loraDown: MLXArray
    @ParameterInfo(key: "lora_up") package var loraUp: MLXArray
    package let strength: Float

    package init(base: QuantizedLinear, loraDown: MLXArray, loraUp: MLXArray, strength: Float) {
        self._loraDown.wrappedValue = loraDown.asType(base.scales.dtype)
        self._loraUp.wrappedValue = loraUp.asType(base.scales.dtype)
        self.strength = strength
        super.init(
            weight: base.weight,
            bias: base.bias,
            scales: base.scales,
            biases: base.biases,
            groupSize: base.groupSize,
            bits: base.bits,
            mode: base.mode,
            globalScale: base.globalScale
        )
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MiniMaxH3RuntimeLoRAMath.project(
            input,
            down: loraDown,
            up: loraUp,
            strength: strength
        )
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }
}

package final class MiniMaxH3RuntimeQKVLoRALinear: Linear {
    @ParameterInfo(key: "query_down") package var queryDown: MLXArray
    @ParameterInfo(key: "query_up") package var queryUp: MLXArray
    @ParameterInfo(key: "key_down") package var keyDown: MLXArray
    @ParameterInfo(key: "key_up") package var keyUp: MLXArray
    @ParameterInfo(key: "value_down") package var valueDown: MLXArray
    @ParameterInfo(key: "value_up") package var valueUp: MLXArray
    package let strength: Float

    package init(
        base: Linear,
        queryDown: MLXArray,
        queryUp: MLXArray,
        keyDown: MLXArray,
        keyUp: MLXArray,
        valueDown: MLXArray,
        valueUp: MLXArray,
        strength: Float
    ) {
        let dtype = base.weight.dtype
        self._queryDown.wrappedValue = queryDown.asType(dtype)
        self._queryUp.wrappedValue = queryUp.asType(dtype)
        self._keyDown.wrappedValue = keyDown.asType(dtype)
        self._keyUp.wrappedValue = keyUp.asType(dtype)
        self._valueDown.wrappedValue = valueDown.asType(dtype)
        self._valueUp.wrappedValue = valueUp.asType(dtype)
        self.strength = strength
        super.init(weight: base.weight, bias: base.bias)
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        return baseOutput + adapterOutput(input).asType(baseOutput.dtype)
    }

    package func evaluateAdapterParameters() {
        MLX.eval(queryDown, queryUp, keyDown, keyUp, valueDown, valueUp)
    }

    private func adapterOutput(_ input: MLXArray) -> MLXArray {
        MLX.concatenated([
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: queryDown,
                up: queryUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: keyDown,
                up: keyUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: valueDown,
                up: valueUp,
                strength: strength
            ),
        ], axis: -1)
    }
}

package final class MiniMaxH3RuntimeQuantizedQKVLoRALinear: QuantizedLinear {
    @ParameterInfo(key: "query_down") package var queryDown: MLXArray
    @ParameterInfo(key: "query_up") package var queryUp: MLXArray
    @ParameterInfo(key: "key_down") package var keyDown: MLXArray
    @ParameterInfo(key: "key_up") package var keyUp: MLXArray
    @ParameterInfo(key: "value_down") package var valueDown: MLXArray
    @ParameterInfo(key: "value_up") package var valueUp: MLXArray
    package let strength: Float

    package init(
        base: QuantizedLinear,
        queryDown: MLXArray,
        queryUp: MLXArray,
        keyDown: MLXArray,
        keyUp: MLXArray,
        valueDown: MLXArray,
        valueUp: MLXArray,
        strength: Float
    ) {
        let dtype = base.scales.dtype
        self._queryDown.wrappedValue = queryDown.asType(dtype)
        self._queryUp.wrappedValue = queryUp.asType(dtype)
        self._keyDown.wrappedValue = keyDown.asType(dtype)
        self._keyUp.wrappedValue = keyUp.asType(dtype)
        self._valueDown.wrappedValue = valueDown.asType(dtype)
        self._valueUp.wrappedValue = valueUp.asType(dtype)
        self.strength = strength
        super.init(
            weight: base.weight,
            bias: base.bias,
            scales: base.scales,
            biases: base.biases,
            groupSize: base.groupSize,
            bits: base.bits,
            mode: base.mode,
            globalScale: base.globalScale
        )
    }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let baseOutput = super.callAsFunction(input)
        let adapterOutput = MLX.concatenated([
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: queryDown,
                up: queryUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: keyDown,
                up: keyUp,
                strength: strength
            ),
            MiniMaxH3RuntimeLoRAMath.project(
                input,
                down: valueDown,
                up: valueUp,
                strength: strength
            ),
        ], axis: -1)
        return baseOutput + adapterOutput.asType(baseOutput.dtype)
    }

    package func evaluateAdapterParameters() {
        MLX.eval(queryDown, queryUp, keyDown, keyUp, valueDown, valueUp)
    }
}

private enum MiniMaxH3RuntimeLoRAMath {
    package static func project(
        _ input: MLXArray,
        down: MLXArray,
        up: MLXArray,
        strength: Float
    ) -> MLXArray {
        MLX.matmul(
            MLX.matmul(input.asType(down.dtype), down.T),
            up.T
        ) * MLXArray(strength).asType(down.dtype)
    }
}
