import MLX
import MLXNN

/// Keeps the base projection in its original precision, including packed MLX weights.
package final class Q35LoRALinear: Linear {
    private let base: Linear
    private let down: MLXArray
    private let up: MLXArray
    private let scale: Float

    package init(base: Linear, down: MLXArray, up: MLXArray, scale: Float) {
        self.base = base
        self.down = down
        self.up = up
        self.scale = scale
        super.init(weight: base.weight, bias: base.bias)
    }

    package override var shape: (Int, Int) { base.shape }

    package override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let output = base(input)
        let delta = (input.asType(down.dtype).matmul(down.T).matmul(up.T) * scale)
        return output + delta.asType(output.dtype)
    }
}
