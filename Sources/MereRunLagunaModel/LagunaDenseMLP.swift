import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package class LagunaFeedForward: Module {
    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("LagunaFeedForward subclasses must implement callAsFunction(_:).")
    }
}

package final class LagunaDenseMLP: LagunaFeedForward {
    @ModuleInfo(key: "gate_proj") package var gateProj: Linear
    @ModuleInfo(key: "up_proj") package var upProj: Linear
    @ModuleInfo(key: "down_proj") package var downProj: Linear

    package init(inputDimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._upProj.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, inputDimensions, bias: false)
        super.init()
    }

    package override func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }

    package func lagunaXSDecodeDownInputs(
        _ x: MLXArray
    ) -> (activated: MLXArray, weight: MLXArray, scales: MLXArray)? {
        guard x.dtype == .bfloat16,
              x.shape == [1, 1, 2_048],
              (type(of: gateProj) == QuantizedLinear.self
                || type(of: gateProj) == PortableQuantizedLinear.self),
              (type(of: upProj) == QuantizedLinear.self
                || type(of: upProj) == PortableQuantizedLinear.self),
              (type(of: downProj) == QuantizedLinear.self
                || type(of: downProj) == PortableQuantizedLinear.self),
              let gate = gateProj as? QuantizedLinear,
              let up = upProj as? QuantizedLinear,
              let down = downProj as? QuantizedLinear,
              gate.mode == .nvfp4,
              up.mode == .nvfp4,
              down.mode == .nvfp4,
              gate.groupSize == 16,
              up.groupSize == 16,
              down.groupSize == 16,
              gate.bits == 4,
              up.bits == 4,
              down.bits == 4,
              gate.bias == nil,
              up.bias == nil,
              down.bias == nil,
              gate.biases == nil,
              up.biases == nil,
              down.biases == nil,
              gate.weight.shape == [512, 256],
              up.weight.shape == [512, 256],
              down.weight.shape == [2_048, 64],
              gate.scales.shape == [512, 128],
              up.scales.shape == [512, 128],
              down.scales.shape == [2_048, 32],
              down.weight.dtype == .uint32,
              down.scales.dtype == .uint8 else {
            return nil
        }
        return (
            MLXNN.silu(gate(x)) * up(x),
            down.weight,
            down.scales
        )
    }
}
