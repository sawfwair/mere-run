import MLX
import MLXNN
import XCTest
@testable import MereRunCore

final class MLXCheckpointTests: MereRunCoreTestCase {
    private final class ScaleModule: Module {
        @ParameterInfo(key: "scale") var scale: MLXArray

        override init() {
            self._scale.wrappedValue = MLXArray(Float(2))
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x * scale
        }
    }

    func testModuleCheckpointPreservesTrainableParameterGradients() {
        let module = ScaleModule()
        let checkpointed = checkpoint(model: module) { module, inputs in
            [module(inputs[0])]
        }
        let valueAndGrad = valueAndGrad(model: module) { _, inputs in
            [sum(checkpointed([inputs[0]])[0])]
        }

        let (_, gradients) = valueAndGrad(module, [MLXArray([Float(3)])])
        let scaleGradient = gradients.flattened().first { key, _ in
            key == "scale"
        }?.1.item(Float.self)

        XCTAssertEqual(try XCTUnwrap(scaleGradient), Float(3), accuracy: Float(0.00001))
    }
}
