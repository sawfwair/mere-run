import MLX
import XCTest

final class MLXStreamConcurrencyContractTests: MereRunCoreTestCase {
    func testAsyncDefaultStreamKeepsCPUAndGPUGraphsValidAcrossExecutorHops() async {
        let graphs = await Stream.withNewDefaultStream {
            let cpuGraph = MLX.multiply(
                MLX.ones([3], type: Float.self, stream: .cpu),
                2,
                stream: .cpu
            )
            let gpuGraph = MLX.multiply(
                MLX.ones([3], type: Float.self, stream: .gpu),
                3,
                stream: .gpu
            )
            await Task.yield()
            return (cpuGraph, gpuGraph)
        }

        let values = await Task.detached {
            MLX.eval(graphs.0, graphs.1)
            return (
                graphs.0.asArray(Float.self),
                graphs.1.asArray(Float.self)
            )
        }.value

        XCTAssertEqual(values.0, [2, 2, 2])
        XCTAssertEqual(values.1, [3, 3, 3])
    }
}
