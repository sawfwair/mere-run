#if os(macOS)
import MereRunTensor
import MLX

/// Immutable native state can remain pending while the next draft runs.
protocol Q35NativeLinearState: AnyObject {
    var convolutionBuffer: M5NativeArrayBuffer { get }
    var recurrentBuffer: M5NativeArrayBuffer { get }
    func waitUntilReady() throws
    func materialized() throws -> (convolution: MLXArray, recurrent: MLXArray)
}

extension Q35NativeLinearState {
    func exported() -> (convolution: MLXArray, recurrent: MLXArray) {
        do { return try materialized() }
        catch { preconditionFailure("Native recurrent state failed: \(error)") }
    }
}

protocol Q35NativeLinearReplay: AnyObject {
    var legacy: Q35LinearVerificationReplay { get }
    var gates: (g: MLXArray, beta: MLXArray) { get }
    func acceptedState(path: [Int]) throws -> any Q35NativeLinearState
}
#endif
