import Foundation
import MLX
import MLXNN
import Cmlx

// MARK: - Internal Helpers (copied from mlx-swift Cmlx+Util.swift)

/// Create a +1 mlx_vector_array containing the given arrays
private func new_mlx_vector_array(_ arrays: [MLXArray]) -> mlx_vector_array {
    withExtendedLifetime(arrays) {
        mlx_vector_array_new_data(arrays.map { $0.ctx }, arrays.count)
    }
}

/// Extract MLXArray values from an mlx_vector_array
private func mlx_vector_array_values(_ vector_array: mlx_vector_array) -> [MLXArray] {
    (0 ..< mlx_vector_array_size(vector_array))
        .map { index in
            var ctx = mlx_array_new()
            mlx_vector_array_get(&ctx, vector_array, index)
            return MLXArray(ctx)
        }
}

/// Create an mlx_closure from a Swift function
private func new_mlx_closure(_ f: @escaping ([MLXArray]) -> [MLXArray]) -> mlx_closure {
    class ClosureCaptureState {
        let f: ([MLXArray]) -> [MLXArray]
        init(_ f: @escaping ([MLXArray]) -> [MLXArray]) {
            self.f = f
        }
    }

    func free(ptr: UnsafeMutableRawPointer?) {
        Unmanaged<ClosureCaptureState>.fromOpaque(ptr!).release()
    }

    let payload = Unmanaged.passRetained(ClosureCaptureState(f)).toOpaque()

    func trampoline(
        resultOut: UnsafeMutablePointer<mlx_vector_array>?,
        vector_array: mlx_vector_array,
        payload: UnsafeMutableRawPointer?
    ) -> Int32 {
        guard let payload, let resultOut else {
            return -1
        }

        let state = Unmanaged<ClosureCaptureState>.fromOpaque(payload).takeUnretainedValue()
        let arrays = mlx_vector_array_values(vector_array)
        let result = state.f(arrays)

        resultOut.pointee = new_mlx_vector_array(result)
        return 0
    }

    return mlx_closure_new_func_payload(trampoline, payload, free)
}

// MARK: - Gradient Checkpointing

/// Wraps a function to use gradient checkpointing.
///
/// During the forward pass, intermediate activations are not stored.
/// During the backward pass, the forward pass is recomputed to get activations.
/// This trades compute for memory - useful for training large models.
///
/// - Parameter f: The function to checkpoint
/// - Returns: A checkpointed version of the function with the same signature
public func checkpoint(
    _ f: @escaping ([MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    // Create the input closure
    let inputClosure = new_mlx_closure(f)

    // Create the checkpointed closure
    var checkpointedClosure = mlx_closure_new()
    mlx_checkpoint(&checkpointedClosure, inputClosure)
    mlx_closure_free(inputClosure)

    // Return a Swift function that uses the checkpointed closure
    return { (inputs: [MLXArray]) -> [MLXArray] in
        let inputs_mlx = new_mlx_vector_array(inputs)
        defer { mlx_vector_array_free(inputs_mlx) }

        var outputs = mlx_vector_array_new()
        mlx_closure_apply(&outputs, checkpointedClosure, inputs_mlx)
        defer { mlx_vector_array_free(outputs) }

        return mlx_vector_array_values(outputs)
    }
}

/// Wraps a module function to use gradient checkpointing with respect to both
/// module trainable parameters and the function inputs.
public func checkpoint<Model: Module>(
    model: Model,
    _ f: @escaping (Model, [MLXArray]) -> [MLXArray]
) -> ([MLXArray]) -> [MLXArray] {
    let parameterKeys = model.trainableParameters().flattened().map(\.0)
    guard !parameterKeys.isEmpty else {
        return checkpoint { [unowned model] inputs in
            f(model, inputs)
        }
    }

    let parameterCount = parameterKeys.count
    let checkpointed = checkpoint { [unowned model] inputs in
        precondition(
            inputs.count >= parameterCount,
            "Checkpointed module call received fewer inputs than trainable parameters."
        )
        let parameterPairs = parameterKeys.enumerated().map { index, key in
            (key, inputs[index])
        }
        model.update(parameters: ModuleParameters.unflattened(parameterPairs))
        return f(model, Array(inputs.dropFirst(parameterCount)))
    }

    return { [unowned model] inputs in
        let parameters = model.trainableParameters().flattened()
        precondition(
            parameters.count == parameterCount,
            "Checkpointed module trainable parameter structure changed after initialization."
        )
        return checkpointed(parameters.map(\.1) + inputs)
    }
}

/// Apply gradient checkpointing to a single-input, single-output function.
/// Convenience wrapper for the common case.
public func checkpoint(
    _ f: @escaping (MLXArray) -> MLXArray
) -> (MLXArray) -> MLXArray {
    let wrapped = checkpoint { (inputs: [MLXArray]) -> [MLXArray] in
        [f(inputs[0])]
    }
    return { input in wrapped([input])[0] }
}
