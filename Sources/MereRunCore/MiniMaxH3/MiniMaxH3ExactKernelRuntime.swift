import Foundation
import MediaIO
import MLX
#if canImport(Darwin)
import Darwin
#endif
#if canImport(IOKit.ps)
import IOKit.ps
#endif

extension MiniMaxH3ExactKernelMode {
    func requiresEagerExecution(sequenceLength: Int) -> Bool {
        switch self {
        case .disabled:
            false
        case .boundaryLayout:
            sequenceLength > MiniMaxH3DenoiseExecutionPolicy.blockwiseSequenceThreshold
        case .affineQ8, .affineQ8MLP, .fastH3Metal:
            true
        }
    }

    static func resolve(environmentValue: String?) throws -> Self {
        switch environmentValue?.lowercased() {
        case nil, "", "disabled": .disabled
        case MiniMaxH3ExactKernelMode.boundaryLayout.rawValue: .boundaryLayout
        case MiniMaxH3ExactKernelMode.affineQ8.rawValue: .affineQ8
        case MiniMaxH3ExactKernelMode.affineQ8MLP.rawValue: .affineQ8MLP
        case MiniMaxH3ExactKernelMode.fastH3Metal.rawValue: .fastH3Metal
        case .some(let value):
            throw MiniMaxH3GeneratorError.invalidOptions(
                "MERERUN_H3_EXACT_KERNELS must be disabled, boundary-layout, "
                    + "affine-q8, affine-q8-mlp, or fasth3-metal, not \(value)"
            )
        }
    }

    static func resolveForRuntime(
        environmentValue: String?,
        usesFastH3VSA: Bool,
        supportsAffineQ8ExactKernels: Bool,
        usesResidentBF16: Bool
    ) throws -> Self {
        let configured = try resolve(environmentValue: environmentValue)
        if environmentValue == nil,
           usesFastH3VSA,
           supportsAffineQ8ExactKernels,
           !usesResidentBF16 {
            return .fastH3Metal
        }
        return configured
    }
}
