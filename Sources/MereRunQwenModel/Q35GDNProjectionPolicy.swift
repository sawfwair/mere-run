import Foundation
import MereRunTensor
import MLX
import MLXNN

/// A separately qualified arithmetic change for the dense 27B fused GDN input.
package enum Q35GDNProjectionPolicy {
    @TaskLocal package static var matrixEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_GDN_MATRIX"] == "1"
    @TaskLocal package static var prefillMatrixEnabled =
        ProcessInfo.processInfo.environment["MERERUN_Q38_GDN_MATRIX_PREFILL"] == "1"
    @TaskLocal package static var matrixLayers = parseMatrixLayers(
        ProcessInfo.processInfo.environment["MERERUN_Q38_GDN_MATRIX_LAYERS"])
    static let matrixArithmetic: GroupAffineArithmetic = {
        switch ProcessInfo.processInfo.environment["MERERUN_Q38_GDN_MATRIX_ARITHMETIC"] {
        case "centered": .centered
        case "compensated": .compensated
        default: .standard
        }
    }()
    private static let logLock = NSLock()
    private nonisolated(unsafe) static var logged = false

    static func parseMatrixLayers(_ value: String?) -> Set<Int>? {
        guard let value else { return nil }
        let fields = value.split(separator: ",", omittingEmptySubsequences: false)
        let layers = fields.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard layers.count == fields.count, layers.allSatisfy((0..<64).contains) else { return [] }
        return Set(layers)
    }

    static func backend(_ lhs: PortableQuantizedLinear, _ rhs: PortableQuantizedLinear,
                        layerIndex: Int? = nil) -> QuantizedVerificationBackend {
        guard matrixEnabled, matrixLayers.map({ allowed in layerIndex.map(allowed.contains) ?? false }) ?? true,
              lhs.verificationBackend == .m5Matrix, rhs.verificationBackend == .m5Matrix,
              lhs.shape.1 == 5_120, lhs.shape.0 + rhs.shape.0 == 16_384,
              lhs.bias == nil, rhs.bias == nil, lhs.bits == 4, rhs.bits == 4,
              lhs.groupSize == 64, rhs.groupSize == 64, lhs.mode == .affine, rhs.mode == .affine else {
            return .reference
        }
        let report = logLock.withLock {
            if logged { return false }
            logged = true
            return true
        }
        if report {
            FileHandle.standardError.write(Data("[q35] gdn_matrix_projection=true prefill=\(prefillMatrixEnabled)\n".utf8))
        }
        return .m5Matrix
    }

    static func project(_ projection: Linear, _ input: MLXArray, layerIndex: Int? = nil) -> MLXArray {
        if QuantizedPrefillContext.mode.usesLateGDNTensor, let layerIndex, layerIndex >= 32 {
            return QuantizedPrefillContext.$mode.withValue(.gdnTensor) {
                projectWithVerificationPolicy(projection, input)
            }
        }
        return projectWithVerificationPolicy(projection, input)
    }

    private static func projectWithVerificationPolicy(_ projection: Linear, _ input: MLXArray) -> MLXArray {
        if let portable = projection as? PortableQuantizedLinear,
           portable.verificationBackend == .m5Matrix, input.size / input.dim(-1) > 16, !prefillMatrixEnabled {
            return QuantizedVerificationContext.$suppressM5.withValue(true) { projection(input) }
        }
        return projection(input)
    }
}
