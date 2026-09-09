import Foundation
import MLX
import MLXFast

enum Gemma4PolarCodebook {
    static func centroids(bits: Int, dim: Int) -> MLXArray {
        MLXArray(values(for: bits).map { $0 / sqrt(Float32(dim)) }, [1 << bits]).asType(.float32)
    }

    static func innerBoundaries(bits: Int, dim: Int) -> MLXArray {
        let values = boundaries(for: bits)
        let scale = Float32(1) / sqrt(Float32(dim))
        return MLXArray(values.dropFirst().dropLast().map { $0 * scale }, [(1 << bits) - 1]).asType(.float32)
    }

    private static func values(for bits: Int) -> [Float32] {
        switch bits {
        case 2:
            return [-1.5104, -0.4528, 0.4528, 1.5104]
        case 3:
            return [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519]
        case 4:
            return [
                -2.7331, -2.0698, -1.6189, -1.2570, -0.9431, -0.6573, -0.3884, -0.1285,
                0.1285, 0.3884, 0.6573, 0.9431, 1.2570, 1.6189, 2.0698, 2.7331,
            ]
        default:
            preconditionFailure("Unsupported PolarKV bit width \(bits).")
        }
    }

    private static func boundaries(for bits: Int) -> [Float32] {
        switch bits {
        case 2:
            return [-5.0, -0.9816, 0.0, 0.9816, 5.0]
        case 3:
            return [-5.0, -1.7479, -1.0499, -0.5005, 0.0, 0.5005, 1.0499, 1.7479, 5.0]
        case 4:
            return [
                -5.0, -2.4015, -1.8443, -1.4380, -1.1001, -0.8002, -0.5229, -0.2585,
                0.0, 0.2585, 0.5229, 0.8002, 1.1001, 1.4380, 1.8443, 2.4015, 5.0,
            ]
        default:
            preconditionFailure("Unsupported PolarKV bit width \(bits).")
        }
    }
}

enum Gemma4PolarRotation {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var matrices: [Int: MLXArray] = [:]

    static func matrix(dim: Int) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }

        if let existing = matrices[dim] {
            return existing
        }

        let scale = Float32(1) / sqrt(Float32(dim))
        let values: [Float32]
        // The Python prototype uses seeded QR; a deterministic Hadamard rotation
        // keeps this native prototype cheap to build for Gemma's power-of-two head dims.
        if dim > 0 && (dim & (dim - 1)) == 0 {
            values = (0..<dim).flatMap { row in
                (0..<dim).map { column in
                    ((row & column).nonzeroBitCount.isMultiple(of: 2) ? scale : -scale)
                }
            }
        } else {
            values = (0..<dim).flatMap { row in
                (0..<dim).map { column in row == column ? Float32(1) : Float32(0) }
            }
        }

        let matrix = MLXArray(values, [dim, dim]).asType(.float32)
        matrices[dim] = matrix
        return matrix
    }
}
