import Foundation
@preconcurrency import MLX

public struct MoGe2FocalShiftSolution: Equatable, Sendable {
    /// Focal length relative to half the image diagonal.
    public let focal: Double
    /// Z translation applied before metric scaling.
    public let shift: Double
    public let iterationCount: Int
    public let residualMeanSquare: Double
}

public enum MoGe2FocalShiftSolver {
    public static func solve(
        affinePoints: [Float],
        validity: [UInt8],
        width: Int,
        height: Int,
        knownFocal: Double? = nil,
        downsampleWidth: Int = 64,
        downsampleHeight: Int = 64,
        maximumIterations: Int = 80
    ) -> MoGe2FocalShiftSolution {
        let targetWidth = min(width, max(1, downsampleWidth))
        let targetHeight = min(height, max(1, downsampleHeight))
        let aspect = Double(width) / Double(height)
        let spanX = aspect / sqrt(1 + aspect * aspect)
        let spanY = 1 / sqrt(1 + aspect * aspect)
        var samples: [(u: Double, v: Double, x: Double, y: Double, z: Double)] = []
        samples.reserveCapacity(targetWidth * targetHeight)

        for outputY in 0..<targetHeight {
            let sourceY = min(height - 1, Int(floor(Double(outputY) * Double(height) / Double(targetHeight))))
            let v = targetHeight == 1
                ? 0
                : -spanY * Double(targetHeight - 1) / Double(targetHeight)
                    + 2 * spanY * Double(targetHeight - 1) / Double(targetHeight)
                    * Double(outputY) / Double(targetHeight - 1)
            for outputX in 0..<targetWidth {
                let sourceX = min(width - 1, Int(floor(Double(outputX) * Double(width) / Double(targetWidth))))
                let pixel = sourceY * width + sourceX
                guard validity[pixel] != 0 else { continue }
                let base = pixel * 3
                let x = Double(affinePoints[base])
                let y = Double(affinePoints[base + 1])
                let z = Double(affinePoints[base + 2])
                guard x.isFinite, y.isFinite, z.isFinite else { continue }
                let u = targetWidth == 1
                    ? 0
                    : -spanX * Double(targetWidth - 1) / Double(targetWidth)
                        + 2 * spanX * Double(targetWidth - 1) / Double(targetWidth)
                        * Double(outputX) / Double(targetWidth - 1)
                samples.append((u, v, x, y, z))
            }
        }
        guard samples.count >= 2 else {
            return MoGe2FocalShiftSolution(
                focal: knownFocal ?? 1,
                shift: 0,
                iterationCount: 0,
                residualMeanSquare: 0
            )
        }

        func focal(at shift: Double) -> Double {
            if let knownFocal { return knownFocal }
            var numerator = 0.0
            var denominator = 0.0
            for sample in samples {
                let divisor = safeDivisor(sample.z + shift)
                let px = sample.x / divisor
                let py = sample.y / divisor
                numerator += px * sample.u + py * sample.v
                denominator += px * px + py * py
            }
            return denominator > 1e-18 ? numerator / denominator : 1
        }

        func residuals(at shift: Double) -> [Double] {
            let f = focal(at: shift)
            var result: [Double] = []
            result.reserveCapacity(samples.count * 2)
            for sample in samples {
                let divisor = safeDivisor(sample.z + shift)
                result.append(f * sample.x / divisor - sample.u)
                result.append(f * sample.y / divisor - sample.v)
            }
            return result
        }

        var shift = 0.0
        var residual = residuals(at: shift)
        var objective = residual.reduce(0) { $0 + $1 * $1 }
        var damping = 1e-3
        var completedIterations = 0
        for iteration in 0..<maximumIterations {
            completedIterations = iteration + 1
            let step = 1e-4 * max(1, abs(shift))
            let plus = residuals(at: shift + step)
            let minus = residuals(at: shift - step)
            var jtj = 0.0
            var jtr = 0.0
            for index in residual.indices {
                let derivative = (plus[index] - minus[index]) / (2 * step)
                jtj += derivative * derivative
                jtr += derivative * residual[index]
            }
            guard jtj.isFinite, jtj > 1e-18 else { break }
            let delta = -jtr / (jtj + damping)
            guard delta.isFinite else { break }
            let candidate = shift + delta
            let candidateResidual = residuals(at: candidate)
            let candidateObjective = candidateResidual.reduce(0) { $0 + $1 * $1 }
            if candidateObjective < objective {
                let previousObjective = objective
                shift = candidate
                residual = candidateResidual
                objective = candidateObjective
                damping = max(1e-12, damping * 0.3)
                if abs(delta) <= 1e-7 * max(1, abs(shift))
                    || abs(previousObjective - objective) <= 1e-6 * max(1, previousObjective) {
                    break
                }
            } else {
                damping = min(1e12, damping * 10)
            }
        }
        return MoGe2FocalShiftSolution(
            focal: focal(at: shift),
            shift: shift,
            iterationCount: completedIterations,
            residualMeanSquare: objective / Double(residual.count)
        )
    }

    private static func safeDivisor(_ value: Double) -> Double {
        if abs(value) >= 1e-9 { return value }
        return value < 0 ? -1e-9 : 1e-9
    }
}

public struct MoGe2PostprocessResult: Sendable {
    public let frame: DenseGeometryFrame
    public let focalShift: MoGe2FocalShiftSolution
    public let metricScale: Double
}

public enum MoGe2Postprocessor {
    public static func process(_ output: MoGe2RawOutput) throws -> MoGe2PostprocessResult {
        precondition(output.points.dim(0) == 1)
        MLX.eval(output.points, output.normals, output.maskProbability, output.metricScale)
        let height = output.points.dim(1)
        let width = output.points.dim(2)
        let affinePoints = output.points.asType(.float32).asArray(Float.self)
        let rawNormals = output.normals.asType(.float32).asArray(Float.self)
        let probabilities = output.maskProbability.asType(.float32).asArray(Float.self)
        let metricScale = Double(output.metricScale.asType(.float32).item(Float.self))
        var validity = probabilities.map { $0 > 0.5 ? UInt8(1) : UInt8(0) }
        let solution = MoGe2FocalShiftSolver.solve(
            affinePoints: affinePoints,
            validity: validity,
            width: width,
            height: height
        )

        let aspect = Double(width) / Double(height)
        let diagonalFactor = sqrt(1 + aspect * aspect)
        let fx = solution.focal / 2 * diagonalFactor / aspect
        let fy = solution.focal / 2 * diagonalFactor
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: width,
            imageHeight: height,
            normalizedFX: fx,
            normalizedFY: fy
        )
        var depth = [Float](repeating: .infinity, count: width * height)
        var normals = rawNormals
        for pixel in 0..<(width * height) {
            let z = (Double(affinePoints[pixel * 3 + 2]) + solution.shift) * metricScale
            if validity[pixel] != 0, z.isFinite, z > 0 {
                depth[pixel] = Float(z)
            } else {
                validity[pixel] = 0
                normals[pixel * 3] = 0
                normals[pixel * 3 + 1] = 0
                normals[pixel * 3 + 2] = 0
            }
        }
        let points = try GeometryProjection.pointMap(depth: depth, validity: validity, intrinsics: intrinsics)
        let frame = try DenseGeometryFrame(
            width: width,
            height: height,
            units: .meters,
            intrinsics: intrinsics,
            depth: depth,
            points: points,
            normals: normals,
            validity: validity,
            confidence: probabilities
        )
        return MoGe2PostprocessResult(frame: frame, focalShift: solution, metricScale: metricScale)
    }
}
