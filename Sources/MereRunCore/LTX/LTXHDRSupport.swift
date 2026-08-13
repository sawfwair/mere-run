import Foundation
import MediaIO
import MLX

public enum LTXHDRColorSpace: String, Sendable, CaseIterable, Hashable {
    case srgbLinear = "srgb-linear"
    case acescg
    case acescct
}

public enum LTXHDRTransfer: String, Sendable, CaseIterable, Hashable {
    case logC3 = "logc3"
    case acesCCT = "acescct"
}

public struct LTXHDROutputFrames: @unchecked Sendable {
    /// Compressed transfer codes in `[F, H, W, 3]`, suitable for an SDR preview.
    public let working: MLXArray
    /// EXR payload matching the requested source/output color-space declaration.
    public let exr: MLXArray
    /// BT.2020 RGB with the HLG OETF already applied.
    public let hlg: MLXArray
}

public struct LTXHDRICLoRAStage2Tiling: Sendable, Hashable {
    public let frameTiles: Int
    public let frameOverlap: Int
    public let heightTiles: Int
    public let heightOverlap: Int
    public let widthTiles: Int
    public let widthOverlap: Int

    public init(
        frameTiles: Int = 2,
        frameOverlap: Int = 8,
        heightTiles: Int = 2,
        heightOverlap: Int = 6,
        widthTiles: Int = 2,
        widthOverlap: Int = 6
    ) {
        precondition(frameTiles > 0 && heightTiles > 0 && widthTiles > 0)
        precondition(frameOverlap >= 0 && heightOverlap >= 0 && widthOverlap >= 0)
        self.frameTiles = frameTiles
        self.frameOverlap = frameOverlap
        self.heightTiles = heightTiles
        self.heightOverlap = heightOverlap
        self.widthTiles = widthTiles
        self.widthOverlap = widthOverlap
    }
}

public struct LTXHDRICLoRAStage2Phase: Sendable, Hashable {
    public let tiling: LTXHDRICLoRAStage2Tiling
    public let sigmas: [Float]
    public let usesICLoRAConditioning: Bool

    public init(
        tiling: LTXHDRICLoRAStage2Tiling = LTXHDRICLoRAStage2Tiling(),
        sigmas: [Float] = [0.909375, 0.725, 0],
        usesICLoRAConditioning: Bool = true
    ) {
        self.tiling = tiling
        self.sigmas = sigmas
        self.usesICLoRAConditioning = usesICLoRAConditioning
    }
}

public struct LTXHDRICLoRAOptions: Sendable, Hashable {
    public let highQuality: Bool
    public let stage2Phases: [LTXHDRICLoRAStage2Phase]

    public init(
        highQuality: Bool = false,
        stage2Phases: [LTXHDRICLoRAStage2Phase] = [LTXHDRICLoRAStage2Phase()]
    ) {
        precondition(!stage2Phases.isEmpty)
        self.highQuality = highQuality
        self.stage2Phases = stage2Phases
    }
}

extension LTXHDROutputFrames {
    func selectingFrames(_ indices: MLXArray) -> LTXHDROutputFrames {
        LTXHDROutputFrames(
            working: MLX.take(working, indices, axis: 0),
            exr: MLX.take(exr, indices, axis: 0),
            hlg: MLX.take(hlg, indices, axis: 0)
        )
    }

    func cropped(width: Int, height: Int) -> LTXHDROutputFrames {
        precondition(width > 0 && height > 0)
        precondition(width <= exr.dim(2) && height <= exr.dim(1))
        return LTXHDROutputFrames(
            working: working[0..., 0..<height, 0..<width, 0...],
            exr: exr[0..., 0..<height, 0..<width, 0...],
            hlg: hlg[0..., 0..<height, 0..<width, 0...]
        )
    }
}

public enum LTXHDRColorPipeline {
    private static let acescgToRec709: [[Float]] = [
        [1.70505000, -0.62179000, -0.08326000],
        [-0.13026000, 1.14080000, -0.01055000],
        [-0.02400000, -0.12897000, 1.15297000],
    ]
    private static let rec709ToACEScg: [[Float]] = invert3x3(acescgToRec709)
    private static let rec709ToRec2020: [[Float]] = [
        [0.62740389, 0.32928304, 0.04331307],
        [0.06909729, 0.91954040, 0.01136232],
        [0.01639144, 0.08801331, 0.89559525],
    ]

    public static func makeConditioningImage(
        _ image: MediaFloatImage,
        colorSpace: LTXHDRColorSpace,
        dtype: DType
    ) -> MLXArray {
        var values = image.rgb
        switch colorSpace {
        case .acescct:
            values = values.map { min(1, max(0, $0)) }
        case .acescg:
            values = compress(values, transfer: .acesCCT)
        case .srgbLinear:
            values = compress(applyMatrix(values, rec709ToACEScg), transfer: .acesCCT)
        }
        let fhwc = MLXArray(values).reshaped(1, image.height, image.width, 3)
        return (fhwc * MLXArray(Float(2)) - MLXArray(Float(1)))
            .transposed(0, 3, 1, 2)
            .reshaped(1, 3, 1, image.height, image.width)
            .asType(dtype)
    }

    public static func decode(
        _ decodedBCFHW: MLXArray,
        transfer: LTXHDRTransfer = .acesCCT,
        exrColorSpace: LTXHDRColorSpace
    ) -> LTXHDROutputFrames {
        precondition(decodedBCFHW.ndim == 5 && decodedBCFHW.dim(0) == 1 && decodedBCFHW.dim(1) == 3)
        let decoded = decodedBCFHW[0, 0..., 0..., 0..., 0...]
            .transposed(1, 2, 3, 0)
            .asType(.float32)
        let working = MLX.clip(
            (decoded + MLXArray(Float(1))) / MLXArray(Float(2)),
            min: MLXArray(Float(0)),
            max: MLXArray(Float(1))
        )
        let nativeLinear = decompress(working, transfer: transfer)
        let rec709Linear: MLXArray
        switch transfer {
        case .acesCCT:
            rec709Linear = applyMatrix(nativeLinear, acescgToRec709)
        case .logC3:
            rec709Linear = nativeLinear
        }
        let clippedRec709 = MLX.maximum(rec709Linear, MLXArray(Float(0)))
        let exr: MLXArray
        switch exrColorSpace {
        case .acescct:
            exr = transfer == .acesCCT
                ? working
                : compress(applyMatrix(clippedRec709, rec709ToACEScg), transfer: .acesCCT)
        case .acescg:
            exr = transfer == .acesCCT
                ? MLX.maximum(nativeLinear, MLXArray(Float(0)))
                : MLX.maximum(applyMatrix(clippedRec709, rec709ToACEScg), MLXArray(Float(0)))
        case .srgbLinear:
            exr = clippedRec709
        }
        let rec2020 = MLX.maximum(
            applyMatrix(clippedRec709, rec709ToRec2020),
            MLXArray(Float(0))
        )
        return LTXHDROutputFrames(
            working: working,
            exr: exr,
            hlg: hlgOETF(sceneLinearRec2020: rec2020)
        )
    }

    public static func compress(_ values: [Float], transfer: LTXHDRTransfer) -> [Float] {
        values.map { compress($0, transfer: transfer) }
    }

    public static func decompress(_ values: [Float], transfer: LTXHDRTransfer) -> [Float] {
        values.map { decompress($0, transfer: transfer) }
    }

    public static func compress(_ input: MLXArray, transfer: LTXHDRTransfer) -> MLXArray {
        let zero = MLXArray(Float(0))
        let one = MLXArray(Float(1))
        let value = MLX.maximum(input.asType(.float32), zero)
        let output: MLXArray
        switch transfer {
        case .logC3:
            let logPart = MLXArray(Float(0.247190))
                * (MLX.log(MLXArray(Float(5.555556)) * value + MLXArray(Float(0.052272)))
                    / MLXArray(Float(log(10.0))))
                + MLXArray(Float(0.385537))
            let linearPart = MLXArray(Float(5.367655)) * value + MLXArray(Float(0.092809))
            output = MLX.where(value .>= MLXArray(Float(0.010591)), logPart, linearPart)
        case .acesCCT:
            let logPart = (
                MLX.log2(MLX.maximum(value, MLXArray(Float(1e-12))))
                    + MLXArray(Float(9.72))
            ) / MLXArray(Float(17.52))
            let linearPart = MLXArray(Float(10.5402377416545)) * value
                + MLXArray(Float(0.0729055341958355))
            output = MLX.where(value .> MLXArray(Float(0.0078125)), logPart, linearPart)
        }
        return MLX.clip(output, min: zero, max: one)
    }

    public static func decompress(_ input: MLXArray, transfer: LTXHDRTransfer) -> MLXArray {
        let value = MLX.clip(
            input.asType(.float32),
            min: MLXArray(Float(0)),
            max: MLXArray(Float(1))
        )
        switch transfer {
        case .logC3:
            let linearFromLog = (
                MLX.pow(
                    MLXArray(Float(10)),
                    (value - MLXArray(Float(0.385537))) / MLXArray(Float(0.247190))
                ) - MLXArray(Float(0.052272))
            ) / MLXArray(Float(5.555556))
            let linearFromLinear = (value - MLXArray(Float(0.092809)))
                / MLXArray(Float(5.367655))
            let cut = Float(5.367655 * 0.010591 + 0.092809)
            return MLX.where(value .>= MLXArray(cut), linearFromLog, linearFromLinear)
        case .acesCCT:
            let linearFromLog = MLX.pow(
                MLXArray(Float(2)),
                value * MLXArray(Float(17.52)) - MLXArray(Float(9.72))
            )
            let linearFromLinear = (value - MLXArray(Float(0.0729055341958355)))
                / MLXArray(Float(10.5402377416545))
            return MLX.where(value .> MLXArray(Float(0.155251141552511)), linearFromLog, linearFromLinear)
        }
    }

    private static func compress(_ input: Float, transfer: LTXHDRTransfer) -> Float {
        let value = max(0, input)
        let output: Float
        switch transfer {
        case .logC3:
            output = value >= 0.010591
                ? 0.247190 * log10(5.555556 * value + 0.052272) + 0.385537
                : 5.367655 * value + 0.092809
        case .acesCCT:
            output = value > 0.0078125
                ? (log2(max(value, 1e-12)) + 9.72) / 17.52
                : 10.5402377416545 * value + 0.0729055341958355
        }
        return min(1, max(0, output))
    }

    private static func decompress(_ input: Float, transfer: LTXHDRTransfer) -> Float {
        let value = min(1, max(0, input))
        switch transfer {
        case .logC3:
            let cut = Float(5.367655 * 0.010591 + 0.092809)
            return value >= cut
                ? (pow(10, (value - 0.385537) / 0.247190) - 0.052272) / 5.555556
                : (value - 0.092809) / 5.367655
        case .acesCCT:
            return value > 0.155251141552511
                ? pow(2, value * 17.52 - 9.72)
                : (value - 0.0729055341958355) / 10.5402377416545
        }
    }

    private static func applyMatrix(_ input: MLXArray, _ matrix: [[Float]]) -> MLXArray {
        let weights = MLXArray(matrix.flatMap { $0 }).reshaped(3, 3)
        return MLX.matmul(input.asType(.float32), weights.T)
    }

    private static func applyMatrix(_ input: [Float], _ matrix: [[Float]]) -> [Float] {
        var output = [Float](repeating: 0, count: input.count)
        for pixel in 0..<(input.count / 3) {
            for row in 0..<3 {
                output[pixel * 3 + row] = (0..<3).reduce(Float(0)) { partial, column in
                    partial + matrix[row][column] * input[pixel * 3 + column]
                }
            }
        }
        return output
    }

    private static func hlgOETF(sceneLinearRec2020: MLXArray) -> MLXArray {
        let a = Float(0.17883277)
        let b = Float(0.28466892)
        let c = Float(0.55991073)
        let whiteSignal = Float(0.75)
        let whiteX = whiteSignal * whiteSignal / 3
        let rolloff = whiteX / (1 - whiteX)
        let linear = sceneLinearRec2020.asType(.float32)
        let mapped = MLX.where(
            linear .<= MLXArray(Float(1)),
            linear * MLXArray(whiteX),
            MLXArray(Float(1))
                - MLXArray(Float(1 - whiteX))
                    * MLX.exp(-MLXArray(rolloff) * (linear - MLXArray(Float(1))))
        )
        let encoded = MLX.where(
            mapped .<= MLXArray(Float(1.0 / 12.0)),
            MLX.sqrt(MLX.maximum(MLXArray(Float(3)) * mapped, MLXArray(Float(0)))),
            MLXArray(a)
                * MLX.log(MLX.maximum(MLXArray(Float(12)) * mapped - MLXArray(b), MLXArray(Float(1e-12))))
                + MLXArray(c)
        )
        return MLX.clip(encoded, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
    }

    private static func invert3x3(_ matrix: [[Float]]) -> [[Float]] {
        let a = Double(matrix[0][0]), b = Double(matrix[0][1]), c = Double(matrix[0][2])
        let d = Double(matrix[1][0]), e = Double(matrix[1][1]), f = Double(matrix[1][2])
        let g = Double(matrix[2][0]), h = Double(matrix[2][1]), i = Double(matrix[2][2])
        let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        return [
            [Float((e * i - f * h) / determinant), Float((c * h - b * i) / determinant), Float((b * f - c * e) / determinant)],
            [Float((f * g - d * i) / determinant), Float((a * i - c * g) / determinant), Float((c * d - a * f) / determinant)],
            [Float((d * h - e * g) / determinant), Float((b * g - a * h) / determinant), Float((a * e - b * d) / determinant)],
        ]
    }
}
