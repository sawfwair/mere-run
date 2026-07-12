import Foundation
import MediaIO
@preconcurrency import MLX

public enum DepthAnything3CameraSemantics: String, Codable, Equatable, Sendable {
    /// Cameras and relative scene scale are predicted entirely by DA3.
    case predictedRelative = "predicted-relative"
    /// Supplied cameras are preserved in the result and depth is divided by a
    /// deterministic similarity-scale estimate from predicted/supplied camera centers.
    case suppliedScaleAlignedRelative = "supplied-scale-aligned-relative"
}

public struct DepthAnything3ViewResult: Sendable {
    public let index: Int
    public let sourceURL: URL
    public let inputIdentity: DepthAnything3InputIdentity
    public let sourceImage: MediaImage
    public let processedImage: MediaImage
    public let preprocessingPlan: DepthAnything3PreprocessingPlan
    public let depth: [Float]
    public let confidence: [Float]
    public let intrinsics: GeometryCameraIntrinsics
    public let extrinsics: GeometryCameraExtrinsics
    public let predictedIntrinsics: GeometryCameraIntrinsics
    public let predictedExtrinsics: GeometryCameraExtrinsics
    public let suppliedCamera: DepthAnything3KnownCamera?
}

struct DepthAnything3PostprocessResult: Sendable {
    let views: [DepthAnything3ViewResult]
    let cameraSemantics: DepthAnything3CameraSemantics
    let cameraScaleAlignment: String
    let depthScaleDivisor: Float
}

public enum DepthAnything3PostprocessingError: Error, Equatable, LocalizedError, Sendable {
    case invalidOutputShape(field: String, expected: [Int], actual: [Int])
    case inputCountMismatch(expected: Int, actual: Int)
    case invalidCamera(field: String, view: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidOutputShape(let field, let expected, let actual):
            "DA3 output '\(field)' expected shape \(expected), found \(actual)."
        case .inputCountMismatch(let expected, let actual):
            "DA3 postprocessing expected \(expected) inputs, found \(actual)."
        case .invalidCamera(let field, let view):
            "DA3 predicted \(field) for view \(view) contains invalid values."
        }
    }
}

enum DepthAnything3Postprocessor {
    static func process(
        raw: DepthAnything3RawOutput,
        sourceURLs: [URL],
        inputIdentities: [DepthAnything3InputIdentity],
        sourceImages: [MediaImage],
        processedImages: [MediaImage],
        preprocessingPlans: [DepthAnything3PreprocessingPlan],
        suppliedCameras: [DepthAnything3KnownCamera]?
    ) throws -> DepthAnything3PostprocessResult {
        let views = processedImages.count
        guard sourceURLs.count == views,
              inputIdentities.count == views,
              sourceImages.count == views,
              preprocessingPlans.count == views else {
            throw DepthAnything3PostprocessingError.inputCountMismatch(
                expected: views,
                actual: [
                    sourceURLs.count,
                    inputIdentities.count,
                    sourceImages.count,
                    preprocessingPlans.count,
                ].min() ?? 0
            )
        }
        if let suppliedCameras, suppliedCameras.count != views {
            throw DepthAnything3PostprocessingError.inputCountMismatch(
                expected: views,
                actual: suppliedCameras.count
            )
        }
        guard let first = processedImages.first else {
            throw DepthAnything3PostprocessingError.inputCountMismatch(expected: 1, actual: 0)
        }
        let height = first.height
        let width = first.width
        try require(raw.depth.shape, field: "depth", expected: [1, views, height, width])
        try require(raw.confidence.shape, field: "confidence", expected: [1, views, height, width])
        try require(raw.extrinsics.shape, field: "extrinsics", expected: [1, views, 3, 4])
        try require(raw.intrinsics.shape, field: "intrinsics", expected: [1, views, 3, 3])
        MLX.eval(raw.depth, raw.confidence, raw.extrinsics, raw.intrinsics)
        let depthValues = raw.depth.asArray(Float.self)
        let confidenceValues = raw.confidence.asArray(Float.self)
        let extrinsicValues = raw.extrinsics.asArray(Float.self)
        let intrinsicValues = raw.intrinsics.asArray(Float.self)
        let predictedExtrinsics = try (0..<views).map { view in
            try extrinsics(values: extrinsicValues, view: view)
        }
        let predictedIntrinsics = try (0..<views).map { view in
            try intrinsics(values: intrinsicValues, view: view, width: width, height: height)
        }

        let scale: Float
        let semantics: DepthAnything3CameraSemantics
        let alignment: String
        if let suppliedCameras {
            scale = similarityScale(
                predicted: predictedExtrinsics,
                supplied: suppliedCameras.map(\.extrinsics)
            )
            semantics = .suppliedScaleAlignedRelative
            alignment = views > 1
                ? "pairwise-median-camera-baseline-scale"
                : "single-view-no-scale-observable"
        } else {
            scale = 1
            semantics = .predictedRelative
            alignment = "predicted-relative"
        }

        let pixelCount = width * height
        var results: [DepthAnything3ViewResult] = []
        results.reserveCapacity(views)
        for view in 0..<views {
            let offset = view * pixelCount
            let depth = Array(depthValues[offset..<(offset + pixelCount)]).map { $0 / scale }
            let confidence = Array(confidenceValues[offset..<(offset + pixelCount)])
            let supplied = suppliedCameras?[view]
            results.append(
                DepthAnything3ViewResult(
                    index: view,
                    sourceURL: sourceURLs[view],
                    inputIdentity: inputIdentities[view],
                    sourceImage: sourceImages[view],
                    processedImage: processedImages[view],
                    preprocessingPlan: preprocessingPlans[view],
                    depth: depth,
                    confidence: confidence,
                    intrinsics: supplied?.intrinsics ?? predictedIntrinsics[view],
                    extrinsics: supplied?.extrinsics ?? predictedExtrinsics[view],
                    predictedIntrinsics: predictedIntrinsics[view],
                    predictedExtrinsics: predictedExtrinsics[view],
                    suppliedCamera: supplied
                )
            )
        }
        return DepthAnything3PostprocessResult(
            views: results,
            cameraSemantics: semantics,
            cameraScaleAlignment: alignment,
            depthScaleDivisor: scale
        )
    }

    private static func require(_ actual: [Int], field: String, expected: [Int]) throws {
        guard actual == expected else {
            throw DepthAnything3PostprocessingError.invalidOutputShape(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }

    private static func intrinsics(
        values: [Float],
        view: Int,
        width: Int,
        height: Int
    ) throws -> GeometryCameraIntrinsics {
        let base = view * 9
        let fx = values[base]
        let fy = values[base + 4]
        let cx = values[base + 2]
        let cy = values[base + 5]
        guard [fx, fy, cx, cy].allSatisfy(\.isFinite), fx > 0, fy > 0 else {
            throw DepthAnything3PostprocessingError.invalidCamera(field: "intrinsics", view: view)
        }
        return GeometryCameraIntrinsics(
            imageWidth: width,
            imageHeight: height,
            normalizedFX: Double(fx) / Double(width),
            normalizedFY: Double(fy) / Double(height),
            normalizedCX: Double(cx) / Double(width),
            normalizedCY: Double(cy) / Double(height)
        )
    }

    private static func extrinsics(
        values: [Float],
        view: Int
    ) throws -> GeometryCameraExtrinsics {
        let base = view * 12
        let rotation = [
            values[base], values[base + 1], values[base + 2],
            values[base + 4], values[base + 5], values[base + 6],
            values[base + 8], values[base + 9], values[base + 10],
        ].map(Double.init)
        let translation = [values[base + 3], values[base + 7], values[base + 11]].map(Double.init)
        guard rotation.allSatisfy(\.isFinite), translation.allSatisfy(\.isFinite) else {
            throw DepthAnything3PostprocessingError.invalidCamera(field: "extrinsics", view: view)
        }
        return try GeometryCameraExtrinsics(rotation: rotation, translation: translation)
    }

    /// Scale-only component of a camera-center similarity alignment. Pairwise
    /// baselines are rotation/translation invariant and the median is stable
    /// under a small number of poor predicted views. The divisor maps DA3's
    /// relative depth scale into the supplied camera-baseline scale.
    private static func similarityScale(
        predicted: [GeometryCameraExtrinsics],
        supplied: [GeometryCameraExtrinsics]
    ) -> Float {
        guard predicted.count == supplied.count, predicted.count > 1 else { return 1 }
        let predictedCenters = predicted.map(cameraCenter)
        let suppliedCenters = supplied.map(cameraCenter)
        var ratios: [Double] = []
        for first in 0..<(predicted.count - 1) {
            for second in (first + 1)..<predicted.count {
                let sourceDistance = distance(suppliedCenters[first], suppliedCenters[second])
                let targetDistance = distance(predictedCenters[first], predictedCenters[second])
                if sourceDistance > 1e-8, targetDistance.isFinite, targetDistance > 1e-8 {
                    ratios.append(targetDistance / sourceDistance)
                }
            }
        }
        guard !ratios.isEmpty else { return 1 }
        ratios.sort()
        let middle = ratios.count / 2
        let median = ratios.count.isMultiple(of: 2)
            ? (ratios[middle - 1] + ratios[middle]) / 2
            : ratios[middle]
        return Float(max(median, 1e-8))
    }

    private static func cameraCenter(_ extrinsics: GeometryCameraExtrinsics) -> [Double] {
        let r = extrinsics.rotation
        let t = extrinsics.translation
        return [
            -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
            -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
            -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]),
        ]
    }

    private static func distance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        sqrt(zip(lhs, rhs).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) })
    }
}
