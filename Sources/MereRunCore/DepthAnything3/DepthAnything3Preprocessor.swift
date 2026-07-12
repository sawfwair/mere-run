import Foundation
import MediaIO
@preconcurrency import MLX

/// A camera supplied for an original input image. Intrinsics describe the
/// original image dimensions and extrinsics are a row-major world-to-camera
/// rigid transform. The contract is Codable so CLI/API clients can share the
/// same camera document without recreating matrix semantics.
public struct DepthAnything3KnownCamera: Codable, Equatable, Sendable {
    public let intrinsics: GeometryCameraIntrinsics
    public let extrinsics: GeometryCameraExtrinsics

    public init(
        intrinsics: GeometryCameraIntrinsics,
        extrinsics: GeometryCameraExtrinsics
    ) {
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
    }
}

/// Exact geometric bookkeeping for DA3's fixed upper-bound-resize path.
public struct DepthAnything3PreprocessingPlan: Codable, Equatable, Sendable {
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let processResolution: Int
    public let boundaryWidth: Int
    public let boundaryHeight: Int
    public let divisibleWidth: Int
    public let divisibleHeight: Int
    public let batchCropLeft: Int
    public let batchCropTop: Int
    public let processedWidth: Int
    public let processedHeight: Int

    public init(
        sourceWidth: Int,
        sourceHeight: Int,
        processResolution: Int,
        boundaryWidth: Int,
        boundaryHeight: Int,
        divisibleWidth: Int,
        divisibleHeight: Int,
        batchCropLeft: Int,
        batchCropTop: Int,
        processedWidth: Int,
        processedHeight: Int
    ) {
        self.sourceWidth = sourceWidth
        self.sourceHeight = sourceHeight
        self.processResolution = processResolution
        self.boundaryWidth = boundaryWidth
        self.boundaryHeight = boundaryHeight
        self.divisibleWidth = divisibleWidth
        self.divisibleHeight = divisibleHeight
        self.batchCropLeft = batchCropLeft
        self.batchCropTop = batchCropTop
        self.processedWidth = processedWidth
        self.processedHeight = processedHeight
    }

    public var sourceToProcessedScaleX: Double {
        Double(divisibleWidth) / Double(sourceWidth)
    }

    public var sourceToProcessedScaleY: Double {
        Double(divisibleHeight) / Double(sourceHeight)
    }
}

/// Native model input plus the CPU images/cameras needed for structured output.
public struct DepthAnything3PreprocessedBatch {
    /// `[1, views, height, width, 3]`, RGB, ImageNet-normalized.
    public let normalizedImages: MLXArray
    public let sourceImages: [MediaImage]
    public let processedImages: [MediaImage]
    public let plans: [DepthAnything3PreprocessingPlan]
    /// Supplied W2C transforms unchanged; only pixel intrinsics are transformed
    /// through resize/crop into processed-image coordinates.
    public let processedKnownCameras: [DepthAnything3KnownCamera]?
    /// Processed intrinsics plus W2C transforms normalized exactly as upstream
    /// before camera-token encoding (first camera identity, median baseline).
    public let conditioningCameras: [DepthAnything3KnownCamera]?
    public let conditioning: DepthAnything3CameraConditioning?

    public init(
        normalizedImages: MLXArray,
        sourceImages: [MediaImage],
        processedImages: [MediaImage],
        plans: [DepthAnything3PreprocessingPlan],
        processedKnownCameras: [DepthAnything3KnownCamera]?,
        conditioningCameras: [DepthAnything3KnownCamera]?,
        conditioning: DepthAnything3CameraConditioning?
    ) {
        self.normalizedImages = normalizedImages
        self.sourceImages = sourceImages
        self.processedImages = processedImages
        self.plans = plans
        self.processedKnownCameras = processedKnownCameras
        self.conditioningCameras = conditioningCameras
        self.conditioning = conditioning
    }
}

public enum DepthAnything3PreprocessingError: Error, Equatable, LocalizedError, Sendable {
    case emptyImages
    case invalidProcessResolution(Int)
    case cameraCountMismatch(expected: Int, actual: Int)
    case cameraImageDimensionMismatch(
        index: Int,
        expectedWidth: Int,
        expectedHeight: Int,
        actualWidth: Int,
        actualHeight: Int
    )

    public var errorDescription: String? {
        switch self {
        case .emptyImages:
            "Depth Anything 3 requires at least one image."
        case .invalidProcessResolution(let value):
            "DA3 process resolution must be at least 14; received \(value)."
        case .cameraCountMismatch(let expected, let actual):
            "DA3 known-camera count must match the \(expected) images; received \(actual)."
        case let .cameraImageDimensionMismatch(index, expectedWidth, expectedHeight, actualWidth, actualHeight):
            "DA3 camera \(index) describes \(actualWidth)x\(actualHeight), not its \(expectedWidth)x\(expectedHeight) image."
        }
    }
}

public enum DepthAnything3Preprocessor {
    public static let defaultProcessResolution = 504
    public static let patchSize = 14

    /// Reproduces pinned upstream `upper_bound_resize`: longest side to the
    /// requested resolution (OpenCV cubic up / area down), each dimension to
    /// the nearest multiple of 14, then a batch-wide center crop to the
    /// smallest shape. Intrinsics follow every resize/crop operation.
    public static func prepare(
        sourceImages images: [MediaImage],
        knownCameras: [DepthAnything3KnownCamera]? = nil,
        processResolution: Int = defaultProcessResolution
    ) throws -> DepthAnything3PreprocessedBatch {
        guard !images.isEmpty else { throw DepthAnything3PreprocessingError.emptyImages }
        try DepthAnything3Limits.validateRequest(
            viewCount: images.count,
            processResolution: processResolution
        )
        try DepthAnything3Limits.validateSourceDimensions(
            images.map { (width: $0.width, height: $0.height) }
        )
        if let knownCameras, knownCameras.count != images.count {
            throw DepthAnything3PreprocessingError.cameraCountMismatch(
                expected: images.count,
                actual: knownCameras.count
            )
        }
        if let knownCameras {
            for index in images.indices {
                try DepthAnything3CameraValidation.validate(knownCameras[index], index: index)
                let image = images[index]
                let intrinsics = knownCameras[index].intrinsics
                guard intrinsics.imageWidth == image.width,
                      intrinsics.imageHeight == image.height else {
                    throw DepthAnything3PreprocessingError.cameraImageDimensionMismatch(
                        index: index,
                        expectedWidth: image.width,
                        expectedHeight: image.height,
                        actualWidth: intrinsics.imageWidth,
                        actualHeight: intrinsics.imageHeight
                    )
                }
            }
        }

        var divisibleImages: [MediaImage] = []
        var provisional: [(boundaryWidth: Int, boundaryHeight: Int, width: Int, height: Int)] = []
        divisibleImages.reserveCapacity(images.count)
        provisional.reserveCapacity(images.count)
        for image in images {
            let longest = max(image.width, image.height)
            let scale = Double(processResolution) / Double(longest)
            let boundaryWidth = max(1, pythonRound(Double(image.width) * scale))
            let boundaryHeight = max(1, pythonRound(Double(image.height) * scale))
            let boundary = try resize(
                image,
                width: boundaryWidth,
                height: boundaryHeight,
                useCubic: scale > 1
            )
            let width = max(patchSize, nearestMultiple(boundaryWidth, of: patchSize))
            let height = max(patchSize, nearestMultiple(boundaryHeight, of: patchSize))
            let useCubic = width > boundaryWidth || height > boundaryHeight
            let divisible = try resize(boundary, width: width, height: height, useCubic: useCubic)
            divisibleImages.append(divisible)
            provisional.append((boundaryWidth, boundaryHeight, width, height))
        }

        guard let processedWidth = provisional.map(\.width).min(),
              let processedHeight = provisional.map(\.height).min() else {
            throw DepthAnything3PreprocessingError.emptyImages
        }
        var processedImages: [MediaImage] = []
        var plans: [DepthAnything3PreprocessingPlan] = []
        processedImages.reserveCapacity(images.count)
        plans.reserveCapacity(images.count)
        for index in images.indices {
            let dimensions = provisional[index]
            let left = max(0, (dimensions.width - processedWidth) / 2)
            let top = max(0, (dimensions.height - processedHeight) / 2)
            processedImages.append(try crop(
                divisibleImages[index],
                left: left,
                top: top,
                width: processedWidth,
                height: processedHeight
            ))
            plans.append(DepthAnything3PreprocessingPlan(
                sourceWidth: images[index].width,
                sourceHeight: images[index].height,
                processResolution: processResolution,
                boundaryWidth: dimensions.boundaryWidth,
                boundaryHeight: dimensions.boundaryHeight,
                divisibleWidth: dimensions.width,
                divisibleHeight: dimensions.height,
                batchCropLeft: left,
                batchCropTop: top,
                processedWidth: processedWidth,
                processedHeight: processedHeight
            ))
        }

        let processedCameras = try knownCameras.map { cameras in
            try cameras.indices.map { index in
                let camera = try transformed(camera: cameras[index], plan: plans[index])
                try DepthAnything3CameraValidation.validate(camera, index: index)
                return camera
            }
        }
        let conditioningCameras = try processedCameras.map(normalizedForConditioning)
        let conditioning = try conditioningCameras.map(makeConditioning)
        return DepthAnything3PreprocessedBatch(
            normalizedImages: normalizedTensor(processedImages),
            sourceImages: images,
            processedImages: processedImages,
            plans: plans,
            processedKnownCameras: processedCameras,
            conditioningCameras: conditioningCameras,
            conditioning: conditioning
        )
    }

    private static func transformed(
        camera: DepthAnything3KnownCamera,
        plan: DepthAnything3PreprocessingPlan
    ) throws -> DepthAnything3KnownCamera {
        let sx = plan.sourceToProcessedScaleX
        let sy = plan.sourceToProcessedScaleY
        let fx = camera.intrinsics.pixelFX * sx
        let fy = camera.intrinsics.pixelFY * sy
        let cx = camera.intrinsics.pixelCX * sx - Double(plan.batchCropLeft)
        let cy = camera.intrinsics.pixelCY * sy - Double(plan.batchCropTop)
        let intrinsics = GeometryCameraIntrinsics(
            imageWidth: plan.processedWidth,
            imageHeight: plan.processedHeight,
            normalizedFX: fx / Double(plan.processedWidth),
            normalizedFY: fy / Double(plan.processedHeight),
            normalizedCX: cx / Double(plan.processedWidth),
            normalizedCY: cy / Double(plan.processedHeight)
        )
        return DepthAnything3KnownCamera(
            intrinsics: intrinsics,
            extrinsics: camera.extrinsics
        )
    }

    /// Matches upstream `_normalize_extrinsics`: make camera zero the world
    /// origin, then divide W2C translations by the lower median camera-center
    /// distance clamped to 0.1.
    private static func normalizedForConditioning(
        _ cameras: [DepthAnything3KnownCamera]
    ) throws -> [DepthAnything3KnownCamera] {
        let matrices = cameras.map { matrix44($0.extrinsics) }
        let firstCameraToWorld = inverseRigid(matrices[0])
        var normalized = matrices.map { multiply44($0, firstCameraToWorld) }
        let distances = normalized.map { matrix -> Double in
            let cameraToWorld = inverseRigid(matrix)
            let x = cameraToWorld[3], y = cameraToWorld[7], z = cameraToWorld[11]
            return sqrt(x * x + y * y + z * z)
        }.sorted()
        let median = max(0.1, distances[(distances.count - 1) / 2])
        for index in normalized.indices {
            normalized[index][3] /= median
            normalized[index][7] /= median
            normalized[index][11] /= median
        }
        return try cameras.indices.map { index in
            let camera = DepthAnything3KnownCamera(
                intrinsics: cameras[index].intrinsics,
                extrinsics: try extrinsics(normalized[index])
            )
            try DepthAnything3CameraValidation.validate(camera, index: index)
            return camera
        }
    }

    private static func makeConditioning(
        _ cameras: [DepthAnything3KnownCamera]
    ) throws -> DepthAnything3CameraConditioning {
        let extrinsicValues = cameras.flatMap { matrix44($0.extrinsics).map(Float.init) }
        let intrinsicValues = cameras.flatMap { camera in
            camera.intrinsics.pixelMatrixRowMajor.map(Float.init)
        }
        return try DepthAnything3CameraConditioning(
            extrinsics: MLXArray(extrinsicValues).reshaped(1, cameras.count, 4, 4),
            intrinsics: MLXArray(intrinsicValues).reshaped(1, cameras.count, 3, 3)
        )
    }

    private static func normalizedTensor(_ images: [MediaImage]) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(images.count * images[0].width * images[0].height * 3)
        let mean: [Float] = [0.485, 0.456, 0.406]
        let deviation: [Float] = [0.229, 0.224, 0.225]
        for image in images {
            for pixel in 0..<(image.width * image.height) {
                let offset = pixel * 4
                values.append((Float(image.rgba8[offset]) / 255 - mean[0]) / deviation[0])
                values.append((Float(image.rgba8[offset + 1]) / 255 - mean[1]) / deviation[1])
                values.append((Float(image.rgba8[offset + 2]) / 255 - mean[2]) / deviation[2])
            }
        }
        return MLXArray(values).reshaped(
            1, images.count, images[0].height, images[0].width, 3
        )
    }

    private static func resize(
        _ image: MediaImage,
        width: Int,
        height: Int,
        useCubic: Bool
    ) throws -> MediaImage {
        guard image.width != width || image.height != height else { return image }
        return try useCubic
            ? cubicResize(image, width: width, height: height)
            : areaResize(image, width: width, height: height)
    }

    /// Bit-compatible port of OpenCV's uint8 INTER_CUBIC path. OpenCV first
    /// quantizes the -0.75 cubic coefficients to signed 11-bit fixed point,
    /// performs the horizontal pass in Int32, then uses Float32 SIMD-style
    /// vertical accumulation before round-to-nearest-even saturation.
    private static func cubicResize(
        _ image: MediaImage,
        width: Int,
        height: Int
    ) throws -> MediaImage {
        let xTable = cubicTable(sourceSize: image.width, destinationSize: width)
        let yTable = cubicTable(sourceSize: image.height, destinationSize: height)
        let channels = 3
        var horizontal = [Int32](
            repeating: 0,
            count: image.height * width * channels
        )
        for sourceY in 0..<image.height {
            for destinationX in 0..<width {
                let entry = xTable[destinationX]
                for channel in 0..<channels {
                    var value: Int32 = 0
                    for tap in 0..<4 {
                        var sourceX = entry.source - 1 + tap
                        while sourceX < 0 { sourceX += 1 }
                        while sourceX >= image.width { sourceX -= 1 }
                        let source = (sourceY * image.width + sourceX) * 4 + channel
                        value += Int32(image.rgba8[source]) * Int32(entry.coefficients[tap])
                    }
                    horizontal[(sourceY * width + destinationX) * channels + channel] = value
                }
            }
        }

        let coefficientScale = Float(1.0 / Double(openCVCoefficientScale * openCVCoefficientScale))
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for destinationY in 0..<height {
            let entry = yTable[destinationY]
            let sourceRows = (0..<4).map {
                min(image.height - 1, max(0, entry.source - 1 + $0))
            }
            let beta = entry.coefficients.map { Float($0) * coefficientScale }
            for destinationX in 0..<width {
                for channel in 0..<channels {
                    let offset = (destinationX * channels) + channel
                    var value = Float(horizontal[sourceRows[3] * width * channels + offset]) * beta[3]
                    value = Float(horizontal[sourceRows[2] * width * channels + offset]) * beta[2] + value
                    value = Float(horizontal[sourceRows[1] * width * channels + offset]) * beta[1] + value
                    value = Float(horizontal[sourceRows[0] * width * channels + offset]) * beta[0] + value
                    rgba[(destinationY * width + destinationX) * 4 + channel] = saturatedUInt8(value)
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    /// Bit-compatible port of OpenCV's generic uint8 INTER_AREA decimator.
    /// Its coefficient tables and both separable accumulation passes are
    /// Float32; a geometrically equivalent Double box average differs by one
    /// level at sparse pixels and is therefore not reference-compatible.
    private static func areaResize(
        _ image: MediaImage,
        width: Int,
        height: Int
    ) throws -> MediaImage {
        let channels = 3
        let xGroups = areaTable(
            sourceSize: image.width,
            destinationSize: width
        )
        let yGroups = areaTable(
            sourceSize: image.height,
            destinationSize: height
        )
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let rowWidth = width * channels
        var buffer = [Float](repeating: 0, count: rowWidth)
        var sum = [Float](repeating: 0, count: rowWidth)
        for destinationY in 0..<height {
            sum.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
            for yEntry in yGroups[destinationY] {
                buffer.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
                for destinationX in 0..<width {
                    for xEntry in xGroups[destinationX] {
                        let source = (yEntry.source * image.width + xEntry.source) * 4
                        let destination = destinationX * channels
                        for channel in 0..<channels {
                            let product = Float(image.rgba8[source + channel]) * xEntry.coefficient
                            buffer[destination + channel] = buffer[destination + channel] + product
                        }
                    }
                }
                for offset in 0..<rowWidth {
                    let product = yEntry.coefficient * buffer[offset]
                    sum[offset] = sum[offset] + product
                }
            }
            for destinationX in 0..<width {
                for channel in 0..<channels {
                    rgba[(destinationY * width + destinationX) * 4 + channel] = saturatedUInt8(
                        sum[destinationX * channels + channel]
                    )
                }
            }
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    private static let openCVCoefficientScale = 1 << 11

    private struct CubicTableEntry {
        let source: Int
        let coefficients: [Int16]
    }

    private static func cubicTable(
        sourceSize: Int,
        destinationSize: Int
    ) -> [CubicTableEntry] {
        let scale = Double(sourceSize) / Double(destinationSize)
        return (0..<destinationSize).map { destination in
            var fraction = Float((Double(destination) + 0.5) * scale - 0.5)
            let source = Int(floor(fraction))
            fraction -= Float(source)
            return CubicTableEntry(
                source: source,
                coefficients: cubicCoefficients(fraction).map { coefficient in
                    let quantized = Int(
                        (coefficient * Float(openCVCoefficientScale)).rounded(.toNearestOrEven)
                    )
                    return Int16(max(Int(Int16.min), min(Int(Int16.max), quantized)))
                }
            )
        }
    }

    private static func cubicCoefficients(_ fraction: Float) -> [Float] {
        let a = Float(-0.75)
        let x1 = fraction + 1
        var c0 = a * x1 - 5 * a
        c0 = c0 * x1 + 8 * a
        c0 = c0 * x1 - 4 * a
        var c1 = (a + 2) * fraction - (a + 3)
        c1 = c1 * fraction * fraction + 1
        let inverse = 1 - fraction
        var c2 = (a + 2) * inverse - (a + 3)
        c2 = c2 * inverse * inverse + 1
        let c3 = 1 - c0 - c1 - c2
        return [c0, c1, c2, c3]
    }

    private struct AreaTableEntry {
        let source: Int
        let coefficient: Float
    }

    private static func areaTable(
        sourceSize: Int,
        destinationSize: Int
    ) -> [[AreaTableEntry]] {
        let scale = Double(sourceSize) / Double(destinationSize)
        return (0..<destinationSize).map { destination in
            let first = Double(destination) * scale
            let end = first + scale
            let cellWidth = min(scale, Double(sourceSize) - first)
            var sourceStart = Int(ceil(first))
            var sourceEnd = Int(floor(end))
            sourceEnd = min(sourceEnd, sourceSize - 1)
            sourceStart = min(sourceStart, sourceEnd)
            var entries: [AreaTableEntry] = []
            if Double(sourceStart) - first > 1e-3 {
                entries.append(AreaTableEntry(
                    source: sourceStart - 1,
                    coefficient: Float((Double(sourceStart) - first) / cellWidth)
                ))
            }
            if sourceStart < sourceEnd {
                for source in sourceStart..<sourceEnd {
                    entries.append(AreaTableEntry(
                        source: source,
                        coefficient: Float(1 / cellWidth)
                    ))
                }
            }
            if end - Double(sourceEnd) > 1e-3 {
                entries.append(AreaTableEntry(
                    source: sourceEnd,
                    coefficient: Float(
                        min(min(end - Double(sourceEnd), 1), cellWidth) / cellWidth
                    )
                ))
            }
            return entries
        }
    }

    private static func crop(
        _ image: MediaImage,
        left: Int,
        top: Int,
        width: Int,
        height: Int
    ) throws -> MediaImage {
        guard left != 0 || top != 0 || width != image.width || height != image.height else {
            return image
        }
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            let sourceStart = ((y + top) * image.width + left) * 4
            let destinationStart = y * width * 4
            rgba.replaceSubrange(
                destinationStart..<(destinationStart + width * 4),
                with: image.rgba8[sourceStart..<(sourceStart + width * 4)]
            )
        }
        return try MediaImage(width: width, height: height, rgba8: rgba)
    }

    private static func pythonRound(_ value: Double) -> Int {
        Int(value.rounded(.toNearestOrEven))
    }

    private static func nearestMultiple(_ value: Int, of multiple: Int) -> Int {
        let down = (value / multiple) * multiple
        let up = down + multiple
        return abs(up - value) <= abs(value - down) ? up : down
    }

    private static func saturatedUInt8(_ value: Float) -> UInt8 {
        UInt8(max(0, min(255, Int(value.rounded(.toNearestOrEven)))))
    }

    private static func matrix44(_ extrinsics: GeometryCameraExtrinsics) -> [Double] {
        let r = extrinsics.rotation, t = extrinsics.translation
        return [
            r[0], r[1], r[2], t[0],
            r[3], r[4], r[5], t[1],
            r[6], r[7], r[8], t[2],
            0, 0, 0, 1,
        ]
    }

    private static func inverseRigid(_ matrix: [Double]) -> [Double] {
        let r = [
            matrix[0], matrix[1], matrix[2],
            matrix[4], matrix[5], matrix[6],
            matrix[8], matrix[9], matrix[10],
        ]
        let t = [matrix[3], matrix[7], matrix[11]]
        let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]]
        let ti = [
            -(rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
            -(rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
            -(rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
        ]
        return [
            rt[0], rt[1], rt[2], ti[0],
            rt[3], rt[4], rt[5], ti[1],
            rt[6], rt[7], rt[8], ti[2],
            0, 0, 0, 1,
        ]
    }

    private static func multiply44(_ lhs: [Double], _ rhs: [Double]) -> [Double] {
        var result = [Double](repeating: 0, count: 16)
        for row in 0..<4 {
            for column in 0..<4 {
                result[row * 4 + column] = (0..<4).reduce(0) {
                    $0 + lhs[row * 4 + $1] * rhs[$1 * 4 + column]
                }
            }
        }
        return result
    }

    private static func extrinsics(_ matrix: [Double]) throws -> GeometryCameraExtrinsics {
        try GeometryCameraExtrinsics(
            rotation: [
                matrix[0], matrix[1], matrix[2],
                matrix[4], matrix[5], matrix[6],
                matrix[8], matrix[9], matrix[10],
            ],
            translation: [matrix[3], matrix[7], matrix[11]]
        )
    }
}
