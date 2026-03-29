import MLX

public final class SharpInitializer {
    public let params: SharpInitializerParameters

    public init(params: SharpInitializerParameters = SharpInitializerParameters()) {
        self.params = params
    }

    public func callAsFunction(image: MLXArray, depth: MLXArray) -> SharpInitializerOutput {
        var workingDepth = depth
        var globalScale: MLXArray?

        if params.normalizeDepth {
            let rescaled = Self.rescaleDepth(workingDepth)
            workingDepth = rescaled.depth
            globalScale = 1.0 / rescaled.depthFactor
        }

        let batchSize = workingDepth.dim(0)
        let imageHeight = workingDepth.dim(2)
        let imageWidth = workingDepth.dim(3)
        let baseHeight = imageHeight / params.stride
        let baseWidth = imageWidth / params.stride

        let firstDisparity: MLXArray
        switch params.firstLayerDepthOption {
        case .surfaceMin:
            firstDisparity = Self.createSurfaceLayer(
                depth: workingDepth[0..., 0..<1, 0..., 0...],
                stride: params.stride,
                poolingMode: .min
            )
        case .surfaceMax:
            firstDisparity = Self.createSurfaceLayer(
                depth: workingDepth[0..., 0..<1, 0..., 0...],
                stride: params.stride,
                poolingMode: .max
            )
        case .baseDepth, .linearDisparity:
            firstDisparity = Self.createDisparityLayers(
                batchSize: batchSize,
                numLayers: 1,
                baseHeight: baseHeight,
                baseWidth: baseWidth,
                baseDepth: params.baseDepth,
                dtype: workingDepth.dtype
            )
        }

        let disparity: MLXArray
        if params.numLayers == 1 {
            disparity = firstDisparity
        } else {
            let followingDepth = workingDepth.dim(1) == 1
                ? workingDepth
                : workingDepth[0..., 1..., 0..., 0...]

            let followingDisparity: MLXArray
            switch params.restLayerDepthOption {
            case .surfaceMin:
                followingDisparity = Self.createSurfaceLayer(
                    depth: followingDepth,
                    stride: params.stride,
                    poolingMode: .min
                )
            case .surfaceMax:
                followingDisparity = Self.createSurfaceLayer(
                    depth: followingDepth,
                    stride: params.stride,
                    poolingMode: .max
                )
            case .baseDepth:
                let parts = (0..<(params.numLayers - 1)).map { _ in
                    Self.createDisparityLayers(
                        batchSize: batchSize,
                        numLayers: 1,
                        baseHeight: baseHeight,
                        baseWidth: baseWidth,
                        baseDepth: params.baseDepth,
                        dtype: workingDepth.dtype
                    )
                }
                followingDisparity = parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 2)
            case .linearDisparity:
                followingDisparity = Self.createDisparityLayers(
                    batchSize: batchSize,
                    numLayers: params.numLayers - 1,
                    baseHeight: baseHeight,
                    baseWidth: baseWidth,
                    baseDepth: params.baseDepth,
                    dtype: workingDepth.dtype
                )
            }

            disparity = MLX.concatenated([firstDisparity, followingDisparity], axis: 2)
        }

        let baseXY = Self.createBaseXY(
            batchSize: batchSize,
            imageHeight: imageHeight,
            imageWidth: imageWidth,
            stride: params.stride,
            numLayers: params.numLayers,
            dtype: workingDepth.dtype
        )

        let disparityScaleFactor = Float(2.0 * params.scaleFactor * Float(params.stride) / Float(imageWidth))
        let baseScales = (1.0 / disparity) * disparityScaleFactor

        let baseQuaternions = MLXArray([Float(1.0), Float(0.0), Float(0.0), Float(0.0)], [1, 4, 1, 1, 1])
            .asType(workingDepth.dtype)
        let baseOpacities = MLXArray([min(1.0 / Float(params.numLayers), 0.5)], [1]).asType(workingDepth.dtype)
        let baseColors = Self.createBaseColors(
            image: image,
            stride: params.stride,
            numLayers: params.numLayers,
            option: params.colorOption
        )

        let normalizedDisparity = params.disparityFactor / workingDepth
        let featureInput = (2.0 * MLX.concatenated([image, normalizedDisparity], axis: 1)) - 1.0

        let baseValues = SharpGaussianBaseValues(
            meanXNDC: baseXY.x,
            meanYNDC: baseXY.y,
            meanInverseZNDC: disparity,
            scales: baseScales,
            quaternions: baseQuaternions,
            colors: baseColors,
            opacities: baseOpacities
        )

        return SharpInitializerOutput(
            gaussianBaseValues: baseValues,
            featureInput: featureInput,
            globalScale: globalScale
        )
    }
}

extension SharpInitializer {
    private enum PoolingMode {
        case min
        case max
    }

    private static func createDisparityLayers(
        batchSize: Int,
        numLayers: Int,
        baseHeight: Int,
        baseWidth: Int,
        baseDepth: Float,
        dtype: DType
    ) -> MLXArray {
        let disparity = linspace(1.0 / baseDepth, 0.0, count: numLayers + 1)
            .asType(dtype)[0..<numLayers]
        var out = disparity.reshaped(1, 1, numLayers, 1, 1)
        out = MLX.repeated(out, count: batchSize, axis: 0)
        out = MLX.repeated(out, count: baseHeight, axis: 3)
        out = MLX.repeated(out, count: baseWidth, axis: 4)
        return out
    }

    private static func createSurfaceLayer(depth: MLXArray, stride: Int, poolingMode: PoolingMode) -> MLXArray {
        let disparity = 1.0 / depth
        let pooled: MLXArray
        switch poolingMode {
        case .min:
            pooled = pool2DStride(disparity, stride: stride, mode: .max)
        case .max:
            pooled = pool2DStride(disparity, stride: stride, mode: .min)
        }
        return pooled.expandedDimensions(axis: 2)
    }

    private static func pool2DStride(_ x: MLXArray, stride: Int, mode: PoolingMode) -> MLXArray {
        guard stride > 1 else { return x }

        let batch = x.dim(0)
        let channels = x.dim(1)
        let height = (x.dim(2) / stride) * stride
        let width = (x.dim(3) / stride) * stride

        var cropped = x[0..., 0..., 0..<height, 0..<width]
        cropped = cropped.reshaped(batch, channels, height / stride, stride, width / stride, stride)

        switch mode {
        case .max:
            return cropped.max(axes: [3, 5])
        case .min:
            return cropped.min(axes: [3, 5])
        }
    }

    private static func avgPool2DStride(_ x: MLXArray, stride: Int) -> MLXArray {
        guard stride > 1 else { return x }

        let batch = x.dim(0)
        let channels = x.dim(1)
        let height = (x.dim(2) / stride) * stride
        let width = (x.dim(3) / stride) * stride

        var cropped = x[0..., 0..., 0..<height, 0..<width]
        cropped = cropped.reshaped(batch, channels, height / stride, stride, width / stride, stride)
        return cropped.mean(axes: [3, 5])
    }

    private static func createBaseXY(
        batchSize: Int,
        imageHeight: Int,
        imageWidth: Int,
        stride: Int,
        numLayers: Int,
        dtype: DType
    ) -> (x: MLXArray, y: MLXArray) {
        let widthIndices = Array(0..<(imageWidth / stride)).map { (Float($0) + 0.5) * Float(stride) }
        let heightIndices = Array(0..<(imageHeight / stride)).map { (Float($0) + 0.5) * Float(stride) }

        let xx = (2.0 * MLXArray(widthIndices).asType(dtype) / Float(imageWidth)) - 1.0
        let yy = (2.0 * MLXArray(heightIndices).asType(dtype) / Float(imageHeight)) - 1.0

        let baseHeight = imageHeight / stride
        let baseWidth = imageWidth / stride

        var xGrid = xx.reshaped(1, 1, 1, 1, baseWidth)
        xGrid = MLX.repeated(xGrid, count: baseHeight, axis: 3)
        xGrid = MLX.repeated(xGrid, count: batchSize, axis: 0)
        xGrid = MLX.repeated(xGrid, count: numLayers, axis: 2)

        var yGrid = yy.reshaped(1, 1, 1, baseHeight, 1)
        yGrid = MLX.repeated(yGrid, count: baseWidth, axis: 4)
        yGrid = MLX.repeated(yGrid, count: batchSize, axis: 0)
        yGrid = MLX.repeated(yGrid, count: numLayers, axis: 2)

        return (xGrid, yGrid)
    }

    private static func createBaseColors(
        image: MLXArray,
        stride: Int,
        numLayers: Int,
        option: SharpColorInitOption
    ) -> MLXArray {
        let pooled = avgPool2DStride(image, stride: stride)
        let batch = pooled.dim(0)
        let channels = pooled.dim(1)
        let height = pooled.dim(2)
        let width = pooled.dim(3)
        let dtype = pooled.dtype

        let grayValue = MLXArray(0.5).asType(dtype)
        let gray = MLX.full([batch, channels, numLayers, height, width], values: grayValue, dtype: dtype)

        switch option {
        case .none:
            return gray
        case .firstLayer:
            let first = pooled.expandedDimensions(axis: 2)
            if numLayers == 1 {
                return first
            }
            let rest = MLX.full(
                [batch, channels, numLayers - 1, height, width],
                values: grayValue,
                dtype: dtype
            )
            return MLX.concatenated([first, rest], axis: 2)
        case .allLayers:
            var all = pooled.expandedDimensions(axis: 2)
            all = MLX.repeated(all, count: numLayers, axis: 2)
            return all
        }
    }

    private static func rescaleDepth(
        _ depth: MLXArray,
        depthMin: Float = 1.0,
        depthMax: Float = 100.0
    ) -> (depth: MLXArray, depthFactor: MLXArray) {
        let currentDepthMin = depth.flattened(start: depth.ndim - 3).min(axis: -1)
        let depthFactor = depthMin / (currentDepthMin + 1e-6)
        let expanded = depthFactor.reshaped(depth.dim(0), 1, 1, 1)
        let scaled = MLX.clip(depth * expanded, max: depthMax)
        return (scaled, depthFactor)
    }
}
