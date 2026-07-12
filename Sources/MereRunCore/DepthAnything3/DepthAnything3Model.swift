@preconcurrency import MLX
import MLXNN

/// Exact native MLX implementation of the pinned Apache-2.0 DA3-Small graph.
///
/// Input is `[batch, views, height, width, 3]` RGB after ImageNet
/// normalization. Height and width must be divisible by 14 for the production
/// configuration. The checkpoint predicts relative—not metric—depth.
public final class DepthAnything3Model: Module {
    public let configuration: DepthAnything3Configuration

    @ModuleInfo(key: "backbone") private var backbone: DepthAnything3Backbone
    @ModuleInfo(key: "head") private var head: DepthAnything3DualDPT
    @ModuleInfo(key: "cam_dec") private var cameraDecoder: DepthAnything3CameraDecoder
    @ModuleInfo(key: "cam_enc") private var cameraEncoder: DepthAnything3CameraEncoder

    public init(configuration: DepthAnything3Configuration = .small) {
        self.configuration = configuration
        self._backbone.wrappedValue = DepthAnything3Backbone(configuration: configuration)
        self._head.wrappedValue = DepthAnything3DualDPT(configuration: configuration)
        self._cameraDecoder.wrappedValue = DepthAnything3CameraDecoder(configuration: configuration)
        self._cameraEncoder.wrappedValue = DepthAnything3CameraEncoder(configuration: configuration)
        super.init()
    }

    public func callAsFunction(
        _ normalizedImages: MLXArray,
        cameraConditioning: DepthAnything3CameraConditioning? = nil,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy = .saddleBalanced
    ) -> DepthAnything3RawOutput {
        precondition(normalizedImages.ndim == 5 && normalizedImages.dim(4) == 3)
        let batch = normalizedImages.dim(0)
        let views = normalizedImages.dim(1)
        let height = normalizedImages.dim(2)
        let width = normalizedImages.dim(3)
        precondition(batch > 0 && views > 0)
        precondition(height.isMultiple(of: configuration.patchSize))
        precondition(width.isMultiple(of: configuration.patchSize))
        if let cameraConditioning {
            precondition(cameraConditioning.extrinsics.dim(0) == batch)
            precondition(cameraConditioning.extrinsics.dim(1) == views)
        }

        let suppliedCameraToken = cameraConditioning.map {
            cameraEncoder($0, imageHeight: height, imageWidth: width)
        }
        let encoded = backbone(
            normalizedImages,
            suppliedCameraToken: suppliedCameraToken,
            referenceViewStrategy: referenceViewStrategy
        )
        let dense = head(
            features: encoded.patchFeatures,
            imageHeight: height,
            imageWidth: width,
            batch: batch,
            views: views
        )
        let cameras = cameraDecoder(
            encoded.cameraToken,
            imageHeight: height,
            imageWidth: width
        )
        return DepthAnything3RawOutput(
            depth: dense.depth,
            confidence: dense.confidence,
            extrinsics: cameras.extrinsics,
            intrinsics: cameras.intrinsics,
            ray: dense.ray,
            rayConfidence: dense.rayConfidence
        )
    }
}
