import Foundation
import MLX
import MLXNN

final class SharpSPNProjectUpsampleBlock: Module, @unchecked Sendable {
    @ModuleInfo(key: "projection") var projection: SharpConv2dNCHW
    @ModuleInfo(key: "upsamplers") var upsamplers: [SharpConvTransposed2dNCHW]

    init(dimIn: Int, dimOut: Int, upsampleLayers: Int, dimIntermediate: Int? = nil) {
        let intermediate = dimIntermediate ?? dimOut

        self._projection.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimIn,
            outputChannels: intermediate,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: false
        )

        var layers: [SharpConvTransposed2dNCHW] = []
        for i in 0..<upsampleLayers {
            layers.append(SharpConvTransposed2dNCHW(
                inputChannels: i == 0 ? intermediate : dimOut,
                outputChannels: dimOut,
                kernelSize: 2,
                stride: 2,
                padding: 0,
                bias: false
            ))
        }
        self._upsamplers.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = projection(x)
        for layer in upsamplers {
            y = layer(y)
        }
        return y
    }
}

public final class SharpSlidingPyramidNetwork: Module, SharpMonodepthEncoder, @unchecked Sendable {
    public let dimsEncoder: [Int]
    public let patchSize: Int
    public let usePatchOverlap: Bool

    @ModuleInfo(key: "patch_encoder") var patchEncoder: SharpVisionTransformer
    @ModuleInfo(key: "image_encoder") var imageEncoder: SharpVisionTransformer

    @ModuleInfo(key: "upsample_latent0") var upsampleLatent0: SharpSPNProjectUpsampleBlock
    @ModuleInfo(key: "upsample_latent1") var upsampleLatent1: SharpSPNProjectUpsampleBlock
    @ModuleInfo(key: "upsample0") var upsample0: SharpSPNProjectUpsampleBlock
    @ModuleInfo(key: "upsample1") var upsample1: SharpSPNProjectUpsampleBlock
    @ModuleInfo(key: "upsample2") var upsample2: SharpSPNProjectUpsampleBlock

    @ModuleInfo(key: "upsample_lowres") var upsampleLowres: SharpConvTransposed2dNCHW
    @ModuleInfo(key: "fuse_lowres") var fuseLowres: SharpConv2dNCHW

    let patchIntermediateFeatureIDs: [Int]

    public init(
        dimsEncoder: [Int],
        patchEncoder: SharpVisionTransformer,
        imageEncoder: SharpVisionTransformer,
        patchIntermediateFeatureIDs: [Int],
        usePatchOverlap: Bool = true
    ) {
        precondition(dimsEncoder.count == 5, "SharpSlidingPyramidNetwork expects exactly five encoder dimensions.")
        precondition(patchIntermediateFeatureIDs.count == 4, "Patch intermediate feature IDs must contain four entries.")

        self.dimsEncoder = dimsEncoder
        self.patchSize = patchEncoder.internalResolution()
        self.usePatchOverlap = usePatchOverlap
        self.patchIntermediateFeatureIDs = patchIntermediateFeatureIDs

        self._patchEncoder.wrappedValue = patchEncoder
        self._imageEncoder.wrappedValue = imageEncoder

        self._upsampleLatent0.wrappedValue = SharpSPNProjectUpsampleBlock(
            dimIn: patchEncoder.config.embedDim,
            dimOut: dimsEncoder[0],
            upsampleLayers: 3,
            dimIntermediate: dimsEncoder[1]
        )
        self._upsampleLatent1.wrappedValue = SharpSPNProjectUpsampleBlock(
            dimIn: patchEncoder.config.embedDim,
            dimOut: dimsEncoder[1],
            upsampleLayers: 2
        )

        self._upsample0.wrappedValue = SharpSPNProjectUpsampleBlock(
            dimIn: patchEncoder.config.embedDim,
            dimOut: dimsEncoder[2],
            upsampleLayers: 1
        )
        self._upsample1.wrappedValue = SharpSPNProjectUpsampleBlock(
            dimIn: patchEncoder.config.embedDim,
            dimOut: dimsEncoder[3],
            upsampleLayers: 1
        )
        self._upsample2.wrappedValue = SharpSPNProjectUpsampleBlock(
            dimIn: patchEncoder.config.embedDim,
            dimOut: dimsEncoder[4],
            upsampleLayers: 1
        )

        self._upsampleLowres.wrappedValue = SharpConvTransposed2dNCHW(
            inputChannels: imageEncoder.config.embedDim,
            outputChannels: dimsEncoder[4],
            kernelSize: 2,
            stride: 2,
            padding: 0,
            bias: true
        )
        self._fuseLowres.wrappedValue = SharpConv2dNCHW(
            inputChannels: dimsEncoder[4] + dimsEncoder[4],
            outputChannels: dimsEncoder[4],
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: true
        )
        super.init()
    }

    public func internalResolution() -> Int {
        patchSize * 4
    }

    public func featureDims() -> [Int] {
        dimsEncoder
    }

    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let batchSize = x.dim(0)

        let (x0, x1, x2) = createPyramid(x)

        let x0Patches: MLXArray
        let x1Patches: MLXArray
        let x2Patches: MLXArray
        let padding: Int

        if usePatchOverlap {
            x0Patches = splitPatches(x0, overlapRatio: 0.25, patchSize: patchSize)
            x1Patches = splitPatches(x1, overlapRatio: 0.5, patchSize: patchSize)
            x2Patches = x2
            padding = 3
        } else {
            x0Patches = splitPatches(x0, overlapRatio: 0.0, patchSize: patchSize)
            x1Patches = splitPatches(x1, overlapRatio: 0.0, patchSize: patchSize)
            x2Patches = x2
            padding = 0
        }

        let x0TileSize = x0Patches.dim(0) / batchSize
        let xPyramidPatches = MLX.concatenated([x0Patches, x1Patches, x2Patches], axis: 0)

        let patchOutputs = patchEncoder(xPyramidPatches)
        let xPyramidEncodings = patchOutputs.features
        let patchIntermediate = patchOutputs.intermediateFeatures

        let latent0Key = patchIntermediateFeatureIDs[0]
        let latent1Key = patchIntermediateFeatureIDs[1]
        guard let latent0Tokens = patchIntermediate[latent0Key],
              let latent1Tokens = patchIntermediate[latent1Key]
        else {
            fatalError("Missing patch encoder intermediate features for SHARP SPN assembly.")
        }

        let latent0Encodings = patchEncoder.reshapeFeature(latent0Tokens)
        let latent1Encodings = patchEncoder.reshapeFeature(latent1Tokens)

        let xLatent0Features = mergePatches(
            latent0Encodings[0..<(batchSize * x0TileSize), 0..., 0..., 0...],
            batchSize: batchSize,
            padding: padding
        )
        let xLatent1Features = mergePatches(
            latent1Encodings[0..<(batchSize * x0TileSize), 0..., 0..., 0...],
            batchSize: batchSize,
            padding: padding
        )

        let x0Count = x0Patches.dim(0)
        let x1Count = x1Patches.dim(0)
        let x2Count = x2Patches.dim(0)

        let x0Encodings = xPyramidEncodings[0..<x0Count, 0..., 0..., 0...]
        let x1Encodings = xPyramidEncodings[x0Count..<(x0Count + x1Count), 0..., 0..., 0...]
        let x2Encodings = xPyramidEncodings[(x0Count + x1Count)..<(x0Count + x1Count + x2Count), 0..., 0..., 0...]

        let x0Features = mergePatches(x0Encodings, batchSize: batchSize, padding: padding)
        let x1Features = mergePatches(x1Encodings, batchSize: batchSize, padding: 2 * padding)
        let x2Features = x2Encodings

        let lowres = imageEncoder(x2Patches).features

        let latent0Up = upsampleLatent0(xLatent0Features)
        let latent1Up = upsampleLatent1(xLatent1Features)

        let up0 = upsample0(x0Features)
        let up1 = upsample1(x1Features)
        let up2 = upsample2(x2Features)

        var lowresUp = upsampleLowres(lowres)
        lowresUp = fuseLowres(MLX.concatenated([up2, lowresUp], axis: 1))

        return [latent0Up, latent1Up, up0, up1, lowresUp]
    }

    private func createPyramid(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let x0 = x
        let x1 = downsampleBy2(x0)
        let x2 = downsampleBy2(x1)
        return (x0, x1, x2)
    }

    private func downsampleBy2(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let channels = x.dim(1)
        let height = (x.dim(2) / 2) * 2
        let width = (x.dim(3) / 2) * 2

        var cropped = x[0..., 0..., 0..<height, 0..<width]
        cropped = cropped.reshaped(batch, channels, height / 2, 2, width / 2, 2)
        return cropped.mean(axes: [3, 5])
    }
}

func splitPatches(_ image: MLXArray, overlapRatio: Float = 0.25, patchSize: Int) -> MLXArray {
    let patchStride = Int(Float(patchSize) * (1.0 - overlapRatio))
    let imageSize = image.dim(3)
    let steps = Int(ceil(Double(imageSize - patchSize) / Double(patchStride))) + 1

    var patches: [MLXArray] = []
    patches.reserveCapacity(steps * steps)

    for j in 0..<steps {
        let j0 = j * patchStride
        let j1 = j0 + patchSize

        for i in 0..<steps {
            let i0 = i * patchStride
            let i1 = i0 + patchSize
            patches.append(image[0..., 0..., j0..<j1, i0..<i1])
        }
    }

    return MLX.concatenated(patches, axis: 0)
}

func mergePatches(_ imagePatches: MLXArray, batchSize: Int, padding: Int = 3) -> MLXArray {
    let steps = Int(sqrt(Double(imagePatches.dim(0) / batchSize)))

    var index = 0
    var rows: [MLXArray] = []
    rows.reserveCapacity(steps)

    for j in 0..<steps {
        var columns: [MLXArray] = []
        columns.reserveCapacity(steps)

        for i in 0..<steps {
            let start = batchSize * index
            let end = batchSize * (index + 1)
            var patch = imagePatches[start..<end, 0..., 0..., 0...]

            if padding != 0 {
                if j != 0 {
                    patch = patch[0..., 0..., padding..., 0...]
                }
                if i != 0 {
                    patch = patch[0..., 0..., 0..., padding...]
                }
                if j != steps - 1 {
                    patch = patch[0..., 0..., 0..<(patch.dim(2) - padding), 0...]
                }
                if i != steps - 1 {
                    patch = patch[0..., 0..., 0..., 0..<(patch.dim(3) - padding)]
                }
            }

            columns.append(patch)
            index += 1
        }

        rows.append(MLX.concatenated(columns, axis: 3))
    }

    return MLX.concatenated(rows, axis: 2)
}

public func createSharpMonodepthEncoder(
    patchEncoderPreset: String,
    imageEncoderPreset: String,
    usePatchOverlap: Bool = true,
    lastEncoder: Int = 256
) -> SharpSlidingPyramidNetwork {
    guard let baseEncoderDims = SharpMonodepthPresets.encoderDims(patchEncoderPreset) else {
        fatalError("Unsupported SHARP monodepth patch encoder preset: \(patchEncoderPreset)")
    }
    guard let patchHookIDs = SharpMonodepthPresets.hookIDs(patchEncoderPreset) else {
        fatalError("Missing SHARP patch hook IDs for preset: \(patchEncoderPreset)")
    }

    let dimsEncoder = [lastEncoder] + baseEncoderDims

    let patchEncoder = createSharpViT(
        preset: patchEncoderPreset,
        intermediateFeatureIDs: patchHookIDs
    )
    let imageEncoder = createSharpViT(
        preset: imageEncoderPreset,
        intermediateFeatureIDs: nil
    )

    return SharpSlidingPyramidNetwork(
        dimsEncoder: dimsEncoder,
        patchEncoder: patchEncoder,
        imageEncoder: imageEncoder,
        patchIntermediateFeatureIDs: patchHookIDs,
        usePatchOverlap: usePatchOverlap
    )
}
