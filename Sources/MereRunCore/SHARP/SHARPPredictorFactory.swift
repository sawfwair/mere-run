import MLX

public func createSharpMonodepthDPT(
    params: SharpMonodepthParameters = SharpMonodepthParameters(),
    numMonodepthLayers: Int = 2
) -> SharpMonodepthDensePredictionTransformer {
    let encoder = createSharpMonodepthEncoder(
        patchEncoderPreset: params.patchEncoderPreset,
        imageEncoderPreset: params.imageEncoderPreset,
        usePatchOverlap: params.usePatchOverlap,
        lastEncoder: params.dimsDecoder.first ?? 256
    )

    let decoder = SharpMultiresConvDecoder(
        dimsEncoder: encoder.featureDims(),
        dimsDecoder: params.dimsDecoder,
        upsamplingMode: .transposedConv
    )

    return SharpMonodepthDensePredictionTransformer(
        encoder: encoder,
        decoder: decoder,
        lastDims: (32, numMonodepthLayers)
    )
}

public func createSharpPredictor(
    params: SharpPredictorParameters = SharpPredictorParameters()
) -> SharpRGBGaussianPredictor {
    precondition(
        params.gaussianDecoder.stride >= params.initializer.stride,
        "gaussianDecoder stride must be >= initializer stride."
    )

    if params.numMonodepthLayers > 1 {
        precondition(
            params.initializer.numLayers == 2,
            "initializer.numLayers must be 2 when numMonodepthLayers > 1."
        )
    }

    let scaleFactor = params.gaussianDecoder.stride / params.initializer.stride

    let composerParams = SharpComposerParameters(
        deltaFactor: params.deltaFactor,
        minScale: params.minScale,
        maxScale: params.maxScale,
        colorActivationType: params.colorActivationType,
        opacityActivationType: params.opacityActivationType,
        colorSpace: params.colorSpace,
        scaleFactor: scaleFactor,
        baseScaleOnPredictedMean: params.baseScaleOnPredictedMean
    )
    let gaussianComposer = SharpGaussianComposer(params: composerParams)

    let monodepthPredictor = createSharpMonodepthDPT(
        params: params.monodepth,
        numMonodepthLayers: params.numMonodepthLayers
    )

    let monodepthAdaptor = createSharpMonodepthAdaptor(
        monodepthPredictor: monodepthPredictor,
        params: params.monodepthAdaptor,
        numMonodepthLayers: params.numMonodepthLayers,
        sortingMonodepth: params.sortingMonodepth
    )

    let gaussianDecoder = createSharpGaussianDecoder(
        params: params.gaussianDecoder,
        dimsDepthFeatures: monodepthAdaptor.getFeatureDims()
    )

    let initializer = SharpInitializer(params: params.initializer)
    let predictionHead = SharpDirectPredictionHead(
        featureDim: gaussianDecoder.dimOut,
        numLayers: initializer.params.numLayers
    )

    let depthDecoderDim = monodepthPredictor.decoder.dimsDecoder.last ?? params.monodepth.dimsDecoder.last ?? 256
    let alignment = createSharpAlignment(params: params.depthAlignment, depthDecoderDim: depthDecoderDim)

    return SharpRGBGaussianPredictor(
        initializer: initializer,
        monodepthModel: monodepthAdaptor,
        featureModel: gaussianDecoder,
        predictionHead: predictionHead,
        gaussianComposer: gaussianComposer,
        scaleMapEstimator: alignment
    )
}
