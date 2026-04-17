import MLX
import XCTest
@testable import MereRunCore

private final class MockMonodepthModel: SharpMonodepthModel {
    func callAsFunction(_ image: MLXArray) -> SharpMonodepthOutput {
        let disparity = MLXArray(Array(repeating: Float(2.0), count: 1 * 1 * 4 * 4), [1, 1, 4, 4]).asType(.float32)
        let features = [MLXArray(Array(repeating: Float(0.0), count: 1 * 8 * 2 * 2), [1, 8, 2, 2]).asType(.float32)]
        return SharpMonodepthOutput(disparity: disparity, outputFeatures: features)
    }

    func internalResolution() -> Int { 1536 }
}

private final class MockFeatureModel: SharpFeatureModel {
    func callAsFunction(_ featureInput: MLXArray, encodings: [MLXArray]) -> SharpImageFeatures {
        _ = encodings
        return SharpImageFeatures(textureFeatures: featureInput, geometryFeatures: featureInput)
    }
}

private final class MockPredictionHead: SharpPredictionHead {
    func callAsFunction(_ imageFeatures: SharpImageFeatures) -> MLXArray {
        _ = imageFeatures
        return MLXArray(Array(repeating: Float(0), count: 1 * 14 * 2 * 2 * 2), [1, 14, 2, 2, 2]).asType(.float32)
    }
}

private final class MockScaleMapEstimator: SharpScaleMapEstimator {
    private let value: Float

    init(value: Float) {
        self.value = value
    }

    func callAsFunction(monodepth: MLXArray, depth: MLXArray, depthDecoderFeatures: MLXArray?) -> MLXArray {
        _ = depth
        _ = depthDecoderFeatures
        return MLX.full(monodepth.shape, values: MLXArray(value).asType(monodepth.dtype), dtype: monodepth.dtype)
    }
}

private final class MockMonodepthPredictorCore: SharpMonodepthPredictorCore {
    func callAsFunction(_ image: MLXArray) -> SharpMonodepthPredictorOutput {
        _ = image
        let disparity = MLXArray(
            [
                1, 4, 2, 7,
                3, 2, 8, 1,
            ],
            [1, 2, 2, 2]
        ).asType(.float32)

        let encoder0 = MLXArray(Array(repeating: Float(0), count: 1 * 4 * 4 * 4), [1, 4, 4, 4]).asType(.float32)
        let encoder1 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 2 * 2), [1, 8, 2, 2]).asType(.float32)
        let decoder = MLXArray(Array(repeating: Float(0), count: 1 * 16 * 2 * 2), [1, 16, 2, 2]).asType(.float32)
        return SharpMonodepthPredictorOutput(
            disparity: disparity,
            encoderFeatures: [encoder0, encoder1],
            decoderFeatures: decoder
        )
    }

    func internalResolution() -> Int { 2048 }
    func encoderFeatureDims() -> [Int] { [4, 8] }
    func decoderFeatureDim() -> Int { 16 }
}

private final class MockMonodepthEncoder: SharpMonodepthEncoder {
    func callAsFunction(_ image: MLXArray) -> [MLXArray] {
        _ = image
        let enc0 = MLXArray(Array(repeating: Float(0), count: 1 * 4 * 8 * 8), [1, 4, 8, 8]).asType(.float32)
        let enc1 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 4 * 4), [1, 8, 4, 4]).asType(.float32)
        let enc2 = MLXArray(Array(repeating: Float(0), count: 1 * 16 * 2 * 2), [1, 16, 2, 2]).asType(.float32)
        let intermediate = MLXArray(Array(repeating: Float(0), count: 1 * 2 * 1 * 1), [1, 2, 1, 1]).asType(.float32)
        return [enc0, enc1, enc2, intermediate]
    }

    func featureDims() -> [Int] { [4, 8, 16] }
    func internalResolution() -> Int { 1536 }
}

final class SHARPPortTests: MereRunCoreTestCase {

    func testSharpMathInverseSigmoidRoundTrip() {
        let x = MLXArray([0.1, 0.25, 0.5, 0.75, 0.9]).asType(.float32)
        let roundTripped = sigmoid(SharpMath.inverseSigmoid(x))

        for i in 0..<5 {
            let actual = roundTripped[i].item(Float.self)
            let expected = x[i].item(Float.self)
            XCTAssertEqual(actual, expected, accuracy: 1e-5)
        }
    }

    func testAffineRangeNormalizerMatchesExpectedMapping() {
        let normalizer = SharpAffineRangeNormalizer(inputRange: (0, 1), outputRange: (-1, 1))
        let x = MLXArray([0.0, 0.5, 1.0], [1, 3, 1, 1]).asType(.float32)
        let y = normalizer(x)

        XCTAssertEqual(y[0, 0, 0, 0].item(Float.self), -1.0, accuracy: 1e-6)
        XCTAssertEqual(y[0, 1, 0, 0].item(Float.self), 0.0, accuracy: 1e-6)
        XCTAssertEqual(y[0, 2, 0, 0].item(Float.self), 1.0, accuracy: 1e-6)
    }

    func testMeanStdNormalizerMatchesExpectedScaling() {
        let normalizer = SharpMeanStdNormalizer(mean: [0.5], std: [0.25])
        let x = MLXArray([0.75], [1, 1, 1, 1]).asType(.float32)
        let y = normalizer(x)
        XCTAssertEqual(y[0, 0, 0, 0].item(Float.self), 1.0, accuracy: 1e-6)
    }

    func testInitializerShapesAndGlobalScale() {
        let image = MLXArray(Array(repeating: Float(0.5), count: 1 * 3 * 4 * 4), [1, 3, 4, 4]).asType(.float32)
        let depth = MLXArray(Array(repeating: Float(2.0), count: 1 * 1 * 4 * 4), [1, 1, 4, 4]).asType(.float32)

        let params = SharpInitializerParameters(
            stride: 2,
            numLayers: 2,
            firstLayerDepthOption: .surfaceMin,
            restLayerDepthOption: .surfaceMin,
            colorOption: .allLayers,
            normalizeDepth: true
        )
        let initializer = SharpInitializer(params: params)
        let output = initializer(image: image, depth: depth)

        XCTAssertEqual(output.featureInput.shape, [1, 4, 4, 4])
        XCTAssertEqual(output.gaussianBaseValues.meanXNDC.shape, [1, 1, 2, 2, 2])
        XCTAssertEqual(output.gaussianBaseValues.colors.shape, [1, 3, 2, 2, 2])

        guard let globalScale = output.globalScale else {
            XCTFail("Expected globalScale for normalized depth.")
            return
        }
        XCTAssertEqual(globalScale.shape, [1])
        XCTAssertEqual(globalScale[0].item(Float.self), 2.0, accuracy: 1e-5)
    }

    func testGaussianComposerPreservesBaseForZeroDelta() {
        let zeros3D = MLXArray(Array(repeating: Float(0), count: 1 * 1 * 1 * 2 * 2), [1, 1, 1, 2, 2]).asType(.float32)
        let ones3D = MLXArray(Array(repeating: Float(1), count: 1 * 1 * 1 * 2 * 2), [1, 1, 1, 2, 2]).asType(.float32)
        let colors = MLXArray(Array(repeating: Float(0.5), count: 1 * 3 * 1 * 2 * 2), [1, 3, 1, 2, 2]).asType(.float32)
        let quaternions = MLXArray([1, 0, 0, 0], [1, 4, 1, 1, 1]).asType(.float32)
        let opacities = MLXArray([0.5], [1]).asType(.float32)

        let base = SharpGaussianBaseValues(
            meanXNDC: zeros3D,
            meanYNDC: zeros3D,
            meanInverseZNDC: ones3D,
            scales: ones3D,
            quaternions: quaternions,
            colors: colors,
            opacities: opacities
        )

        let delta = MLXArray(Array(repeating: Float(0), count: 1 * 14 * 1 * 2 * 2), [1, 14, 1, 2, 2]).asType(.float32)
        let composer = SharpGaussianComposer()
        let gaussians = composer(delta: delta, baseValues: base, globalScale: MLXArray([2.0]).asType(.float32))

        XCTAssertEqual(gaussians.meanVectors.shape, [1, 4, 3])
        XCTAssertEqual(gaussians.singularValues.shape, [1, 4, 3])
        XCTAssertEqual(gaussians.quaternions.shape, [1, 4, 4])
        XCTAssertEqual(gaussians.colors.shape, [1, 4, 3])
        XCTAssertEqual(gaussians.opacities.shape, [1, 4])

        let z = gaussians.meanVectors[0, 0, 2].item(Float.self)
        XCTAssertEqual(z, 1.998, accuracy: 1e-2)

        let scale = gaussians.singularValues[0, 0, 0].item(Float.self)
        XCTAssertEqual(scale, 1.998, accuracy: 1e-4)

        let color = gaussians.colors[0, 0, 0].item(Float.self)
        XCTAssertGreaterThan(color, 0.20)
        XCTAssertLessThan(color, 0.30)

        let opacity = gaussians.opacities[0, 0].item(Float.self)
        XCTAssertEqual(opacity, 0.5, accuracy: 1e-5)
    }

    func testDepthAlignmentUsesEstimatorWhenDepthProvided() {
        let monodepth = MLXArray(Array(repeating: Float(1.0), count: 1 * 1 * 2 * 2), [1, 1, 2, 2]).asType(.float32)
        let depth = MLXArray(Array(repeating: Float(4.0), count: 1 * 1 * 2 * 2), [1, 1, 2, 2]).asType(.float32)

        let alignment = SharpDepthAlignment(scaleMapEstimator: MockScaleMapEstimator(value: 3.0))
        let result = alignment(monodepth: monodepth, depth: depth, depthDecoderFeatures: nil)

        XCTAssertEqual(result.alignmentMap[0, 0, 0, 0].item(Float.self), 3.0, accuracy: 1e-5)
        XCTAssertEqual(result.alignedMonodepth[0, 0, 0, 0].item(Float.self), 3.0, accuracy: 1e-5)
    }

    func testLearnedAlignmentMapShape() {
        let alignment = SharpLearnedAlignment(
            steps: 4,
            stride: 2,
            baseWidth: 8,
            depthDecoderFeatures: false,
            depthDecoderDim: 8,
            activationType: .exp
        )

        let monodepth = MLXArray(Array(repeating: Float(2.0), count: 1 * 1 * 16 * 16), [1, 1, 16, 16]).asType(.float32)
        let depth = MLXArray(Array(repeating: Float(4.0), count: 1 * 1 * 16 * 16), [1, 1, 16, 16]).asType(.float32)

        let map = alignment(monodepth: monodepth, depth: depth, depthDecoderFeatures: nil)
        XCTAssertEqual(map.shape, [1, 1, 16, 16])
    }

    func testLearnedAlignmentSupportsDepthDecoderFeatures() {
        let params = SharpAlignmentParameters(
            stride: 4,
            frozen: false,
            steps: 4,
            activationType: .exp,
            depthDecoderFeatures: true,
            baseWidth: 8
        )
        let alignment = createSharpAlignment(params: params, depthDecoderDim: 4)

        let monodepth = MLXArray(Array(repeating: Float(2.0), count: 1 * 1 * 16 * 16), [1, 1, 16, 16]).asType(.float32)
        let depth = MLXArray(Array(repeating: Float(4.0), count: 1 * 1 * 16 * 16), [1, 1, 16, 16]).asType(.float32)
        let decoderFeatures = MLXArray(Array(repeating: Float(0.25), count: 1 * 4 * 4 * 4), [1, 4, 4, 4]).asType(.float32)

        let map = alignment(monodepth: monodepth, depth: depth, depthDecoderFeatures: decoderFeatures)
        XCTAssertEqual(map.shape, [1, 1, 16, 16])
    }

    func testMonodepthAdaptorSortsTwoLayerDisparity() {
        let adaptor = SharpMonodepthWithEncodingAdaptor(
            monodepthPredictor: MockMonodepthPredictorCore(),
            returnEncoderFeatures: true,
            returnDecoderFeatures: true,
            numMonodepthLayers: 2,
            sortingMonodepth: true
        )
        let image = MLXArray(Array(repeating: Float(0), count: 1 * 3 * 4 * 4), [1, 3, 4, 4]).asType(.float32)
        let output = adaptor(image)

        XCTAssertEqual(output.disparity.shape, [1, 2, 2, 2])
        XCTAssertEqual(output.disparity[0, 0, 0, 0].item(Float.self), 3, accuracy: 1e-6)
        XCTAssertEqual(output.disparity[0, 1, 0, 0].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(output.outputFeatures.count, 3)
    }

    func testMonodepthAdaptorFeatureSelectionAndDims() {
        let adaptor = createSharpMonodepthAdaptor(
            monodepthPredictor: MockMonodepthPredictorCore(),
            params: SharpMonodepthAdaptorParameters(encoderFeatures: true, decoderFeatures: false),
            numMonodepthLayers: 1,
            sortingMonodepth: false
        )
        let image = MLXArray(Array(repeating: Float(0), count: 1 * 3 * 4 * 4), [1, 3, 4, 4]).asType(.float32)
        let output = adaptor(image)

        XCTAssertEqual(adaptor.getFeatureDims(), [4, 8])
        XCTAssertEqual(adaptor.internalResolution(), 2048)
        XCTAssertEqual(output.outputFeatures.count, 2)
        XCTAssertEqual(output.decoderFeatures?.shape, [1, 16, 2, 2])
    }

    func testMonodepthDPTCoreOutputAndAdaptorIntegration() {
        let encoder = MockMonodepthEncoder()
        let decoder = SharpMultiresConvDecoder(
            dimsEncoder: [4, 8, 16],
            dimsDecoder: [8, 8, 8],
            upsamplingMode: .transposedConv
        )
        let monodepth = SharpMonodepthDensePredictionTransformer(
            encoder: encoder,
            decoder: decoder,
            lastDims: (8, 1)
        )

        let image = MLXArray(Array(repeating: Float(0), count: 1 * 3 * 8 * 8), [1, 3, 8, 8]).asType(.float32)
        let output = monodepth(image)
        XCTAssertEqual(output.disparity.shape, [1, 1, 16, 16])
        XCTAssertEqual(output.encoderFeatures.count, 3)
        XCTAssertEqual(output.decoderFeatures.shape, [1, 8, 8, 8])
        XCTAssertEqual(output.intermediateFeatures.count, 1)
        XCTAssertEqual(monodepth.encoderFeatureDims(), [4, 8, 16])
        XCTAssertEqual(monodepth.decoderFeatureDim(), 8)
        XCTAssertEqual(monodepth.internalResolution(), 1536)

        let adaptor = createSharpMonodepthAdaptor(
            monodepthPredictor: monodepth,
            params: SharpMonodepthAdaptorParameters(encoderFeatures: true, decoderFeatures: true),
            numMonodepthLayers: 1,
            sortingMonodepth: false
        )
        let adapted = adaptor(image)
        XCTAssertEqual(adapted.disparity.shape, [1, 1, 16, 16])
        XCTAssertEqual(adapted.outputFeatures.count, 4)
    }

    func testPredictorPipelineContracts() {
        let image = MLXArray(Array(repeating: Float(0.5), count: 1 * 3 * 4 * 4), [1, 3, 4, 4]).asType(.float32)
        let disparityFactor = MLXArray([Float(2.0)]).asType(.float32)

        let initializer = SharpInitializer(params: SharpInitializerParameters(
            stride: 2,
            numLayers: 2,
            firstLayerDepthOption: .surfaceMin,
            restLayerDepthOption: .surfaceMin,
            colorOption: .allLayers,
            normalizeDepth: true
        ))
        let predictor = SharpRGBGaussianPredictor(
            initializer: initializer,
            monodepthModel: MockMonodepthModel(),
            featureModel: MockFeatureModel(),
            predictionHead: MockPredictionHead(),
            gaussianComposer: SharpGaussianComposer()
        )

        let gaussians = predictor(image: image, disparityFactor: disparityFactor, depth: nil)

        XCTAssertEqual(predictor.internalResolution(), 1536)
        XCTAssertEqual(gaussians.meanVectors.shape, [1, 8, 3])
        XCTAssertEqual(gaussians.singularValues.shape, [1, 8, 3])
        XCTAssertEqual(gaussians.quaternions.shape, [1, 8, 4])
        XCTAssertEqual(gaussians.colors.shape, [1, 8, 3])
        XCTAssertEqual(gaussians.opacities.shape, [1, 8])
    }

    func testDirectPredictionHeadOutputShape() {
        let head = SharpDirectPredictionHead(featureDim: 32, numLayers: 2)
        let texture = MLXArray(Array(repeating: Float(0), count: 1 * 32 * 8 * 8), [1, 32, 8, 8]).asType(.float32)
        let geometry = MLXArray(Array(repeating: Float(0), count: 1 * 32 * 8 * 8), [1, 32, 8, 8]).asType(.float32)

        let output = head(SharpImageFeatures(textureFeatures: texture, geometryFeatures: geometry))
        XCTAssertEqual(output.shape, [1, 14, 2, 8, 8])
    }

    func testMultiresConvDecoderOutputShape() {
        let decoder = SharpMultiresConvDecoder(
            dimsEncoder: [4, 8, 16],
            dimsDecoder: [8, 8, 8],
            upsamplingMode: .transposedConv
        )

        let enc0 = MLXArray(Array(repeating: Float(0), count: 1 * 4 * 16 * 16), [1, 4, 16, 16]).asType(.float32)
        let enc1 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 8 * 8), [1, 8, 8, 8]).asType(.float32)
        let enc2 = MLXArray(Array(repeating: Float(0), count: 1 * 16 * 4 * 4), [1, 16, 4, 4]).asType(.float32)

        let output = decoder([enc0, enc1, enc2])
        XCTAssertEqual(output.shape, [1, 8, 16, 16])
    }

    func testGaussianDecoderFeatureOutputShapes() {
        let params = SharpGaussianDecoderParameters(
            dimIn: 5,
            dimOut: 32,
            normType: .groupNorm,
            normNumGroups: 8,
            stride: 2,
            dimsDecoder: [16, 16, 16],
            useDepthInput: true,
            upsamplingMode: .transposedConv,
            imageEncoderType: .skipConvKernel2
        )
        let decoder = createSharpGaussianDecoder(params: params, dimsDepthFeatures: [8, 8, 8])

        let inputFeatures = MLXArray(Array(repeating: Float(0), count: 1 * 5 * 16 * 16), [1, 5, 16, 16]).asType(.float32)
        let enc0 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 8 * 8), [1, 8, 8, 8]).asType(.float32)
        let enc1 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 4 * 4), [1, 8, 4, 4]).asType(.float32)
        let enc2 = MLXArray(Array(repeating: Float(0), count: 1 * 8 * 2 * 2), [1, 8, 2, 2]).asType(.float32)

        let features = decoder(inputFeatures, encodings: [enc0, enc1, enc2])
        XCTAssertEqual(features.textureFeatures.shape, [1, 32, 8, 8])
        XCTAssertEqual(features.geometryFeatures.shape, [1, 32, 8, 8])
    }

    func testSharpViTOutputsFeaturesAndIntermediates() {
        let config = SharpViTConfig(
            imageSize: 16,
            patchSize: 4,
            inChannels: 3,
            embedDim: 64,
            depth: 2,
            numHeads: 4,
            initValues: 1e-5
        )
        let vit = SharpVisionTransformer(config: config, intermediateFeatureIDs: [0, 1])
        let image = MLXArray(Array(repeating: Float(0), count: 1 * 3 * 16 * 16), [1, 3, 16, 16]).asType(.float32)

        let output = vit(image)
        XCTAssertEqual(output.features.shape, [1, 64, 4, 4])
        XCTAssertEqual(output.intermediateFeatures.count, 2)
        XCTAssertEqual(output.intermediateFeatures[0]?.shape, [1, 17, 64])
        XCTAssertEqual(output.intermediateFeatures[1]?.shape, [1, 17, 64])
    }

    func testSlidingPyramidEncoderTinyPresetShapes() {
        let patch = createSharpViT(preset: "tiny16_64", intermediateFeatureIDs: [0, 1, 2, 3])
        let image = createSharpViT(preset: "tiny16_64", intermediateFeatureIDs: nil)
        let encoder = SharpSlidingPyramidNetwork(
            dimsEncoder: [8, 8, 16, 32, 32],
            patchEncoder: patch,
            imageEncoder: image,
            patchIntermediateFeatureIDs: [0, 1, 2, 3],
            usePatchOverlap: false
        )

        let input = MLXArray(Array(repeating: Float(0), count: 1 * 3 * 64 * 64), [1, 3, 64, 64]).asType(.float32)
        let output = encoder(input)

        XCTAssertEqual(encoder.internalResolution(), 64)
        XCTAssertEqual(encoder.featureDims(), [8, 8, 16, 32, 32])
        XCTAssertEqual(output.count, 5)
        XCTAssertEqual(output[0].shape, [1, 8, 32, 32])
        XCTAssertEqual(output[1].shape, [1, 8, 16, 16])
        XCTAssertEqual(output[2].shape, [1, 16, 8, 8])
        XCTAssertEqual(output[3].shape, [1, 32, 4, 4])
        XCTAssertEqual(output[4].shape, [1, 32, 2, 2])
    }

    func testCreateSharpPredictorTinyPresetRunsForward() {
        var params = SharpPredictorParameters()
        params.monodepth.patchEncoderPreset = "tiny16_64"
        params.monodepth.imageEncoderPreset = "tiny16_64"
        params.monodepth.dimsDecoder = [32, 32, 32, 32, 32]
        params.monodepth.usePatchOverlap = false
        params.gaussianDecoder = SharpGaussianDecoderParameters(
            dimIn: 5,
            dimOut: 16,
            normType: .groupNorm,
            normNumGroups: 8,
            stride: 2,
            dimsDecoder: [16, 16, 16, 16, 16],
            useDepthInput: true,
            upsamplingMode: .transposedConv,
            imageEncoderType: .skipConvKernel2
        )
        params.depthAlignment = SharpAlignmentParameters(
            stride: 1,
            frozen: false,
            steps: 2,
            activationType: .exp,
            depthDecoderFeatures: false,
            baseWidth: 8
        )
        params.numMonodepthLayers = 2
        params.sortingMonodepth = false

        let predictor = createSharpPredictor(params: params)
        let image = MLXArray(Array(repeating: Float(0.5), count: 1 * 3 * 64 * 64), [1, 3, 64, 64]).asType(.float32)
        let disparityFactor = MLXArray([Float(1.0)]).asType(.float32)

        let output = predictor(image: image, disparityFactor: disparityFactor, depth: nil)
        XCTAssertEqual(predictor.internalResolution(), 64)
        XCTAssertEqual(output.meanVectors.shape, [1, 2048, 3])
        XCTAssertEqual(output.singularValues.shape, [1, 2048, 3])
        XCTAssertEqual(output.quaternions.shape, [1, 2048, 4])
        XCTAssertEqual(output.colors.shape, [1, 2048, 3])
        XCTAssertEqual(output.opacities.shape, [1, 2048])
    }
}
