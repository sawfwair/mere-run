import Foundation
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
#endif

public final class SHARPNativePredictor: @unchecked Sendable {
    public enum SHARPNativePredictorError: Swift.Error, LocalizedError {
        case unsupportedPlatform
        case predictorCreationFailed
        case incompatiblePredictorStructure
        case checkpointNotFound(URL)
        case imageLoadFailed(URL)
        case checkpointLoadFailed(String)
        case noMappedWeights(String)
        case outputWriteFailed(URL, String)

        public var errorDescription: String? {
            switch self {
            case .unsupportedPlatform:
                return "Native SHARP requires CoreGraphics/ImageIO."
            case .predictorCreationFailed:
                return "Failed to create SHARP predictor."
            case .incompatiblePredictorStructure:
                return "Unexpected SHARP predictor structure."
            case .checkpointNotFound(let url):
                return "Native SHARP checkpoint not found at \(url.path)."
            case .imageLoadFailed(let url):
                return "Failed to load image at \(url.path)."
            case .checkpointLoadFailed(let reason):
                return "Failed to load native SHARP checkpoint: \(reason)"
            case .noMappedWeights(let module):
                return "No SHARP weights mapped for module: \(module)"
            case .outputWriteFailed(let url, let reason):
                return "Failed to write PLY at \(url.path): \(reason)"
            }
        }
    }

    public struct InferenceStatus: Sendable {
        public let stage: String
        public let progress: Double

        public init(stage: String, progress: Double) {
            self.stage = stage
            self.progress = progress
        }
    }

    private let lock = NSLock()
    private var predictor: SharpRGBGaussianPredictor?
    private var loadedCheckpointPath: String?

    public init() {}

    @discardableResult
    public func predict(
        inputImageURL: URL,
        checkpointURL: URL,
        outputPLYURL: URL,
        status: (@Sendable (InferenceStatus) -> Void)? = nil
    ) throws -> URL {
        lock.lock()
        defer { lock.unlock() }

        status?(InferenceStatus(stage: "Loading model…", progress: 0.0))
        try ensureModelLoaded(checkpointURL: checkpointURL)

        status?(InferenceStatus(stage: "Preprocessing…", progress: 0.15))
        let preprocessed = try Self.preprocessImage(inputImageURL)

        guard let predictor else {
            throw SHARPNativePredictorError.predictorCreationFailed
        }

        status?(InferenceStatus(stage: "Running inference…", progress: 0.45))
        let disparityFactor = MLXArray([preprocessed.focalPx / Float(preprocessed.originalWidth)]).asType(.float32)
        var gaussians = predictor(
            image: preprocessed.image,
            disparityFactor: disparityFactor,
            depth: nil
        )

        status?(InferenceStatus(stage: "Postprocessing…", progress: 0.8))
        gaussians = Self.applyApproximateUnprojection(
            gaussians,
            focalPx: preprocessed.focalPx,
            width: preprocessed.originalWidth,
            height: preprocessed.originalHeight
        )

        status?(InferenceStatus(stage: "Saving PLY…", progress: 0.92))
        try Self.writePLY(gaussians: gaussians, to: outputPLYURL)

        status?(InferenceStatus(stage: "Done", progress: 1.0))
        return outputPLYURL
    }

    // MARK: - Model Loading

    private func ensureModelLoaded(checkpointURL: URL) throws {
        let normalizedPath = checkpointURL.standardizedFileURL.path
        if loadedCheckpointPath == normalizedPath, predictor != nil {
            return
        }

        guard FileManager.default.fileExists(atPath: normalizedPath) else {
            throw SHARPNativePredictorError.checkpointNotFound(checkpointURL)
        }

        let predictor = createSharpPredictor()
        guard let monodepthAdaptor = predictor.monodepthModel as? SharpMonodepthWithEncodingAdaptor,
              let monodepthCore = monodepthAdaptor.monodepthPredictor as? SharpMonodepthDensePredictionTransformer,
              let monodepthEncoder = monodepthCore.encoder as? SharpSlidingPyramidNetwork,
              let featureModel = predictor.featureModel as? SharpGaussianDensePredictionTransformer,
              let predictionHead = predictor.predictionHead as? SharpDirectPredictionHead
        else {
            throw SHARPNativePredictorError.incompatiblePredictorStructure
        }

        let arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: checkpointURL)
        } catch {
            throw SHARPNativePredictorError.checkpointLoadFailed(error.localizedDescription)
        }

        let monodepthCoreParams = Dictionary(uniqueKeysWithValues: monodepthCore.parameters().flattened())
        let monodepthEncoderParams = Dictionary(uniqueKeysWithValues: monodepthEncoder.parameters().flattened())
        let featureModelParams = Dictionary(uniqueKeysWithValues: featureModel.parameters().flattened())
        let predictionHeadParams = Dictionary(uniqueKeysWithValues: predictionHead.parameters().flattened())

        var monodepthCoreUpdates: [String: MLXArray] = [:]
        var monodepthEncoderUpdates: [String: MLXArray] = [:]
        var featureModelUpdates: [String: MLXArray] = [:]
        var predictionHeadUpdates: [String: MLXArray] = [:]

        for (sourceKey, sourceValue) in arrays {
            if sourceKey.hasPrefix("monodepth_model.monodepth_predictor.encoder.") {
                let raw = String(sourceKey.dropFirst("monodepth_model.monodepth_predictor.encoder.".count))
                if let mapped = Self.mapMonodepthEncoderKey(raw),
                   let resolved = Self.resolveCandidateKey(mapped, targetParameters: monodepthEncoderParams),
                   let adapted = Self.adapt(value: sourceValue, to: monodepthEncoderParams[resolved]!)
                {
                    monodepthEncoderUpdates[resolved] = adapted
                }
                continue
            }

            if sourceKey.hasPrefix("monodepth_model.monodepth_predictor.") {
                let raw = String(sourceKey.dropFirst("monodepth_model.monodepth_predictor.".count))
                if let mapped = Self.mapMonodepthCoreKey(raw),
                   let resolved = Self.resolveCandidateKey(mapped, targetParameters: monodepthCoreParams),
                   let adapted = Self.adapt(value: sourceValue, to: monodepthCoreParams[resolved]!)
                {
                    monodepthCoreUpdates[resolved] = adapted
                }
                continue
            }

            if sourceKey.hasPrefix("feature_model.") {
                let raw = String(sourceKey.dropFirst("feature_model.".count))
                if let mapped = Self.mapFeatureModelKey(raw),
                   let resolved = Self.resolveCandidateKey(mapped, targetParameters: featureModelParams),
                   let adapted = Self.adapt(value: sourceValue, to: featureModelParams[resolved]!)
                {
                    featureModelUpdates[resolved] = adapted
                }
                continue
            }

            if sourceKey.hasPrefix("prediction_head.") {
                let raw = String(sourceKey.dropFirst("prediction_head.".count))
                if let resolved = Self.resolveCandidateKey(raw, targetParameters: predictionHeadParams),
                   let adapted = Self.adapt(value: sourceValue, to: predictionHeadParams[resolved]!)
                {
                    predictionHeadUpdates[resolved] = adapted
                }
            }
        }

        if monodepthCoreUpdates.isEmpty {
            throw SHARPNativePredictorError.noMappedWeights("monodepth_core")
        }
        if monodepthEncoderUpdates.isEmpty {
            throw SHARPNativePredictorError.noMappedWeights("monodepth_encoder")
        }
        if featureModelUpdates.isEmpty {
            throw SHARPNativePredictorError.noMappedWeights("feature_model")
        }
        if predictionHeadUpdates.isEmpty {
            throw SHARPNativePredictorError.noMappedWeights("prediction_head")
        }

        try monodepthCore.update(parameters: ModuleParameters.unflattened(monodepthCoreUpdates), verify: .none)
        try monodepthEncoder.update(parameters: ModuleParameters.unflattened(monodepthEncoderUpdates), verify: .none)
        try featureModel.update(parameters: ModuleParameters.unflattened(featureModelUpdates), verify: .none)
        try predictionHead.update(parameters: ModuleParameters.unflattened(predictionHeadUpdates), verify: .none)

        self.predictor = predictor
        self.loadedCheckpointPath = normalizedPath
    }

    private static func mapMonodepthCoreKey(_ key: String) -> String? {
        if let mappedHead = mapMonodepthHeadKey(key) {
            return mappedHead
        }
        var mapped = key
        mapped = mapMultiresDecoderKey(mapped)
        mapped = mapFeatureFusionKey(mapped)
        mapped = mapResidualBlockKey(mapped)
        return mapped
    }

    private static func mapMonodepthEncoderKey(_ key: String) -> String? {
        if let mapped = mapProjectUpsampleSequentialKey(key) {
            return mapped.replacingOccurrences(of: ".attn.", with: ".attention.")
        }

        var mapped = key.replacingOccurrences(of: ".attn.", with: ".attention.")
        if mapped.hasPrefix("fuse_lowres.") {
            mapped = mapped.replacingOccurrences(of: "fuse_lowres.", with: "fuse_lowres.conv.")
        } else if mapped.hasPrefix("upsample_lowres.") {
            mapped = mapped.replacingOccurrences(of: "upsample_lowres.", with: "upsample_lowres.conv_transposed.")
        }
        return mapped
    }

    private static func mapFeatureModelKey(_ key: String) -> String? {
        var mapped = key

        if mapped.hasPrefix("image_encoder.conv.") {
            mapped = mapped.replacingOccurrences(of: "image_encoder.conv.", with: "image_encoder.conv.conv.")
        }

        if mapped.hasPrefix("texture_head.") || mapped.hasPrefix("geometry_head.") {
            let components = mapped.split(separator: ".", omittingEmptySubsequences: false)
            if components.count >= 3 {
                let head = String(components[0])
                let block = String(components[1])
                let rest = components.dropFirst(2).joined(separator: ".")
                switch block {
                case "0":
                    mapped = "\(head).residual_1.\(rest)"
                case "1":
                    mapped = "\(head).residual_2.\(rest)"
                case "3":
                    mapped = "\(head).conv_out.\(rest)"
                default:
                    break
                }
            }
        }

        mapped = mapMultiresDecoderKey(mapped)
        mapped = mapFeatureFusionKey(mapped)
        mapped = mapResidualBlockKey(mapped)
        return mapped
    }

    private static func mapMonodepthHeadKey(_ key: String) -> String? {
        let components = key.split(separator: ".", omittingEmptySubsequences: false)
        guard components.count == 3, components[0] == "head" else { return nil }
        guard let index = Int(components[1]) else { return nil }
        let parameter = String(components[2])

        switch index {
        case 0:
            return "head_conv_0.conv.\(parameter)"
        case 1:
            return "head_deconv.conv_transposed.\(parameter)"
        case 2:
            return "head_conv_1.conv.\(parameter)"
        case 4:
            return "head_conv_2.conv.\(parameter)"
        default:
            return nil
        }
    }

    private static func mapProjectUpsampleSequentialKey(_ key: String) -> String? {
        let components = key.split(separator: ".", omittingEmptySubsequences: false)
        guard components.count == 3 else { return nil }
        let block = String(components[0])
        let blockNames = Set(["upsample_latent0", "upsample_latent1", "upsample0", "upsample1", "upsample2"])
        guard blockNames.contains(block), let index = Int(components[1]) else { return nil }
        let parameter = String(components[2])

        if index == 0 {
            return "\(block).projection.conv.\(parameter)"
        }
        return "\(block).upsamplers.\(index - 1).conv_transposed.\(parameter)"
    }

    private static func mapMultiresDecoderKey(_ key: String) -> String {
        let components = key.split(separator: ".", omittingEmptySubsequences: false)
        if components.count == 4,
           components[0] == "decoder",
           components[1] == "convs",
           components[3] == "weight"
        {
            return "decoder.convs.\(components[2]).conv.conv.weight"
        }
        return key
    }

    private static func mapFeatureFusionKey(_ key: String) -> String {
        var mapped = key
        if mapped.hasSuffix(".deconv.weight") {
            mapped = String(mapped.dropLast(".deconv.weight".count)) + ".deconv.transposed.conv_transposed.weight"
        }

        let components = mapped.split(separator: ".", omittingEmptySubsequences: false)
        if components.count >= 5 {
            for i in 0..<(components.count - 3) {
                let isResnet = (components[i] == "resnet1" || components[i] == "resnet2")
                if isResnet, components[i + 1] == "residual" {
                    let paramIndex = String(components[i + 2])
                    let paramName = String(components[i + 3])
                    let prefix = components[..<i].joined(separator: ".")
                    let suffix = components.dropFirst(i + 4).joined(separator: ".")

                    let mappedBlock: String?
                    switch paramIndex {
                    case "1":
                        mappedBlock = "conv_1.conv.\(paramName)"
                    case "3":
                        mappedBlock = "conv_2.conv.\(paramName)"
                    default:
                        mappedBlock = nil
                    }

                    if let mappedBlock {
                        if prefix.isEmpty {
                            return suffix.isEmpty ? mappedBlock : "\(mappedBlock).\(suffix)"
                        }
                        if suffix.isEmpty {
                            return "\(prefix).\(components[i]).\(mappedBlock)"
                        }
                        return "\(prefix).\(components[i]).\(mappedBlock).\(suffix)"
                    }
                }
            }
        }
        return mapped
    }

    private static func mapResidualBlockKey(_ key: String) -> String {
        let components = key.split(separator: ".", omittingEmptySubsequences: false)
        guard components.count >= 3 else { return key }

        // Map residual_block_2d sequence indices:
        // [norm1, relu, conv1, norm2, relu, conv2]
        if let residualIndex = components.firstIndex(of: "residual"),
           residualIndex + 2 < components.count
        {
            let stage = String(components[residualIndex + 1])
            let parameter = String(components[residualIndex + 2])
            let prefix = components[..<residualIndex].joined(separator: ".")
            let suffix = components.dropFirst(residualIndex + 3).joined(separator: ".")

            let mappedStage: String?
            switch stage {
            case "0":
                mappedStage = "norm_1.group_norm.\(parameter)"
            case "2":
                mappedStage = "conv_1.conv.\(parameter)"
            case "3":
                mappedStage = "norm_2.group_norm.\(parameter)"
            case "5":
                mappedStage = "conv_2.conv.\(parameter)"
            default:
                mappedStage = nil
            }

            if let mappedStage {
                if prefix.isEmpty {
                    return suffix.isEmpty ? mappedStage : "\(mappedStage).\(suffix)"
                }
                if suffix.isEmpty {
                    return "\(prefix).\(mappedStage)"
                }
                return "\(prefix).\(mappedStage).\(suffix)"
            }
        }

        if key.hasSuffix(".shortcut.weight") {
            return String(key.dropLast(".shortcut.weight".count)) + ".shortcut.conv.weight"
        }
        if key.hasSuffix(".shortcut.bias") {
            return String(key.dropLast(".shortcut.bias".count)) + ".shortcut.conv.bias"
        }

        return key
    }

    private static func resolveCandidateKey(
        _ candidate: String,
        targetParameters: [String: MLXArray]
    ) -> String? {
        if targetParameters[candidate] != nil {
            return candidate
        }

        guard let dot = candidate.lastIndex(of: ".") else {
            return nil
        }
        let base = String(candidate[..<dot])
        let parameter = String(candidate[candidate.index(after: dot)...])
        if parameter != "weight" && parameter != "bias" && parameter != "running_mean" && parameter != "running_var" {
            return nil
        }

        let attempts = [
            "\(base).conv.\(parameter)",
            "\(base).conv.conv.\(parameter)",
            "\(base).conv_transposed.\(parameter)",
            "\(base).transposed.conv_transposed.\(parameter)",
            "\(base).group_norm.\(parameter)",
            "\(base).batch_norm.\(parameter)",
            "\(base).instance_norm.\(parameter)",
        ]
        return attempts.first(where: { targetParameters[$0] != nil })
    }

    private static func adapt(value: MLXArray, to target: MLXArray) -> MLXArray? {
        if value.shape == target.shape {
            return value.asType(target.dtype)
        }

        if value.ndim == 4 {
            let permutations = [
                (0, 2, 3, 1),  // OIHW -> OHWI
                (1, 2, 3, 0),  // IOHW -> OHWI
                (2, 3, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 0, 1),
                (3, 2, 1, 0),
            ]
            for perm in permutations {
                let permuted = value.transposed(perm.0, perm.1, perm.2, perm.3)
                if permuted.shape == target.shape {
                    return permuted.asType(target.dtype)
                }
            }
        }

        if value.ndim == 2 {
            let transposed = value.transposed(1, 0)
            if transposed.shape == target.shape {
                return transposed.asType(target.dtype)
            }
        }

        return nil
    }

    // MARK: - Preprocess / Postprocess

    #if canImport(CoreGraphics)
    private struct PreprocessedInput {
        let image: MLXArray
        let originalWidth: Int
        let originalHeight: Int
        let focalPx: Float
    }

    private static func preprocessImage(_ imageURL: URL) throws -> PreprocessedInput {
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw SHARPNativePredictorError.imageLoadFailed(imageURL)
        }

        let width = image.width
        let height = image.height
        let focalPx = inferFocalLengthPx(source: source, width: width, height: height)

        let tensor = try QwenImageIO.resizedCenterCropPixelArray(
            from: image,
            width: 1536,
            height: 1536,
            addBatchDimension: true,
            dtype: .float32
        )

        return PreprocessedInput(
            image: tensor,
            originalWidth: width,
            originalHeight: height,
            focalPx: focalPx
        )
    }

    private static func inferFocalLengthPx(source: CGImageSource, width: Int, height: Int) -> Float {
        var focal35mm: Float?
        var focalMm: Float?

        if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any]
        {
            if let value = exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber {
                focal35mm = value.floatValue
            }
            if let value = exif[kCGImagePropertyExifFocalLength] as? NSNumber {
                focalMm = value.floatValue
            }
        }

        var focalEquivalent35 = focal35mm ?? 0
        if focalEquivalent35 < 1 {
            focalEquivalent35 = focalMm ?? 30.0
            if focalEquivalent35 < 10 {
                focalEquivalent35 *= 8.4
            }
        }
        if focalEquivalent35 < 1 {
            focalEquivalent35 = 30
        }

        let diagPixels = sqrt(Float(width * width + height * height))
        let diagMm: Float = sqrt(36.0 * 36.0 + 24.0 * 24.0)
        return focalEquivalent35 * (diagPixels / diagMm)
    }
    #else
    private static func preprocessImage(_ imageURL: URL) throws -> Never {
        _ = imageURL
        throw SHARPNativePredictorError.unsupportedPlatform
    }
    #endif

    private static func applyApproximateUnprojection(
        _ gaussians: SharpGaussians3D,
        focalPx: Float,
        width: Int,
        height: Int
    ) -> SharpGaussians3D {
        let sx = Float(width) / max(2.0 * focalPx, 1e-4)
        let sy = Float(height) / max(2.0 * focalPx, 1e-4)
        let scale = MLXArray([sx, sy, 1.0], [1, 1, 3]).asType(gaussians.meanVectors.dtype)

        let meanVectors = gaussians.meanVectors * scale
        let singularValues = gaussians.singularValues * scale

        return SharpGaussians3D(
            meanVectors: meanVectors,
            singularValues: singularValues,
            quaternions: gaussians.quaternions,
            colors: gaussians.colors,
            opacities: gaussians.opacities
        )
    }

    // MARK: - PLY Export

    private static func writePLY(gaussians: SharpGaussians3D, to url: URL) throws {
        let means = gaussians.meanVectors.reshaped(-1, 3)
        let colorsLinear = gaussians.colors.reshaped(-1, 3)
        let colorsSRGB = SharpColorSpaceOps.linearRGBToSRGB(colorsLinear)
        let shCoeff0: Float = sqrt(1.0 / (4.0 * Float.pi))
        let sh0 = (colorsSRGB - 0.5) / shCoeff0

        let opacities = MLX.clip(gaussians.opacities.reshaped(-1), min: 1e-6, max: 1.0 - 1e-6)
        let opacityLogits = MLX.log(opacities / (1.0 - opacities))
        let scaleLogits = MLX.log(MLX.maximum(gaussians.singularValues.reshaped(-1, 3), 1e-8))
        let quaternions = gaussians.quaternions.reshaped(-1, 4)

        let meanArray = floatArray(means)
        let shArray = floatArray(sh0)
        let opacityArray = floatArray(opacityLogits)
        let scaleArray = floatArray(scaleLogits)
        let quaternionArray = floatArray(quaternions)

        let count = meanArray.count / 3
        guard count > 0 else {
            throw SHARPNativePredictorError.outputWriteFailed(url, "No gaussians were produced.")
        }

        var data = Data()
        data.append(contentsOf: Data(plyHeader(vertexCount: count).utf8))

        for i in 0..<count {
            appendFloat32(meanArray[i * 3], to: &data)
            appendFloat32(meanArray[i * 3 + 1], to: &data)
            appendFloat32(meanArray[i * 3 + 2], to: &data)

            appendFloat32(shArray[i * 3], to: &data)
            appendFloat32(shArray[i * 3 + 1], to: &data)
            appendFloat32(shArray[i * 3 + 2], to: &data)

            appendFloat32(opacityArray[i], to: &data)

            appendFloat32(scaleArray[i * 3], to: &data)
            appendFloat32(scaleArray[i * 3 + 1], to: &data)
            appendFloat32(scaleArray[i * 3 + 2], to: &data)

            appendFloat32(quaternionArray[i * 4], to: &data)
            appendFloat32(quaternionArray[i * 4 + 1], to: &data)
            appendFloat32(quaternionArray[i * 4 + 2], to: &data)
            appendFloat32(quaternionArray[i * 4 + 3], to: &data)
        }

        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } catch {
            throw SHARPNativePredictorError.outputWriteFailed(url, error.localizedDescription)
        }
    }

    private static func plyHeader(vertexCount: Int) -> String {
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex \(vertexCount)",
            "property float x",
            "property float y",
            "property float z",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ].joined(separator: "\n")
    }

    private static func appendFloat32(_ value: Float, to data: inout Data) {
        var little = value.bitPattern.littleEndian
        withUnsafeBytes(of: &little) { raw in
            data.append(contentsOf: raw)
        }
    }

    private static func floatArray(_ value: MLXArray) -> [Float] {
        let cast = value.asType(.float32)
        MLX.eval(cast)
        return cast.asData().data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }
}
