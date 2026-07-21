import Foundation
import MediaIO
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
#endif

public struct SAM31SegmentationBox: Codable, Hashable, Sendable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float

    public init(x1: Float, y1: Float, x2: Float, y2: Float) {
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    }
}

public struct SAM31SegmentationDetection: Codable, Hashable, Sendable {
    public let objectID: String?
    public let label: String
    public let promptKind: SAM31PromptKind?
    public let score: Float
    public let box: SAM31SegmentationBox
    public let maskAreaPixels: Int
    public let maskPath: String?
    public let candidateIndex: Int?

    public init(
        objectID: String? = nil,
        label: String,
        promptKind: SAM31PromptKind? = nil,
        score: Float,
        box: SAM31SegmentationBox,
        maskAreaPixels: Int,
        maskPath: String? = nil,
        candidateIndex: Int? = nil
    ) {
        self.objectID = objectID
        self.label = label
        self.promptKind = promptKind
        self.score = score
        self.box = box
        self.maskAreaPixels = maskAreaPixels
        self.maskPath = maskPath
        self.candidateIndex = candidateIndex
    }
}

public struct SAM31SegmentationMetadata: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let modelID: String
    public let inputImagePath: String
    public let annotatedImagePath: String
    public let jsonOutputPath: String
    public let prompts: [String]
    public let threshold: Float
    public let resolution: Int
    public let detections: [SAM31SegmentationDetection]

    public init(
        schemaVersion: Int = 2,
        modelID: String,
        inputImagePath: String,
        annotatedImagePath: String,
        jsonOutputPath: String,
        prompts: [String],
        threshold: Float,
        resolution: Int,
        detections: [SAM31SegmentationDetection]
    ) {
        self.schemaVersion = schemaVersion
        self.modelID = modelID
        self.inputImagePath = inputImagePath
        self.annotatedImagePath = annotatedImagePath
        self.jsonOutputPath = jsonOutputPath
        self.prompts = prompts
        self.threshold = threshold
        self.resolution = resolution
        self.detections = detections
    }
}

public struct SAM31SegmentationRun: Hashable, Sendable {
    public let modelID: String
    public let annotatedImageURL: URL
    public let jsonOutputURL: URL
    public let detections: [SAM31SegmentationDetection]
    public let metadata: SAM31SegmentationMetadata

    public init(
        modelID: String,
        annotatedImageURL: URL,
        jsonOutputURL: URL,
        detections: [SAM31SegmentationDetection],
        metadata: SAM31SegmentationMetadata
    ) {
        self.modelID = modelID
        self.annotatedImageURL = annotatedImageURL
        self.jsonOutputURL = jsonOutputURL
        self.detections = detections
        self.metadata = metadata
    }
}

public final class SAM31ImageSegmenter: @unchecked Sendable {
    public enum SegmenterError: LocalizedError, Sendable {
        case unsupportedPlatform
        case invalidModelRoot(URL, details: [String])
        case invalidImage(URL)
        case failedToCreateOutputDirectory(URL)
        case failedToWriteImage(URL)
        case failedToWriteMask(URL)
        case failedToEncodeMetadata(String)
        case failedToLoadWeights(URL, reason: String)
        case failedToMapWeights
        case interactivePromptingUnavailable

        public var errorDescription: String? {
            switch self {
            case .unsupportedPlatform:
                return "Native SAM 3.1 segmentation requires CoreGraphics/ImageIO."
            case .invalidModelRoot(let url, let details):
                var lines = ["Invalid SAM 3.1 model directory: \(url.path)"]
                lines.append(contentsOf: details.map { "  - \($0)" })
                return lines.joined(separator: "\n")
            case .invalidImage(let url):
                return "Failed to load image: \(url.path)"
            case .failedToCreateOutputDirectory(let url):
                return "Failed to create output directory: \(url.path)"
            case .failedToWriteImage(let url):
                return "Failed to write annotated image: \(url.path)"
            case .failedToWriteMask(let url):
                return "Failed to write mask image: \(url.path)"
            case .failedToEncodeMetadata(let reason):
                return "Failed to encode segmentation metadata: \(reason)"
            case .failedToLoadWeights(let url, let reason):
                return "Failed to load SAM 3.1 weights from \(url.path): \(reason)"
            case .failedToMapWeights:
                return "Failed to map any SAM 3.1 detector weights into the native model."
            case .interactivePromptingUnavailable:
                return "This SAM 3.1 model does not include the interactive prompt path required for box or point prompting."
            }
        }
    }

    private struct LoadedState {
        let config: SAM31ModelConfig
        let tokenizer: SAM31Tokenizer?
        let model: SAM31Model
    }

    struct PreparedDetection: Hashable, Sendable {
        let objectID: String?
        let label: String
        let promptKind: SAM31PromptKind?
        let score: Float
        let box: SAM31SegmentationBox
        let maskAreaPixels: Int
        let binaryMask: [UInt8]
        let candidateIndex: Int?
        let maskPath: String?
    }

    public let modelRootURL: URL
    public let modelID: String

    private let expectedModelID: String?
    private let fileManager: FileManager
    private var loadedState: LoadedState?

    public init(
        modelRootURL: URL,
        expectedModelID: String? = nil,
        fileManager: FileManager = .default
    ) throws {
        self.modelRootURL = modelRootURL.standardizedFileURL
        self.expectedModelID = expectedModelID
        self.fileManager = fileManager

        let report = MereRunModelValidator.validate(
            modelRoot: self.modelRootURL,
            expectedModelID: expectedModelID,
            fileManager: fileManager
        )
        guard report.isValid else {
            throw SegmenterError.invalidModelRoot(self.modelRootURL, details: report.errors)
        }

        if let manifestID = report.manifest?.id {
            self.modelID = manifestID
        } else {
            self.modelID = expectedModelID ?? self.modelRootURL.lastPathComponent
        }
    }

    deinit {
        unload()
    }

    public func unload() {
        loadedState = nil
        clearCache()
    }

    public func clearCache() {
        clearGPUMemory()
    }

    public static func defaultAnnotatedOutputURL(for imageURL: URL) -> URL {
        let parent = imageURL.deletingLastPathComponent()
        let stem = imageURL.deletingPathExtension().lastPathComponent
        let ext = imageURL.pathExtension.isEmpty ? "png" : imageURL.pathExtension
        return parent.appendingPathComponent("\(stem)_segmented").appendingPathExtension(ext)
    }

    public static func defaultJSONOutputURL(for imageURL: URL) -> URL {
        let parent = imageURL.deletingLastPathComponent()
        let stem = imageURL.deletingPathExtension().lastPathComponent
        return parent.appendingPathComponent("\(stem)_segmented").appendingPathExtension("json")
    }

    public func segment(
        imageURL: URL,
        prompts: [String],
        annotatedImageURL: URL,
        jsonOutputURL: URL,
        threshold: Float = 0.3,
        resolution: Int = 1008,
        showBoxes: Bool = false,
        showLabels: Bool = false
    ) throws -> SAM31SegmentationRun {
        try segment(
            imageURL: imageURL,
            promptSet: SAM31PromptSet(textPrompts: prompts),
            annotatedImageURL: annotatedImageURL,
            jsonOutputURL: jsonOutputURL,
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels
        )
    }

    public func segment(
        imageURL: URL,
        promptSet: SAM31PromptSet,
        annotatedImageURL: URL,
        jsonOutputURL: URL,
        threshold: Float = 0.3,
        resolution: Int = 1008,
        showBoxes: Bool = false,
        showLabels: Bool = false,
        multimask: Bool = false,
        maskOutputDirectoryURL: URL? = nil
    ) throws -> SAM31SegmentationRun {
        guard resolution > 0 else {
            throw SegmenterError.failedToEncodeMetadata("Resolution must be greater than zero.")
        }
        guard !promptSet.isEmpty else {
            throw SegmenterError.failedToEncodeMetadata("At least one prompt is required.")
        }
        defer {
            clearCache()
        }

        #if canImport(CoreGraphics)
        guard let cgImage = QwenVLImageLoader.loadCGImage(url: imageURL) else {
            throw SegmenterError.invalidImage(imageURL)
        }
        let imageWidth = cgImage.width
        let imageHeight = cgImage.height
        let pixelValues = try Self.preprocessImage(cgImage: cgImage, resolution: resolution)
        #else
        let mediaImage: MediaImage
        do {
            mediaImage = try MediaImageIO.decode(imageURL)
        } catch {
            throw SegmenterError.invalidImage(imageURL)
        }
        let imageWidth = mediaImage.width
        let imageHeight = mediaImage.height
        let pixelValues = try Self.preprocessImage(image: mediaImage, resolution: resolution)
        #endif

        let state = try ensureLoaded()
        let maxObjects = min(state.config.trackerConfig?.multiplexCount ?? 16, state.config.maxNumObjects)
        let normalizedPrompts = try promptSet.normalized(maxObjects: maxObjects)
        guard !normalizedPrompts.isEmpty else {
            throw SegmenterError.failedToEncodeMetadata("At least one non-empty prompt is required.")
        }

        let visionContext = state.model.detectorModel.prepareVision(pixelValues)
        MLX.eval(visionContext.src, visionContext.posFlat)

        var preparedDetections: [PreparedDetection] = []
        preparedDetections.reserveCapacity(normalizedPrompts.count * 4)

        for promptObject in normalizedPrompts {
            switch promptObject.promptKind {
            case .text:
                guard let textPrompt = promptObject.textPrompt else { continue }
                guard let tokenizer = state.tokenizer else {
                    throw SegmenterError.failedToEncodeMetadata(
                        "Text prompts require tokenizer.json and tokenizer_config.json in the SAM 3.1 model root."
                    )
                }
                let tokens = tokenizer.encode(
                    prompts: [textPrompt],
                    maxLength: state.config.detectorConfig.textConfig.maxPositionEmbeddings
                )
                let inputsEmbeds = state.model.detectorModel.getInputEmbeddings(
                    tokens.inputIDs,
                    attentionMask: tokens.attentionMask
                )
                let output = state.model.detectorModel.detect(
                    visionContext: visionContext,
                    inputsEmbeds: inputsEmbeds,
                    attentionMask: tokens.attentionMask
                )
                MLX.eval(output.predLogits, output.predBoxes, output.predMasks, output.presenceLogits)
                preparedDetections.append(
                    contentsOf: Self.postprocess(
                        output: output,
                        prompt: textPrompt,
                        imageWidth: imageWidth,
                        imageHeight: imageHeight,
                        threshold: threshold,
                        nmsThreshold: state.config.detNMSThresh,
                        objectID: promptObject.objectID,
                        promptKind: .text
                    )
                )
            case .box, .point, .mask:
                guard let trackerModel = state.model.trackerModel else {
                    throw SegmenterError.interactivePromptingUnavailable
                }
                let interactive = try Self.segmentInteractivePrompt(
                    trackerModel: trackerModel,
                    featurePyramid: visionContext.featurePyramid,
                    promptObject: promptObject,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    threshold: threshold,
                    multimask: multimask
                )
                preparedDetections.append(contentsOf: interactive)
            }
        }

        preparedDetections.sort { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.label < rhs.label
            }
            return lhs.score > rhs.score
        }

        try createParentDirectoryIfNeeded(for: annotatedImageURL)
        try createParentDirectoryIfNeeded(for: jsonOutputURL)
        if let maskOutputDirectoryURL {
            try fileManager.createDirectory(at: maskOutputDirectoryURL, withIntermediateDirectories: true)
        }

        let exportedDetections = try Self.writeMaskArtifacts(
            preparedDetections,
            width: imageWidth,
            height: imageHeight,
            outputDirectoryURL: maskOutputDirectoryURL
        )

        #if canImport(CoreGraphics)
        let annotatedImage = try Self.renderAnnotatedImage(
            baseImage: cgImage,
            detections: exportedDetections,
            showBoxes: showBoxes,
            showLabels: showLabels
        )
        try Self.writeImage(annotatedImage, to: annotatedImageURL)
        #else
        let annotatedImage = try Self.renderAnnotatedImage(
            baseImage: mediaImage,
            detections: exportedDetections,
            showBoxes: showBoxes,
            showLabels: showLabels
        )
        try MediaImageIO.writePNG(annotatedImage, to: annotatedImageURL)
        #endif

        let detections = exportedDetections.map {
            SAM31SegmentationDetection(
                objectID: $0.objectID,
                label: $0.label,
                promptKind: $0.promptKind,
                score: $0.score,
                box: $0.box,
                maskAreaPixels: $0.maskAreaPixels,
                maskPath: $0.maskPath,
                candidateIndex: $0.candidateIndex
            )
        }
        let metadata = SAM31SegmentationMetadata(
            modelID: modelID,
            inputImagePath: imageURL.standardizedFileURL.path,
            annotatedImagePath: annotatedImageURL.standardizedFileURL.path,
            jsonOutputPath: jsonOutputURL.standardizedFileURL.path,
            prompts: normalizedPrompts.map(\.label),
            threshold: threshold,
            resolution: resolution,
            detections: detections
        )
        try Self.writeMetadata(metadata, to: jsonOutputURL)

        return SAM31SegmentationRun(
            modelID: modelID,
            annotatedImageURL: annotatedImageURL.standardizedFileURL,
            jsonOutputURL: jsonOutputURL.standardizedFileURL,
            detections: detections,
            metadata: metadata
        )
    }

    private func createParentDirectoryIfNeeded(for url: URL) throws {
        let directory = url.deletingLastPathComponent()
        do {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        } catch {
            throw SegmenterError.failedToCreateOutputDirectory(directory)
        }
    }

    private func ensureLoaded() throws -> LoadedState {
        if let loadedState {
            return loadedState
        }

        let resources = SAM31Resources(modelRootURL: modelRootURL)
        let config = try SAM31ModelConfig.load(from: resources.configURL)
        let tokenizer: SAM31Tokenizer?
        let tokenizerConfigURL = resources.tokenizerRootURL.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = resources.tokenizerRootURL.appendingPathComponent("tokenizer.json")
        if fileManager.fileExists(atPath: tokenizerConfigURL.path),
           fileManager.fileExists(atPath: tokenizerDataURL.path) {
            tokenizer = try SAM31Tokenizer.load(from: resources.tokenizerRootURL)
        } else {
            tokenizer = nil
        }
        let model = SAM31Model(config: config)
        try Self.loadWeights(resources: resources, into: model)

        let state = LoadedState(config: config, tokenizer: tokenizer, model: model)
        self.loadedState = state
        return state
    }

    private func clearGPUMemory(synchronize: Bool = true) {
        if synchronize {
            Stream.gpu.synchronize()
        }
        MLX.eval(MLXArray([]))
        Memory.clearCache()
    }

    private static func loadWeights(resources: SAM31Resources, into model: SAM31Model) throws {
        let targetParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var anyMapped = false
        var mappedKeys = Set<String>()

        func applyArrays(_ arrays: [String: MLXArray]) throws {
            var updates: [(String, MLXArray)] = []
            updates.reserveCapacity(arrays.count)

            for (key, value) in arrays {
                let resolvedKey = mapCheckpointKey(key)
                guard let target = targetParameters[resolvedKey] else { continue }
                guard let adapted = adaptWeight(value, to: target) else { continue }
                updates.append((resolvedKey, adapted))
                mappedKeys.insert(resolvedKey)
            }

            if !updates.isEmpty {
                anyMapped = true
                try model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
            }
        }

        do {
            if FileManager.default.fileExists(atPath: resources.weightsIndexURL.path) {
                let data = try Data(contentsOf: resources.weightsIndexURL)
                let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: data)
                let baseURL = resources.weightsIndexURL.deletingLastPathComponent()
                for shard in index.shardFilenames {
                    let shardURL = baseURL.appendingPathComponent(shard)
                    let arrays = try MLX.loadArrays(url: shardURL)
                    try applyArrays(arrays)
                }
            } else {
                let arrays = try MLX.loadArrays(url: resources.weightsURL)
                try applyArrays(arrays)
            }
        } catch {
            throw SegmenterError.failedToLoadWeights(resources.modelRootURL, reason: error.localizedDescription)
        }

        guard anyMapped else {
            throw SegmenterError.failedToMapWeights
        }
        let missingTrackerKeys = targetParameters.keys
            .filter { $0.hasPrefix("tracker_model.") && !mappedKeys.contains($0) }
            .sorted()
        guard missingTrackerKeys.isEmpty else {
            let preview = missingTrackerKeys.prefix(12).joined(separator: ", ")
            throw SegmenterError.failedToLoadWeights(
                resources.modelRootURL,
                reason: "Missing \(missingTrackerKeys.count) native tracker weights: \(preview)"
            )
        }
    }

    private static func adaptWeight(_ value: MLXArray, to target: MLXArray) -> MLXArray? {
        if value.shape == target.shape {
            return value.asType(target.dtype)
        }

        if value.ndim == 4 {
            let permutations = [
                (0, 2, 3, 1),
                (1, 2, 3, 0),
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

    static func mapCheckpointKey(_ key: String) -> String {
        if key.contains("tracker_model.interactive_sam_mask_decoder."),
           key.contains(".output_hypernetworks_mlps.")
            || key.contains(".iou_prediction_head.")
            || key.contains(".pred_obj_score_head.") {
            return key
                .replacingOccurrences(of: ".proj_in.", with: ".layer1.")
                .replacingOccurrences(of: ".layers.0.", with: ".layer2.")
                .replacingOccurrences(of: ".proj_out.", with: ".layer3.")
        }
        if key.contains(".scale_layers.0.") {
            return key.replacingOccurrences(of: ".scale_layers.0.", with: ".scale_layers.first.")
        }
        if key.contains(".scale_layers.2.") {
            return key.replacingOccurrences(of: ".scale_layers.2.", with: ".scale_layers.second.")
        }
        return key
    }

    private static func segmentInteractivePrompt(
        trackerModel: SAM31TrackerModel,
        featurePyramid: SAM31VisionFeaturePyramid,
        promptObject: SAM31PromptObject,
        imageWidth: Int,
        imageHeight: Int,
        threshold: Float,
        multimask: Bool
    ) throws -> [PreparedDetection] {
        let coarse = featurePyramid.interactiveFeatures[featurePyramid.interactiveFeatures.count - 1]
        let coarseHeight = coarse.dim(1)
        let coarseWidth = coarse.dim(2)

        let pointTensor: SAM31PointPromptTensor?
        let boxTensor: MLXArray?
        switch promptObject.promptKind {
        case .point:
            pointTensor = makePointPromptTensor(
                promptObject.pointPrompts,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
            boxTensor = makeBoxPromptTensor(
                promptObject.boxPrompt,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
        case .box:
            pointTensor = makePointPromptTensor(
                promptObject.pointPrompts,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
            boxTensor = makeBoxPromptTensor(
                promptObject.boxPrompt,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
        case .mask:
            pointTensor = makePointPromptTensor(
                promptObject.pointPrompts,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
            boxTensor = makeBoxPromptTensor(
                promptObject.boxPrompt,
                targetWidth: coarseWidth,
                targetHeight: coarseHeight,
                imageWidth: imageWidth,
                imageHeight: imageHeight
            )
        case .text:
            return []
        }

        let maskTensor: MLXArray?
        if let path = promptObject.maskPrompt?.path {
            let image = try MediaImageIO.decode(URL(fileURLWithPath: path).standardizedFileURL)
            let values = Self.binaryMaskPromptValues(from: image)
            maskTensor = MLXArray(values).reshaped(1, image.height, image.width, 1)
        } else {
            maskTensor = nil
        }

        let output = trackerModel.segment(
            featurePyramid: featurePyramid,
            boxPrompt: boxTensor,
            pointPrompt: pointTensor,
            maskPrompt: maskTensor,
            multimaskOutput: multimask
        )
        MLX.eval(output.predMasks, output.iouScores, output.objectScores)

        let masks = floatArray(from: output.predMasks)
        let iouScores = floatArray(from: output.iouScores)
        let maskCount = output.predMasks.dim(output.predMasks.ndim - 3)
        let maskHeight = output.predMasks.dim(output.predMasks.ndim - 2)
        let maskWidth = output.predMasks.dim(output.predMasks.ndim - 1)

        var detections: [PreparedDetection] = []
        var bestCandidate: PreparedDetection?
        detections.reserveCapacity(maskCount)

        for index in 0..<maskCount {
            let score = min(1, max(0, iouScores[index]))
            let maskBase = index * maskHeight * maskWidth
            let maskSlice = Array(masks[maskBase..<(maskBase + maskHeight * maskWidth)])
            let resized = resizeMask(
                maskSlice,
                sourceWidth: maskWidth,
                sourceHeight: maskHeight,
                targetWidth: imageWidth,
                targetHeight: imageHeight
            )

            var binaryMask = [UInt8](repeating: 0, count: imageWidth * imageHeight)
            var area = 0
            for pixelIndex in 0..<resized.count {
                if resized[pixelIndex] > 0 {
                    binaryMask[pixelIndex] = 1
                    area += 1
                }
            }
            binaryMask = largestConnectedComponent(binaryMask: binaryMask, width: imageWidth, height: imageHeight)
            binaryMask = fillSmallHoles(
                binaryMask: binaryMask,
                width: imageWidth,
                height: imageHeight,
                maximumArea: max(64, (imageWidth * imageHeight) / 200)
            )
            area = binaryMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
            guard area > 0 else { continue }

            let candidate = PreparedDetection(
                objectID: promptObject.objectID,
                label: promptObject.label,
                promptKind: promptObject.promptKind,
                score: score,
                box: maskBoundingBox(binaryMask: binaryMask, width: imageWidth, height: imageHeight),
                maskAreaPixels: area,
                binaryMask: binaryMask,
                candidateIndex: multimask ? index : nil,
                maskPath: nil
            )
            if bestCandidate == nil || candidate.score > bestCandidate!.score {
                bestCandidate = candidate
            }
            guard score >= threshold else { continue }
            detections.append(candidate)
        }

        let sorted = detections.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return (lhs.candidateIndex ?? 0) < (rhs.candidateIndex ?? 0)
            }
            return lhs.score > rhs.score
        }
        if !sorted.isEmpty {
            return sorted
        }
        if !multimask, let bestCandidate {
            return [bestCandidate]
        }
        return []
    }

    private static func makeBoxPromptTensor(
        _ boxPrompt: SAM31PromptBox?,
        targetWidth: Int,
        targetHeight: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> MLXArray? {
        guard let boxPrompt else { return nil }
        let promptWidth = targetWidth * 14
        let promptHeight = targetHeight * 14
        let scaleX = Float(promptWidth) / Float(max(imageWidth, 1))
        let scaleY = Float(promptHeight) / Float(max(imageHeight, 1))
        return MLXArray([
            boxPrompt.x1 * scaleX,
            boxPrompt.y1 * scaleY,
            boxPrompt.x2 * scaleX,
            boxPrompt.y2 * scaleY,
        ], [1, 1, 4]).asType(.float32)
    }

    private static func makePointPromptTensor(
        _ points: [SAM31PromptPoint],
        targetWidth: Int,
        targetHeight: Int,
        imageWidth: Int,
        imageHeight: Int
    ) -> SAM31PointPromptTensor? {
        guard !points.isEmpty else { return nil }
        let promptWidth = targetWidth * 14
        let promptHeight = targetHeight * 14
        let scaleX = Float(promptWidth) / Float(max(imageWidth, 1))
        let scaleY = Float(promptHeight) / Float(max(imageHeight, 1))

        var coords: [Float] = []
        coords.reserveCapacity(points.count * 2)
        var labels: [Int32] = []
        labels.reserveCapacity(points.count)
        for point in points {
            coords.append(point.x * scaleX)
            coords.append(point.y * scaleY)
            labels.append(point.isPositive ? 1 : 0)
        }
        return SAM31PointPromptTensor(
            coords: MLXArray(coords, [1, points.count, 2]).asType(.float32),
            labels: MLXArray(labels, [1, points.count]).asType(.int32)
        )
    }

    private static func maskBoundingBox(binaryMask: [UInt8], width: Int, height: Int) -> SAM31SegmentationBox {
        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0
        var found = false

        for y in 0..<height {
            for x in 0..<width where binaryMask[y * width + x] != 0 {
                minX = Swift.min(minX, x)
                minY = Swift.min(minY, y)
                maxX = Swift.max(maxX, x)
                maxY = Swift.max(maxY, y)
                found = true
            }
        }

        guard found else {
            return SAM31SegmentationBox(x1: 0, y1: 0, x2: 0, y2: 0)
        }
        return SAM31SegmentationBox(
            x1: Float(minX),
            y1: Float(minY),
            x2: Float(maxX),
            y2: Float(maxY)
        )
    }

    private static func largestConnectedComponent(binaryMask: [UInt8], width: Int, height: Int) -> [UInt8] {
        guard width > 0, height > 0 else { return binaryMask }
        var visited = [UInt8](repeating: 0, count: binaryMask.count)
        var bestComponent: [Int] = []
        let neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for startIndex in 0..<binaryMask.count where binaryMask[startIndex] != 0 && visited[startIndex] == 0 {
            var queue = [startIndex]
            var component: [Int] = []
            visited[startIndex] = 1

            while !queue.isEmpty {
                let current = queue.removeLast()
                component.append(current)
                let x = current % width
                let y = current / width

                for (dx, dy) in neighbors {
                    let nx = x + dx
                    let ny = y + dy
                    guard nx >= 0, ny >= 0, nx < width, ny < height else { continue }
                    let nextIndex = ny * width + nx
                    guard binaryMask[nextIndex] != 0, visited[nextIndex] == 0 else { continue }
                    visited[nextIndex] = 1
                    queue.append(nextIndex)
                }
            }

            if component.count > bestComponent.count {
                bestComponent = component
            }
        }

        guard !bestComponent.isEmpty else { return binaryMask }
        var filtered = [UInt8](repeating: 0, count: binaryMask.count)
        for index in bestComponent {
            filtered[index] = 1
        }
        return filtered
    }

    static func fillSmallHoles(
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        maximumArea: Int
    ) -> [UInt8] {
        guard width > 0,
              height > 0,
              binaryMask.count == width * height,
              maximumArea > 0 else {
            return binaryMask
        }
        var result = binaryMask
        var visited = [UInt8](repeating: 0, count: binaryMask.count)
        let neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for startIndex in binaryMask.indices
        where binaryMask[startIndex] == 0 && visited[startIndex] == 0 {
            var queue = [startIndex]
            var cursor = 0
            var component: [Int] = []
            var touchesBorder = false
            visited[startIndex] = 1

            while cursor < queue.count {
                let index = queue[cursor]
                cursor += 1
                component.append(index)
                let x = index % width
                let y = index / width
                if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                    touchesBorder = true
                }
                for (dx, dy) in neighbors {
                    let nextX = x + dx
                    let nextY = y + dy
                    guard nextX >= 0,
                          nextX < width,
                          nextY >= 0,
                          nextY < height else {
                        continue
                    }
                    let nextIndex = (nextY * width) + nextX
                    guard binaryMask[nextIndex] == 0,
                          visited[nextIndex] == 0 else {
                        continue
                    }
                    visited[nextIndex] = 1
                    queue.append(nextIndex)
                }
            }

            if !touchesBorder && component.count <= maximumArea {
                for index in component {
                    result[index] = 1
                }
            }
        }
        return result
    }

    private static func writeMaskArtifacts(
        _ detections: [PreparedDetection],
        width: Int,
        height: Int,
        outputDirectoryURL: URL?
    ) throws -> [PreparedDetection] {
        guard let outputDirectoryURL else { return detections }
        return try detections.enumerated().map { index, detection in
            let base = slugify(detection.objectID ?? detection.label)
            let suffix = detection.candidateIndex.map { "_mask_\($0)" } ?? "_mask"
            let url = outputDirectoryURL.appendingPathComponent("\(base)\(suffix)_\(index)").appendingPathExtension("png")
            #if canImport(CoreGraphics)
            try writeMaskImage(binaryMask: detection.binaryMask, width: width, height: height, to: url)
            #else
            try writeMaskImagePortable(binaryMask: detection.binaryMask, width: width, height: height, to: url)
            #endif
            return PreparedDetection(
                objectID: detection.objectID,
                label: detection.label,
                promptKind: detection.promptKind,
                score: detection.score,
                box: detection.box,
                maskAreaPixels: detection.maskAreaPixels,
                binaryMask: detection.binaryMask,
                candidateIndex: detection.candidateIndex,
                maskPath: url.path
            )
        }
    }

    static func postprocess(
        output: SAM31DetectorOutput,
        prompt: String,
        imageWidth: Int,
        imageHeight: Int,
        threshold: Float,
        nmsThreshold: Float,
        objectID: String? = nil,
        promptKind: SAM31PromptKind? = nil
    ) -> [PreparedDetection] {
        let scoresTensor = MLX.sigmoid(output.predLogits) * MLX.sigmoid(output.presenceLogits)
        MLX.eval(scoresTensor)

        let scores = floatArray(from: scoresTensor)
        let boxes = floatArray(from: output.predBoxes)
        let masks = floatArray(from: output.predMasks)

        let boxCount = output.predBoxes.dim(output.predBoxes.ndim - 2)
        let maskHeight = output.predMasks.dim(output.predMasks.ndim - 2)
        let maskWidth = output.predMasks.dim(output.predMasks.ndim - 1)

        var detections: [PreparedDetection] = []
        detections.reserveCapacity(boxCount)

        for index in 0..<boxCount {
            let score = scores[index]
            guard score > threshold else { continue }

            let boxBase = index * 4
            let scaledBox = SAM31SegmentationBox(
                x1: clamp(boxes[boxBase] * Float(imageWidth), min: 0, max: Float(imageWidth)),
                y1: clamp(boxes[boxBase + 1] * Float(imageHeight), min: 0, max: Float(imageHeight)),
                x2: clamp(boxes[boxBase + 2] * Float(imageWidth), min: 0, max: Float(imageWidth)),
                y2: clamp(boxes[boxBase + 3] * Float(imageHeight), min: 0, max: Float(imageHeight))
            )

            let maskBase = index * maskHeight * maskWidth
            let maskSlice = Array(masks[maskBase..<(maskBase + maskHeight * maskWidth)])
            let resized = resizeMask(
                maskSlice,
                sourceWidth: maskWidth,
                sourceHeight: maskHeight,
                targetWidth: imageWidth,
                targetHeight: imageHeight
            )

            var binaryMask = [UInt8](repeating: 0, count: imageWidth * imageHeight)
            var area = 0
            for pixelIndex in 0..<resized.count {
                if resized[pixelIndex] > 0 {
                    binaryMask[pixelIndex] = 1
                    area += 1
                }
            }

            detections.append(
                PreparedDetection(
                    objectID: objectID,
                    label: prompt,
                    promptKind: promptKind,
                    score: score,
                    box: scaledBox,
                    maskAreaPixels: area,
                    binaryMask: binaryMask,
                    candidateIndex: nil,
                    maskPath: nil
                )
            )
        }

        guard !detections.isEmpty else {
            return []
        }

        let suppressed = nonMaximumSuppress(detections, threshold: nmsThreshold)
        guard let objectID else { return suppressed }
        return suppressed.enumerated().map { index, detection in
            PreparedDetection(
                objectID: suppressed.count == 1 ? objectID : "\(objectID)-\(index + 1)",
                label: detection.label,
                promptKind: detection.promptKind,
                score: detection.score,
                box: detection.box,
                maskAreaPixels: detection.maskAreaPixels,
                binaryMask: detection.binaryMask,
                candidateIndex: detection.candidateIndex,
                maskPath: detection.maskPath
            )
        }
    }

    static func nonMaximumSuppress(_ detections: [PreparedDetection], threshold: Float) -> [PreparedDetection] {
        guard detections.count > 1 else { return detections }

        let ordered = detections.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.label < rhs.label
            }
            return lhs.score > rhs.score
        }

        var kept: [PreparedDetection] = []
        kept.reserveCapacity(ordered.count)
        for candidate in ordered {
            let overlapsExisting = kept.contains { existing in
                iou(candidate.box, existing.box) > threshold
            }
            if !overlapsExisting {
                kept.append(candidate)
            }
        }
        return kept
    }

    static func iou(_ lhs: SAM31SegmentationBox, _ rhs: SAM31SegmentationBox) -> Float {
        let x1 = max(lhs.x1, rhs.x1)
        let y1 = max(lhs.y1, rhs.y1)
        let x2 = min(lhs.x2, rhs.x2)
        let y2 = min(lhs.y2, rhs.y2)
        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let lhsArea = max(0, lhs.x2 - lhs.x1) * max(0, lhs.y2 - lhs.y1)
        let rhsArea = max(0, rhs.x2 - rhs.x1) * max(0, rhs.y2 - rhs.y1)
        let union = lhsArea + rhsArea - intersection
        guard union > 1e-6 else { return 0 }
        return intersection / union
    }

    static func resizeMask(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        guard sourceWidth > 0, sourceHeight > 0, targetWidth > 0, targetHeight > 0 else {
            return []
        }
        if sourceWidth == targetWidth, sourceHeight == targetHeight {
            return source
        }

        let scaleX = Float(sourceWidth) / Float(targetWidth)
        let scaleY = Float(sourceHeight) / Float(targetHeight)
        var output = [Float](repeating: 0, count: targetWidth * targetHeight)

        for y in 0..<targetHeight {
            let srcY = max(0, min(Float(sourceHeight - 1), (Float(y) + 0.5) * scaleY - 0.5))
            let y0 = Int(floor(srcY))
            let y1 = min(y0 + 1, sourceHeight - 1)
            let yLerp = srcY - Float(y0)

            for x in 0..<targetWidth {
                let srcX = max(0, min(Float(sourceWidth - 1), (Float(x) + 0.5) * scaleX - 0.5))
                let x0 = Int(floor(srcX))
                let x1 = min(x0 + 1, sourceWidth - 1)
                let xLerp = srcX - Float(x0)

                let topLeft = source[y0 * sourceWidth + x0]
                let topRight = source[y0 * sourceWidth + x1]
                let bottomLeft = source[y1 * sourceWidth + x0]
                let bottomRight = source[y1 * sourceWidth + x1]

                let top = topLeft + (topRight - topLeft) * xLerp
                let bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp
                output[y * targetWidth + x] = top + (bottom - top) * yLerp
            }
        }

        return output
    }

    private static func clamp(_ value: Float, min: Float, max: Float) -> Float {
        Swift.max(min, Swift.min(max, value))
    }

    private static func floatArray(from array: MLXArray) -> [Float] {
        let cast = array.asType(.float32)
        MLX.eval(cast)
        return cast.asData().data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
    }

    private static let overlayColors: [(UInt8, UInt8, UInt8)] = [
        (31, 120, 181),
        (255, 128, 13),
        (43, 161, 43),
        (214, 38, 41),
        (148, 102, 189),
        (140, 87, 74),
    ]

    private static func preprocessImage(image: MediaImage, resolution: Int) throws -> MLXArray {
        let resized = try MediaImageIO.resized(image, width: resolution, height: resolution)
        var floats = [Float](repeating: 0, count: resolution * resolution * 3)
        for index in 0..<(resolution * resolution) {
            let src = index * 4
            let dst = index * 3
            floats[dst] = Float(resized.rgba8[src]) / 127.5 - 1.0
            floats[dst + 1] = Float(resized.rgba8[src + 1]) / 127.5 - 1.0
            floats[dst + 2] = Float(resized.rgba8[src + 2]) / 127.5 - 1.0
        }
        return MLXArray(floats, [1, resolution, resolution, 3]).asType(.float32)
    }

    private static func renderAnnotatedImage(
        baseImage: MediaImage,
        detections: [PreparedDetection],
        showBoxes: Bool,
        showLabels: Bool
    ) throws -> MediaImage {
        var bytes = baseImage.rgba8
        for (index, detection) in detections.enumerated() {
            let color = overlayColors[index % overlayColors.count]
            blendMask(
                detection.binaryMask,
                into: &bytes,
                width: baseImage.width,
                height: baseImage.height,
                color: color
            )
            if showBoxes {
                drawBox(
                    detection.box,
                    into: &bytes,
                    width: baseImage.width,
                    height: baseImage.height,
                    color: color
                )
            }
            if showLabels {
                drawLabelMarker(
                    detection.box,
                    into: &bytes,
                    width: baseImage.width,
                    height: baseImage.height,
                    color: color
                )
            }
        }
        return try MediaImage(width: baseImage.width, height: baseImage.height, rgba8: bytes)
    }

    private static func blendMask(
        _ binaryMask: [UInt8],
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        let count = min(binaryMask.count, width * height)
        for index in 0..<count where binaryMask[index] != 0 {
            let offset = index * 4
            bytes[offset] = UInt8((UInt16(bytes[offset]) + UInt16(color.0)) / 2)
            bytes[offset + 1] = UInt8((UInt16(bytes[offset + 1]) + UInt16(color.1)) / 2)
            bytes[offset + 2] = UInt8((UInt16(bytes[offset + 2]) + UInt16(color.2)) / 2)
            bytes[offset + 3] = 255
        }
    }

    private static func drawBox(
        _ box: SAM31SegmentationBox,
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        let minX = max(0, min(width - 1, Int(box.x1.rounded(.down))))
        let minY = max(0, min(height - 1, Int(box.y1.rounded(.down))))
        let maxX = max(0, min(width - 1, Int(box.x2.rounded(.up))))
        let maxY = max(0, min(height - 1, Int(box.y2.rounded(.up))))
        for inset in 0..<2 {
            drawHorizontalLine(
                y: minY + inset,
                x1: minX,
                x2: maxX,
                into: &bytes,
                width: width,
                height: height,
                color: color
            )
            drawHorizontalLine(
                y: maxY - inset,
                x1: minX,
                x2: maxX,
                into: &bytes,
                width: width,
                height: height,
                color: color
            )
            drawVerticalLine(
                x: minX + inset,
                y1: minY,
                y2: maxY,
                into: &bytes,
                width: width,
                height: height,
                color: color
            )
            drawVerticalLine(
                x: maxX - inset,
                y1: minY,
                y2: maxY,
                into: &bytes,
                width: width,
                height: height,
                color: color
            )
        }
    }

    private static func drawLabelMarker(
        _ box: SAM31SegmentationBox,
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        let minX = max(0, min(width - 1, Int(box.x1.rounded(.down))))
        let minY = max(0, min(height - 1, Int(box.y1.rounded(.down))))
        let markerWidth = max(8, min(48, width - minX))
        let markerHeight = max(4, min(12, height - minY))
        for y in minY..<min(height, minY + markerHeight) {
            for x in minX..<min(width, minX + markerWidth) {
                setPixel(x: x, y: y, in: &bytes, width: width, height: height, color: color)
            }
        }
    }

    private static func drawHorizontalLine(
        y: Int,
        x1: Int,
        x2: Int,
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        guard y >= 0, y < height else { return }
        let lower = max(0, x1)
        let upper = min(width - 1, x2)
        guard lower <= upper else { return }
        for x in lower...upper {
            setPixel(x: x, y: y, in: &bytes, width: width, height: height, color: color)
        }
    }

    private static func drawVerticalLine(
        x: Int,
        y1: Int,
        y2: Int,
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        guard x >= 0, x < width else { return }
        let lower = max(0, y1)
        let upper = min(height - 1, y2)
        guard lower <= upper else { return }
        for y in lower...upper {
            setPixel(x: x, y: y, in: &bytes, width: width, height: height, color: color)
        }
    }

    private static func setPixel(
        x: Int,
        y: Int,
        in bytes: inout [UInt8],
        width: Int,
        height: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        guard x >= 0, y >= 0, x < width, y < height else { return }
        let offset = ((y * width) + x) * 4
        bytes[offset] = color.0
        bytes[offset + 1] = color.1
        bytes[offset + 2] = color.2
        bytes[offset + 3] = 255
    }

    private static func writeMaskImagePortable(binaryMask: [UInt8], width: Int, height: Int, to url: URL) throws {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let count = min(binaryMask.count, width * height)
        for index in 0..<count {
            let value: UInt8 = binaryMask[index] == 0 ? 0 : 255
            let offset = index * 4
            rgba[offset] = value
            rgba[offset + 1] = value
            rgba[offset + 2] = value
            rgba[offset + 3] = 255
        }
        do {
            try MediaImageIO.writePNG(try MediaImage(width: width, height: height, rgba8: rgba), to: url)
        } catch {
            throw SegmenterError.failedToWriteMask(url)
        }
    }

    #if canImport(CoreGraphics)
    private static func preprocessImage(cgImage: CGImage, resolution: Int) throws -> MLXArray {
        let resized = try QwenImageIO.resizedCGImage(from: cgImage, width: resolution, height: resolution)
        let width = resized.width
        let height = resized.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel

        var buffer = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        let ok = buffer.withUnsafeMutableBytes { ptr -> Bool in
            guard let baseAddress = ptr.baseAddress else { return false }
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(resized, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        guard ok else {
            throw SegmenterError.invalidImage(URL(fileURLWithPath: "<buffer>"))
        }

        var floats = [Float](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            let src = index * bytesPerPixel
            let dst = index * 3
            floats[dst] = Float(buffer[src]) / 127.5 - 1.0
            floats[dst + 1] = Float(buffer[src + 1]) / 127.5 - 1.0
            floats[dst + 2] = Float(buffer[src + 2]) / 127.5 - 1.0
        }
        return MLXArray(floats, [1, height, width, 3]).asType(.float32)
    }

    private static func renderAnnotatedImage(
        baseImage: CGImage,
        detections: [PreparedDetection],
        showBoxes: Bool,
        showLabels: Bool
    ) throws -> CGImage {
        let width = baseImage.width
        let height = baseImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

        var bytes = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let didCreate = bytes.withUnsafeMutableBytes { ptr -> Bool in
            guard let baseAddress = ptr.baseAddress else { return false }
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                return false
            }
            context.draw(baseImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard didCreate else {
            throw SegmenterError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
        }

        for (index, detection) in detections.enumerated() {
            let (r, g, b) = overlayColor(for: detection, index: index)
            for pixelIndex in 0..<(width * height) where detection.binaryMask[pixelIndex] != 0 {
                let base = pixelIndex * bytesPerPixel
                bytes[base] = UInt8(Float(bytes[base]) * 0.55 + Float(r) * 0.45)
                bytes[base + 1] = UInt8(Float(bytes[base + 1]) * 0.55 + Float(g) * 0.45)
                bytes[base + 2] = UInt8(Float(bytes[base + 2]) * 0.55 + Float(b) * 0.45)
                bytes[base + 3] = 255
            }

            if showBoxes {
                drawBox(
                    into: &bytes,
                    width: width,
                    height: height,
                    box: detection.box,
                    color: (r, g, b),
                    lineWidth: 2
                )
            }
        }

        if showLabels {
            drawLabels(into: &bytes, width: width, height: height, detections: detections)
        }

        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw SegmenterError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
        }
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            throw SegmenterError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
        }
        return image
    }

    private static func overlayColor(for detection: PreparedDetection, index: Int) -> (UInt8, UInt8, UInt8) {
        let seed = detection.objectID ?? detection.label
        let hash = seed.isEmpty ? index : abs(seed.hashValue)
        return overlayColors[hash % overlayColors.count]
    }

    private static func drawLabels(
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        detections: [PreparedDetection]
    ) {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        bytes.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo
                  )
            else {
                return
            }

            context.setAllowsAntialiasing(true)
            context.setShouldAntialias(true)
            let fontSize = CGFloat(max(12, min(18, height / 48)))
            let font = CTFontCreateWithName("Menlo" as CFString, fontSize, nil)

            for (index, detection) in detections.enumerated() {
                let label = detection.label.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !label.isEmpty else { continue }

                let (r, g, b) = overlayColor(for: detection, index: index)
                let attributes: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                ]
                let line = CTLineCreateWithAttributedString(NSAttributedString(string: label, attributes: attributes))

                var ascent: CGFloat = 0
                var descent: CGFloat = 0
                var leading: CGFloat = 0
                let textWidth = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading)).rounded(.up)
                let textHeight = (ascent + descent + leading).rounded(.up)
                let paddingX: CGFloat = 6
                let paddingY: CGFloat = 4
                let labelWidth = min(CGFloat(width), textWidth + paddingX * 2)
                let labelHeight = min(CGFloat(height), textHeight + paddingY * 2)
                let desiredX = CGFloat(max(0, min(width - 1, Int(detection.box.x1.rounded(.down)))))
                let desiredTopY = CGFloat(max(0, min(height - 1, Int(detection.box.y1.rounded(.down))))) - labelHeight - 4
                let rectX = min(max(0, desiredX), max(0, CGFloat(width) - labelWidth))
                let rectTopY = max(0, desiredTopY)
                let rectY = CGFloat(height) - rectTopY - labelHeight

                context.setFillColor(CGColor(
                    red: CGFloat(r) / 255,
                    green: CGFloat(g) / 255,
                    blue: CGFloat(b) / 255,
                    alpha: 0.92
                ))
                context.fill(CGRect(x: rectX, y: rectY, width: labelWidth, height: labelHeight))

                context.textPosition = CGPoint(
                    x: rectX + paddingX,
                    y: rectY + paddingY + descent
                )
                CTLineDraw(line, context)
            }
        }
    }

    private static func drawBox(
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        box: SAM31SegmentationBox,
        color: (UInt8, UInt8, UInt8),
        lineWidth: Int
    ) {
        let x1 = max(0, min(width - 1, Int(box.x1.rounded(.down))))
        let y1 = max(0, min(height - 1, Int(box.y1.rounded(.down))))
        let x2 = max(0, min(width - 1, Int(box.x2.rounded(.down))))
        let y2 = max(0, min(height - 1, Int(box.y2.rounded(.down))))
        guard x2 > x1, y2 > y1 else { return }

        for thickness in 0..<lineWidth {
            let top = min(height - 1, y1 + thickness)
            let bottom = max(0, y2 - thickness)
            for x in x1...x2 {
                writePixel(&bytes, width: width, height: height, x: x, y: top, color: color)
                writePixel(&bytes, width: width, height: height, x: x, y: bottom, color: color)
            }

            let left = min(width - 1, x1 + thickness)
            let right = max(0, x2 - thickness)
            for y in y1...y2 {
                writePixel(&bytes, width: width, height: height, x: left, y: y, color: color)
                writePixel(&bytes, width: width, height: height, x: right, y: y, color: color)
            }
        }
    }

    private static func writePixel(
        _ bytes: inout [UInt8],
        width: Int,
        height: Int,
        x: Int,
        y: Int,
        color: (UInt8, UInt8, UInt8)
    ) {
        guard (0..<width).contains(x), (0..<height).contains(y) else { return }
        let offset = (y * width + x) * 4
        bytes[offset] = color.0
        bytes[offset + 1] = color.1
        bytes[offset + 2] = color.2
        bytes[offset + 3] = 255
    }

    private static func writeMaskImage(
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        to url: URL
    ) throws {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue

        var bytes = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        for pixelIndex in 0..<(width * height) where binaryMask[pixelIndex] != 0 {
            let offset = pixelIndex * bytesPerPixel
            bytes[offset] = 255
            bytes[offset + 1] = 255
            bytes[offset + 2] = 255
            bytes[offset + 3] = 255
        }

        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else {
            throw SegmenterError.failedToWriteMask(url)
        }
        guard let image = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw SegmenterError.failedToWriteMask(url)
        }
        try writeImage(image, to: url)
    }

    private static func writeImage(_ image: CGImage, to url: URL) throws {
        let utType = UTType(filenameExtension: url.pathExtension.lowercased()) ?? .png
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, utType.identifier as CFString, 1, nil) else {
            throw SegmenterError.failedToWriteImage(url)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw SegmenterError.failedToWriteImage(url)
        }
    }
    #endif

    private static func writeMetadata(_ metadata: SAM31SegmentationMetadata, to url: URL) throws {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(metadata)
            try data.write(to: url, options: [.atomic])
        } catch {
            throw SegmenterError.failedToEncodeMetadata(error.localizedDescription)
        }
    }

    static func binaryMaskPromptValues(from image: MediaImage) -> [Float] {
        stride(from: 0, to: image.rgba8.count, by: 4).map { offset in
            let foreground = max(
                image.rgba8[offset],
                max(image.rgba8[offset + 1], image.rgba8[offset + 2])
            )
            return foreground >= 128 ? 1 : 0
        }
    }

    private static func slugify(_ value: String) -> String {
        let lowered = value.lowercased()
        let parts = lowered.split { !$0.isLetter && !$0.isNumber }
        let joined = parts.joined(separator: "-")
        return joined.isEmpty ? "object" : joined
    }
}
