import Foundation
import MediaIO
@preconcurrency import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
#endif

public struct FalconPerceptionCenter: Codable, Hashable, Sendable {
    public let x: Float
    public let y: Float
}

public struct FalconPerceptionSize: Codable, Hashable, Sendable {
    public let h: Float
    public let w: Float
}

public struct FalconPerceptionBoundingBox: Codable, Hashable, Sendable {
    public let x1: Float
    public let y1: Float
    public let x2: Float
    public let y2: Float
}

public struct FalconPerceptionDetection: Codable, Hashable, Sendable {
    public let label: String
    public let xy: FalconPerceptionCenter
    public let hw: FalconPerceptionSize
    public let box: FalconPerceptionBoundingBox
    public let score: Float?
    public let maskPath: String?

    public init(
        label: String,
        xy: FalconPerceptionCenter,
        hw: FalconPerceptionSize,
        box: FalconPerceptionBoundingBox,
        score: Float? = nil,
        maskPath: String? = nil
    ) {
        self.label = label
        self.xy = xy
        self.hw = hw
        self.box = box
        self.score = score
        self.maskPath = maskPath
    }
}

public struct FalconPerceptionGroundingMetadata: Codable, Hashable, Sendable {
    public let schemaVersion: Int
    public let modelID: String
    public let inputImagePath: String
    public let annotatedImagePath: String
    public let jsonOutputPath: String
    public let queries: [String]
    public let detections: [FalconPerceptionDetection]
}

public struct FalconPerceptionGroundingRun: Hashable, Sendable {
    public let modelID: String
    public let annotatedImageURL: URL
    public let jsonOutputURL: URL
    public let detections: [FalconPerceptionDetection]
    public let metadata: FalconPerceptionGroundingMetadata
}

public final class FalconPerceptionGrounder: @unchecked Sendable {
    public enum GrounderError: LocalizedError, Sendable {
        case invalidModelRoot(URL, details: [String])
        case unsupportedPlatform
        case invalidImage(URL)
        case failedToCreateOutputDirectory(URL)
        case failedToWriteImage(URL)
        case failedToWriteMask(URL)
        case failedToEncodeMetadata(String)
        case failedToLoadWeights(URL, reason: String)
        case failedToMapWeights

        public var errorDescription: String? {
            switch self {
            case .invalidModelRoot(let url, let details):
                var lines = ["Invalid Falcon Perception model directory: \(url.path)"]
                lines.append(contentsOf: details.map { "  - \($0)" })
                return lines.joined(separator: "\n")
            case .unsupportedPlatform:
                return "Falcon Perception grounding requires CoreGraphics/ImageIO."
            case .invalidImage(let url):
                return "Failed to load image: \(url.path)"
            case .failedToCreateOutputDirectory(let url):
                return "Failed to create output directory: \(url.path)"
            case .failedToWriteImage(let url):
                return "Failed to write annotated image: \(url.path)"
            case .failedToWriteMask(let url):
                return "Failed to write mask image: \(url.path)"
            case .failedToEncodeMetadata(let reason):
                return "Failed to encode grounding metadata: \(reason)"
            case .failedToLoadWeights(let url, let reason):
                return "Failed to load Falcon Perception weights from \(url.path): \(reason)"
            case .failedToMapWeights:
                return "Failed to map any Falcon Perception checkpoint weights into the native model."
            }
        }
    }

    private struct LoadedState {
        let config: FalconPerceptionModelConfig
        let tokenizer: FalconPerceptionTokenizer
        let processor: FalconPerceptionProcessor
        let model: FalconPerceptionModel
    }

    private struct PreparedDetection: Hashable, Sendable {
        let label: String
        let xy: FalconPerceptionCenter
        let hw: FalconPerceptionSize
        let box: FalconPerceptionBoundingBox
        let score: Float?
        let binaryMask: [UInt8]?
        let maskPath: String?
    }

    private static let overlayColors: [(UInt8, UInt8, UInt8)] = [
        (32, 138, 255),
        (255, 99, 71),
        (52, 199, 89),
        (255, 204, 0),
        (175, 82, 222),
        (255, 149, 0),
        (90, 200, 250),
        (255, 45, 85),
    ]

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
            throw GrounderError.invalidModelRoot(self.modelRootURL, details: report.errors)
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
        loadedState?.model.resetGroundingState()
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
        return parent.appendingPathComponent("\(stem)_grounded").appendingPathExtension(ext)
    }

    public static func defaultJSONOutputURL(for imageURL: URL) -> URL {
        let parent = imageURL.deletingLastPathComponent()
        let stem = imageURL.deletingPathExtension().lastPathComponent
        return parent.appendingPathComponent("\(stem)_grounded").appendingPathExtension("json")
    }

    public func ground(
        imageURL: URL,
        queries: [String],
        annotatedImageURL: URL,
        jsonOutputURL: URL,
        maskOutputDirectoryURL: URL? = nil,
        maxNewTokens: Int = 512,
        segmentationThreshold: Float = 0.5
    ) throws -> FalconPerceptionGroundingRun {
        let normalizedQueries = queries.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }.filter { !$0.isEmpty }
        guard !normalizedQueries.isEmpty else {
            throw GrounderError.failedToEncodeMetadata("At least one non-empty query is required.")
        }
        defer {
            clearCache()
        }

        let imageWidth: Int
        let imageHeight: Int
        #if canImport(CoreGraphics)
        guard let baseImage = QwenVLImageLoader.loadCGImage(url: imageURL) else {
            throw GrounderError.invalidImage(imageURL)
        }
        imageWidth = baseImage.width
        imageHeight = baseImage.height
        #else
        let baseImage: MediaImage
        do {
            baseImage = try MediaImageIO.decode(imageURL)
            imageWidth = baseImage.width
            imageHeight = baseImage.height
        } catch {
            throw GrounderError.invalidImage(imageURL)
        }
        #endif

        let state = try ensureLoaded()
        var preparedDetections: [PreparedDetection] = []
        preparedDetections.reserveCapacity(normalizedQueries.count * 4)

        for query in normalizedQueries {
            preparedDetections.append(
                contentsOf: try groundSingleQuery(
                    query: query,
                    imageURL: imageURL,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    state: state,
                    maxNewTokens: maxNewTokens,
                    segmentationThreshold: segmentationThreshold
                )
            )
            state.model.resetGroundingState()
            clearCache()
        }

        try createParentDirectoryIfNeeded(for: annotatedImageURL)
        try createParentDirectoryIfNeeded(for: jsonOutputURL)
        if let maskOutputDirectoryURL {
            do {
                try fileManager.createDirectory(at: maskOutputDirectoryURL, withIntermediateDirectories: true)
            } catch {
                throw GrounderError.failedToCreateOutputDirectory(maskOutputDirectoryURL)
            }
        }

        let exportedDetections = try Self.writeMaskArtifacts(
            preparedDetections,
            width: imageWidth,
            height: imageHeight,
            outputDirectoryURL: maskOutputDirectoryURL
        )
        #if canImport(CoreGraphics)
        let annotatedImage = try Self.renderAnnotatedImage(baseImage: baseImage, detections: exportedDetections)
        try Self.writeImage(annotatedImage, to: annotatedImageURL)
        #else
        let annotatedImage = try Self.renderAnnotatedImage(baseImage: baseImage, detections: exportedDetections)
        do {
            try MediaImageIO.writePNG(annotatedImage, to: annotatedImageURL)
        } catch {
            throw GrounderError.failedToWriteImage(annotatedImageURL)
        }
        #endif

        let detections = exportedDetections.map {
            FalconPerceptionDetection(
                label: $0.label,
                xy: $0.xy,
                hw: $0.hw,
                box: $0.box,
                score: $0.score,
                maskPath: $0.maskPath
            )
        }
        let metadata = FalconPerceptionGroundingMetadata(
            schemaVersion: 1,
            modelID: modelID,
            inputImagePath: imageURL.standardizedFileURL.path,
            annotatedImagePath: annotatedImageURL.standardizedFileURL.path,
            jsonOutputPath: jsonOutputURL.standardizedFileURL.path,
            queries: normalizedQueries,
            detections: detections
        )
        try Self.writeMetadata(metadata, to: jsonOutputURL)

        return FalconPerceptionGroundingRun(
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
            throw GrounderError.failedToCreateOutputDirectory(directory)
        }
    }

    private func ensureLoaded() throws -> LoadedState {
        if let loadedState {
            return loadedState
        }

        let resources = FalconPerceptionResources(rootURL: modelRootURL)
        let config = try FalconPerceptionModelConfig.load(from: resources.configURL)
        let tokenizer = try FalconPerceptionTokenizer.load(from: resources.tokenizerRootURL)
        let processor = FalconPerceptionProcessor(tokenizer: tokenizer, config: config)
        let model = FalconPerceptionModel(config: config)
        try Self.loadWeights(resources: resources, into: model)

        let state = LoadedState(
            config: config,
            tokenizer: tokenizer,
            processor: processor,
            model: model
        )
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

    private func groundSingleQuery(
        query: String,
        imageURL: URL,
        imageWidth: Int,
        imageHeight: Int,
        state: LoadedState,
        maxNewTokens: Int,
        segmentationThreshold: Float
    ) throws -> [PreparedDetection] {
        let runtimeTraceLogURL = ProcessInfo.processInfo.environment["MERERUN_FALCON_TRACE_RUNTIME_LOG"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        func trace(_ line: String) {
            guard let runtimeTraceLogURL else { return }
            let data = (line + "\n").data(using: .utf8) ?? Data()
            if fileManager.fileExists(atPath: runtimeTraceLogURL.path) {
                if let handle = try? FileHandle(forWritingTo: runtimeTraceLogURL) {
                    _ = try? handle.seekToEnd()
                    try? handle.write(contentsOf: data)
                    try? handle.close()
                }
            } else {
                fileManager.createFile(atPath: runtimeTraceLogURL.path, contents: data)
            }
        }

        let processed = try state.processor.process(imageURL: imageURL, query: query)
        let model = state.model
        defer {
            model.resetGroundingState()
        }
        let config = state.config
        model.resetGroundingState()
        trace("TRACE runtime query: \(query)")
        trace("TRACE runtime processed_size: \(processed.processedSize.width) \(processed.processedSize.height)")
        trace("TRACE runtime input_count: \(processed.inputIDs.dim(1))")
        let positionData = FalconPerceptionModel.computePositionData(
            inputIDs: processed.inputIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        model.prepareGroundingPrefill(positionData: positionData)
        let caches = model.makeCaches()
        let inputsEmbeds = model.makeInputEmbeddings(
            inputIDs: processed.inputIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        var logits = model.forward(
            inputIDs: processed.inputIDs,
            inputsEmbeds: inputsEmbeds,
            caches: caches,
            lastPositionOnly: true
        )
        MLX.eval(logits)
        model.finishGroundingPrefill()
        trace("TRACE runtime prefill_cache_offset: \(caches.first??.offset ?? 0)")

        let gridHW = processed.imageGridHW.asArray(Int32.self).map(Int.init)
        let gridH = gridHW.indices.contains(0) ? gridHW[0] : 0
        let gridW = gridHW.indices.contains(1) ? gridHW[1] : 0

        let disableSegmentationFeatures = ProcessInfo.processInfo.environment["MERERUN_FALCON_DISABLE_SEGM_FEATURES"] == "1"
        var segmentationFeatures: MLXArray?
        if !disableSegmentationFeatures, let hiddenState = model.lastHiddenState {
            segmentationFeatures = model.computeSegmentationFeatures(
                hiddenState: hiddenState,
                inputIDs: processed.inputIDs,
                pixelValues: processed.pixelValues,
                gridH: gridH,
                gridW: gridW
            )
            if let segmentationFeatures {
                MLX.eval(segmentationFeatures)
                trace("TRACE runtime segm_features_shape: \(segmentationFeatures.shape)")
            }
        } else if disableSegmentationFeatures {
            trace("TRACE runtime segm_features_shape: disabled")
        }

        var detections: [PreparedDetection] = []
        detections.reserveCapacity(8)

        var currentXY: FalconPerceptionCenter?
        var currentHW: FalconPerceptionSize?
        var currentMask: [UInt8]?

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, logits.dim(1) - 1]
            if runtimeTraceLogURL != nil {
                let logitsValues = lastLogits.asArray(Float.self)
                let topTokens = logitsValues.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(5)
                    .map { index, value in
                        "\(index):\(state.tokenizer.decode(token: index))=\(String(format: "%.4f", value))"
                    }
                    .joined(separator: ", ")
                trace("TRACE runtime top5 \(topTokens)")
            }
            let tokenID = Int(MLX.argMax(lastLogits).item(Int32.self))
            trace("TRACE runtime token: \(tokenID) \(state.tokenizer.decode(token: tokenID))")
            if tokenID == config.eosID {
                trace("TRACE runtime hit_eos")
                break
            }

            var encodedCoordXY: MLXArray?
            var encodedSizeHW: MLXArray?

            if let hiddenState = model.lastHiddenState {
                let hiddenLast = hiddenState[0, hiddenState.dim(1) - 1]
                let hiddenForDecode = hiddenLast.reshaped(1, hiddenLast.dim(0))
                // Trace-only decodes: the coordinate and size heads plus two
                // GPU readbacks per generated token, so they must not run
                // when tracing is disabled.
                if runtimeTraceLogURL != nil {
                    let coordLogitsForTrace = model.decodeCoordinates(from: hiddenForDecode)
                    let coordBinsForTrace = MLX.argMax(coordLogitsForTrace, axis: -1).asArray(Int32.self)
                    let coordNumBins = max(1, coordLogitsForTrace.dim(-1) - 1)
                    let traceX = Float(coordBinsForTrace[0]) / Float(coordNumBins)
                    let traceY = Float(coordBinsForTrace[1]) / Float(coordNumBins)
                    let sizeLogitsForTrace = model.decodeSizes(from: hiddenForDecode)
                    let sizeValuesForTrace = model.processSizes(sizeLogitsForTrace).asArray(Float.self)
                    trace(
                        String(
                            format: "TRACE runtime hidden_decoded x=%.6f y=%.6f h=%.6f w=%.6f",
                            traceX,
                            traceY,
                            sizeValuesForTrace[0],
                            sizeValuesForTrace[1]
                        )
                    )
                }

                if tokenID == config.coordTokenID {
                    if let currentXY, let currentHW {
                        trace(String(format: "TRACE runtime append_before_coord xy=(%.6f,%.6f) hw=(%.6f,%.6f)", currentXY.x, currentXY.y, currentHW.h, currentHW.w))
                        detections.append(
                            PreparedDetection(
                                label: query,
                                xy: currentXY,
                                hw: currentHW,
                                box: Self.boundingBox(xy: currentXY, hw: currentHW),
                                score: nil,
                                binaryMask: currentMask,
                                maskPath: nil
                            )
                        )
                        currentMask = nil
                    }

                    let coordLogits = model.decodeCoordinates(from: hiddenForDecode)
                    let predBins = MLX.argMax(coordLogits, axis: -1).asArray(Int32.self)
                    let numBins = max(1, coordLogits.dim(-1) - 1)
                    let x = Float(predBins[0]) / Float(numBins)
                    let y = Float(predBins[1]) / Float(numBins)
                    let xy = FalconPerceptionCenter(x: x, y: y)
                    currentXY = xy
                    encodedCoordXY = MLXArray([x, y], [1, 2]).asType(.float32)
                    trace(String(format: "TRACE runtime coord xy=(%.6f,%.6f)", x, y))
                } else if tokenID == config.sizeTokenID {
                    let sizeLogits = model.decodeSizes(from: hiddenForDecode)
                    let sizeValues = model.processSizes(sizeLogits).asArray(Float.self)
                    let hw = FalconPerceptionSize(h: sizeValues[0], w: sizeValues[1])
                    currentHW = hw
                    encodedSizeHW = MLXArray([hw.h, hw.w], [1, 2]).asType(.float32)
                    trace(String(format: "TRACE runtime size hw=(%.6f,%.6f)", hw.h, hw.w))
                } else if tokenID == config.segTokenID {
                    if let segmentationFeatures {
                        let mask = model.decodeSegmentationMask(
                            segHidden: hiddenLast,
                            segmentationFeatures: segmentationFeatures,
                            outputHeight: imageHeight,
                            outputWidth: imageWidth,
                            threshold: segmentationThreshold
                        )
                        if let mask {
                            MLX.eval(mask)
                            currentMask = Self.binaryMask(from: mask)
                            let maskPixels = currentMask?.reduce(0) { partial, value in
                                partial + Int(value)
                            } ?? 0
                            trace("TRACE runtime seg mask_pixels=\(maskPixels)")
                        }
                    }

                    if let currentXY, let currentHW {
                        trace(String(format: "TRACE runtime append_on_seg xy=(%.6f,%.6f) hw=(%.6f,%.6f)", currentXY.x, currentXY.y, currentHW.h, currentHW.w))
                        detections.append(
                            PreparedDetection(
                                label: query,
                                xy: currentXY,
                                hw: currentHW,
                                box: Self.boundingBox(xy: currentXY, hw: currentHW),
                                score: nil,
                                binaryMask: currentMask,
                                maskPath: nil
                            )
                        )
                    }
                    currentXY = nil
                    currentHW = nil
                    currentMask = nil
                }
            }

            let tokenArray = MLXArray([Int32(tokenID)], [1, 1])
            var tokenEmbeds = model.embedTokens(tokenArray)
            if tokenID == config.coordTokenID, let encodedCoordXY {
                tokenEmbeds = model.encodeCoordinates(
                    into: tokenEmbeds,
                    inputIDs: tokenArray,
                    coordXY: encodedCoordXY
                )
            } else if tokenID == config.sizeTokenID, let encodedSizeHW {
                tokenEmbeds = model.encodeSizes(
                    into: tokenEmbeds,
                    inputIDs: tokenArray,
                    sizeHW: encodedSizeHW
                )
            }

            logits = model.forward(
                inputIDs: tokenArray,
                inputsEmbeds: tokenEmbeds,
                caches: caches,
                lastPositionOnly: true
            )
            asyncEval(logits)
            trace("TRACE runtime cached_forward_offset=\(caches.first??.offset ?? 0)")
        }

        if let currentXY, let currentHW {
            trace(String(format: "TRACE runtime append_final xy=(%.6f,%.6f) hw=(%.6f,%.6f)", currentXY.x, currentXY.y, currentHW.h, currentHW.w))
            detections.append(
                PreparedDetection(
                    label: query,
                    xy: currentXY,
                    hw: currentHW,
                    box: Self.boundingBox(xy: currentXY, hw: currentHW),
                    score: nil,
                    binaryMask: currentMask,
                    maskPath: nil
                )
            )
        }

        trace("TRACE runtime detection_count: \(detections.count)")
        return detections
    }

    static func loadWeights(resources: FalconPerceptionResources, into model: FalconPerceptionModel) throws {
        let targetParameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var anyMapped = false
        var pendingUpdates: [(String, MLXArray)] = []

        func applyArrays(_ arrays: [String: MLXArray]) throws {
            for (key, value) in arrays {
                for (mappedKey, mappedValue) in mapCheckpointKey(key: key, value: value) {
                    if assignManuallyMappedWeight(mappedValue, for: mappedKey, into: model) {
                        anyMapped = true
                        continue
                    }
                    guard let target = targetParameters[mappedKey] else { continue }
                    guard let adapted = adaptWeight(mappedValue, to: target) else { continue }
                    pendingUpdates.append((mappedKey, adapted))
                }
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
            throw GrounderError.failedToLoadWeights(resources.rootURL, reason: error.localizedDescription)
        }

        if !pendingUpdates.isEmpty {
            anyMapped = true
            try model.update(parameters: ModuleParameters.unflattened(pendingUpdates), verify: .none)
        }

        guard anyMapped else {
            throw GrounderError.failedToMapWeights
        }
    }

    static func assignManuallyMappedWeight(
        _ value: MLXArray,
        for mappedKey: String,
        into model: FalconPerceptionModel
    ) -> Bool {
        guard let match = mappedKey.firstMatch(
            of: /^language_model\.model\.layers\.(\d+)\.(self_attn\._norm_w_in|self_attn\._norm_w_qk|mlp\._norm_w)$/
        ), let layerIndex = Int(match.1),
        model.languageModel.model.layers.indices.contains(layerIndex) else {
            return false
        }

        let suffix = String(match.2)
        let layer = model.languageModel.model.layers[layerIndex]

        switch suffix {
        case "self_attn._norm_w_in":
            guard value.shape == layer.selfAttn.normWIn.shape else { return false }
            return (try? layer.selfAttn.update(
                parameters: ModuleParameters.unflattened([
                    ("_norm_w_in", value.asType(layer.selfAttn.normWIn.dtype))
                ]),
                verify: .none
            )) != nil
        case "self_attn._norm_w_qk":
            guard value.shape == layer.selfAttn.normWQK.shape else { return false }
            return (try? layer.selfAttn.update(
                parameters: ModuleParameters.unflattened([
                    ("_norm_w_qk", value.asType(layer.selfAttn.normWQK.dtype))
                ]),
                verify: .none
            )) != nil
        case "mlp._norm_w":
            guard value.shape == layer.mlp.normW.shape else { return false }
            return (try? layer.mlp.update(
                parameters: ModuleParameters.unflattened([
                    ("_norm_w", value.asType(layer.mlp.normW.dtype))
                ]),
                verify: .none
            )) != nil
        default:
            return false
        }
    }

    static func remapAnyUpKeySuffix(_ suffix: String) -> String? {
        let blockMap: [String: String] = [
            "0.weight": "norm1.weight",
            "0.bias": "norm1.bias",
            "2.weight": "conv1.weight",
            "3.weight": "norm2.weight",
            "3.bias": "norm2.bias",
            "5.weight": "conv2.weight",
        ]

        let encoders = ["image_encoder", "key_encoder", "query_encoder", "aggregation"]
        for encoder in encoders {
            guard suffix.hasPrefix(encoder + ".") else { continue }
            let remainder = String(suffix.dropFirst(encoder.count + 1))
            if remainder == "0.weight" {
                return encoder + ".conv.weight"
            }
            if let match = remainder.firstMatch(of: /^(\d+)\.block\.(.+)$/),
               let blockIndex = Int(match.1),
               let mapped = blockMap[String(match.2)] {
                return "\(encoder).blocks.\(blockIndex - 1).\(mapped)"
            }
            if let match = remainder.firstMatch(of: /^(\d+)\.shortcut\.weight$/),
               let blockIndex = Int(match.1) {
                return "\(encoder).blocks.\(blockIndex - 1).shortcut.weight"
            }
        }

        if suffix.hasPrefix("key_features_encoder.") {
            let remainder = String(suffix.dropFirst("key_features_encoder.".count))
            if remainder == "0.basis" {
                return "key_features_encoder.lfu.basis"
            }
            if let match = remainder.firstMatch(of: /^(\d+)\.block\.(.+)$/),
               let blockIndex = Int(match.1),
               let mapped = blockMap[String(match.2)] {
                return "key_features_encoder.blocks.\(blockIndex - 1).\(mapped)"
            }
        }

        if suffix == "cross_decode.conv2d.weight" {
            return "cross_decode.conv.weight"
        }
        if suffix == "cross_decode.cross_attn.norm_q.weight" {
            return "cross_decode.cross_attn.norm_q.weight"
        }
        if suffix == "cross_decode.cross_attn.norm_k.weight" {
            return "cross_decode.cross_attn.norm_k.weight"
        }
        if suffix.hasPrefix("cross_decode.cross_attn.attention.in_proj_") {
            return suffix
        }
        if suffix == "rope.freqs" {
            return "rope.freqs"
        }

        return nil
    }

    static func mapCheckpointKey(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key.hasPrefix("itok_upsampler.") {
            let suffix = String(key.dropFirst("itok_upsampler.".count))
            if suffix == "cross_decode.cross_attn.attention.in_proj_weight" {
                let parts = split(value, parts: 3, axis: 0)
                guard parts.count == 3 else { return [] }
                return [
                    ("itok_upsampler.cross_decode.cross_attn.q_proj.weight", parts[0]),
                    ("itok_upsampler.cross_decode.cross_attn.k_proj.weight", parts[1]),
                ]
            }
            if suffix == "cross_decode.cross_attn.attention.in_proj_bias" {
                let parts = split(value, parts: 3, axis: 0)
                guard parts.count == 3 else { return [] }
                return [
                    ("itok_upsampler.cross_decode.cross_attn.q_proj.bias", parts[0]),
                    ("itok_upsampler.cross_decode.cross_attn.k_proj.bias", parts[1]),
                ]
            }
            guard let remappedSuffix = remapAnyUpKeySuffix(suffix) else {
                return []
            }

            var mappedValue = value
            if remappedSuffix.contains("lfu.basis") || (mappedValue.ndim == 4 && !remappedSuffix.contains("norm")) {
                mappedValue = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(mappedValue)
            }
            return [("itok_upsampler." + remappedSuffix, mappedValue)]
        }

        var mappedKey = key
        if key.hasPrefix("tok_embeddings.") {
            mappedKey = key.replacingOccurrences(of: "tok_embeddings.", with: "language_model.model.embed_tokens.", options: [], range: nil)
        } else if key.hasPrefix("img_projector.") {
            mappedKey = key.replacingOccurrences(of: "img_projector.", with: "language_model.model.img_projector.", options: [], range: nil)
        } else if key.hasPrefix("norm.") {
            mappedKey = key.replacingOccurrences(of: "norm.", with: "language_model.model.norm.", options: [], range: nil)
        } else if key.hasPrefix("output.") {
            mappedKey = key.replacingOccurrences(of: "output.", with: "language_model.lm_head.", options: [], range: nil)
        } else if key == "freqs_cis_golden" {
            mappedKey = "language_model.model.freqs_cis_golden"
        } else if key.hasPrefix("layers.") {
            mappedKey = key.replacingOccurrences(of: "layers.", with: "language_model.model.layers.", options: [], range: nil)
            mappedKey = mappedKey.replacingOccurrences(of: ".attention.", with: ".self_attn.")
            mappedKey = mappedKey.replacingOccurrences(of: ".feed_forward.", with: ".mlp.")
        }

        var mappedValue = value
        if mappedKey.contains(".w13.") {
            mappedValue = reorderW13Weights(value)
        }
        if mappedKey.contains("conv_segm.weight") {
            mappedValue = HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(mappedValue)
        }
        return [(mappedKey, mappedValue)]
    }

    private static func reorderW13Weights(_ value: MLXArray) -> MLXArray {
        guard value.ndim >= 2 else { return value }
        let rowCount = value.dim(0)
        guard rowCount > 1 else { return value }

        var rows: [MLXArray] = []
        rows.reserveCapacity(rowCount)
        for row in stride(from: 0, to: rowCount, by: 2) {
            rows.append(value[row].reshaped(1, value.dim(1)))
        }
        for row in stride(from: 1, to: rowCount, by: 2) {
            rows.append(value[row].reshaped(1, value.dim(1)))
        }
        return concatenated(rows, axis: 0)
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

    static func boundingBox(
        xy: FalconPerceptionCenter,
        hw: FalconPerceptionSize
    ) -> FalconPerceptionBoundingBox {
        let halfWidth = hw.w / 2
        let halfHeight = hw.h / 2
        return FalconPerceptionBoundingBox(
            x1: min(max(0, xy.x - halfWidth), 1),
            y1: min(max(0, xy.y - halfHeight), 1),
            x2: min(max(0, xy.x + halfWidth), 1),
            y2: min(max(0, xy.y + halfHeight), 1)
        )
    }

    static func binaryMask(from mask: MLXArray) -> [UInt8] {
        mask.asType(.int32).asArray(Int32.self).map { $0 == 0 ? 0 : 1 }
    }

    #if canImport(CoreGraphics)
    private static func renderAnnotatedImage(
        baseImage: CGImage,
        detections: [PreparedDetection]
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
            throw GrounderError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
        }

        for (index, detection) in detections.enumerated() {
            let color = overlayColor(for: detection, index: index)
            if let mask = detection.binaryMask {
                for pixelIndex in 0..<(width * height) where mask[pixelIndex] != 0 {
                    let base = pixelIndex * bytesPerPixel
                    bytes[base] = UInt8(Float(bytes[base]) * 0.55 + Float(color.0) * 0.45)
                    bytes[base + 1] = UInt8(Float(bytes[base + 1]) * 0.55 + Float(color.1) * 0.45)
                    bytes[base + 2] = UInt8(Float(bytes[base + 2]) * 0.55 + Float(color.2) * 0.45)
                    bytes[base + 3] = 255
                }
            }

            drawBox(
                into: &bytes,
                width: width,
                height: height,
                box: detection.box,
                color: color,
                lineWidth: 2
            )
        }

        drawLabels(into: &bytes, width: width, height: height, detections: detections)

        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            throw GrounderError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
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
            throw GrounderError.failedToWriteImage(URL(fileURLWithPath: "<memory>"))
        }
        return image
    }

    private static func overlayColor(for detection: PreparedDetection, index: Int) -> (UInt8, UInt8, UInt8) {
        let seed = detection.label
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

                let color = overlayColor(for: detection, index: index)
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
                let desiredX = CGFloat(Int((detection.box.x1 * Float(width)).rounded(.down)))
                let desiredTopY = CGFloat(Int((detection.box.y1 * Float(height)).rounded(.down))) - labelHeight - 4
                let rectX = min(max(0, desiredX), max(0, CGFloat(width) - labelWidth))
                let rectTopY = max(0, desiredTopY)
                let rectY = CGFloat(height) - rectTopY - labelHeight

                context.setFillColor(CGColor(
                    red: CGFloat(color.0) / 255,
                    green: CGFloat(color.1) / 255,
                    blue: CGFloat(color.2) / 255,
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
        box: FalconPerceptionBoundingBox,
        color: (UInt8, UInt8, UInt8),
        lineWidth: Int
    ) {
        let x1 = max(0, min(width - 1, Int((box.x1 * Float(width)).rounded(.down))))
        let y1 = max(0, min(height - 1, Int((box.y1 * Float(height)).rounded(.down))))
        let x2 = max(0, min(width - 1, Int((box.x2 * Float(width)).rounded(.down))))
        let y2 = max(0, min(height - 1, Int((box.y2 * Float(height)).rounded(.down))))
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

    private static func writeMaskArtifacts(
        _ detections: [PreparedDetection],
        width: Int,
        height: Int,
        outputDirectoryURL: URL?
    ) throws -> [PreparedDetection] {
        guard let outputDirectoryURL else { return detections }
        return try detections.enumerated().map { index, detection in
            guard let binaryMask = detection.binaryMask else { return detection }
            let base = slugify(detection.label)
            let url = outputDirectoryURL.appendingPathComponent("\(base)_mask_\(index)").appendingPathExtension("png")
            try writeMaskImage(binaryMask: binaryMask, width: width, height: height, to: url)
            return PreparedDetection(
                label: detection.label,
                xy: detection.xy,
                hw: detection.hw,
                box: detection.box,
                score: detection.score,
                binaryMask: binaryMask,
                maskPath: url.path
            )
        }
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
            throw GrounderError.failedToWriteMask(url)
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
            throw GrounderError.failedToWriteMask(url)
        }
        try writeImage(image, to: url)
    }

    private static func writeImage(_ image: CGImage, to url: URL) throws {
        let utType = UTType(filenameExtension: url.pathExtension.lowercased()) ?? .png
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, utType.identifier as CFString, 1, nil) else {
            throw GrounderError.failedToWriteImage(url)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw GrounderError.failedToWriteImage(url)
        }
    }
    #else
    private static func renderAnnotatedImage(
        baseImage: MediaImage,
        detections: [PreparedDetection]
    ) throws -> MediaImage {
        let width = baseImage.width
        let height = baseImage.height
        var bytes = baseImage.rgba8

        for (index, detection) in detections.enumerated() {
            let color = overlayColor(for: detection, index: index)
            if let mask = detection.binaryMask {
                for pixelIndex in 0..<(width * height) where mask[pixelIndex] != 0 {
                    let base = pixelIndex * 4
                    bytes[base] = UInt8(Float(bytes[base]) * 0.55 + Float(color.0) * 0.45)
                    bytes[base + 1] = UInt8(Float(bytes[base + 1]) * 0.55 + Float(color.1) * 0.45)
                    bytes[base + 2] = UInt8(Float(bytes[base + 2]) * 0.55 + Float(color.2) * 0.45)
                    bytes[base + 3] = 255
                }
            }

            drawBox(
                into: &bytes,
                width: width,
                height: height,
                box: detection.box,
                color: color,
                lineWidth: 2
            )
        }

        return try MediaImage(width: width, height: height, rgba8: bytes)
    }

    private static func overlayColor(for detection: PreparedDetection, index: Int) -> (UInt8, UInt8, UInt8) {
        let seed = detection.label
        let hash = seed.isEmpty ? index : abs(seed.hashValue)
        return overlayColors[hash % overlayColors.count]
    }

    private static func drawBox(
        into bytes: inout [UInt8],
        width: Int,
        height: Int,
        box: FalconPerceptionBoundingBox,
        color: (UInt8, UInt8, UInt8),
        lineWidth: Int
    ) {
        let x1 = max(0, min(width - 1, Int((box.x1 * Float(width)).rounded(.down))))
        let y1 = max(0, min(height - 1, Int((box.y1 * Float(height)).rounded(.down))))
        let x2 = max(0, min(width - 1, Int((box.x2 * Float(width)).rounded(.down))))
        let y2 = max(0, min(height - 1, Int((box.y2 * Float(height)).rounded(.down))))
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

    private static func writeMaskArtifacts(
        _ detections: [PreparedDetection],
        width: Int,
        height: Int,
        outputDirectoryURL: URL?
    ) throws -> [PreparedDetection] {
        guard let outputDirectoryURL else { return detections }
        return try detections.enumerated().map { index, detection in
            guard let binaryMask = detection.binaryMask else { return detection }
            let base = slugify(detection.label)
            let url = outputDirectoryURL.appendingPathComponent("\(base)_mask_\(index)").appendingPathExtension("png")
            try writeMaskImage(binaryMask: binaryMask, width: width, height: height, to: url)
            return PreparedDetection(
                label: detection.label,
                xy: detection.xy,
                hw: detection.hw,
                box: detection.box,
                score: detection.score,
                binaryMask: binaryMask,
                maskPath: url.path
            )
        }
    }

    private static func writeMaskImage(
        binaryMask: [UInt8],
        width: Int,
        height: Int,
        to url: URL
    ) throws {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for pixelIndex in 0..<(width * height) where binaryMask[pixelIndex] != 0 {
            let offset = pixelIndex * 4
            rgba[offset] = 255
            rgba[offset + 1] = 255
            rgba[offset + 2] = 255
            rgba[offset + 3] = 255
        }

        do {
            try MediaImageIO.writePNG(MediaImage(width: width, height: height, rgba8: rgba), to: url)
        } catch {
            throw GrounderError.failedToWriteMask(url)
        }
    }
    #endif

    private static func writeMetadata(_ metadata: FalconPerceptionGroundingMetadata, to url: URL) throws {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(metadata)
            try data.write(to: url, options: [.atomic])
        } catch {
            throw GrounderError.failedToEncodeMetadata(error.localizedDescription)
        }
    }

    private static func slugify(_ value: String) -> String {
        let lowered = value.lowercased()
        let parts = lowered.split { !$0.isLetter && !$0.isNumber }
        let joined = parts.joined(separator: "-")
        return joined.isEmpty ? "object" : joined
    }
}
