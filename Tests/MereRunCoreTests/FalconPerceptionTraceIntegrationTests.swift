import Foundation
import XCTest
import MLX
import MLXNN
@testable import MereRunCore

final class FalconPerceptionTraceIntegrationTests: MereRunCoreTestCase {
    func testLocalCheckpointCachedPresenceStepMatchesFullForward() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_FALCON_ASSERT_CACHE_PARITY"] == "1" else {
            throw XCTSkip("Set MERERUN_FALCON_ASSERT_CACHE_PARITY=1 to run the cached-vs-full Falcon parity diagnostic.")
        }
        guard let modelRoot = env["MERERUN_FALCON_PERCEPTION_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_MODEL_ROOT to run Falcon checkpoint parity coverage.")
        }
        guard let imagePath = env["MERERUN_FALCON_PERCEPTION_TRACE_IMAGE"], !imagePath.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_TRACE_IMAGE to run Falcon checkpoint parity coverage.")
        }

        let query = env["MERERUN_FALCON_PERCEPTION_TRACE_QUERY"] ?? "person"
        let modelURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path),
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("Falcon parity assets are missing on disk.")
        }

        let resources = FalconPerceptionResources(rootURL: modelURL)
        let config = try FalconPerceptionModelConfig.load(from: resources.configURL)
        let tokenizer = try FalconPerceptionTokenizer.load(from: resources.tokenizerRootURL)
        let processor = FalconPerceptionProcessor(tokenizer: tokenizer, config: config)
        let processed = try processor.process(imageURL: imageURL, query: query)

        let cachedModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: cachedModel)
        let fullModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: fullModel)

        let promptPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: processed.inputIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        cachedModel.resetGroundingState()
        cachedModel.prepareGroundingPrefill(positionData: promptPositionData)
        let caches = cachedModel.makeCaches()
        let promptEmbeds = cachedModel.makeInputEmbeddings(
            inputIDs: processed.inputIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        var cachedLogits = cachedModel.forward(
            inputIDs: processed.inputIDs,
            inputsEmbeds: promptEmbeds,
            caches: caches
        )
        MLX.eval(cachedLogits)
        cachedModel.finishGroundingPrefill()

        let presenceToken: Int32 = 268
        let presenceIDs = MLXArray([presenceToken], [1, 1])
        let presenceEmbeds = cachedModel.embedTokens(presenceIDs)
        cachedLogits = cachedModel.forward(
            inputIDs: presenceIDs,
            inputsEmbeds: presenceEmbeds,
            caches: caches
        )
        MLX.eval(cachedLogits)

        let promptTokenIDs = processed.inputIDs.asArray(Int32.self)
        let fullTokenIDs = promptTokenIDs + [presenceToken]
        let fullIDs = MLXArray(fullTokenIDs, [1, fullTokenIDs.count])
        let fullPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: fullIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        let fullEmbeds = fullModel.makeInputEmbeddings(
            inputIDs: fullIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        let fullLogits = fullModel.forward(
            inputIDs: fullIDs,
            inputsEmbeds: fullEmbeds,
            mask: fullPositionData.attentionMask,
            caches: nil,
            positionIDs: fullPositionData.positionIDs,
            posHW: fullPositionData.posHW
        )
        MLX.eval(fullLogits)

        let cachedStep = cachedLogits[0, cachedLogits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let fullStep = fullLogits[0, fullLogits.dim(1) - 1].asType(.float32).asArray(Float.self)

        XCTAssertEqual(cachedStep.count, fullStep.count)
        let cachedTop = cachedStep.enumerated().max { $0.element < $1.element }?.offset
        let fullTop = fullStep.enumerated().max { $0.element < $1.element }?.offset
        XCTAssertEqual(cachedTop, fullTop, "Cached decode and full forward should agree on the next token after <|presence|>.")

        let maxDiff = zip(cachedStep, fullStep).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThan(
            maxDiff,
            0.05,
            "Loaded Falcon checkpoint should keep cached decode close to full-sequence forward for the first perception step."
        )
    }

    func testSwiftTraceAgainstLocalFalconModel() throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelRoot = env["MERERUN_FALCON_PERCEPTION_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_MODEL_ROOT to run Falcon trace coverage.")
        }
        guard let imagePath = env["MERERUN_FALCON_PERCEPTION_TRACE_IMAGE"], !imagePath.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_TRACE_IMAGE to run Falcon trace coverage.")
        }

        let query = env["MERERUN_FALCON_PERCEPTION_TRACE_QUERY"] ?? "person"
        let maxSteps = Int(env["MERERUN_FALCON_PERCEPTION_TRACE_STEPS"] ?? "6") ?? 6
        let traceLogURL = env["MERERUN_FALCON_PERCEPTION_TRACE_LOG"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }

        func log(_ line: String) {
            if let traceLogURL {
                let data = (line + "\n").data(using: .utf8) ?? Data()
                if FileManager.default.fileExists(atPath: traceLogURL.path) {
                    if let handle = try? FileHandle(forWritingTo: traceLogURL) {
                        try? handle.seekToEnd()
                        try? handle.write(contentsOf: data)
                        try? handle.close()
                    }
                } else {
                    FileManager.default.createFile(atPath: traceLogURL.path, contents: data)
                }
            }
            print(line)
        }

        let modelURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path),
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("Falcon trace assets are missing on disk.")
        }

        let resources = FalconPerceptionResources(rootURL: modelURL)
        let config = try FalconPerceptionModelConfig.load(from: resources.configURL)
        let tokenizer = try FalconPerceptionTokenizer.load(from: resources.tokenizerRootURL)
        let processor = FalconPerceptionProcessor(tokenizer: tokenizer, config: config)
        let model = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: model)

        let processed = try processor.process(imageURL: imageURL, query: query)
        let grid = processed.imageGridHW.asArray(Int32.self).map(Int.init)
        let inputIDs = processed.inputIDs.asArray(Int32.self).map(Int.init)

        log("TRACE swift query: \(query)")
        log("TRACE swift processed_size: \(processed.processedSize.width) \(processed.processedSize.height)")
        log("TRACE swift grid_hw: \(grid)")
        log("TRACE swift input_count: \(inputIDs.count)")
        log("TRACE swift input_prefix: \(Array(inputIDs.prefix(48)))")

        let positionData = FalconPerceptionModel.computePositionData(
            inputIDs: processed.inputIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        log("TRACE swift rope_delta: \(positionData.ropeDelta)")
        log("TRACE swift position_prefix: \(Array(positionData.positionIDs.asArray(Int32.self).prefix(48)))")

        model.resetGroundingState()
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
            caches: caches
        )
        MLX.eval(logits)
        model.finishGroundingPrefill()

        var segmentationFeatures: MLXArray?
        if let hiddenState = model.lastHiddenState {
            let gridH = grid.indices.contains(0) ? grid[0] : 0
            let gridW = grid.indices.contains(1) ? grid[1] : 0
            segmentationFeatures = model.computeSegmentationFeatures(
                hiddenState: hiddenState,
                inputIDs: processed.inputIDs,
                pixelValues: processed.pixelValues,
                gridH: gridH,
                gridW: gridW
            )
            if let segmentationFeatures {
                MLX.eval(segmentationFeatures)
                log("TRACE swift segm_features_shape: \(segmentationFeatures.shape)")
            } else {
                log("TRACE swift segm_features_shape: nil")
            }
        }

        Self.printStep(
            prefix: "TRACE swift prefill",
            logits: logits,
            model: model,
            tokenizer: tokenizer,
            log: log
        )

        var pendingCoordXY: MLXArray?
        var pendingSizeHW: MLXArray?
        var currentXY: FalconPerceptionCenter?
        var currentHW: FalconPerceptionSize?
        var detections: [(xy: FalconPerceptionCenter, hw: FalconPerceptionSize, maskPixels: Int?)] = []

        for step in 0..<maxSteps {
            let lastLogits = logits[0, logits.dim(1) - 1]
            let tokenID = Int(MLX.argMax(lastLogits).item(Int32.self))
            let tokenString = tokenizer.decode(token: tokenID)
            let hiddenLast = model.lastHiddenState?[0, model.lastHiddenState!.dim(1) - 1]
            let decoded = hiddenLast.map { Self.decodeHidden($0, with: model) }

            log("TRACE swift step \(step) token: \(tokenID) \(tokenString)")
            if let decoded {
                log(
                    "TRACE swift step \(step) hidden_decoded: "
                        + String(format: "x=%.6f y=%.6f h=%.6f w=%.6f",
                                 decoded.x, decoded.y, decoded.h, decoded.w)
                )
            }

            if tokenID == config.eosID {
                log("TRACE swift step \(step) hit_eos")
                break
            }

            if let decoded {
                if tokenID == config.coordTokenID {
                    currentXY = FalconPerceptionCenter(x: decoded.x, y: decoded.y)
                    pendingCoordXY = MLXArray([decoded.x, decoded.y], [1, 2]).asType(.float32)
                } else if tokenID == config.sizeTokenID {
                    currentHW = FalconPerceptionSize(h: decoded.h, w: decoded.w)
                    pendingSizeHW = MLXArray([decoded.h, decoded.w], [1, 2]).asType(.float32)
                } else if tokenID == config.segTokenID {
                    var maskPixels: Int?
                    if let segmentationFeatures, let hiddenLast {
                        let mask = model.decodeSegmentationMask(
                            segHidden: hiddenLast,
                            segmentationFeatures: segmentationFeatures,
                            outputHeight: processed.processedSize.height,
                            outputWidth: processed.processedSize.width,
                            threshold: 0.5
                        )
                        if let mask {
                            MLX.eval(mask)
                            maskPixels = mask.asType(DType.int32).asArray(Int32.self).reduce(0) { partial, value in
                                partial + Int(value)
                            }
                        }
                    }
                    log(
                        "TRACE swift step \(step) committed: "
                            + (currentXY.map { String(format: "xy=(%.6f,%.6f)", $0.x, $0.y) } ?? "xy=nil")
                            + " "
                            + (currentHW.map { String(format: "hw=(%.6f,%.6f)", $0.h, $0.w) } ?? "hw=nil")
                            + " "
                            + (maskPixels.map { "mask_pixels=\($0)" } ?? "mask_pixels=nil")
                    )
                    if let currentXY, let currentHW {
                        detections.append((xy: currentXY, hw: currentHW, maskPixels: maskPixels))
                    }
                    currentXY = nil
                    currentHW = nil
                }
            }

            let tokenArray = MLXArray([Int32(tokenID)], [1, 1])
            var tokenEmbeds = model.embedTokens(tokenArray)
            if tokenID == config.coordTokenID, let pendingCoordXY {
                tokenEmbeds = model.encodeCoordinates(
                    into: tokenEmbeds,
                    inputIDs: tokenArray,
                    coordXY: pendingCoordXY
                )
            } else if tokenID == config.sizeTokenID, let pendingSizeHW {
                tokenEmbeds = model.encodeSizes(
                    into: tokenEmbeds,
                    inputIDs: tokenArray,
                    sizeHW: pendingSizeHW
                )
            }

            logits = model.forward(
                inputIDs: tokenArray,
                inputsEmbeds: tokenEmbeds,
                caches: caches
            )
            MLX.eval(logits)
        }

        if let currentXY, let currentHW {
            detections.append((xy: currentXY, hw: currentHW, maskPixels: nil))
        }
        log("TRACE swift detection_count: \(detections.count)")
        for (index, detection) in detections.enumerated() {
            log(
                "TRACE swift detection \(index): "
                    + String(format: "xy=(%.6f,%.6f) hw=(%.6f,%.6f)",
                             detection.xy.x, detection.xy.y, detection.hw.h, detection.hw.w)
                    + " "
                    + (detection.maskPixels.map { "mask_pixels=\($0)" } ?? "mask_pixels=nil")
            )
        }
    }

    func testLocalCheckpointLayerParityTrace() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_FALCON_TRACE_LAYER_PARITY"] == "1" else {
            throw XCTSkip("Set MERERUN_FALCON_TRACE_LAYER_PARITY=1 to run Falcon layer parity tracing.")
        }
        guard let modelRoot = env["MERERUN_FALCON_PERCEPTION_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_MODEL_ROOT to run Falcon layer parity tracing.")
        }
        guard let imagePath = env["MERERUN_FALCON_PERCEPTION_TRACE_IMAGE"], !imagePath.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_TRACE_IMAGE to run Falcon layer parity tracing.")
        }

        let traceLogURL = env["MERERUN_FALCON_PERCEPTION_TRACE_LOG"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        func log(_ line: String) {
            if let traceLogURL {
                let data = (line + "\n").data(using: .utf8) ?? Data()
                if FileManager.default.fileExists(atPath: traceLogURL.path) {
                    if let handle = try? FileHandle(forWritingTo: traceLogURL) {
                        try? handle.seekToEnd()
                        try? handle.write(contentsOf: data)
                        try? handle.close()
                    }
                } else {
                    FileManager.default.createFile(atPath: traceLogURL.path, contents: data)
                }
            }
            print(line)
        }

        let query = env["MERERUN_FALCON_PERCEPTION_TRACE_QUERY"] ?? "person"
        let modelURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path),
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("Falcon layer parity assets are missing on disk.")
        }

        let resources = FalconPerceptionResources(rootURL: modelURL)
        let config = try FalconPerceptionModelConfig.load(from: resources.configURL)
        let tokenizer = try FalconPerceptionTokenizer.load(from: resources.tokenizerRootURL)
        let processor = FalconPerceptionProcessor(tokenizer: tokenizer, config: config)
        let processed = try processor.process(imageURL: imageURL, query: query)

        let cachedModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: cachedModel)
        let fullModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: fullModel)
        cachedModel.languageModel.model.captureLayerOutputs = true
        fullModel.languageModel.model.captureLayerOutputs = true

        let promptPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: processed.inputIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        cachedModel.resetGroundingState()
        cachedModel.prepareGroundingPrefill(positionData: promptPositionData)
        let caches = cachedModel.makeCaches()
        let promptEmbeds = cachedModel.makeInputEmbeddings(
            inputIDs: processed.inputIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        _ = cachedModel.forward(
            inputIDs: processed.inputIDs,
            inputsEmbeds: promptEmbeds,
            caches: caches
        )
        cachedModel.finishGroundingPrefill()

        let presenceToken: Int32 = 268
        let presenceIDs = MLXArray([presenceToken], [1, 1])
        let presenceEmbeds = cachedModel.embedTokens(presenceIDs)
        let cachedLogits = cachedModel.forward(
            inputIDs: presenceIDs,
            inputsEmbeds: presenceEmbeds,
            caches: caches
        )
        MLX.eval(cachedLogits)

        let promptTokenIDs = processed.inputIDs.asArray(Int32.self)
        let fullTokenIDs = promptTokenIDs + [presenceToken]
        let fullIDs = MLXArray(fullTokenIDs, [1, fullTokenIDs.count])
        let fullPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: fullIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        let fullEmbeds = fullModel.makeInputEmbeddings(
            inputIDs: fullIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        let fullLogits = fullModel.forward(
            inputIDs: fullIDs,
            inputsEmbeds: fullEmbeds,
            mask: fullPositionData.attentionMask,
            caches: nil,
            positionIDs: fullPositionData.positionIDs,
            posHW: fullPositionData.posHW
        )
        MLX.eval(fullLogits)

        let cachedLayers = cachedModel.languageModel.model.capturedLayerOutputs
        let fullLayers = fullModel.languageModel.model.capturedLayerOutputs
        log("TRACE layer-parity query: \(query)")
        log("TRACE layer-parity cached_layers: \(cachedLayers.count)")
        log("TRACE layer-parity full_layers: \(fullLayers.count)")

        let layerCount = min(cachedLayers.count, fullLayers.count)
        var firstDivergingLayer: Int?
        for layerIndex in 0..<layerCount {
            let cachedValues = cachedLayers[layerIndex][0, cachedLayers[layerIndex].dim(1) - 1]
                .asType(.float32)
                .asArray(Float.self)
            let fullValues = fullLayers[layerIndex][0, fullLayers[layerIndex].dim(1) - 1]
                .asType(.float32)
                .asArray(Float.self)
            let maxDiff = zip(cachedValues, fullValues).map { abs($0 - $1) }.max() ?? .infinity
            let meanDiff = zip(cachedValues, fullValues).reduce(Float.zero) { partial, pair in
                partial + abs(pair.0 - pair.1)
            } / Float(max(1, cachedValues.count))
            log(String(format: "TRACE layer-parity layer=%d max_diff=%.6f mean_diff=%.6f", layerIndex, maxDiff, meanDiff))
            if firstDivergingLayer == nil, maxDiff > 0.01 {
                firstDivergingLayer = layerIndex
            }
        }

        let cachedStep = cachedLogits[0, cachedLogits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let fullStep = fullLogits[0, fullLogits.dim(1) - 1].asType(.float32).asArray(Float.self)
        let cachedTop = cachedStep.enumerated().max { $0.element < $1.element }?.offset ?? -1
        let fullTop = fullStep.enumerated().max { $0.element < $1.element }?.offset ?? -1
        let logitsMaxDiff = zip(cachedStep, fullStep).map { abs($0 - $1) }.max() ?? .infinity
        log("TRACE layer-parity logits cached_top=\(cachedTop) full_top=\(fullTop) logits_max_diff=\(logitsMaxDiff)")
        log("TRACE layer-parity first_diverging_layer=\(firstDivergingLayer.map(String.init) ?? "none")")

        XCTAssertEqual(cachedLayers.count, fullLayers.count)
    }

    func testLocalCheckpointLayer0AttentionTrace() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["MERERUN_FALCON_TRACE_LAYER0_ATTENTION"] == "1" else {
            throw XCTSkip("Set MERERUN_FALCON_TRACE_LAYER0_ATTENTION=1 to run Falcon layer-0 attention tracing.")
        }
        guard let modelRoot = env["MERERUN_FALCON_PERCEPTION_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_MODEL_ROOT to run Falcon layer-0 attention tracing.")
        }
        guard let imagePath = env["MERERUN_FALCON_PERCEPTION_TRACE_IMAGE"], !imagePath.isEmpty else {
            throw XCTSkip("Set MERERUN_FALCON_PERCEPTION_TRACE_IMAGE to run Falcon layer-0 attention tracing.")
        }

        let traceLogURL = env["MERERUN_FALCON_PERCEPTION_TRACE_LOG"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
        func log(_ line: String) {
            if let traceLogURL {
                let data = (line + "\n").data(using: .utf8) ?? Data()
                if FileManager.default.fileExists(atPath: traceLogURL.path) {
                    if let handle = try? FileHandle(forWritingTo: traceLogURL) {
                        try? handle.seekToEnd()
                        try? handle.write(contentsOf: data)
                        try? handle.close()
                    }
                } else {
                    FileManager.default.createFile(atPath: traceLogURL.path, contents: data)
                }
            }
            print(line)
        }

        let query = env["MERERUN_FALCON_PERCEPTION_TRACE_QUERY"] ?? "person"
        let modelURL = URL(fileURLWithPath: modelRoot).standardizedFileURL
        let imageURL = URL(fileURLWithPath: imagePath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path),
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw XCTSkip("Falcon layer-0 attention trace assets are missing on disk.")
        }

        let resources = FalconPerceptionResources(rootURL: modelURL)
        let config = try FalconPerceptionModelConfig.load(from: resources.configURL)
        let tokenizer = try FalconPerceptionTokenizer.load(from: resources.tokenizerRootURL)
        let processor = FalconPerceptionProcessor(tokenizer: tokenizer, config: config)
        let processed = try processor.process(imageURL: imageURL, query: query)

        let cachedModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: cachedModel)
        let fullModel = FalconPerceptionModel(config: config)
        try Self.loadWeightsForTrace(resources: resources, into: fullModel)
        cachedModel.languageModel.model.layers[0].selfAttn.captureDebugStages = true
        fullModel.languageModel.model.layers[0].selfAttn.captureDebugStages = true

        let promptPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: processed.inputIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        cachedModel.resetGroundingState()
        cachedModel.prepareGroundingPrefill(positionData: promptPositionData)
        let caches = cachedModel.makeCaches()
        let promptEmbeds = cachedModel.makeInputEmbeddings(
            inputIDs: processed.inputIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        _ = cachedModel.forward(
            inputIDs: processed.inputIDs,
            inputsEmbeds: promptEmbeds,
            caches: caches
        )
        cachedModel.finishGroundingPrefill()

        let presenceToken: Int32 = 268
        let presenceIDs = MLXArray([presenceToken], [1, 1])
        let presenceEmbeds = cachedModel.embedTokens(presenceIDs)
        _ = cachedModel.forward(
            inputIDs: presenceIDs,
            inputsEmbeds: presenceEmbeds,
            caches: caches
        )

        let promptTokenIDs = processed.inputIDs.asArray(Int32.self)
        let fullTokenIDs = promptTokenIDs + [presenceToken]
        let fullIDs = MLXArray(fullTokenIDs, [1, fullTokenIDs.count])
        let fullPositionData = FalconPerceptionModel.computePositionData(
            inputIDs: fullIDs,
            config: config,
            imageGridHW: processed.imageGridHW
        )
        let fullEmbeds = fullModel.makeInputEmbeddings(
            inputIDs: fullIDs,
            pixelValues: processed.pixelValues,
            imageGridHW: processed.imageGridHW
        )
        _ = fullModel.forward(
            inputIDs: fullIDs,
            inputsEmbeds: fullEmbeds,
            mask: fullPositionData.attentionMask,
            caches: nil,
            positionIDs: fullPositionData.positionIDs,
            posHW: fullPositionData.posHW
        )

        guard let cachedCapture = cachedModel.languageModel.model.layers.first?.selfAttn.lastDebugCapture,
              let fullCapture = fullModel.languageModel.model.layers.first?.selfAttn.lastDebugCapture else {
            XCTFail("Expected both cached and full layer-0 attention captures.")
            return
        }

        func lastTokenVector(_ tensor: MLXArray) -> [Float] {
            tensor[0, tensor.dim(1) - 1].asType(.float32).asArray(Float.self)
        }

        func lastTokenAcrossHeads(_ tensor: MLXArray) -> [Float] {
            let headCount = tensor.dim(1)
            let lastIndex = tensor.dim(2) - 1
            var values: [Float] = []
            values.reserveCapacity(max(1, headCount) * max(1, tensor.dim(3)))
            for head in 0..<headCount {
                values.append(contentsOf: tensor[0, head, lastIndex].asType(.float32).asArray(Float.self))
            }
            return values
        }

        func firstQueryMaskRow(_ tensor: MLXArray) -> [Float] {
            tensor[0, 0, 0].asType(.float32).asArray(Float.self)
        }

        func lastQueryMaskRow(_ tensor: MLXArray) -> [Float] {
            tensor[0, 0, tensor.dim(2) - 1].asType(.float32).asArray(Float.self)
        }

        func arrayDiffs(_ lhs: [Float], _ rhs: [Float]) -> (max: Float, mean: Float) {
            let pairs = zip(lhs, rhs)
            let maxDiff = pairs.map { abs($0 - $1) }.max() ?? .infinity
            let meanDiff = zip(lhs, rhs).reduce(Float.zero) { partial, pair in
                partial + abs(pair.0 - pair.1)
            } / Float(max(1, min(lhs.count, rhs.count)))
            return (maxDiff, meanDiff)
        }

        let stages: [(String, [Float]?, [Float]?)] = [
            ("input", cachedCapture.input.map(lastTokenVector), fullCapture.input.map(lastTokenVector)),
            ("normalized_input", cachedCapture.normalizedInput.map(lastTokenVector), fullCapture.normalizedInput.map(lastTokenVector)),
            ("qkv_projected", cachedCapture.qkvProjected.map(lastTokenVector), fullCapture.qkvProjected.map(lastTokenVector)),
            ("queries_before_rope", cachedCapture.queriesBeforeRoPE.map(lastTokenAcrossHeads), fullCapture.queriesBeforeRoPE.map(lastTokenAcrossHeads)),
            ("keys_before_rope", cachedCapture.keysBeforeRoPE.map(lastTokenAcrossHeads), fullCapture.keysBeforeRoPE.map(lastTokenAcrossHeads)),
            ("values_before_cache", cachedCapture.valuesBeforeCache.map(lastTokenAcrossHeads), fullCapture.valuesBeforeCache.map(lastTokenAcrossHeads)),
            ("queries_after_rope", cachedCapture.queriesAfterRoPE.map(lastTokenAcrossHeads), fullCapture.queriesAfterRoPE.map(lastTokenAcrossHeads)),
            ("keys_after_rope", cachedCapture.keysAfterRoPE.map(lastTokenAcrossHeads), fullCapture.keysAfterRoPE.map(lastTokenAcrossHeads)),
            ("keys_after_cache", cachedCapture.keysAfterCache.map { $0.asType(.float32).asArray(Float.self) }, fullCapture.keysAfterCache.map { $0.asType(.float32).asArray(Float.self) }),
            ("values_after_cache", cachedCapture.valuesAfterCache.map { $0.asType(.float32).asArray(Float.self) }, fullCapture.valuesAfterCache.map { $0.asType(.float32).asArray(Float.self) }),
            ("attention_mask_row", cachedCapture.attentionMask.map(firstQueryMaskRow), fullCapture.attentionMask.map(lastQueryMaskRow)),
            ("attention_output", cachedCapture.attentionOutput.map(lastTokenVector), fullCapture.attentionOutput.map(lastTokenVector)),
            ("projected_output", cachedCapture.projectedOutput.map(lastTokenVector), fullCapture.projectedOutput.map(lastTokenVector)),
        ]

        log("TRACE layer0-attn query: \(query)")
        var firstDivergingStage: String?
        for (name, cachedValues, fullValues) in stages {
            guard let cachedValues, let fullValues else {
                log("TRACE layer0-attn stage=\(name) missing_capture")
                continue
            }
            if cachedValues.count != fullValues.count {
                log("TRACE layer0-attn stage=\(name) shape_mismatch cached_count=\(cachedValues.count) full_count=\(fullValues.count)")
                if firstDivergingStage == nil {
                    firstDivergingStage = name
                }
                continue
            }
            let diffs = arrayDiffs(cachedValues, fullValues)
            log(String(format: "TRACE layer0-attn stage=%@ max_diff=%.6f mean_diff=%.6f", name, diffs.max, diffs.mean))
            if firstDivergingStage == nil, diffs.max > 0.01 {
                firstDivergingStage = name
            }
        }
        log("TRACE layer0-attn first_diverging_stage=\(firstDivergingStage ?? "none")")

        XCTAssertNotNil(cachedCapture.projectedOutput)
        XCTAssertNotNil(fullCapture.projectedOutput)
    }

    private static func printStep(
        prefix: String,
        logits: MLXArray,
        model: FalconPerceptionModel,
        tokenizer: FalconPerceptionTokenizer,
        log: (String) -> Void
    ) {
        let lastLogits = logits[0, logits.dim(1) - 1]
        let tokenID = Int(MLX.argMax(lastLogits).item(Int32.self))
        let tokenString = tokenizer.decode(token: tokenID)
        log("\(prefix) next_token: \(tokenID) \(tokenString)")
        if let hiddenLast = model.lastHiddenState?[0, model.lastHiddenState!.dim(1) - 1] {
            let decoded = decodeHidden(hiddenLast, with: model)
            log(
                "\(prefix) hidden_decoded: "
                    + String(format: "x=%.6f y=%.6f h=%.6f w=%.6f",
                             decoded.x, decoded.y, decoded.h, decoded.w)
            )
        }
    }

    private static func decodeHidden(
        _ hiddenLast: MLXArray,
        with model: FalconPerceptionModel
    ) -> (x: Float, y: Float, h: Float, w: Float) {
        let hiddenForDecode = hiddenLast.reshaped(1, hiddenLast.dim(0))
        let coordLogits = model.decodeCoordinates(from: hiddenForDecode)
        let coordBins = MLX.argMax(coordLogits, axis: -1).asArray(Int32.self)
        let coordDenominator = Float(max(1, coordLogits.dim(-1) - 1))
        let sizeLogits = model.decodeSizes(from: hiddenForDecode)
        let sizeValues = model.processSizes(sizeLogits).asArray(Float.self)
        return (
            x: Float(coordBins[0]) / coordDenominator,
            y: Float(coordBins[1]) / coordDenominator,
            h: sizeValues[0],
            w: sizeValues[1]
        )
    }

    private static func loadWeightsForTrace(
        resources: FalconPerceptionResources,
        into model: FalconPerceptionModel
    ) throws {
        try FalconPerceptionGrounder.loadWeights(resources: resources, into: model)
    }
}
