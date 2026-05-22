import Foundation

public final class SAM31VideoTracker: @unchecked Sendable {
    public enum TrackingError: LocalizedError, Sendable {
        case unsupportedPlatform
        case initFrameOutOfRange(Int)

        public var errorDescription: String? {
            switch self {
            case .unsupportedPlatform:
                return "Video tracking requires a supported MediaIO video backend."
            case .initFrameOutOfRange(let frame):
                return "Initial tracking frame \(frame) is outside the extracted video frame range."
            }
        }
    }

    private let segmenter: SAM31ImageSegmenter
    private let fileManager: FileManager

    private struct TrackingState: Sendable {
        let trackedObject: SAM31TrackedObject
        let seedPromptObject: SAM31PromptObject
        let seedResult: SAM31TrackingObjectResult
    }

    public init(segmenter: SAM31ImageSegmenter, fileManager: FileManager = .default) {
        self.segmenter = segmenter
        self.fileManager = fileManager
    }

    public func track(
        videoURL: URL,
        promptSet: SAM31PromptSet,
        outputVideoURL: URL,
        jsonOutputURL: URL? = nil,
        initFrameIndex: Int = 0,
        endFrameIndex: Int? = nil,
        threshold: Float = 0.3,
        resolution: Int = 1008,
        showBoxes: Bool = false,
        showLabels: Bool = false,
        maskOutputDirectoryURL: URL? = nil,
        seedFrameSearchLimit: Int = 0
    ) throws -> SAM31TrackingRun {
        let tempRoot = fileManager.temporaryDirectory
            .appendingPathComponent("mererun-sam31-track", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        let framesDir = tempRoot.appendingPathComponent("frames", isDirectory: true)
        let annotatedDir = tempRoot.appendingPathComponent("annotated", isDirectory: true)
        let frameJSONDir = tempRoot.appendingPathComponent("json", isDirectory: true)
        try fileManager.createDirectory(at: annotatedDir, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: frameJSONDir, withIntermediateDirectories: true)

        let asset = try SAM31VideoIO.extractFrames(from: videoURL, into: framesDir, endFrame: endFrameIndex)
        guard initFrameIndex >= 0, initFrameIndex < asset.frameURLs.count else {
            throw TrackingError.initFrameOutOfRange(initFrameIndex)
        }

        var annotatedFrameURLs = asset.frameURLs
        var resultsByFrame: [Int: [SAM31TrackingObjectResult]] = [:]

        let normalizedPrompts = try promptSet.normalized()
        var seedFrameIndex = initFrameIndex
        var seedRun = try runSeedSegmentation(
            frameIndex: seedFrameIndex,
            frameURLs: asset.frameURLs,
            promptSet: promptSet,
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            annotatedDir: annotatedDir,
            frameJSONDir: frameJSONDir,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )
        if seedRun.detections.isEmpty && seedFrameSearchLimit > 0 {
            let lastCandidateFrame = min(asset.frameURLs.count - 1, initFrameIndex + seedFrameSearchLimit)
            if lastCandidateFrame > initFrameIndex {
                for candidateFrameIndex in (initFrameIndex + 1)...lastCandidateFrame {
                    let candidateRun = try runSeedSegmentation(
                        frameIndex: candidateFrameIndex,
                        frameURLs: asset.frameURLs,
                        promptSet: promptSet,
                        threshold: threshold,
                        resolution: resolution,
                        showBoxes: showBoxes,
                        showLabels: showLabels,
                        annotatedDir: annotatedDir,
                        frameJSONDir: frameJSONDir,
                        maskOutputDirectoryURL: maskOutputDirectoryURL
                    )
                    if !candidateRun.detections.isEmpty {
                        seedFrameIndex = candidateFrameIndex
                        seedRun = candidateRun
                        break
                    }
                }
            }
        }
        annotatedFrameURLs[seedFrameIndex] = seedRun.annotatedImageURL

        let seedDetections: [String: SAM31SegmentationDetection] = Dictionary(uniqueKeysWithValues: normalizeSeedDetections(seedRun.detections).compactMap { detection in
            guard let objectID = detection.objectID else { return nil }
            return (objectID, detection)
        })
        let trackingStates = normalizedPrompts.compactMap { promptObject -> TrackingState? in
            guard let detection = seedDetections[promptObject.objectID] else { return nil }
            let trackedObject = SAM31TrackedObject(
                objectID: promptObject.objectID,
                label: promptObject.label,
                promptKind: promptObject.promptKind,
                seedFrameIndex: seedFrameIndex,
                textPrompt: promptObject.textPrompt,
                seedBox: promptObject.boxPrompt?.segmentationBox ?? detection.box,
                seedPoints: promptObject.pointPrompts
            )
            let seedResult = SAM31TrackingObjectResult(
                objectID: promptObject.objectID,
                label: promptObject.label,
                score: detection.score,
                visible: detection.maskAreaPixels > 0,
                box: detection.box,
                maskAreaPixels: detection.maskAreaPixels,
                maskPath: detection.maskPath
            )
            return TrackingState(
                trackedObject: trackedObject,
                seedPromptObject: promptObject,
                seedResult: seedResult
            )
        }
        let trackedObjects = trackingStates.map { $0.trackedObject }
        resultsByFrame[seedFrameIndex] = trackingStates.map { $0.seedResult }

        if trackingStates.isEmpty {
            for frameIndex in asset.frameURLs.indices where frameIndex != seedFrameIndex {
                resultsByFrame[frameIndex] = []
            }
        }

        let trackedByObjectID: [String: SAM31TrackedObject] = Dictionary(uniqueKeysWithValues: trackedObjects.map { ($0.objectID, $0) })
        try propagate(
            frameIndices: Array((seedFrameIndex + 1)..<asset.frameURLs.count),
            frameURLs: asset.frameURLs,
            annotatedFrameURLs: &annotatedFrameURLs,
            resultsByFrame: &resultsByFrame,
            trackingStates: trackingStates,
            trackedByObjectID: trackedByObjectID,
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            annotatedDir: annotatedDir,
            frameJSONDir: frameJSONDir,
            maskOutputDirectoryURL: maskOutputDirectoryURL
        )
        if seedFrameIndex > 0 {
            try propagate(
                frameIndices: Array(stride(from: seedFrameIndex - 1, through: 0, by: -1)),
                frameURLs: asset.frameURLs,
                annotatedFrameURLs: &annotatedFrameURLs,
                resultsByFrame: &resultsByFrame,
                trackingStates: trackingStates,
                trackedByObjectID: trackedByObjectID,
                threshold: threshold,
                resolution: resolution,
                showBoxes: showBoxes,
                showLabels: showLabels,
                annotatedDir: annotatedDir,
                frameJSONDir: frameJSONDir,
                maskOutputDirectoryURL: maskOutputDirectoryURL
            )
        }

        try SAM31VideoIO.writeVideo(frameURLs: annotatedFrameURLs, fps: asset.fps, to: outputVideoURL)

        let orderedFrames = resultsByFrame.keys.sorted().map { frameIndex in
            SAM31TrackingFrameResult(
                frameIndex: frameIndex,
                timestampSeconds: Double(frameIndex) / asset.fps,
                detections: resultsByFrame[frameIndex] ?? []
            )
        }
        let run = SAM31TrackingRun(
            modelID: segmenter.modelID,
            inputVideoPath: videoURL.standardizedFileURL.path,
            annotatedVideoPath: outputVideoURL.standardizedFileURL.path,
            jsonOutputPath: jsonOutputURL?.standardizedFileURL.path,
            fps: asset.fps,
            frameWidth: asset.frameWidth,
            frameHeight: asset.frameHeight,
            initFrameIndex: seedFrameIndex,
            objects: trackedObjects,
            frames: orderedFrames
        )
        if let jsonOutputURL {
            try fileManager.createDirectory(at: jsonOutputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(run).write(to: jsonOutputURL, options: [.atomic])
        }
        return run
    }

    private func runSeedSegmentation(
        frameIndex: Int,
        frameURLs: [URL],
        promptSet: SAM31PromptSet,
        threshold: Float,
        resolution: Int,
        showBoxes: Bool,
        showLabels: Bool,
        annotatedDir: URL,
        frameJSONDir: URL,
        maskOutputDirectoryURL: URL?
    ) throws -> SAM31SegmentationRun {
        let maskDirectory = maskOutputDirectoryURL?.appendingPathComponent(String(format: "frame_%05d", frameIndex), isDirectory: true)
        return try segmenter.segment(
            imageURL: frameURLs[frameIndex],
            promptSet: promptSet,
            annotatedImageURL: annotatedDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("png"),
            jsonOutputURL: frameJSONDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("json"),
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            multimask: false,
            maskOutputDirectoryURL: maskDirectory
        )
    }

    private func propagate(
        frameIndices: [Int],
        frameURLs: [URL],
        annotatedFrameURLs: inout [URL],
        resultsByFrame: inout [Int: [SAM31TrackingObjectResult]],
        trackingStates: [TrackingState],
        trackedByObjectID: [String: SAM31TrackedObject],
        threshold: Float,
        resolution: Int,
        showBoxes: Bool,
        showLabels: Bool,
        annotatedDir: URL,
        frameJSONDir: URL,
        maskOutputDirectoryURL: URL?
    ) throws {
        guard let seedFrameIndex = trackingStates.first?.trackedObject.seedFrameIndex,
              let seedResults = resultsByFrame[seedFrameIndex]
        else {
            return
        }

        var previousResultsByObject = Dictionary(uniqueKeysWithValues: seedResults.map { ($0.objectID, $0) })
        let statesByObjectID = Dictionary(uniqueKeysWithValues: trackingStates.map { ($0.trackedObject.objectID, $0) })
        for frameIndex in frameIndices {
            let orderedStates = trackingStates.sorted { $0.trackedObject.objectID < $1.trackedObject.objectID }
            let promptObjects = orderedStates.compactMap { state -> SAM31PromptObject? in
                guard let previous = previousResultsByObject[state.trackedObject.objectID] else { return nil }
                return propagatedPromptObject(for: state, previousResult: previous)
            }

            let maskDirectory = maskOutputDirectoryURL?.appendingPathComponent(String(format: "frame_%05d", frameIndex), isDirectory: true)
            let initialRun = try segmenter.segment(
                imageURL: frameURLs[frameIndex],
                promptSet: promptSet(for: promptObjects),
                annotatedImageURL: annotatedDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("png"),
                jsonOutputURL: frameJSONDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("json"),
                threshold: threshold,
                resolution: resolution,
                showBoxes: showBoxes,
                showLabels: showLabels,
                multimask: false,
                maskOutputDirectoryURL: maskDirectory
            )
            let stabilizedRun = try stabilizedRunIfNeeded(
                initialRun,
                imageURL: frameURLs[frameIndex],
                trackingStates: orderedStates,
                previousResultsByObject: previousResultsByObject,
                threshold: threshold,
                resolution: resolution,
                showBoxes: showBoxes,
                showLabels: showLabels,
                annotatedDir: annotatedDir,
                frameJSONDir: frameJSONDir,
                maskOutputDirectoryURL: maskOutputDirectoryURL,
                frameIndex: frameIndex
            )
            annotatedFrameURLs[frameIndex] = stabilizedRun.annotatedImageURL

            let detectionsByObjectID: [String: SAM31SegmentationDetection] = Dictionary(uniqueKeysWithValues: stabilizedRun.detections.compactMap { detection in
                guard let objectID = detection.objectID else { return nil }
                return (objectID, detection)
            })

            let frameResults = orderedStates.map { state -> SAM31TrackingObjectResult in
                let object = state.trackedObject
                if let detection = detectionsByObjectID[object.objectID] {
                    return SAM31TrackingObjectResult(
                        objectID: object.objectID,
                        label: trackedByObjectID[object.objectID]?.label ?? object.label,
                        score: detection.score,
                        visible: detection.maskAreaPixels > 0,
                        box: detection.box,
                        maskAreaPixels: detection.maskAreaPixels,
                        maskPath: detection.maskPath
                    )
                }
                let previous = previousResultsByObject[object.objectID]
                let seed = statesByObjectID[object.objectID]?.seedResult
                return SAM31TrackingObjectResult(
                    objectID: object.objectID,
                    label: trackedByObjectID[object.objectID]?.label ?? object.label,
                    score: 0,
                    visible: false,
                    box: previous?.box ?? seed?.box ?? SAM31SegmentationBox(x1: 0, y1: 0, x2: 0, y2: 0),
                    maskAreaPixels: 0,
                    maskPath: nil
                )
            }

            resultsByFrame[frameIndex] = frameResults
            previousResultsByObject = Dictionary(uniqueKeysWithValues: frameResults.map { ($0.objectID, $0) })
        }
    }

    private func normalizeSeedDetections(_ detections: [SAM31SegmentationDetection]) -> [SAM31SegmentationDetection] {
        detections.enumerated().map { index, detection in
            guard detection.objectID == nil else { return detection }
            return SAM31SegmentationDetection(
                objectID: defaultObjectID(for: detection.label, index: index),
                label: detection.label,
                promptKind: detection.promptKind,
                score: detection.score,
                box: detection.box,
                maskAreaPixels: detection.maskAreaPixels,
                maskPath: detection.maskPath,
                candidateIndex: detection.candidateIndex
            )
        }
    }

    private func defaultObjectID(for label: String, index: Int) -> String {
        let parts = label.lowercased().split { !$0.isLetter && !$0.isNumber }
        let base = parts.joined(separator: "-")
        return base.isEmpty ? "object-\(index + 1)" : "\(base)-\(index + 1)"
    }

    private func propagatedPromptObject(
        for state: TrackingState,
        previousResult: SAM31TrackingObjectResult
    ) -> SAM31PromptObject {
        let fallbackBox = state.trackedObject.seedBox ?? state.seedResult.box
        let previousBox = previousResult.visible && previousResult.maskAreaPixels > 0 ? previousResult.box : fallbackBox
        let propagatedBox = SAM31PromptBox(
            x1: previousBox.x1,
            y1: previousBox.y1,
            x2: previousBox.x2,
            y2: previousBox.y2,
            label: state.trackedObject.objectID
        )
        return SAM31PromptObject(
            objectID: state.trackedObject.objectID,
            label: state.trackedObject.label,
            promptKind: .box,
            boxPrompt: propagatedBox
        )
    }

    private func promptSet(for promptObjects: [SAM31PromptObject]) -> SAM31PromptSet {
        var promptSet = SAM31PromptSet()
        for promptObject in promptObjects {
            switch promptObject.promptKind {
            case .text:
                if let textPrompt = promptObject.textPrompt, !textPrompt.isEmpty {
                    promptSet.textPrompts.append(textPrompt)
                }
            case .box:
                if let boxPrompt = promptObject.boxPrompt {
                    promptSet.boxPrompts.append(
                        SAM31PromptBox(
                            x1: boxPrompt.x1,
                            y1: boxPrompt.y1,
                            x2: boxPrompt.x2,
                            y2: boxPrompt.y2,
                            label: promptObject.objectID
                        )
                    )
                }
            case .point:
                promptSet.pointPrompts.append(
                    contentsOf: promptObject.pointPrompts.map { point in
                        SAM31PromptPoint(
                            x: point.x,
                            y: point.y,
                            isPositive: point.isPositive,
                            label: promptObject.objectID
                        )
                    }
                )
            }
        }
        return promptSet
    }

    private func stabilizedRunIfNeeded(
        _ initialRun: SAM31SegmentationRun,
        imageURL: URL,
        trackingStates: [TrackingState],
        previousResultsByObject: [String: SAM31TrackingObjectResult],
        threshold: Float,
        resolution: Int,
        showBoxes: Bool,
        showLabels: Bool,
        annotatedDir: URL,
        frameJSONDir: URL,
        maskOutputDirectoryURL: URL?,
        frameIndex: Int
    ) throws -> SAM31SegmentationRun {
        let detectionsByObjectID: [String: SAM31SegmentationDetection] = Dictionary(uniqueKeysWithValues: initialRun.detections.compactMap { detection in
            guard let objectID = detection.objectID else { return nil }
            return (objectID, detection)
        })
        let fallbackPromptObjects = trackingStates.compactMap { state -> SAM31PromptObject? in
            let previousResult = previousResultsByObject[state.trackedObject.objectID] ?? state.seedResult
            let detection = detectionsByObjectID[state.trackedObject.objectID]
            guard needsFallback(
                detection: detection,
                previousResult: previousResult,
                seedResult: state.seedResult,
                threshold: threshold
            ) else {
                return nil
            }
            return fallbackPromptObject(for: state)
        }
        guard !fallbackPromptObjects.isEmpty else {
            return initialRun
        }

        var mergedPromptObjects: [SAM31PromptObject] = []
        mergedPromptObjects.reserveCapacity(trackingStates.count)
        let fallbackByObjectID: [String: SAM31PromptObject] = Dictionary(uniqueKeysWithValues: fallbackPromptObjects.map { ($0.objectID, $0) })
        for state in trackingStates {
            if let fallbackPrompt = fallbackByObjectID[state.trackedObject.objectID] {
                mergedPromptObjects.append(fallbackPrompt)
            } else {
                let previousResult = previousResultsByObject[state.trackedObject.objectID] ?? state.seedResult
                mergedPromptObjects.append(propagatedPromptObject(for: state, previousResult: previousResult))
            }
        }

        let rerunMaskDirectory = maskOutputDirectoryURL?.appendingPathComponent(String(format: "frame_%05d", frameIndex), isDirectory: true)
        return try segmenter.segment(
            imageURL: imageURL,
            promptSet: promptSet(for: mergedPromptObjects),
            annotatedImageURL: annotatedDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("png"),
            jsonOutputURL: frameJSONDir.appendingPathComponent(String(format: "frame_%05d", frameIndex)).appendingPathExtension("json"),
            threshold: threshold,
            resolution: resolution,
            showBoxes: showBoxes,
            showLabels: showLabels,
            multimask: false,
            maskOutputDirectoryURL: rerunMaskDirectory
        )
    }

    private func fallbackPromptObject(for state: TrackingState) -> SAM31PromptObject {
        switch state.seedPromptObject.promptKind {
        case .box:
            let seedBox = state.seedPromptObject.boxPrompt?.segmentationBox ?? state.trackedObject.seedBox ?? state.seedResult.box
            return SAM31PromptObject(
                objectID: state.trackedObject.objectID,
                label: state.trackedObject.label,
                promptKind: .box,
                boxPrompt: SAM31PromptBox(
                    x1: seedBox.x1,
                    y1: seedBox.y1,
                    x2: seedBox.x2,
                    y2: seedBox.y2,
                    label: state.trackedObject.objectID
                )
            )
        case .point:
            return SAM31PromptObject(
                objectID: state.trackedObject.objectID,
                label: state.trackedObject.label,
                promptKind: .point,
                pointPrompts: state.seedPromptObject.pointPrompts.map { point in
                    SAM31PromptPoint(
                        x: point.x,
                        y: point.y,
                        isPositive: point.isPositive,
                        label: state.trackedObject.objectID
                    )
                }
            )
        case .text:
            return SAM31PromptObject(
                objectID: state.trackedObject.objectID,
                label: state.trackedObject.label,
                promptKind: .box,
                boxPrompt: SAM31PromptBox(
                    x1: state.seedResult.box.x1,
                    y1: state.seedResult.box.y1,
                    x2: state.seedResult.box.x2,
                    y2: state.seedResult.box.y2,
                    label: state.trackedObject.objectID
                )
            )
        }
    }

    private func needsFallback(
        detection: SAM31SegmentationDetection?,
        previousResult: SAM31TrackingObjectResult,
        seedResult: SAM31TrackingObjectResult,
        threshold: Float
    ) -> Bool {
        guard let detection else { return true }
        guard detection.maskAreaPixels > 0 else { return true }

        let previousArea = max(previousResult.maskAreaPixels, 1)
        let seedArea = max(seedResult.maskAreaPixels, 1)
        let candidateArea = detection.maskAreaPixels
        let maxReferenceArea = max(previousArea, seedArea)

        if detection.score < max(0.01, threshold * 0.4) {
            return true
        }
        if candidateArea > Int(Float(maxReferenceArea) * 1.75) {
            return true
        }
        if previousResult.visible && previousResult.maskAreaPixels > 0 {
            let overlap = SAM31ImageSegmenter.iou(detection.box, previousResult.box)
            if overlap < 0.05 {
                return true
            }
        }
        return false
    }
}
