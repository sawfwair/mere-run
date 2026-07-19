import Foundation
import MediaIO

public enum SCAIL2MaskPreparationError: LocalizedError, Sendable {
    case inputNotFound(URL)
    case outputAlreadyExists(URL)
    case previewFrameOutOfRange(frame: Int, frameCount: Int)
    case emptyReference(subjectID: String)
    case missingSeed(subjectID: String)
    case frameMismatch(expected: Int, actual: Int, artifact: String)
    case invalidDimensions(expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
    case frameRateMismatch(expected: Double, actual: Double, artifact: String)

    public var errorDescription: String? {
        switch self {
        case .inputNotFound(let url):
            "Mask preparation input not found: \(url.path)"
        case .outputAlreadyExists(let url):
            "Mask output is immutable once written; choose a new directory because this manifest exists: \(url.path)"
        case .previewFrameOutOfRange(let frame, let frameCount):
            "Preview frame \(frame) is outside the normalized driving range 0..<\(frameCount)."
        case .emptyReference(let subjectID):
            "Reference segmentation for subject \(subjectID) produced an empty mask."
        case .missingSeed(let subjectID):
            "No driving-video seed was found for subject \(subjectID)."
        case .frameMismatch(let expected, let actual, let artifact):
            "\(artifact) has \(actual) frames; expected \(expected)."
        case .invalidDimensions(let expectedWidth, let expectedHeight, let actualWidth, let actualHeight):
            "Artifact dimensions are \(actualWidth)x\(actualHeight); expected \(expectedWidth)x\(expectedHeight)."
        case .frameRateMismatch(let expected, let actual, let artifact):
            "\(artifact) has \(actual) FPS; expected \(expected) FPS."
        }
    }
}

public final class SCAIL2MaskPreparer: @unchecked Sendable {
    private struct SubjectTrack {
        let subject: SCAIL2MaskSubject
        let base: SAM31TrackingRun
        let corrections: [(correction: SCAIL2MaskCorrection, run: SAM31TrackingRun)]
    }

    private let segmenter: SAM31ImageSegmenter
    private let fileManager: FileManager

    public init(segmenter: SAM31ImageSegmenter, fileManager: FileManager = .default) {
        self.segmenter = segmenter
        self.fileManager = fileManager
    }

    public static func decodePlan(at planURL: URL) throws -> SCAIL2MaskPlan {
        let data = try Data(contentsOf: planURL)
        let plan = try JSONDecoder().decode(SCAIL2MaskPlan.self, from: data)
            .resolvingPaths(relativeTo: planURL)
        try plan.validate()
        return plan
    }

    public static func validateInputFiles(
        for plan: SCAIL2MaskPlan,
        fileManager: FileManager = .default
    ) throws {
        let paths = [plan.drivingVideo]
            + plan.subjects.map(\.referenceImage)
            + plan.corrections.compactMap(\.paintedBinaryCorrectionPNG)
        for path in paths {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard fileManager.fileExists(atPath: url.path) else {
                throw SCAIL2MaskPreparationError.inputNotFound(url)
            }
        }
    }

    public func prepare(
        plan: SCAIL2MaskPlan,
        outputDirectoryURL: URL,
        previewFrame: Int? = nil,
        modelRevision: String
    ) throws -> SCAIL2MaskPreparationResult {
        try plan.validate()
        try Self.validateInputFiles(for: plan, fileManager: fileManager)
        let outputDirectoryURL = outputDirectoryURL.standardizedFileURL
        let manifestURL = outputDirectoryURL.appendingPathComponent("manifest.json")
        guard !fileManager.fileExists(atPath: manifestURL.path) else {
            throw SCAIL2MaskPreparationError.outputAlreadyExists(manifestURL)
        }
        try fileManager.createDirectory(at: outputDirectoryURL, withIntermediateDirectories: true)

        let temporaryRoot = fileManager.temporaryDirectory
            .appendingPathComponent("mererun-scail2-masks", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: temporaryRoot, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporaryRoot) }

        let driving = try normalizeDriving(
            plan: plan,
            outputDirectoryURL: outputDirectoryURL,
            temporaryRoot: temporaryRoot,
            emitProxy: previewFrame == nil
        )
        if let previewFrame {
            guard driving.frames.indices.contains(previewFrame) else {
                throw SCAIL2MaskPreparationError.previewFrameOutOfRange(
                    frame: previewFrame,
                    frameCount: driving.frames.count
                )
            }
            return try preparePreview(
                plan: plan,
                previewFrame: previewFrame,
                drivingFrames: driving.frames,
                outputDirectoryURL: outputDirectoryURL,
                temporaryRoot: temporaryRoot,
                manifestURL: manifestURL,
                modelRevision: modelRevision
            )
        }
        return try prepareFull(
            plan: plan,
            drivingFrames: driving.frames,
            drivingProxyURL: driving.proxyURL!,
            outputDirectoryURL: outputDirectoryURL,
            temporaryRoot: temporaryRoot,
            manifestURL: manifestURL,
            modelRevision: modelRevision
        )
    }

    public static func correctionIndex(
        for frameIndex: Int,
        corrections: [SCAIL2MaskCorrection]
    ) -> Int? {
        guard !corrections.isEmpty else { return nil }
        let ordered = corrections.enumerated().sorted {
            if $0.element.frameIndex == $1.element.frameIndex {
                return $0.offset < $1.offset
            }
            return $0.element.frameIndex < $1.element.frameIndex
        }
        var selected = 0
        if ordered.count > 1 {
            for index in 0..<(ordered.count - 1) {
                let left = ordered[index].element.frameIndex
                let right = ordered[index + 1].element.frameIndex
                let boundary = left + ((right - left) / 2)
                if frameIndex > boundary {
                    selected = index + 1
                }
            }
        }
        return ordered[selected].offset
    }

    public static func correctionPropagationRange(
        at index: Int,
        corrections: [SCAIL2MaskCorrection]
    ) -> ClosedRange<Int>? {
        let ordered = corrections.sorted {
            $0.frameIndex < $1.frameIndex
        }
        guard ordered.indices.contains(index) else { return nil }
        let correction = ordered[index]
        let start: Int
        if index > ordered.startIndex {
            let previous = ordered[index - 1].frameIndex
            start = previous + ((correction.frameIndex - previous) / 2) + 1
        } else {
            start = 0
        }
        let end: Int
        if index < ordered.index(before: ordered.endIndex) {
            let next = ordered[index + 1].frameIndex
            end = correction.frameIndex + ((next - correction.frameIndex) / 2)
        } else {
            end = .max
        }
        return start...end
    }

    private func preparePreview(
        plan: SCAIL2MaskPlan,
        previewFrame: Int,
        drivingFrames: [URL],
        outputDirectoryURL: URL,
        temporaryRoot: URL,
        manifestURL: URL,
        modelRevision: String
    ) throws -> SCAIL2MaskPreparationResult {
        let references = try prepareReferences(
            plan: plan,
            outputDirectoryURL: outputDirectoryURL,
            temporaryRoot: temporaryRoot
        )
        var drivingMasks: [(color: SCAIL2SubjectColor, mask: [UInt8])] = []
        var warnings: [SCAIL2MaskQualityWarning] = []
        for subject in plan.subjects {
            let run = try segment(
                imageURL: drivingFrames[previewFrame],
                prompt: subject.drivingSelector.promptObject(subjectID: subject.id),
                stem: "preview-\(subject.id)",
                temporaryRoot: temporaryRoot,
                threshold: plan.threshold,
                resolution: plan.resolution
            )
            guard let detection = run.detections.first,
                  let maskPath = detection.maskPath else {
                warnings.append(
                    SCAIL2MaskQualityWarning(
                        code: "subject_missing",
                        subjectID: subject.id,
                        frameIndex: previewFrame,
                        message: "Subject \(subject.id) was not detected in preview frame \(previewFrame)."
                    )
                )
                drivingMasks.append(
                    (subject.color, [UInt8](repeating: 0, count: plan.width * plan.height))
                )
                continue
            }
            drivingMasks.append(
                (
                    subject.color,
                    try loadBinaryMask(
                        URL(fileURLWithPath: maskPath),
                        width: plan.width,
                        height: plan.height,
                        allowEmpty: true
                    )
                )
            )
        }

        let composition = try SCAIL2Palette.compose(
            width: plan.width,
            height: plan.height,
            subjectMasks: drivingMasks
        )
        if composition.overlapPixelCount > 0 {
            warnings.append(overlapWarning(pixelCount: composition.overlapPixelCount, frameIndex: previewFrame))
        }
        let driver = try MediaImageIO.decode(drivingFrames[previewFrame])
        let overlay = try SCAIL2Palette.overlay(image: driver, paletteMask: composition.image)
        let overlayURL = outputDirectoryURL
            .appendingPathComponent(String(format: "preview-frame-%05d.png", previewFrame))
        try MediaImageIO.writePNG(overlay, to: overlayURL)
        let contactSheetURL = outputDirectoryURL.appendingPathComponent("contact-sheet.png")
        try MediaImageIO.writePNG(
            try makeContactSheet(images: references.overlays + [overlay]),
            to: contactSheetURL
        )
        let quality = SCAIL2MaskQualityReport(
            blockingErrors: [],
            warnings: warnings,
            overlapPixelCount: composition.overlapPixelCount,
            paletteRoundTripValidated: true
        )
        let qualityURL = outputDirectoryURL.appendingPathComponent("quality.json")
        try writeJSON(quality, to: qualityURL)

        let artifactURLs = references.artifactURLs + [overlayURL, contactSheetURL, qualityURL]
        let artifacts = try artifactURLs.map {
            try artifact(for: $0, relativeTo: outputDirectoryURL)
        }
        let manifest = SCAIL2MaskManifest(
            status: "preview_ready",
            previewFrame: previewFrame,
            modelID: segmenter.modelID,
            modelRevision: modelRevision,
            drivingSourcePath: plan.drivingVideo,
            drivingProxyPath: nil,
            drivingMaskPath: nil,
            overlayPreviewPath: relativePath(overlayURL, to: outputDirectoryURL),
            contactSheetPath: relativePath(contactSheetURL, to: outputDirectoryURL),
            trackingPath: nil,
            qualityPath: relativePath(qualityURL, to: outputDirectoryURL),
            frameCount: drivingFrames.count,
            fps: plan.fps,
            width: plan.width,
            height: plan.height,
            subjects: references.manifests,
            corrections: plan.corrections.map(SCAIL2MaskCorrectionRecord.init),
            artifacts: artifacts
        )
        try writeJSON(manifest, to: manifestURL)
        return SCAIL2MaskPreparationResult(
            manifestPath: manifestURL.path,
            status: manifest.status,
            preview: true,
            frameCount: drivingFrames.count,
            warnings: warnings.count
        )
    }

    private func prepareFull(
        plan: SCAIL2MaskPlan,
        drivingFrames: [URL],
        drivingProxyURL: URL,
        outputDirectoryURL: URL,
        temporaryRoot: URL,
        manifestURL: URL,
        modelRevision: String
    ) throws -> SCAIL2MaskPreparationResult {
        let references = try prepareReferences(
            plan: plan,
            outputDirectoryURL: outputDirectoryURL,
            temporaryRoot: temporaryRoot
        )
        let tracks = try plan.subjects.map { subject in
            try track(
                subject: subject,
                corrections: plan.corrections.filter { $0.subjectID == subject.id },
                plan: plan,
                drivingProxyURL: drivingProxyURL,
                temporaryRoot: temporaryRoot
            )
        }

        let paletteFrameDirectory = temporaryRoot.appendingPathComponent("palette-frames", isDirectory: true)
        let overlayFrameDirectory = temporaryRoot.appendingPathComponent("overlay-frames", isDirectory: true)
        try fileManager.createDirectory(at: paletteFrameDirectory, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: overlayFrameDirectory, withIntermediateDirectories: true)

        var paletteFrameURLs: [URL] = []
        var overlayFrameURLs: [URL] = []
        var warnings: [SCAIL2MaskQualityWarning] = []
        var overlapPixelCount = 0
        var previousGeometry: [String: (area: Int, centroidX: Float, centroidY: Float)] = [:]
        var gapsBySubject: [String: [Int]] = [:]

        for frameIndex in drivingFrames.indices {
            var masks: [(color: SCAIL2SubjectColor, mask: [UInt8])] = []
            for track in tracks {
                let resolved = try resolvedMask(
                    for: frameIndex,
                    track: track,
                    width: plan.width,
                    height: plan.height
                )
                masks.append((track.subject.color, resolved.mask))
                if !resolved.visible {
                    gapsBySubject[track.subject.id, default: []].append(frameIndex)
                }
                if resolved.score < max(0.01, plan.threshold * 0.6), resolved.visible {
                    warnings.append(
                        SCAIL2MaskQualityWarning(
                            code: "weak_score",
                            subjectID: track.subject.id,
                            frameIndex: frameIndex,
                            message: "Subject \(track.subject.id) has weak mask confidence at frame \(frameIndex)."
                        )
                    )
                }
                if let geometry = resolved.geometry,
                   let previous = previousGeometry[track.subject.id] {
                    let areaRatio = Float(geometry.area) / Float(max(previous.area, 1))
                    let dx = geometry.centroidX - previous.centroidX
                    let dy = geometry.centroidY - previous.centroidY
                    let distance = sqrt((dx * dx) + (dy * dy))
                    if areaRatio < 0.4 || areaRatio > 2.5 || distance > 0.25 {
                        warnings.append(
                            SCAIL2MaskQualityWarning(
                                code: "abrupt_change",
                                subjectID: track.subject.id,
                                frameIndex: frameIndex,
                                message: "Subject \(track.subject.id) changes abruptly at frame \(frameIndex)."
                            )
                        )
                    }
                }
                if let geometry = resolved.geometry {
                    previousGeometry[track.subject.id] = geometry
                }
            }

            let composition = try SCAIL2Palette.compose(
                width: plan.width,
                height: plan.height,
                subjectMasks: masks
            )
            overlapPixelCount += composition.overlapPixelCount
            if composition.overlapPixelCount > 0 {
                warnings.append(overlapWarning(pixelCount: composition.overlapPixelCount, frameIndex: frameIndex))
            }
            let paletteURL = paletteFrameDirectory
                .appendingPathComponent(String(format: "frame_%05d.png", frameIndex))
            try MediaImageIO.writePNG(composition.image, to: paletteURL)
            paletteFrameURLs.append(paletteURL)

            let driver = try MediaImageIO.decode(drivingFrames[frameIndex])
            let overlay = try SCAIL2Palette.overlay(image: driver, paletteMask: composition.image)
            let overlayURL = overlayFrameDirectory
                .appendingPathComponent(String(format: "frame_%05d.png", frameIndex))
            try MediaImageIO.writePNG(overlay, to: overlayURL)
            overlayFrameURLs.append(overlayURL)
        }

        for subject in plan.subjects {
            for range in contiguousRanges(gapsBySubject[subject.id] ?? []) {
                warnings.append(
                    SCAIL2MaskQualityWarning(
                        code: "disappearance",
                        subjectID: subject.id,
                        range: range,
                        message: "Subject \(subject.id) is absent in frames \(range.lowerBound)...\(range.upperBound)."
                    )
                )
            }
        }

        let drivingMaskURL = outputDirectoryURL.appendingPathComponent("driving-mask.mov")
        try MediaVideoIO.writePaletteVideo(frameURLs: paletteFrameURLs, fps: plan.fps, to: drivingMaskURL)
        try validatePaletteVideo(
            drivingMaskURL,
            expectedFrameCount: drivingFrames.count,
            expectedFPS: plan.fps,
            width: plan.width,
            height: plan.height,
            tolerance: plan.paletteTolerance,
            temporaryRoot: temporaryRoot
        )

        let overlayPreviewURL = outputDirectoryURL.appendingPathComponent("overlay-preview.mp4")
        try MediaVideoIO.writeVideo(frameURLs: overlayFrameURLs, fps: plan.fps, to: overlayPreviewURL)
        let contactSheetURL = outputDirectoryURL.appendingPathComponent("contact-sheet.png")
        let contactFrames = try contactSheetIndices(frameCount: overlayFrameURLs.count).map { index in
            try MediaImageIO.decode(overlayFrameURLs[index])
        }
        try MediaImageIO.writePNG(
            try makeContactSheet(images: references.overlays + contactFrames),
            to: contactSheetURL
        )

        let tracking = SCAIL2MaskTrackingReport(
            modelID: segmenter.modelID,
            frameCount: drivingFrames.count,
            fps: plan.fps,
            subjects: tracks.map {
                SCAIL2MaskTrackingSubject(
                    id: $0.subject.id,
                    color: $0.subject.color,
                    seedFrameIndex: $0.base.objects.first?.seedFrameIndex,
                    frames: $0.base.frames
                )
            }
        )
        let trackingURL = outputDirectoryURL.appendingPathComponent("tracking.json")
        try writeJSON(tracking, to: trackingURL)
        let quality = SCAIL2MaskQualityReport(
            blockingErrors: [],
            warnings: warnings,
            overlapPixelCount: overlapPixelCount,
            paletteRoundTripValidated: true
        )
        let qualityURL = outputDirectoryURL.appendingPathComponent("quality.json")
        try writeJSON(quality, to: qualityURL)

        let artifactURLs = references.artifactURLs + [
            drivingProxyURL,
            drivingMaskURL,
            overlayPreviewURL,
            contactSheetURL,
            trackingURL,
            qualityURL,
        ]
        let artifacts = try artifactURLs.map {
            try artifact(for: $0, relativeTo: outputDirectoryURL)
        }
        let subjectManifests = references.manifests.map { reference in
            let seed = tracks.first { $0.subject.id == reference.id }?.base.objects.first?.seedFrameIndex
            return SCAIL2MaskSubjectManifest(
                id: reference.id,
                color: reference.color,
                referenceImagePath: reference.referenceImagePath,
                referenceMaskPath: reference.referenceMaskPath,
                seedFrameIndex: seed,
                gapRanges: contiguousRanges(gapsBySubject[reference.id] ?? [])
            )
        }
        let manifest = SCAIL2MaskManifest(
            status: "ready",
            previewFrame: nil,
            modelID: segmenter.modelID,
            modelRevision: modelRevision,
            drivingSourcePath: plan.drivingVideo,
            drivingProxyPath: relativePath(drivingProxyURL, to: outputDirectoryURL),
            drivingMaskPath: relativePath(drivingMaskURL, to: outputDirectoryURL),
            overlayPreviewPath: relativePath(overlayPreviewURL, to: outputDirectoryURL),
            contactSheetPath: relativePath(contactSheetURL, to: outputDirectoryURL),
            trackingPath: relativePath(trackingURL, to: outputDirectoryURL),
            qualityPath: relativePath(qualityURL, to: outputDirectoryURL),
            frameCount: drivingFrames.count,
            fps: plan.fps,
            width: plan.width,
            height: plan.height,
            subjects: subjectManifests,
            corrections: plan.corrections.map(SCAIL2MaskCorrectionRecord.init),
            artifacts: artifacts
        )
        try writeJSON(manifest, to: manifestURL)
        return SCAIL2MaskPreparationResult(
            manifestPath: manifestURL.path,
            status: manifest.status,
            preview: false,
            frameCount: drivingFrames.count,
            warnings: warnings.count
        )
    }

    private func normalizeDriving(
        plan: SCAIL2MaskPlan,
        outputDirectoryURL: URL,
        temporaryRoot: URL,
        emitProxy: Bool
    ) throws -> (frames: [URL], proxyURL: URL?) {
        let extractedDirectory = temporaryRoot.appendingPathComponent("source-frames", isDirectory: true)
        let source = try MediaVideoIO.extractFrames(
            from: URL(fileURLWithPath: plan.drivingVideo),
            into: extractedDirectory
        )
        let start = plan.inSeconds ?? 0
        let sourceDuration = Double(source.frameURLs.count) / source.fps
        let end = min(plan.outSeconds ?? sourceDuration, sourceDuration)
        let outputFrameCount = max(1, Int(((end - start) * plan.fps).rounded(.down)))
        let normalizedDirectory = temporaryRoot.appendingPathComponent("normalized-frames", isDirectory: true)
        try fileManager.createDirectory(at: normalizedDirectory, withIntermediateDirectories: true)
        var normalizedFrames: [URL] = []
        normalizedFrames.reserveCapacity(outputFrameCount)
        for frameIndex in 0..<outputFrameCount {
            let timestamp = start + (Double(frameIndex) / plan.fps)
            let sourceIndex = min(
                source.frameURLs.count - 1,
                max(0, Int((timestamp * source.fps).rounded()))
            )
            let image = try MediaImageIO.decode(source.frameURLs[sourceIndex])
            let normalized = try MediaImageIO.centerCropped(image, width: plan.width, height: plan.height)
            let url = normalizedDirectory
                .appendingPathComponent(String(format: "frame_%05d.png", frameIndex))
            try MediaImageIO.writePNG(normalized, to: url)
            normalizedFrames.append(url)
        }
        guard emitProxy else { return (normalizedFrames, nil) }

        let proxyURL = outputDirectoryURL.appendingPathComponent("driving-proxy.mp4")
        let silentURL = temporaryRoot.appendingPathComponent("driving-proxy-silent.mp4")
        try MediaVideoIO.writeVideo(frameURLs: normalizedFrames, fps: plan.fps, to: silentURL)
        let drivingURL = URL(fileURLWithPath: plan.drivingVideo)
        if MediaVideoIO.hasAudioTrack(drivingURL) {
            let duration = Double(normalizedFrames.count) / plan.fps
            let audio = try MediaAudioIO.decodeSegment(
                drivingURL,
                startTime: start,
                duration: duration,
                targetSampleRate: 48_000,
                channels: 2
            )
            let targetSampleCount = Int((duration * 48_000).rounded()) * 2
            let samples = Array(audio.samples.prefix(targetSampleCount))
                + [Float](repeating: 0, count: max(0, targetSampleCount - audio.samples.count))
            let audioURL = temporaryRoot.appendingPathComponent("driving-audio.wav")
            try MediaAudioIO.writeFloatWAV(
                samples: Array(samples.prefix(targetSampleCount)),
                sampleRate: 48_000,
                channels: 2,
                to: audioURL
            )
            try MediaVideoIO.mux(
                videoURL: silentURL,
                audioURL: audioURL,
                outputURL: proxyURL,
                audioBitRate: 192_000
            )
        } else {
            try fileManager.moveItem(at: silentURL, to: proxyURL)
        }
        let verificationDirectory = temporaryRoot.appendingPathComponent("proxy-verification", isDirectory: true)
        let verified = try MediaVideoIO.extractFrames(from: proxyURL, into: verificationDirectory)
        try validateSequence(
            verified,
            expectedFrameCount: normalizedFrames.count,
            expectedFPS: plan.fps,
            width: plan.width,
            height: plan.height,
            artifact: "Normalized driving proxy"
        )
        return (normalizedFrames, proxyURL)
    }

    private func prepareReferences(
        plan: SCAIL2MaskPlan,
        outputDirectoryURL: URL,
        temporaryRoot: URL
    ) throws -> (
        manifests: [SCAIL2MaskSubjectManifest],
        overlays: [MediaImage],
        artifactURLs: [URL]
    ) {
        var manifests: [SCAIL2MaskSubjectManifest] = []
        var overlays: [MediaImage] = []
        var artifactURLs: [URL] = []
        for subject in plan.subjects {
            let referenceURL = URL(fileURLWithPath: subject.referenceImage)
            let run = try segment(
                imageURL: referenceURL,
                prompt: subject.referenceSelector.promptObject(subjectID: subject.id),
                stem: "reference-\(subject.id)",
                temporaryRoot: temporaryRoot,
                threshold: plan.threshold,
                resolution: plan.resolution
            )
            guard let detection = run.detections.first,
                  detection.maskAreaPixels > 0,
                  let maskPath = detection.maskPath else {
                throw SCAIL2MaskPreparationError.emptyReference(subjectID: subject.id)
            }
            let referenceImage = try MediaImageIO.decode(referenceURL)
            let binary = try loadBinaryMask(
                URL(fileURLWithPath: maskPath),
                width: referenceImage.width,
                height: referenceImage.height,
                allowEmpty: false
            )
            let palette = try SCAIL2Palette.compose(
                width: referenceImage.width,
                height: referenceImage.height,
                subjectMasks: [(subject.color, binary)]
            ).image
            let maskURL = outputDirectoryURL
                .appendingPathComponent("reference-\(subject.id)-mask.png")
            let overlayURL = outputDirectoryURL
                .appendingPathComponent("reference-\(subject.id)-overlay.png")
            try MediaImageIO.writePNG(palette, to: maskURL)
            let overlay = try SCAIL2Palette.overlay(image: referenceImage, paletteMask: palette)
            try MediaImageIO.writePNG(overlay, to: overlayURL)
            overlays.append(overlay)
            artifactURLs.append(contentsOf: [maskURL, overlayURL])
            manifests.append(
                SCAIL2MaskSubjectManifest(
                    id: subject.id,
                    color: subject.color,
                    referenceImagePath: subject.referenceImage,
                    referenceMaskPath: relativePath(maskURL, to: outputDirectoryURL),
                    seedFrameIndex: nil,
                    gapRanges: []
                )
            )
        }
        return (manifests, overlays, artifactURLs)
    }

    private func track(
        subject: SCAIL2MaskSubject,
        corrections: [SCAIL2MaskCorrection],
        plan: SCAIL2MaskPlan,
        drivingProxyURL: URL,
        temporaryRoot: URL
    ) throws -> SubjectTrack {
        let tracker = SAM31VideoTracker(segmenter: segmenter, fileManager: fileManager)
        let baseRoot = temporaryRoot.appendingPathComponent("track-\(subject.id)", isDirectory: true)
        let base = try tracker.track(
            videoURL: drivingProxyURL,
            promptSet: SAM31PromptSet(
                objectPrompts: [subject.drivingSelector.promptObject(subjectID: subject.id)]
            ),
            outputVideoURL: baseRoot.appendingPathComponent("overlay.mp4"),
            jsonOutputURL: baseRoot.appendingPathComponent("tracking.json"),
            threshold: plan.threshold,
            resolution: plan.resolution,
            maskOutputDirectoryURL: baseRoot.appendingPathComponent("masks", isDirectory: true),
            seedFrameSearchLimit: plan.seedFrameSearchLimit
        )
        var correctionRuns: [(SCAIL2MaskCorrection, SAM31TrackingRun)] = []
        let orderedCorrections = corrections.sorted(by: { $0.frameIndex < $1.frameIndex })
        for (correctionIndex, correction) in orderedCorrections.enumerated() {
            let correctionRoot = temporaryRoot
                .appendingPathComponent("correction-\(subject.id)-\(correction.frameIndex)", isDirectory: true)
            let maskURL = try correction.paintedBinaryCorrectionPNG.map { path -> URL in
                let image = try MediaImageIO.decode(URL(fileURLWithPath: path))
                let resized = try MediaImageIO.resized(image, width: plan.width, height: plan.height)
                let output = correctionRoot.appendingPathComponent("authoritative.png")
                try MediaImageIO.writePNG(resized, to: output)
                return output
            }
            let prompt = correction.selector.promptObject(subjectID: subject.id, maskURL: maskURL)
            let propagationRange = Self.correctionPropagationRange(
                at: correctionIndex,
                corrections: orderedCorrections
            )!
            let run = try tracker.track(
                videoURL: drivingProxyURL,
                promptSet: SAM31PromptSet(objectPrompts: [prompt]),
                outputVideoURL: correctionRoot.appendingPathComponent("overlay.mp4"),
                jsonOutputURL: correctionRoot.appendingPathComponent("tracking.json"),
                initFrameIndex: correction.frameIndex,
                startFrameIndex: propagationRange.lowerBound,
                endFrameIndex: propagationRange.upperBound == .max
                    ? nil
                    : propagationRange.upperBound,
                threshold: plan.threshold,
                resolution: plan.resolution,
                maskOutputDirectoryURL: correctionRoot.appendingPathComponent("masks", isDirectory: true)
            )
            guard !run.objects.isEmpty else {
                throw SCAIL2MaskPreparationError.missingSeed(subjectID: subject.id)
            }
            correctionRuns.append((correction, run))
        }
        let effectiveBase: SAM31TrackingRun
        if base.objects.isEmpty {
            guard let correctionRun = correctionRuns.first?.1 else {
                throw SCAIL2MaskPreparationError.missingSeed(subjectID: subject.id)
            }
            effectiveBase = correctionRun
        } else {
            effectiveBase = base
        }
        return SubjectTrack(subject: subject, base: effectiveBase, corrections: correctionRuns)
    }

    private func segment(
        imageURL: URL,
        prompt: SAM31PromptObject,
        stem: String,
        temporaryRoot: URL,
        threshold: Float,
        resolution: Int
    ) throws -> SAM31SegmentationRun {
        let root = temporaryRoot.appendingPathComponent(stem, isDirectory: true)
        return try segmenter.segment(
            imageURL: imageURL,
            promptSet: SAM31PromptSet(objectPrompts: [prompt]),
            annotatedImageURL: root.appendingPathComponent("overlay.png"),
            jsonOutputURL: root.appendingPathComponent("result.json"),
            threshold: threshold,
            resolution: resolution,
            maskOutputDirectoryURL: root.appendingPathComponent("masks", isDirectory: true)
        )
    }

    private func resolvedMask(
        for frameIndex: Int,
        track: SubjectTrack,
        width: Int,
        height: Int
    ) throws -> (
        mask: [UInt8],
        visible: Bool,
        score: Float,
        geometry: (area: Int, centroidX: Float, centroidY: Float)?
    ) {
        let correctionIndex = Self.correctionIndex(
            for: frameIndex,
            corrections: track.corrections.map(\.correction)
        )
        let run = correctionIndex.map { track.corrections[$0].run } ?? track.base
        let correction = correctionIndex.map { track.corrections[$0].correction }
        if correction?.frameIndex == frameIndex,
           let path = correction?.paintedBinaryCorrectionPNG {
            let mask = try loadBinaryMask(
                URL(fileURLWithPath: path),
                width: width,
                height: height,
                allowEmpty: false
            )
            return (mask, true, 1, maskGeometry(mask, width: width, height: height))
        }
        guard let frame = run.frames.first(where: { $0.frameIndex == frameIndex }),
              let detection = frame.detections.first(where: { $0.objectID == track.subject.id }),
              detection.visible,
              let maskPath = detection.maskPath else {
            return ([UInt8](repeating: 0, count: width * height), false, 0, nil)
        }
        let mask = try loadBinaryMask(
            URL(fileURLWithPath: maskPath),
            width: width,
            height: height,
            allowEmpty: true
        )
        return (
            mask,
            mask.contains(1),
            detection.score,
            maskGeometry(mask, width: width, height: height)
        )
    }

    private func loadBinaryMask(
        _ url: URL,
        width: Int,
        height: Int,
        allowEmpty: Bool
    ) throws -> [UInt8] {
        let decoded = try MediaImageIO.decode(url)
        let resized = decoded.width == width && decoded.height == height
            ? decoded
            : try MediaImageIO.resized(decoded, width: width, height: height)
        var mask = [UInt8](repeating: 0, count: width * height)
        var active = 0
        for pixelIndex in mask.indices {
            let offset = pixelIndex * 4
            let value = max(resized.rgba8[offset], max(resized.rgba8[offset + 1], resized.rgba8[offset + 2]))
            if value >= 128 {
                mask[pixelIndex] = 1
                active += 1
            }
        }
        if active == 0 && !allowEmpty {
            throw SCAIL2PaletteError.invalidBinaryMask(path: url.path)
        }
        return mask
    }

    private func validatePaletteVideo(
        _ url: URL,
        expectedFrameCount: Int,
        expectedFPS: Double,
        width: Int,
        height: Int,
        tolerance: UInt8,
        temporaryRoot: URL
    ) throws {
        let directory = temporaryRoot.appendingPathComponent("palette-round-trip", isDirectory: true)
        let sequence = try MediaVideoIO.extractFrames(from: url, into: directory)
        try validateSequence(
            sequence,
            expectedFrameCount: expectedFrameCount,
            expectedFPS: expectedFPS,
            width: width,
            height: height,
            artifact: "Driving mask video"
        )
        for frameURL in sequence.frameURLs {
            _ = try SCAIL2Palette.snapped(try MediaImageIO.decode(frameURL), tolerance: tolerance)
        }
    }

    private func validateSequence(
        _ sequence: VideoFrameSequence,
        expectedFrameCount: Int,
        expectedFPS: Double?,
        width: Int,
        height: Int,
        artifact: String
    ) throws {
        guard sequence.frameURLs.count == expectedFrameCount else {
            throw SCAIL2MaskPreparationError.frameMismatch(
                expected: expectedFrameCount,
                actual: sequence.frameURLs.count,
                artifact: artifact
            )
        }
        if let expectedFPS, abs(sequence.fps - expectedFPS) > 0.01 {
            throw SCAIL2MaskPreparationError.frameRateMismatch(
                expected: expectedFPS,
                actual: sequence.fps,
                artifact: artifact
            )
        }
        guard sequence.frameWidth == width, sequence.frameHeight == height else {
            throw SCAIL2MaskPreparationError.invalidDimensions(
                expectedWidth: width,
                expectedHeight: height,
                actualWidth: sequence.frameWidth,
                actualHeight: sequence.frameHeight
            )
        }
    }

    private func artifact(for url: URL, relativeTo outputDirectoryURL: URL) throws -> SCAIL2MaskArtifact {
        SCAIL2MaskArtifact(
            kind: url.deletingPathExtension().lastPathComponent,
            path: relativePath(url, to: outputDirectoryURL),
            sha256: try ModelArtifactPin.fileSHA256(url),
            byteCount: try ModelArtifactPin.fileByteCount(url)
        )
    }

    private func writeJSON<Value: Encodable>(_ value: Value, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: [.atomic])
    }

    private func relativePath(_ url: URL, to directory: URL) -> String {
        let prefix = directory.standardizedFileURL.path + "/"
        let path = url.standardizedFileURL.path
        return path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
    }

    private func contiguousRanges(_ frames: [Int]) -> [ClosedRange<Int>] {
        let ordered = Array(Set(frames)).sorted()
        guard let first = ordered.first else { return [] }
        var result: [ClosedRange<Int>] = []
        var start = first
        var previous = first
        for frame in ordered.dropFirst() {
            if frame != previous + 1 {
                result.append(start...previous)
                start = frame
            }
            previous = frame
        }
        result.append(start...previous)
        return result
    }

    private func overlapWarning(pixelCount: Int, frameIndex: Int) -> SCAIL2MaskQualityWarning {
        SCAIL2MaskQualityWarning(
            code: "subject_overlap",
            frameIndex: frameIndex,
            message: "\(pixelCount) subject mask pixels overlap at frame \(frameIndex)."
        )
    }

    private func maskGeometry(
        _ mask: [UInt8],
        width: Int,
        height: Int
    ) -> (area: Int, centroidX: Float, centroidY: Float)? {
        var area = 0
        var sumX = 0
        var sumY = 0
        for index in mask.indices where mask[index] != 0 {
            area += 1
            sumX += index % width
            sumY += index / width
        }
        guard area > 0 else { return nil }
        return (
            area,
            Float(sumX) / Float(area * max(width, 1)),
            Float(sumY) / Float(area * max(height, 1))
        )
    }

    private func contactSheetIndices(frameCount: Int) -> [Int] {
        guard frameCount > 1 else { return [0] }
        return Array(Set([0, frameCount / 4, frameCount / 2, (frameCount * 3) / 4, frameCount - 1])).sorted()
    }

    private func makeContactSheet(images: [MediaImage]) throws -> MediaImage {
        let thumbnailWidth = 320
        let thumbnailHeight = 180
        let columns = min(3, max(1, images.count))
        let rows = Int(ceil(Double(images.count) / Double(columns)))
        var rgba = [UInt8](repeating: 24, count: columns * thumbnailWidth * rows * thumbnailHeight * 4)
        for alphaIndex in stride(from: 3, to: rgba.count, by: 4) {
            rgba[alphaIndex] = 255
        }
        for (index, image) in images.enumerated() {
            let thumbnail = try MediaImageIO.centerCropped(
                image,
                width: thumbnailWidth,
                height: thumbnailHeight
            )
            let originX = (index % columns) * thumbnailWidth
            let originY = (index / columns) * thumbnailHeight
            for y in 0..<thumbnailHeight {
                let sourceStart = y * thumbnailWidth * 4
                let destinationStart = ((originY + y) * columns * thumbnailWidth + originX) * 4
                rgba.replaceSubrange(
                    destinationStart..<(destinationStart + thumbnailWidth * 4),
                    with: thumbnail.rgba8[sourceStart..<(sourceStart + thumbnailWidth * 4)]
                )
            }
        }
        return try MediaImage(
            width: columns * thumbnailWidth,
            height: rows * thumbnailHeight,
            rgba8: rgba
        )
    }
}
