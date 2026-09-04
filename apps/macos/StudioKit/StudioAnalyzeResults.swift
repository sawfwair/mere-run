import CoreGraphics
import Foundation

// The result documents the CLI writes for the input-first tasks, decoded into one shape the
// Analyze canvas can draw. These mirror what `MereRunCore` encodes today — camelCase Swift
// property names for the vision writers, snake_case for `speech diarize` — so a decode failure
// here means the CLI changed, not that the app guessed.

// MARK: - vision ground

/// `mere.run vision ground --json-output` (`FalconPerceptionGroundingMetadata`).
///
/// Its boxes are **normalized** 0…1 in the input image's frame; `xy` is the box centre and `hw`
/// is height-then-width. `score` is optional and today's grounder leaves it unset.
package struct StudioVisionGroundDocument: Decodable, Equatable {
    package struct Box: Decodable, Equatable {
        package let x1: Double
        package let y1: Double
        package let x2: Double
        package let y2: Double
    }

    package struct Detection: Decodable, Equatable {
        package let label: String
        package let box: Box
        package let score: Double?
        package let maskPath: String?
    }

    package let schemaVersion: Int
    package let modelID: String
    package let inputImagePath: String
    package let annotatedImagePath: String?
    package let queries: [String]
    package let detections: [Detection]
}

// MARK: - vision segment

/// `mere.run vision segment --json-output` (`SAM31SegmentationMetadata`).
///
/// Its boxes are **absolute pixels**, xyxy, in the input image's frame.
package struct StudioVisionSegmentDocument: Decodable, Equatable {
    package struct Box: Decodable, Equatable {
        package let x1: Double
        package let y1: Double
        package let x2: Double
        package let y2: Double
    }

    package struct Detection: Decodable, Equatable {
        package let objectID: String?
        package let label: String
        package let promptKind: String?
        package let score: Double
        package let box: Box
        package let maskAreaPixels: Int
        package let maskPath: String?
    }

    package let schemaVersion: Int
    package let modelID: String
    package let inputImagePath: String
    package let annotatedImagePath: String?
    package let prompts: [String]
    package let threshold: Double
    package let detections: [Detection]
}

// MARK: - vision track

/// `mere.run vision track --json-output` (`SAM31TrackingRun`).
///
/// Boxes are **absolute pixels** in `frameWidth` × `frameHeight`. Every frame carries an entry for
/// every tracked object, including the ones it lost (`visible: false`, `score: 0`).
package struct StudioVisionTrackDocument: Decodable, Equatable {
    package struct Box: Decodable, Equatable {
        package let x1: Double
        package let y1: Double
        package let x2: Double
        package let y2: Double
    }

    package struct TrackedObject: Decodable, Equatable {
        package let objectID: String
        package let label: String
        package let seedFrameIndex: Int
    }

    package struct FrameDetection: Decodable, Equatable {
        package let objectID: String
        package let label: String
        package let score: Double
        package let visible: Bool
        package let box: Box
        package let maskPath: String?
    }

    package struct Frame: Decodable, Equatable {
        package let frameIndex: Int
        package let timestampSeconds: Double
        package let detections: [FrameDetection]
    }

    package let schemaVersion: Int
    package let modelID: String
    package let inputVideoPath: String
    package let annotatedVideoPath: String?
    package let fps: Double
    package let frameWidth: Int
    package let frameHeight: Int
    package let objects: [TrackedObject]
    package let frames: [Frame]

    package var duration: TimeInterval {
        guard fps > 0 else { return 0 }
        return Double(frames.count) / fps
    }
}

// MARK: - speech diarize

/// `mere.run speech diarize --format json` (`SpeechDiarizationPayload`), the only one of these
/// documents the CLI writes in snake_case.
package struct StudioDiarizationDocument: Decodable, Equatable {
    package struct Segment: Decodable, Equatable {
        package let speaker: String
        package let speakerIndex: Int
        package let startSeconds: Double
        package let endSeconds: Double

        package enum CodingKeys: String, CodingKey {
            case speaker
            case speakerIndex = "speaker_index"
            case startSeconds = "start_seconds"
            case endSeconds = "end_seconds"
        }
    }

    package let schemaVersion: Int
    package let model: String
    package let durationSeconds: Double
    package let speakerCount: Int
    package let segments: [Segment]

    package enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model
        case durationSeconds = "duration_seconds"
        case speakerCount = "speaker_count"
        case segments
    }
}

// MARK: - speech transcribe

/// What `mere.run speech transcribe` writes: the whole transcript, a blank line, then one
/// `[MM:SS.mmm --> MM:SS.mmm] text` line per alignment (an hours field appears past an hour).
/// It is text, not JSON, so it is parsed rather than decoded.
package struct StudioTranscriptDocument: Equatable {
    package struct Segment: Equatable, Identifiable {
        package let index: Int
        package let start: TimeInterval
        package let end: TimeInterval
        package let text: String

        package var id: Int { index }
    }

    package let text: String
    package let segments: [Segment]

    package var duration: TimeInterval {
        segments.last?.end ?? 0
    }

    /// Parses the transcript the CLI wrote. A file with no timestamp lines still yields its text.
    package static func parse(_ raw: String) -> StudioTranscriptDocument {
        var segments: [Segment] = []
        var prose: [String] = []
        for line in raw.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let segment = parseSegment(trimmed, index: segments.count) {
                segments.append(segment)
            } else if !trimmed.isEmpty, segments.isEmpty {
                prose.append(trimmed)
            }
        }
        let text = prose.isEmpty
            ? segments.map(\.text).joined(separator: " ")
            : prose.joined(separator: "\n")
        return StudioTranscriptDocument(text: text, segments: segments)
    }

    /// "[00:04.120 --> 00:07.400] and that is the whole idea."
    private static func parseSegment(_ line: String, index: Int) -> Segment? {
        guard line.hasPrefix("["), let close = line.firstIndex(of: "]") else { return nil }
        let stamps = String(line[line.index(after: line.startIndex)..<close])
        let parts = stamps.components(separatedBy: "-->").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 2,
              let start = seconds(parts[0]),
              let end = seconds(parts[1]) else { return nil }
        let text = String(line[line.index(after: close)...]).trimmingCharacters(in: .whitespaces)
        return Segment(index: index, start: start, end: end, text: text)
    }

    /// "MM:SS.mmm" or "HH:MM:SS.mmm" to seconds.
    package static func seconds(_ stamp: String) -> TimeInterval? {
        let fields = stamp.components(separatedBy: ":")
        guard (2...3).contains(fields.count) else { return nil }
        var total: TimeInterval = 0
        for field in fields.dropLast() {
            guard let value = Double(field) else { return nil }
            total = total * 60 + value
        }
        guard let last = Double(fields[fields.count - 1]) else { return nil }
        return total * 60 + last
    }
}

// MARK: - One shape the canvas draws

/// One thing a run found: what it is, how sure the model was, and where it is in the input's own
/// pixels (origin top-left), with the mask PNG the run wrote for it when there is one.
package struct StudioAnalyzeDetection: Identifiable, Equatable {
    package let id: Int
    package let label: String
    package let confidence: Double?
    /// The input's pixel space, origin top-left.
    package let box: CGRect
    package let maskURL: URL?

    /// "[246, 307, 717, 758]" — the same xyxy the result document carries, rounded to pixels.
    package var boxDescription: String {
        let values = [box.minX, box.minY, box.maxX, box.maxY].map { Int($0.rounded()) }
        return "[\(values.map(String.init).joined(separator: ", "))]"
    }

    package var confidenceDescription: String? {
        guard let confidence else { return nil }
        return String(format: "%.2f", confidence)
    }
}

/// The result document of the run the Analyze canvas is showing.
package enum StudioAnalyzeDocument: Equatable {
    case ground(StudioVisionGroundDocument)
    case segmentation(StudioVisionSegmentDocument)
    case tracking(StudioVisionTrackDocument)
    case diarization(StudioDiarizationDocument)
    case transcript(StudioTranscriptDocument)

    /// Decodes whichever document `data` holds. The vision writers all emit a JSON object with a
    /// distinguishing key (`queries`, `prompts`, `frames`), so the shape identifies itself; a
    /// payload that is not JSON at all is read as a transcript.
    package static func decode(_ data: Data) -> StudioAnalyzeDocument? {
        let decoder = JSONDecoder()
        if let document = try? decoder.decode(StudioVisionTrackDocument.self, from: data) {
            return .tracking(document)
        }
        if let document = try? decoder.decode(StudioVisionSegmentDocument.self, from: data) {
            return .segmentation(document)
        }
        if let document = try? decoder.decode(StudioVisionGroundDocument.self, from: data) {
            return .ground(document)
        }
        if let document = try? decoder.decode(StudioDiarizationDocument.self, from: data) {
            return .diarization(document)
        }
        guard let text = String(data: data, encoding: .utf8), !text.isBlank else { return nil }
        let transcript = StudioTranscriptDocument.parse(text)
        return transcript.segments.isEmpty && transcript.text.isEmpty ? nil : .transcript(transcript)
    }

    /// The model the run used, as the document records it.
    package var modelID: String? {
        switch self {
        case .ground(let document): return document.modelID
        case .segmentation(let document): return document.modelID
        case .tracking(let document): return document.modelID
        case .diarization(let document): return document.model
        case .transcript: return nil
        }
    }

    /// The pixel size the document itself knows, when it records one (only `track` does).
    package var reportedInputSize: CGSize? {
        guard case .tracking(let document) = self else { return nil }
        return CGSize(width: document.frameWidth, height: document.frameHeight)
    }

    /// The detections to draw, in the input's pixel space.
    ///
    /// - Parameters:
    ///   - imageSize: the input's pixel size. `ground` stores normalized boxes, so its detections
    ///     cannot be placed without it; the pixel-space documents ignore it.
    ///   - frame: which tracked frame to read, for `track`.
    package func detections(imageSize: CGSize, frame: Int = 0) -> [StudioAnalyzeDetection] {
        switch self {
        case .ground(let document):
            return document.detections.enumerated().map { index, detection in
                StudioAnalyzeDetection(
                    id: index,
                    label: detection.label,
                    confidence: detection.score,
                    box: CGRect(
                        x: detection.box.x1 * imageSize.width,
                        y: detection.box.y1 * imageSize.height,
                        width: (detection.box.x2 - detection.box.x1) * imageSize.width,
                        height: (detection.box.y2 - detection.box.y1) * imageSize.height
                    ),
                    maskURL: detection.maskPath.map { URL(fileURLWithPath: $0) }
                )
            }
        case .segmentation(let document):
            return document.detections.enumerated().map { index, detection in
                StudioAnalyzeDetection(
                    id: index,
                    label: detection.label,
                    confidence: detection.score,
                    box: CGRect(
                        x: detection.box.x1,
                        y: detection.box.y1,
                        width: detection.box.x2 - detection.box.x1,
                        height: detection.box.y2 - detection.box.y1
                    ),
                    maskURL: detection.maskPath.map { URL(fileURLWithPath: $0) }
                )
            }
        case .tracking(let document):
            guard let match = document.frames.first(where: { $0.frameIndex == frame })
                ?? document.frames.first else { return [] }
            return match.detections.enumerated().compactMap { index, detection in
                guard detection.visible else { return nil }
                return StudioAnalyzeDetection(
                    id: index,
                    label: detection.label,
                    confidence: detection.score,
                    box: CGRect(
                        x: detection.box.x1,
                        y: detection.box.y1,
                        width: detection.box.x2 - detection.box.x1,
                        height: detection.box.y2 - detection.box.y1
                    ),
                    maskURL: detection.maskPath.map { URL(fileURLWithPath: $0) }
                )
            }
        case .diarization, .transcript:
            return []
        }
    }

    /// The result panel's header: what the run found, in the task's own words.
    package func summary(detectionCount: Int) -> String {
        switch self {
        case .ground, .segmentation:
            return detectionCount == 1 ? "1 object found" : "\(detectionCount) objects found"
        case .tracking(let document):
            let objects = document.objects.count == 1 ? "1 object" : "\(document.objects.count) objects"
            return "\(objects) across \(document.frames.count) frames"
        case .diarization(let document):
            let speakers = document.speakerCount == 1 ? "1 speaker" : "\(document.speakerCount) speakers"
            return "\(speakers) · \(document.segments.count) turns"
        case .transcript(let document):
            let count = document.segments.count
            return count == 1 ? "1 segment" : "\(count) segments"
        }
    }

    /// Spoken turns for the transcript and timeline views, whichever document carries them.
    package var speechSegments: [StudioAnalyzeSpeechSegment] {
        switch self {
        case .transcript(let document):
            return document.segments.map {
                StudioAnalyzeSpeechSegment(
                    id: $0.index, speaker: nil, start: $0.start, end: $0.end, text: $0.text
                )
            }
        case .diarization(let document):
            return document.segments.enumerated().map { index, segment in
                StudioAnalyzeSpeechSegment(
                    id: index,
                    speaker: "Speaker \(segment.speakerIndex + 1)",
                    start: segment.startSeconds,
                    end: segment.endSeconds,
                    text: ""
                )
            }
        default:
            return []
        }
    }
}

/// One spoken turn: a transcript line, or a diarized speaker turn.
package struct StudioAnalyzeSpeechSegment: Identifiable, Equatable {
    package let id: Int
    package let speaker: String?
    package let start: TimeInterval
    package let end: TimeInterval
    package let text: String

    /// "0:04" — the same clock the players use.
    package var startDescription: String {
        StudioTimeFormat.string(start)
    }
}

// MARK: - Where a run writes its result document

/// Where a Studio vision run puts the documents the Analyze canvas reads: a `.json` sidecar
/// beside the annotated output, and a `-masks` directory next to it. Both are the CLI's own
/// `--json-output` and `--mask-output-dir`; Studio only chooses the paths.
package enum StudioVisionResultPaths {
    package static func apply(to draft: inout CommandDraft, wantsMasks: Bool) {
        guard !draft.outputPath.isBlank else { return }
        let stem = URL(fileURLWithPath: draft.outputPath).deletingPathExtension()
        if draft.visionJSONOutputPath.isBlank {
            draft.visionJSONOutputPath = stem.appendingPathExtension("json").path
        }
        if wantsMasks, draft.visionMaskOutputDirectory.isBlank {
            draft.visionMaskOutputDirectory = maskDirectory(forStem: stem)
        }
    }

    /// Moves the sidecars of a replayed run beside its new output. Paths the user chose by hand
    /// (anything that was not derived from the previous output) are left alone.
    package static func rederive(in draft: inout CommandDraft, previousOutputPath: String) {
        guard !previousOutputPath.isBlank else { return }
        let previousStem = URL(fileURLWithPath: previousOutputPath).deletingPathExtension()
        let wantsMasks = !draft.visionMaskOutputDirectory.isBlank
        if draft.visionJSONOutputPath == previousStem.appendingPathExtension("json").path {
            draft.visionJSONOutputPath = ""
        }
        if draft.visionMaskOutputDirectory == maskDirectory(forStem: previousStem) {
            draft.visionMaskOutputDirectory = ""
        }
        apply(to: &draft, wantsMasks: wantsMasks)
    }

    private static func maskDirectory(forStem stem: URL) -> String {
        stem.deletingLastPathComponent()
            .appendingPathComponent("\(stem.lastPathComponent)-masks", isDirectory: true)
            .path
    }
}

// MARK: - Pixels to points

/// Placing pixel-space result geometry on the view that shows the input.
package enum StudioAnalyzeGeometry {
    /// Where a pixel-space rect lands inside a view of `displaySize` that shows the whole image
    /// with no letterboxing — the Analyze canvas sizes its media container to the input's own
    /// aspect ratio, so one uniform scale maps both axes.
    package static func viewRect(for pixelBox: CGRect, imageSize: CGSize, displaySize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scaleX = displaySize.width / imageSize.width
        let scaleY = displaySize.height / imageSize.height
        return CGRect(
            x: pixelBox.minX * scaleX,
            y: pixelBox.minY * scaleY,
            width: pixelBox.width * scaleX,
            height: pixelBox.height * scaleY
        )
    }

    /// The rect an image of `imageSize` occupies when aspect-fitted into `bounds`, for containers
    /// that do letterbox (a video player, a preview whose height is capped).
    package static func fittedRect(imageSize: CGSize, in bounds: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0, bounds.width > 0, bounds.height > 0 else {
            return .zero
        }
        let scale = min(bounds.width / imageSize.width, bounds.height / imageSize.height)
        let fitted = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        return CGRect(
            x: (bounds.width - fitted.width) / 2,
            y: (bounds.height - fitted.height) / 2,
            width: fitted.width,
            height: fitted.height
        )
    }
}

// MARK: - Result documents
/// Where a run's result document lives.
package enum StudioAnalyzeDocumentSource {
    private static let documentExtensions = ["json", "txt", "vtt", "srt", "jsonl"]
    /// The receipt roles that name a structured result, most specific first.
    private static let documentRoles = [StudioArtifactRole.detections, StudioArtifactRole.tracking]

    /// The artifact holding the run's structured result: the `--json-output` sidecar for the
    /// vision tasks, the transcript the speech tasks write. A run whose receipt named its
    /// sidecars says which file that is; extension order is the fallback for the rest.
    package static func url(for item: StudioLibraryItem) -> URL? {
        let artifacts = item.allArtifactURLs
        for role in documentRoles {
            if let match = artifacts.first(where: { item.artifactRole(for: $0) == role }) {
                return match
            }
        }
        for pathExtension in documentExtensions {
            if let match = artifacts.first(where: { $0.pathExtension.lowercased() == pathExtension }) {
                return match
            }
        }
        return nil
    }
}
