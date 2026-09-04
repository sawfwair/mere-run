import Foundation
import UniformTypeIdentifiers

// The Analyze archetype's declarative surface: which result views an input-first task can switch
// between, and which contextual next steps its result offers. The views in
// `StudioAnalyzeCanvas.swift` render from these declarations, so a task's shape lives in one
// place and is testable without SwiftUI.

/// One of the views the Analyze input strip switches between. The set a task offers depends on
/// what its result actually is, not on the domain it lives in.
enum StudioAnalyzeResultView: String, CaseIterable, Identifiable, Hashable {
    /// Detection rectangles drawn over the input.
    case boxes
    /// Segmentation masks composited over the input.
    case masks
    /// The result overlaid on the video with a scrubber.
    case video
    /// Depth or another single-channel map rendered as an image.
    case depth
    /// Landmarks and keypoints over the input.
    case points
    /// A dense vector field (optical flow, embeddings).
    case vectors
    /// A reconstructed scene or point cloud.
    case scene
    /// A georeferenced raster over the input.
    case map
    /// Spoken text with timestamps.
    case transcript
    /// Segments laid out along the input's duration.
    case timeline
    /// The audio result with the waveform player.
    case audio
    /// Separated stems.
    case stems
    /// Plain text (a caption, OCR, a rewritten document).
    case text
    /// One number and how it was reached.
    case score
    /// The raw result document, monospaced.
    case json

    var id: String { rawValue }

    /// The segment's label, exactly as the design draws it.
    var title: String {
        switch self {
        case .boxes: return "Boxes"
        case .masks: return "Masks"
        case .video: return "Video"
        case .depth: return "Depth"
        case .points: return "Points"
        case .vectors: return "Vectors"
        case .scene: return "Scene"
        case .map: return "Map"
        case .transcript: return "Transcript"
        case .timeline: return "Timeline"
        case .audio: return "Audio"
        case .stems: return "Stems"
        case .text: return "Text"
        case .score: return "Score"
        case .json: return "JSON"
        }
    }
}

/// What the Analyze canvas renders on the left: the input the task was pointed at.
enum StudioAnalyzeInputKind: Equatable {
    case image
    case video
    case audio
    /// A file the canvas can only name (a text corpus, a GeoTIFF the app does not decode).
    case file
}

/// What one contextual next step does.
enum StudioAnalyzeNextActionKind: Equatable {
    /// Continue in a sibling task, carrying this task's input when the target accepts it.
    case openTask(StudioTask)
    /// Write part of the result somewhere the user picks.
    case save(StudioAnalyzeSaveKind)
}

/// Which artifact a "Save…" next step writes.
enum StudioAnalyzeSaveKind: Equatable {
    /// The result document the run wrote (`--json-output`).
    case json
    /// The text the run produced (a transcript, a caption).
    case text
    /// The primary media output (an annotated image, a tracked clip, enhanced audio).
    case media
}

/// One button in the result panel's action row.
struct StudioAnalyzeNextAction: Identifiable, Equatable {
    let title: String
    let kind: StudioAnalyzeNextActionKind

    var id: String { title }

    static func open(_ title: String, _ task: StudioTask) -> StudioAnalyzeNextAction {
        StudioAnalyzeNextAction(title: title, kind: .openTask(task))
    }

    static func save(_ title: String, _ kind: StudioAnalyzeSaveKind) -> StudioAnalyzeNextAction {
        StudioAnalyzeNextAction(title: title, kind: .save(kind))
    }
}

/// The Analyze archetype for one task: an input on the left, one result on the right, and the
/// steps that continue from it. Every input-first task — the ones that take a file, run a single
/// pass over it, and show what the model found — declares one.
struct StudioAnalyzeArchetype: Equatable {
    let task: StudioTask
    let inputKind: StudioAnalyzeInputKind
    /// The strip's view switch, in order; the first is the default.
    let views: [StudioAnalyzeResultView]
    let nextActions: [StudioAnalyzeNextAction]

    var defaultView: StudioAnalyzeResultView {
        views.first ?? .json
    }

    /// The tasks this archetype's next steps can hand off to.
    var siblingTasks: [StudioTask] {
        nextActions.compactMap { action in
            if case .openTask(let task) = action.kind { return task }
            return nil
        }
    }
}

extension StudioTask {
    /// The Analyze archetype this task renders with, or nil for tasks that are not input-first
    /// (Generate, Compose, Chat, and the Project, Session, and Manage tasks).
    var analyzeArchetype: StudioAnalyzeArchetype? {
        StudioAnalyzeArchetype.archetypes[self]
    }

    /// Whether this task renders the Analyze archetype rather than the generation feed.
    var isAnalyzeTask: Bool {
        analyzeArchetype != nil
    }
}

extension StudioAnalyzeArchetype {
    // swiftlint:disable:next function_body_length
    static let archetypes: [StudioTask: StudioAnalyzeArchetype] = {
        var table: [StudioTask: StudioAnalyzeArchetype] = [:]

        func add(
            _ task: StudioTask,
            _ inputKind: StudioAnalyzeInputKind,
            _ views: [StudioAnalyzeResultView],
            _ nextActions: [StudioAnalyzeNextAction]
        ) {
            table[task] = StudioAnalyzeArchetype(
                task: task, inputKind: inputKind, views: views, nextActions: nextActions
            )
        }

        // Vision
        add(.visionRead, .image, [.text], [
            .open("Find objects", .visionFind),
            .save("Save text", .text)
        ])
        add(.visionFind, .image, [.boxes, .masks, .json], [
            .open("Segment these", .visionSegment),
            .open("Track in video", .visionTrack),
            .save("Save JSON", .json)
        ])
        add(.visionSegment, .image, [.boxes, .masks, .json], [
            .open("Track in video", .visionTrack),
            .open("Read this image", .visionRead),
            .save("Save JSON", .json)
        ])
        add(.visionTrack, .video, [.video, .json], [
            .open("Segment a frame", .visionSegment),
            .save("Save JSON", .json)
        ])
        add(.visionDepth, .image, [.depth, .json], [
            .open("Find objects", .visionFind),
            .save("Save JSON", .json)
        ])
        add(.visionPose, .image, [.points, .json], [
            .open("Detect faces", .visionFaces),
            .save("Save JSON", .json)
        ])
        add(.visionFaces, .image, [.boxes, .points, .json], [
            .open("Pose landmarks", .visionPose),
            .save("Save JSON", .json)
        ])
        add(.visionFlow, .image, [.vectors, .json], [
            .save("Save flow", .media)
        ])
        add(.visionGeometry, .image, [.scene, .json], [
            .save("Save scene", .media)
        ])
        add(.visionLive, .video, [.video, .json], [
            .open("Track a clip", .visionTrack),
            .save("Save JSON", .json)
        ])

        // Audio
        add(.audioTranscribe, .audio, [.transcript, .timeline, .json], [
            .open("Who spoke", .audioWhoSpoke),
            .open("Enhance audio", .audioEnhance),
            .save("Save transcript", .text)
        ])
        add(.audioWhoSpoke, .audio, [.transcript, .timeline, .json], [
            .open("Transcribe", .audioTranscribe),
            .save("Save JSON", .json)
        ])
        add(.audioEnhance, .audio, [.audio, .json], [
            .open("Transcribe", .audioTranscribe),
            .open("Separate stems", .audioSeparate),
            .save("Save audio", .media)
        ])
        add(.audioSeparate, .audio, [.stems, .json], [
            .open("Transcribe", .audioTranscribe),
            .save("Save stems", .media)
        ])

        // Text
        add(.textEmbeddings, .file, [.vectors, .json], [
            .save("Save JSON", .json)
        ])
        add(.textAnonymize, .file, [.text], [
            .save("Save text", .text)
        ])

        // Earth
        for task in [StudioTask.earthFlood, .earthFire, .earthTessera, .earthOlmoEarth] {
            add(task, .file, [.map, .json], [.save("Save JSON", .json)])
        }

        // Sound analysis
        add(.soundScore, .audio, [.score], [
            .open("Transcribe", .audioTranscribe),
            .save("Save JSON", .json)
        ])
        add(.soundCondition, .audio, [.audio, .json], [
            .save("Save audio", .media)
        ])

        return table
    }()
}

/// Carrying one task's input into the sibling task a next step opens.
///
/// A handoff only carries the file when the target genuinely takes it: "Track in video" from a
/// still image opens Track with the prompt but an empty well, because Track needs a clip. The
/// prompt always carries, so the user never retypes what they were looking for.
struct StudioAnalyzeHandoff: Equatable {
    let task: StudioTask
    let inputPath: String
    let prompt: String

    /// Whether `target` accepts `url` as its input, which decides if the well is carried over.
    static func carriesInput(_ url: URL, to target: StudioTask) -> Bool {
        guard let mode = target.mode else {
            // A task without a composer keeps whatever the shared draft holds, and its own form
            // validates the file; carrying is only refused when the extension is unreadable.
            return UTType(filenameExtension: url.pathExtension) != nil
        }
        return mode.attachmentSlots.contains { $0.accepts(url) }
    }

    /// The handoff a next step produces, or nil when there is nothing to carry.
    static func make(to target: StudioTask, inputPath: String, prompt: String) -> StudioAnalyzeHandoff {
        let trimmed = inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let carried = trimmed.isEmpty || carriesInput(URL(fileURLWithPath: trimmed), to: target)
            ? trimmed
            : ""
        return StudioAnalyzeHandoff(task: target, inputPath: carried, prompt: prompt)
    }

    /// Applies the handoff to the draft the target task is about to show.
    func apply(to draft: inout StudioDraft) {
        guard let mode = task.mode else { return }
        draft.prompt = prompt
        guard !inputPath.isEmpty else { return }
        _ = draft.attach(dropped: [URL(fileURLWithPath: inputPath)], for: mode)
    }
}
