import Foundation
import UniformTypeIdentifiers

// The Analyze archetype's declarative surface: which result views an input-first task can switch
// between, and which contextual next steps its result offers. The views in
// `StudioAnalyzeCanvas.swift` render from these declarations, so a task's shape lives in one
// place and is testable without SwiftUI.

/// One of the views the Analyze input strip switches between. The set a task offers depends on
/// what its result actually is, not on the domain it lives in.
package enum StudioAnalyzeResultView: String, CaseIterable, Identifiable, Hashable {
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
    /// What a model understood about a recording: tempo, key, meter, a caption.
    case analysis
    /// Transcribed notes on a piano roll.
    case notes
    /// A tensor's header: dtype, shape, and size, for an `.npy` or safetensors output.
    case tensor
    /// Dataset candidates found under a folder.
    case candidates
    /// A structured report (a run plan's preflight or materialization).
    case report
    /// Text with the spans a model marked in it.
    case spans
    /// The raw result document, monospaced.
    case json

    package var id: String { rawValue }

    /// The segment's label, exactly as the design draws it.
    package var title: String {
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
        case .analysis: return "Analysis"
        case .notes: return "Notes"
        case .tensor: return "Tensor"
        case .candidates: return "Candidates"
        case .report: return "Report"
        case .spans: return "Spans"
        case .json: return "JSON"
        }
    }
}

/// What the Analyze canvas renders on the left: the input the task was pointed at.
package enum StudioAnalyzeInputKind: Equatable {
    case image
    case video
    case audio
    /// A file the canvas can only name (a text corpus, a GeoTIFF the app does not decode).
    case file
    /// A folder the run scans or reads.
    case directory
    /// Typed text: the input column is an editor bound to the command's positional.
    case text
    /// The run takes no input at all (`image validate`); the result column stands alone.
    case none
}

/// What one contextual next step does.
package enum StudioAnalyzeNextActionKind: Equatable {
    /// Continue in a sibling task, carrying this task's input when the target accepts it.
    case openTask(StudioTask)
    /// Write part of the result somewhere the user picks.
    case save(StudioAnalyzeSaveKind)
}

/// Which artifact a "Save…" next step writes.
package enum StudioAnalyzeSaveKind: Equatable {
    /// The result document the run wrote (`--json-output`).
    case json
    /// The text the run produced (a transcript, a caption).
    case text
    /// The primary media output (an annotated image, a tracked clip, enhanced audio).
    case media
}

/// One button in the result panel's action row.
package struct StudioAnalyzeNextAction: Identifiable, Equatable {
    package let title: String
    package let kind: StudioAnalyzeNextActionKind

    package var id: String { title }

    static func open(_ title: String, _ task: StudioTask) -> StudioAnalyzeNextAction {
        StudioAnalyzeNextAction(title: title, kind: .openTask(task))
    }

    package static func save(_ title: String, _ kind: StudioAnalyzeSaveKind) -> StudioAnalyzeNextAction {
        StudioAnalyzeNextAction(title: title, kind: .save(kind))
    }
}

/// The Analyze archetype for one task: an input on the left, one result on the right, and the
/// steps that continue from it. Every input-first task — the ones that take a file, run a single
/// pass over it, and show what the model found — declares one.
package struct StudioAnalyzeArchetype: Equatable {
    /// What one of a task's templates takes and shows when it differs from the task's own
    /// declaration: Image ▸ Datasets scans a folder to discover, reads a plan file to check it,
    /// and takes nothing at all to validate a model.
    package struct Variant: Equatable {
        package let inputKind: StudioAnalyzeInputKind
        package let views: [StudioAnalyzeResultView]

        package init(inputKind: StudioAnalyzeInputKind, views: [StudioAnalyzeResultView]) {
            self.inputKind = inputKind
            self.views = views
        }
    }

    package let task: StudioTask
    package let inputKind: StudioAnalyzeInputKind
    /// The strip's view switch, in order; the first is the default.
    package let views: [StudioAnalyzeResultView]
    package let nextActions: [StudioAnalyzeNextAction]
    /// Per-template exceptions to `inputKind` and `views`, keyed by the variant template.
    package var variants: [CommandTemplateID: Variant] = [:]

    package init(
        task: StudioTask,
        inputKind: StudioAnalyzeInputKind,
        views: [StudioAnalyzeResultView],
        nextActions: [StudioAnalyzeNextAction],
        variants: [CommandTemplateID: Variant] = [:]
    ) {
        self.task = task
        self.inputKind = inputKind
        self.views = views
        self.nextActions = nextActions
        self.variants = variants
    }

    package var defaultView: StudioAnalyzeResultView {
        views.first ?? .json
    }

    /// The input the chosen variant takes: its own declaration, else the task's.
    package func inputKind(for templateID: CommandTemplateID?) -> StudioAnalyzeInputKind {
        templateID.flatMap { variants[$0]?.inputKind } ?? inputKind
    }

    /// The views the chosen variant offers: its own declaration, else the task's.
    package func views(for templateID: CommandTemplateID?) -> [StudioAnalyzeResultView] {
        templateID.flatMap { variants[$0]?.views } ?? views
    }

    /// The tasks this archetype's next steps can hand off to.
    package var siblingTasks: [StudioTask] {
        nextActions.compactMap { action in
            if case .openTask(let task) = action.kind { return task }
            return nil
        }
    }
}

extension StudioTask {
    /// The Analyze archetype this task renders with, or nil for tasks that are not input-first
    /// (Generate, Compose, Chat, and the Project, Session, and Manage tasks).
    package var analyzeArchetype: StudioAnalyzeArchetype? {
        StudioAnalyzeArchetype.archetypes[self]
    }

    /// Whether this task renders the Analyze archetype rather than the generation feed.
    package var isAnalyzeTask: Bool {
        analyzeArchetype != nil
    }
}

extension StudioAnalyzeArchetype {
    // swiftlint:disable:next function_body_length
    package static let archetypes: [StudioTask: StudioAnalyzeArchetype] = {
        var table: [StudioTask: StudioAnalyzeArchetype] = [:]

        func add(
            _ task: StudioTask,
            _ inputKind: StudioAnalyzeInputKind,
            _ views: [StudioAnalyzeResultView],
            _ nextActions: [StudioAnalyzeNextAction],
            variants: [CommandTemplateID: Variant] = [:]
        ) {
            table[task] = StudioAnalyzeArchetype(
                task: task, inputKind: inputKind, views: views, nextActions: nextActions, variants: variants
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
        ], variants: [
            .visionDepthVideo: Variant(inputKind: .video, views: [.video, .json])
        ])
        add(.visionPose, .image, [.points, .json], [
            .open("Detect faces", .visionFaces),
            .save("Save JSON", .json)
        ])
        add(.visionFaces, .image, [.boxes, .points, .json], [
            .open("Pose landmarks", .visionPose),
            .save("Save JSON", .json)
        ], variants: [
            .visionFaceEmbed: Variant(inputKind: .image, views: [.vectors, .json]),
            .visionFaceCompare: Variant(inputKind: .image, views: [.score, .json]),
            .visionFaceBatch: Variant(inputKind: .image, views: [.json])
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

        // Music
        add(.musicAnalyze, .audio, [.analysis, .json], [
            .open("Transcribe notes", .musicTranscribe),
            .open("Separate stems", .musicSeparate),
            .save("Save JSON", .json)
        ])
        add(.musicTranscribe, .audio, [.notes, .json], [
            .open("Analyze", .musicAnalyze),
            .save("Save MIDI", .media)
        ])
        add(.musicSeparate, .audio, [.stems, .json], [
            .open("Analyze", .musicAnalyze),
            .save("Save stems", .media)
        ])

        // Text: Embeddings and Anonymize take typed text, not a file.
        add(.textDecide, .file, [.json], [.save("Save JSON", .json)])
        add(.textEmbeddings, .text, [.vectors, .json], [
            .save("Save JSON", .json)
        ])
        add(.textAnonymize, .text, [.spans, .text], [
            .save("Save text", .text)
        ])

        // Earth: the CLI writes safetensors, so the result is the tensor's header, not a raster.
        for task in [StudioTask.earthFlood, .earthFire, .earthTessera, .earthOlmoEarth] {
            add(task, .file, [.tensor, .json], [.save("Save tensor", .media)])
        }

        // Sound analysis
        add(.soundScore, .audio, [.score], [
            .open("Transcribe", .audioTranscribe),
            .save("Save JSON", .json)
        ])
        add(.soundEncode, .audio, [.tensor, .json], [
            .open("Decode latents", .soundDecode),
            .save("Save latents", .media)
        ])
        add(.soundDecode, .file, [.audio, .json], [
            .open("Score against a prompt", .soundScore),
            .save("Save audio", .media)
        ])

        // Image datasets: three variants behind one task control segment. Training on a found
        // dataset is a Project handoff, which a next step (input-first siblings only) cannot be;
        // the candidates renderer offers it on each row instead.
        add(.imageDatasets, .directory, [.candidates, .json], [
            .save("Save JSON", .json)
        ], variants: [
            .imageRunPlan: Variant(inputKind: .file, views: [.report, .json]),
            .imageValidate: Variant(inputKind: .none, views: [.json])
        ])

        return table
    }()
}

/// The Generate archetype for a task that has no `StudioMode`: prompt first, a feed of results.
/// The attachment slots and chips come from the task's contract (`StudioTaskSchema`); this
/// declares only what the contract cannot say — whether the composer shows a prompt, and what
/// the finished card leads with.
package struct StudioGenerateArchetype: Equatable {
    package let task: StudioTask
    /// The command takes free text (a positional prompt or `--prompt`), so the composer shows the
    /// prompt field.
    package let hasPrompt: Bool
    /// What the finished card's first tile is.
    package let primaryOutput: StudioOutputFileKind

    package static let archetypes: [StudioTask: StudioGenerateArchetype] = [
        .soundFoley: StudioGenerateArchetype(task: .soundFoley, hasPrompt: true, primaryOutput: .audio),
        .soundCondition: StudioGenerateArchetype(task: .soundCondition, hasPrompt: true, primaryOutput: .other),
        .threeDFromImage: StudioGenerateArchetype(task: .threeDFromImage, hasPrompt: false, primaryOutput: .model3D),
    ]
}

extension StudioTask {
    /// The Generate archetype this mode-less task renders with, or nil for every other task
    /// (a mode-backed Generate task declares its shape through the mode).
    package var generateArchetype: StudioGenerateArchetype? {
        StudioGenerateArchetype.archetypes[self]
    }
}

/// Carrying one task's input into the sibling task a next step opens.
///
/// A handoff only carries the file when the target genuinely takes it: "Track in video" from a
/// still image opens Track with the prompt but an empty well, because Track needs a clip. The
/// prompt always carries, so the user never retypes what they were looking for.
package struct StudioAnalyzeHandoff: Equatable {
    package let task: StudioTask
    package let inputPath: String
    package let prompt: String
    /// What the source found, as box prompts on the carried input: "Segment these" from Find
    /// opens Segment with the detected boxes already drawn. Empty unless the input carries, since
    /// the boxes are in that picture's pixels.
    package let regionPrompts: [StudioRegionPrompt]

    package init(task: StudioTask, inputPath: String, prompt: String, regionPrompts: [StudioRegionPrompt] = []) {
        self.task = task
        self.inputPath = inputPath
        self.prompt = prompt
        self.regionPrompts = regionPrompts
    }

    /// Whether `target` accepts `url` as its input, which decides if the well is carried over.
    package static func carriesInput(_ url: URL, to target: StudioTask) -> Bool {
        guard let mode = target.mode else {
            // A task without a composer keeps whatever the shared draft holds, and its own form
            // validates the file; carrying is only refused when the extension is unreadable.
            return UTType(filenameExtension: url.pathExtension) != nil
        }
        return mode.attachmentSlots.contains { $0.accepts(url) }
    }

    /// The handoff a next step produces, or nil when there is nothing to carry.
    ///
    /// - Parameter detections: the source result's boxes in the input's pixels, carried as box
    ///   prompts when the target draws prompts (Segment, Track) and the input itself carries.
    package static func make(
        to target: StudioTask,
        inputPath: String,
        prompt: String,
        detections: [StudioAnalyzeDetection] = []
    ) -> StudioAnalyzeHandoff {
        let trimmed = inputPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let carried = trimmed.isEmpty || carriesInput(URL(fileURLWithPath: trimmed), to: target)
            ? trimmed
            : ""
        let prompts = !carried.isEmpty && target.drawsRegionPrompts
            ? detections.map { StudioRegionPrompt.box($0.box, label: $0.label) }
            : []
        return StudioAnalyzeHandoff(task: target, inputPath: carried, prompt: prompt, regionPrompts: prompts)
    }

    /// Applies the handoff to the draft the target task is about to show.
    package func apply(to draft: inout StudioDraft, source: StudioScopeSource) {
        guard let mode = task.mode else { return }
        draft.prompt = prompt
        guard !inputPath.isEmpty else { return }
        _ = draft.attach(dropped: [URL(fileURLWithPath: inputPath)], for: mode, source: source)
        // Prompts drawn on the target's previous input do not belong on this one.
        if task.drawsRegionPrompts {
            draft.visionRegionPrompts = regionPrompts.isEmpty ? nil : regionPrompts
        }
    }
}

extension StudioTask {
    /// Whether this task's input can carry boxes and points drawn on it (the CLI's `--box` and
    /// `--point`): Segment on a still, Track on a clip's seed frame.
    package var drawsRegionPrompts: Bool {
        self == .visionSegment || self == .visionTrack
    }
}

/// How tall the input may be on the Analyze canvas: whatever the column has above the composer
/// once the input strip and the drawing toolbar have their rows, so a portrait picture fits the
/// visible area instead of running under the composer. Capped so a tall window never gets a
/// picture wider than the column can use, and floored so a short window still shows something
/// to draw on (the canvas scrolls then).
package enum StudioAnalyzeMediaLayout {
    package static let maximumHeight: CGFloat = 520
    package static let minimumHeight: CGFloat = 220

    package static func mediaHeight(availableHeight: CGFloat, chromeHeight: CGFloat) -> CGFloat {
        min(maximumHeight, max(minimumHeight, availableHeight - chromeHeight))
    }
}
