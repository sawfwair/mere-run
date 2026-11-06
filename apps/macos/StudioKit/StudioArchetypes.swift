import Foundation
import UniformTypeIdentifiers

// Every task declares the surface archetype that shells it. The archetype decides the chrome
// (Library column, inspector, composer) and which canvas the shared task workspace renders; the
// per-task tables in `StudioAnalyzeSchema.swift` and `StudioTaskSchema.swift` fill in the rest.
// Generate and Analyze tasks use a shared workspace; the other archetypes have purpose-built
// surfaces. The contract-backed tasks keep one draft across the workspace, Command view, and Run.

/// The six shells of the v2 design (`docs/macos-studio-v2.md` §6).
package enum StudioSurfaceArchetype: String, CaseIterable, Hashable {
    /// Prompt first, a feed of results, the composer pinned at the bottom.
    case generate
    /// A thread list and turns with a per-turn model.
    case converse
    /// Input first, prompt optional, a result renderer specific to the output.
    case analyze
    /// A long-lived process with transport controls and live state.
    case session
    /// Multi-stage with persisted state, dashboards, and compare.
    case project
    /// A list and a detail with actions and confirmations.
    case manage
}

extension StudioTask {
    /// The shell this task renders in. Exhaustive so a new task must say what it is.
    package var archetype: StudioSurfaceArchetype {
        switch self {
        case .imageGenerate, .videoGenerate, .videoEdit, .videoCosmos, .musicCompose, .soundGenerate,
             .voiceSpeak, .voiceEdit, .soundFoley, .threeDFromImage:
            return .generate
        case .chatChat, .chatCode:
            return .converse
        case .visionRead, .visionFind, .visionSegment, .visionTrack, .visionDocument, .visionDepth, .visionPose,
             .visionFaces, .visionFlow, .visionGeometry, .audioTranscribe, .audioWhoSpoke, .audioEnhance,
             .musicAnalyze, .musicTranscribe, .musicSeparate, .soundScore, .soundLatents,
             .textEmbeddings, .textAnonymize, .textDecide, .textClassify, .textExtract,
             .earthFlood, .earthFire, .earthTessera,
             .earthOlmoEarth, .imageDatasets:
            return .analyze
        case .musicRealtime, .audioLive, .visionLive, .serverServing, .serverMusic, .serverVision, .serverWorld:
            return .session
        case .imageTrain, .chatTrain, .musicTrain, .videoSubjects:
            return .project
        case .voiceVoices, .modelsInstalled, .modelsLocations, .modelsHealth, .modelsBenchmarks,
             .modelsAdapters, .runsRuns, .pluginsCatalog:
            return .manage
        }
    }

    /// Tasks with a contract-backed `StudioTaskDraft`. Generate and Analyze tasks use the shared
    /// workspace; Live, Voices, and Training use their own Session, Manage, and Project surfaces
    /// over the same draft.
    package static let taskDraftTasks: Set<StudioTask> = [
        .videoEdit, .videoCosmos, .voiceEdit,
        .soundFoley, .soundScore, .soundLatents,
        .musicAnalyze, .musicTranscribe,
        .visionDocument, .visionDepth, .visionPose, .visionFaces, .visionFlow, .visionGeometry, .visionLive,
        .audioWhoSpoke, .audioEnhance, .musicSeparate, .audioLive, .voiceVoices,
        .textEmbeddings, .textAnonymize, .imageDatasets,
        .threeDFromImage,
        .earthFlood, .earthFire, .earthTessera, .earthOlmoEarth,
        .imageTrain, .chatTrain, .musicTrain,
    ]

    /// Whether this task's draft is a `StudioTaskDraft` (a contract form for its template)
    /// rather than the prompt tasks' `StudioDraft` or a Session, Project, or Manage page's state.
    package var usesTaskDraft: Bool {
        Self.taskDraftTasks.contains(self)
    }

    /// Generate, Converse, and Analyze show the Library column and inspector. Other archetypes
    /// fill the detail area with their own surface.
    package var showsPromptChrome: Bool {
        (mode != nil || usesTaskDraft) && [.generate, .converse, .analyze].contains(archetype)
    }

    /// The templates this task can run, in picker order. One for most tasks; Vision ▸ Faces or
    /// 3D ▸ From image offer several as a variant chip. Templates that list or delete records
    /// (`speech profile list`) or open a viewer (`image visualize-run`) are not runs, so they are
    /// left to the Manage and Project pages that own them.
    package var variantTemplates: [CommandTemplate] {
        let templates = commandTemplates.filter { !Self.nonVariantTemplates.contains($0.id) }
        guard let primary = primaryTemplateID, let index = templates.firstIndex(where: { $0.id == primary }) else {
            return templates
        }
        var ordered = templates
        ordered.insert(ordered.remove(at: index), at: 0)
        return ordered
    }

    /// The template a fresh draft starts on when the task has several: the one the task's page
    /// opened on (Detect for Faces, TRELLIS.2 for 3D, Discover for Datasets). Catalog order
    /// decides the rest of the picker.
    package var primaryTemplateID: CommandTemplateID? {
        switch self {
        case .visionFaces: return .visionFaceDetect
        case .visionDepth: return .visionDepth
        case .visionGeometry: return .visionGeometry
        case .audioEnhance: return .audioEnhance
        case .videoEdit: return .videoRetake
        case .soundLatents: return .sfxAEEncode
        case .threeDFromImage: return .imageReconstruct3DTrellis2
        case .imageDatasets: return .imageDatasetDiscover
        case .voiceVoices: return .speechProfileCreate
        case .audioLive: return .speechListen
        default: return nil
        }
    }

    private static let nonVariantTemplates: Set<CommandTemplateID> = [
        .speechProfileList, .speechProfileDelete, .imageVisualizeRun,
    ]
}

/// The words and glyph a task's surface shows before it has anything to show: the empty
/// canvas's title and guidance, the composer's placeholder, and the example prompts. Mode-backed
/// tasks read theirs from the mode; the rest are authored here in the same voice.
package struct StudioTaskPresentation: Equatable {
    package let title: String
    package let systemImage: String
    package let emptyTitle: String
    package let emptyMessage: String
    package let promptPlaceholder: String
    package let examplePrompts: [String]
    /// The empty state offers an attach button before anything else can happen.
    package let requiresAttachment: Bool
    /// That button's title ("Choose audio…").
    package let attachLabel: String

    package init(
        title: String,
        systemImage: String,
        emptyTitle: String,
        emptyMessage: String,
        promptPlaceholder: String = "",
        examplePrompts: [String] = [],
        requiresAttachment: Bool = false,
        attachLabel: String = "Choose file…"
    ) {
        self.title = title
        self.systemImage = systemImage
        self.emptyTitle = emptyTitle
        self.emptyMessage = emptyMessage
        self.promptPlaceholder = promptPlaceholder
        self.examplePrompts = examplePrompts
        self.requiresAttachment = requiresAttachment
        self.attachLabel = attachLabel
    }

    /// A prompt mode's presentation for the wells the model its draft runs takes
    /// (`StudioMode.emptyMessage(slots:)`).
    package init(mode: StudioMode, slots: [StudioAttachmentSlot]) {
        let base = StudioTaskPresentation(mode: mode)
        self.init(
            title: base.title, systemImage: base.systemImage, emptyTitle: base.emptyTitle,
            emptyMessage: mode.emptyMessage(slots: slots), promptPlaceholder: base.promptPlaceholder,
            examplePrompts: base.examplePrompts, requiresAttachment: base.requiresAttachment, attachLabel: base.attachLabel
        )
    }

    package init(mode: StudioMode) {
        let attachLabel: String
        switch mode {
        case .listen: attachLabel = "Choose audio…"
        case .track: attachLabel = "Choose video…"
        default: attachLabel = "Choose image…"
        }
        self.init(
            title: mode.title,
            systemImage: mode.systemImage,
            emptyTitle: mode.emptyTitle,
            emptyMessage: mode.emptyMessage,
            promptPlaceholder: mode.promptPlaceholder,
            examplePrompts: mode.examplePrompts,
            requiresAttachment: mode.requiresAttachment,
            attachLabel: attachLabel
        )
    }

    /// The same presentation with the attach button named after what a task's first slot takes.
    package func attaching(_ slot: StudioAttachmentSlot?) -> StudioTaskPresentation {
        guard let slot else { return self }
        let noun: String
        if slot.acceptedTypes.contains(.folder) {
            noun = "folder"
        } else if slot.acceptedTypes.contains(.image) {
            noun = "image"
        } else if slot.acceptedTypes.contains(.audio) {
            noun = "audio"
        } else if slot.acceptedTypes.contains(where: { $0 == .movie || $0 == .video || $0 == .audiovisualContent }) {
            noun = "video"
        } else {
            noun = "file"
        }
        return StudioTaskPresentation(
            title: title, systemImage: systemImage, emptyTitle: emptyTitle, emptyMessage: emptyMessage,
            promptPlaceholder: promptPlaceholder, examplePrompts: examplePrompts,
            requiresAttachment: requiresAttachment, attachLabel: "Choose \(noun)…"
        )
    }
}

extension StudioTask {
    // swiftlint:disable:next function_body_length
    package var presentation: StudioTaskPresentation {
        if let mode { return StudioTaskPresentation(mode: mode) }
        switch self {
        case .imageDatasets:
            return StudioTaskPresentation(
                title: title, systemImage: "folder.badge.questionmark",
                emptyTitle: "Find training data in a folder",
                emptyMessage: "Point at a folder of images and captions to discover datasets, check a run plan, or validate an image model.",
                requiresAttachment: true
            )
        case .imageTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Train an image adapter",
                emptyMessage: "Choose a dataset and an output folder, then start training. The dashboard follows every step."
            )
        case .videoEdit:
            return StudioTaskPresentation(
                title: title, systemImage: "timeline.selection",
                emptyTitle: "Rework part of a clip",
                emptyMessage: "Attach a clip and describe the change. Retake regenerates a timed region. Dub-It carries a performance onto a new scene.",
                promptPlaceholder: "Describe the replacement",
                examplePrompts: ["The speaker turns toward the window and smiles", "Rain starts falling halfway through"],
                requiresAttachment: true,
                attachLabel: "Choose video…"
            )
        case .videoCosmos:
            return StudioTaskPresentation(
                title: title, systemImage: "sparkles.tv",
                emptyTitle: "Generate with Cosmos3",
                emptyMessage: "Describe a scene to make an image or a clip, or attach an image or video to continue from it. Choose the mode in the inspector.",
                promptPlaceholder: "Describe the scene",
                examplePrompts: ["A rover crossing a windswept alien plain", "A robot arm stacking blocks on a workbench"]
            )
        case .videoSubjects:
            return StudioTaskPresentation(
                title: title, systemImage: "person.and.background.dotted",
                emptyTitle: "Animate a subject",
                emptyMessage: "Choose a subject image and a driving clip, prepare masks, then animate."
            )
        case .musicRealtime:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.path",
                emptyTitle: "Play music live",
                emptyMessage: "Start a session and steer it with prompts while it plays."
            )
        case .musicAnalyze:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.and.magnifyingglass",
                emptyTitle: "Analyze a song",
                emptyMessage: "Attach a track to read its tempo, key, meter, and a caption of what it sounds like.",
                requiresAttachment: true
            )
        case .musicTranscribe:
            return StudioTaskPresentation(
                title: title, systemImage: "pianokeys",
                emptyTitle: "Transcribe a recording to MIDI",
                emptyMessage: "Attach a recording to transcribe it to MIDI, with the instruments it hears.",
                requiresAttachment: true
            )
        case .musicSeparate:
            return StudioTaskPresentation(
                title: title, systemImage: "square.stack.3d.down.forward",
                emptyTitle: "Separate a mix into stems",
                emptyMessage: "Attach a track to separate it into vocals, drums, bass, and the rest.",
                requiresAttachment: true
            )
        case .musicTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Train a music adapter",
                emptyMessage: "Assemble a manifest of clips and an output folder, then start training."
            )
        case .soundFoley:
            return StudioTaskPresentation(
                title: title, systemImage: "film.stack",
                emptyTitle: "Add sound to a video",
                emptyMessage: "Attach a clip and describe the sound. The effect follows the timing of what happens on screen.",
                promptPlaceholder: "Describe the sound for the clip",
                examplePrompts: ["Footsteps on wet gravel", "A door slamming in a hallway", "Rain on a tent"],
                requiresAttachment: true
            )
        case .soundLatents:
            return StudioTaskPresentation(
                title: title, systemImage: "arrow.down.right.and.arrow.up.left",
                emptyTitle: "Work with sound latents",
                emptyMessage: "Encode audio to latents, decode latents to a WAV, or save a text-conditioning tensor that later runs can reuse. Choose which in the variant menu.",
                requiresAttachment: true
            )
        case .soundScore:
            return StudioTaskPresentation(
                title: title, systemImage: "gauge.with.needle",
                emptyTitle: "Score a sound against a description",
                emptyMessage: "Attach audio and describe the sound you expect. CLAP scores how well they match.",
                promptPlaceholder: "What should the sound be?",
                examplePrompts: ["a dog barking", "rain on a tin roof"],
                requiresAttachment: true
            )
        case .voiceEdit:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.badge.plus",
                emptyTitle: "Edit speech with an instruction",
                emptyMessage: "Describe the speech you want. Attach a reference recording to keep its voice.",
                promptPlaceholder: "Describe the speech and how it sounds",
                examplePrompts: ["Say \"Welcome back\" warmly, a little slower", "Read the line as a whisper"],
                attachLabel: "Choose audio…"
            )
        case .voiceVoices:
            return StudioTaskPresentation(
                title: title, systemImage: "person.wave.2",
                emptyTitle: "Save the voices you use",
                emptyMessage: "Save a reference recording as a named voice, then choose it in Speak."
            )
        case .threeDFromImage:
            return StudioTaskPresentation(
                title: title, systemImage: "cube.transparent",
                emptyTitle: "Turn an image into 3D",
                emptyMessage: "Attach an image, or ordered views, to reconstruct a mesh or generate a Gaussian splat.",
                requiresAttachment: true
            )
        case .chatTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Train a text adapter",
                emptyMessage: "Choose a JSONL dataset and an output folder, then start training."
            )
        case .visionDocument:
            return StudioTaskPresentation(
                title: title, systemImage: "doc.text.viewfinder",
                emptyTitle: "Read the text in a document",
                emptyMessage: "Attach one or more page images in order. Unlimited-OCR-2 reads them as one document. LightOnOCR, GLM-OCR, and Infinity-Parser2 read each page.",
                requiresAttachment: true
            )
        case .visionDepth:
            return StudioTaskPresentation(
                title: title, systemImage: "square.3.layers.3d.down.right",
                emptyTitle: "Estimate depth",
                emptyMessage: "Attach an image or a clip to estimate depth for every pixel.",
                requiresAttachment: true
            )
        case .visionPose:
            return StudioTaskPresentation(
                title: title, systemImage: "figure.stand",
                emptyTitle: "Mark body, hand, and face landmarks",
                emptyMessage: "Attach an image to mark body, hand, and face landmarks.",
                requiresAttachment: true
            )
        case .visionFaces:
            return StudioTaskPresentation(
                title: title, systemImage: "face.dashed",
                emptyTitle: "Detect and compare faces",
                emptyMessage: "Attach an image to detect faces, embed one, or compare two.",
                requiresAttachment: true
            )
        case .visionFlow:
            return StudioTaskPresentation(
                title: title, systemImage: "arrow.triangle.2.circlepath",
                emptyTitle: "Measure the motion between frames",
                emptyMessage: "Attach two equal-size images to estimate dense optical flow from one to the other.",
                requiresAttachment: true
            )
        case .visionGeometry:
            return StudioTaskPresentation(
                title: title, systemImage: "view.3d",
                emptyTitle: "Recover scene geometry",
                emptyMessage: "Attach an image, or several views, to estimate metric depth, normals, and a point cloud.",
                requiresAttachment: true
            )
        case .visionLive:
            return StudioTaskPresentation(
                title: title, systemImage: "video.badge.waveform",
                emptyTitle: "Track objects live from a camera",
                emptyMessage: "Name what to follow, then start. When you stop, the annotated clip appears in the Library.",
                promptPlaceholder: "What should mere.run track?",
                examplePrompts: ["the person", "the red mug"]
            )
        case .audioWhoSpoke:
            return StudioTaskPresentation(
                title: title, systemImage: "person.2.wave.2",
                emptyTitle: "Tell speakers apart",
                emptyMessage: "Attach a recording to find who spoke when.",
                requiresAttachment: true
            )
        case .audioEnhance:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.badge.magnifyingglass",
                emptyTitle: "Extend a recording's bandwidth",
                emptyMessage: "Attach narrow-band audio to extend it to 48 kHz.",
                requiresAttachment: true
            )
        case .audioLive:
            return StudioTaskPresentation(
                title: title, systemImage: "mic.badge.plus",
                emptyTitle: "Transcribe as you speak",
                emptyMessage: "Choose an input and start. Words appear as they're heard."
            )
        case .textEmbeddings:
            return StudioTaskPresentation(
                title: title, systemImage: "point.3.connected.trianglepath.dotted",
                emptyTitle: "Compare texts by meaning",
                emptyMessage: "Type one text per line to embed them and compare how close they are.",
                promptPlaceholder: "One text per line"
            )
        case .textAnonymize:
            return StudioTaskPresentation(
                title: title, systemImage: "eye.slash",
                emptyTitle: "Remove personal information",
                emptyMessage: "Paste text to find and replace personal information.",
                promptPlaceholder: "Paste the text to protect"
            )
        case .textDecide:
            return StudioTaskPresentation(
                title: title, systemImage: "checklist",
                emptyTitle: "Decide with evidence",
                emptyMessage: "Frame the question, add the options and facts, and ask for a ranked answer."
            )
        case .textClassify:
            return StudioTaskPresentation(
                title: title, systemImage: "tag",
                emptyTitle: "Classify text",
                emptyMessage: "Add text and label tasks to classify with GLiNER2.5 Decide."
            )
        case .textExtract:
            return StudioTaskPresentation(
                title: title, systemImage: "text.viewfinder",
                emptyTitle: "Extract information from text",
                emptyMessage: "Add text and an extraction schema for GLiNER2.5 Decide."
            )
        case .earthFlood:
            return StudioTaskPresentation(
                title: title, systemImage: "water.waves",
                emptyTitle: "Map flooded ground",
                emptyMessage: "Attach a normalized S2L2A, S1RTC, and DEM tile batch to segment flooded ground.",
                requiresAttachment: true
            )
        case .earthFire:
            return StudioTaskPresentation(
                title: title, systemImage: "flame",
                emptyTitle: "Map burned ground",
                emptyMessage: "Attach a normalized S2L2A, S1RTC, and DEM tile batch to segment burned ground.",
                requiresAttachment: true
            )
        case .earthTessera:
            return StudioTaskPresentation(
                title: title, systemImage: "square.stack.3d.down.right",
                emptyTitle: "Embed a year of observations",
                emptyMessage: "Attach raw Sentinel-1/2 observations with their day-of-year tensors.",
                requiresAttachment: true
            )
        case .earthOlmoEarth:
            return StudioTaskPresentation(
                title: title, systemImage: "globe.europe.africa",
                emptyTitle: "Embed multisensor observations",
                emptyMessage: "Attach a tile bundle with a TIMESTAMPS tensor to encode it.",
                requiresAttachment: true
            )
        case .modelsInstalled, .modelsLocations, .modelsHealth, .modelsBenchmarks, .modelsAdapters:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Manage the models on this Mac",
                emptyMessage: "Download, locate, check, and benchmark the models Studio runs."
            )
        case .serverServing, .serverMusic, .serverVision, .serverWorld:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Serve a model locally",
                emptyMessage: "Start the server and point your tools at it."
            )
        case .runsRuns:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Inspect every run",
                emptyMessage: "Inspect local and Relay runs and fetch what they produced."
            )
        case .pluginsCatalog:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Extend mere.run",
                emptyMessage: "Install verified companion tools and run them here."
            )
        case .imageGenerate, .videoGenerate, .musicCompose, .soundGenerate, .voiceSpeak, .chatChat, .chatCode,
             .visionRead, .visionFind, .visionSegment, .visionTrack, .audioTranscribe:
            // Mode-backed; handled above. Listed so a task that loses its mode must say what it shows.
            return StudioTaskPresentation(title: title, systemImage: domain.systemImage, emptyTitle: title, emptyMessage: "")
        }
    }
}
