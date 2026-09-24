import Foundation
import UniformTypeIdentifiers

// Every task declares the surface archetype that shells it. The archetype decides the chrome
// (Library column, inspector, composer) and which canvas the shared task workspace renders; the
// per-task tables in `StudioAnalyzeSchema.swift` and `StudioTaskSchema.swift` fill in the rest.
// Until a task's page PR lands it keeps its bespoke page (`usesLegacyPage`), so a task can move
// onto the shared workspace by flipping one line here.

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
        case .imageGenerate, .videoGenerate, .musicCompose, .soundGenerate, .voiceSpeak,
             .soundFoley, .soundCondition, .threeDFromImage:
            return .generate
        case .chatChat, .chatCode:
            return .converse
        case .visionRead, .visionFind, .visionSegment, .visionTrack, .visionDepth, .visionPose, .visionFaces,
             .visionFlow, .visionGeometry, .audioTranscribe, .audioWhoSpoke, .audioEnhance, .audioSeparate,
             .musicAnalyze, .musicTranscribe, .musicSeparate, .soundScore, .soundEncode, .soundDecode,
             .textEmbeddings, .textAnonymize, .textDecide, .earthFlood, .earthFire, .earthTessera,
             .earthOlmoEarth, .imageDatasets:
            return .analyze
        case .musicRealtime, .audioLive, .visionLive, .serverServing, .serverMusic, .serverVision:
            return .session
        case .imageTrain, .chatTrain, .musicTrain, .videoSubjects:
            return .project
        case .voiceVoices, .modelsInstalled, .modelsLocations, .modelsHealth, .modelsBenchmarks,
             .modelsAdapters, .runsRuns, .pluginsCatalog:
            return .manage
        }
    }

    /// The mode-less tasks that have moved off their bespoke pages onto a `StudioTaskDraft`. A
    /// page PR adds its task here and deletes its page: an Analyze or Generate task then renders
    /// the shared task workspace (`StudioTaskWorkspace`), and a Session or Manage task its own
    /// page over the same draft, so the Command view, Library restoration, and Stop read one
    /// value for all of them.
    package static let migratedTasks: Set<StudioTask> = [
        .soundFoley, .soundCondition, .soundEncode, .soundDecode, .soundScore,
        .musicAnalyze, .musicTranscribe,
        .visionDepth, .visionPose, .visionFaces, .visionFlow, .visionGeometry, .visionLive,
        .audioWhoSpoke, .audioEnhance, .audioSeparate, .musicSeparate, .audioLive, .voiceVoices,
    ]

    /// Temporary gate: true while this task still renders its bespoke page. A mode-backed task
    /// never does — its prompt workspace is the shared surface already.
    package var usesLegacyPage: Bool {
        mode == nil && !Self.migratedTasks.contains(self)
    }

    /// Whether this task's draft is a `StudioTaskDraft` (a contract form for its template)
    /// rather than the prompt tasks' `StudioDraft` or a page's own state.
    package var usesTaskDraft: Bool {
        mode == nil && !usesLegacyPage
    }

    /// Generate, Converse, and Analyze show the Library column and the inspector. A task that
    /// still renders its legacy page takes the full width whatever its archetype says, so the
    /// shell is unchanged until the page moves.
    package var showsPromptChrome: Bool {
        guard !usesLegacyPage else { return false }
        return [.generate, .converse, .analyze].contains(archetype)
    }

    /// The templates this task can run, in picker order. One for most tasks; Vision ▸ Faces or
    /// 3D ▸ From image offer several as a variant chip. Templates that list or delete records
    /// (`speech profile list`) or open a viewer (`image visualize-run`) are not runs, so they are
    /// left to the Manage and Project pages that own them; `audio edit` is instruction-first
    /// speech generation with an optional reference, not a restoration of the attached audio, so
    /// Audio ▸ Enhance does not offer it (the Command Console still does).
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
        case .threeDFromImage: return .imageReconstruct3DTrellis2
        case .imageDatasets: return .imageDatasetDiscover
        case .voiceVoices: return .speechProfileCreate
        case .audioLive: return .speechListen
        default: return nil
        }
    }

    private static let nonVariantTemplates: Set<CommandTemplateID> = [
        .speechProfileList, .speechProfileDelete, .imageVisualizeRun, .audioEdit,
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
                emptyTitle: "Find what's worth training on.",
                emptyMessage: "Point at a folder of images and captions to discover datasets, check a run plan, or validate an image model.",
                requiresAttachment: true
            )
        case .imageTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Teach the image model a look.",
                emptyMessage: "Choose a dataset and an output folder, then start training. The dashboard follows every step."
            )
        case .videoSubjects:
            return StudioTaskPresentation(
                title: title, systemImage: "person.and.background.dotted",
                emptyTitle: "Animate a subject.",
                emptyMessage: "Pick a subject image and a driving clip, prepare masks, then animate."
            )
        case .musicRealtime:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.path",
                emptyTitle: "Play music live.",
                emptyMessage: "Start a session and steer it with prompts while it plays."
            )
        case .musicAnalyze:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.and.magnifyingglass",
                emptyTitle: "Hear what's in the song.",
                emptyMessage: "Attach a track to read its tempo, key, meter, and a caption of what it sounds like.",
                requiresAttachment: true
            )
        case .musicTranscribe:
            return StudioTaskPresentation(
                title: title, systemImage: "pianokeys",
                emptyTitle: "Turn audio into notes.",
                emptyMessage: "Attach a recording to transcribe it to MIDI, with the instruments it hears.",
                requiresAttachment: true
            )
        case .musicSeparate, .audioSeparate:
            return StudioTaskPresentation(
                title: title, systemImage: "square.stack.3d.down.forward",
                emptyTitle: "Pull the mix apart.",
                emptyMessage: "Attach a track to separate it into vocals, drums, bass, and the rest.",
                requiresAttachment: true
            )
        case .musicTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Teach the music model a style.",
                emptyMessage: "Assemble a manifest of clips and an output folder, then start training."
            )
        case .soundFoley:
            return StudioTaskPresentation(
                title: title, systemImage: "film.stack",
                emptyTitle: "Give the picture a sound.",
                emptyMessage: "Attach a clip and describe the sound; the effect lands in sync with what happens on screen.",
                promptPlaceholder: "Describe the sound the clip should make...",
                examplePrompts: ["Footsteps on wet gravel", "A door slamming in a hallway", "Rain on a tent"],
                requiresAttachment: true
            )
        case .soundCondition:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.badge.plus",
                emptyTitle: "Encode a sound description.",
                emptyMessage: "Describe a sound and save the conditioning tensor a later run can reuse.",
                promptPlaceholder: "Describe the sound...",
                examplePrompts: ["Heavy wooden door creaking open", "Distant thunder rolling over hills"]
            )
        case .soundEncode:
            return StudioTaskPresentation(
                title: title, systemImage: "arrow.down.right.and.arrow.up.left",
                emptyTitle: "Compress a sound to latents.",
                emptyMessage: "Attach audio to encode it with the sound autoencoder.",
                requiresAttachment: true
            )
        case .soundDecode:
            return StudioTaskPresentation(
                title: title, systemImage: "arrow.up.left.and.arrow.down.right",
                emptyTitle: "Bring latents back to sound.",
                emptyMessage: "Attach a latents file to decode it to a WAV.",
                requiresAttachment: true
            )
        case .soundScore:
            return StudioTaskPresentation(
                title: title, systemImage: "gauge.with.needle",
                emptyTitle: "Score a sound against words.",
                emptyMessage: "Attach audio and describe what it should be; CLAP says how well they match.",
                promptPlaceholder: "What should the sound be?",
                examplePrompts: ["a dog barking", "rain on a tin roof"],
                requiresAttachment: true
            )
        case .voiceVoices:
            return StudioTaskPresentation(
                title: title, systemImage: "person.wave.2",
                emptyTitle: "Keep the voices you use.",
                emptyMessage: "Save a reference recording as a named voice and pick it from Speak."
            )
        case .threeDFromImage:
            return StudioTaskPresentation(
                title: title, systemImage: "cube.transparent",
                emptyTitle: "Lift a picture into 3D.",
                emptyMessage: "Attach a single image (or ordered views) and reconstruct a mesh.",
                requiresAttachment: true
            )
        case .chatTrain:
            return StudioTaskPresentation(
                title: title, systemImage: "slider.horizontal.3",
                emptyTitle: "Teach the text model a voice.",
                emptyMessage: "Choose a JSONL dataset and an output folder, then start training."
            )
        case .visionDepth:
            return StudioTaskPresentation(
                title: title, systemImage: "square.3.layers.3d.down.right",
                emptyTitle: "See how far things are.",
                emptyMessage: "Attach an image or a clip to estimate depth for every pixel.",
                requiresAttachment: true
            )
        case .visionPose:
            return StudioTaskPresentation(
                title: title, systemImage: "figure.stand",
                emptyTitle: "Find the body in the frame.",
                emptyMessage: "Attach an image to mark body, hand, and face landmarks.",
                requiresAttachment: true
            )
        case .visionFaces:
            return StudioTaskPresentation(
                title: title, systemImage: "face.dashed",
                emptyTitle: "Find every face.",
                emptyMessage: "Attach an image to detect faces, embed one, or compare two.",
                requiresAttachment: true
            )
        case .visionFlow:
            return StudioTaskPresentation(
                title: title, systemImage: "arrow.triangle.2.circlepath",
                emptyTitle: "Measure the motion between frames.",
                emptyMessage: "Attach two equal-size images to estimate dense optical flow from one to the other.",
                requiresAttachment: true
            )
        case .visionGeometry:
            return StudioTaskPresentation(
                title: title, systemImage: "view.3d",
                emptyTitle: "Recover the scene.",
                emptyMessage: "Attach an image, or several views, to estimate metric depth, normals, and a point cloud.",
                requiresAttachment: true
            )
        case .visionLive:
            return StudioTaskPresentation(
                title: title, systemImage: "video.badge.waveform",
                emptyTitle: "Track through the camera.",
                emptyMessage: "Name what to follow and start; the annotated clip lands in the Library when you stop.",
                promptPlaceholder: "What should mere.run track?",
                examplePrompts: ["the person", "the red mug"]
            )
        case .audioWhoSpoke:
            return StudioTaskPresentation(
                title: title, systemImage: "person.2.wave.2",
                emptyTitle: "Tell the voices apart.",
                emptyMessage: "Attach a recording to find who spoke when.",
                requiresAttachment: true
            )
        case .audioEnhance:
            return StudioTaskPresentation(
                title: title, systemImage: "waveform.badge.magnifyingglass",
                emptyTitle: "Give a recording more air.",
                emptyMessage: "Attach narrow-band audio to extend it to 48 kHz.",
                requiresAttachment: true
            )
        case .audioLive:
            return StudioTaskPresentation(
                title: title, systemImage: "mic.badge.plus",
                emptyTitle: "Transcribe as you speak.",
                emptyMessage: "Pick an input and start; words appear as they are heard."
            )
        case .textEmbeddings:
            return StudioTaskPresentation(
                title: title, systemImage: "point.3.connected.trianglepath.dotted",
                emptyTitle: "Measure meaning.",
                emptyMessage: "Type one text per line to embed them and compare how close they are.",
                promptPlaceholder: "One text per line..."
            )
        case .textAnonymize:
            return StudioTaskPresentation(
                title: title, systemImage: "eye.slash",
                emptyTitle: "Take the names out.",
                emptyMessage: "Paste text to find and replace personal information.",
                promptPlaceholder: "Paste the text to protect..."
            )
        case .textDecide:
            return StudioTaskPresentation(
                title: title, systemImage: "checklist",
                emptyTitle: "Decide with evidence.",
                emptyMessage: "Frame the question, add the options and facts, and ask for a ranked answer."
            )
        case .earthFlood:
            return StudioTaskPresentation(
                title: title, systemImage: "water.waves",
                emptyTitle: "Map the flood.",
                emptyMessage: "Attach a normalized S2L2A, S1RTC, and DEM tile batch to segment flooded ground.",
                requiresAttachment: true
            )
        case .earthFire:
            return StudioTaskPresentation(
                title: title, systemImage: "flame",
                emptyTitle: "Map the burn.",
                emptyMessage: "Attach a normalized S2L2A, S1RTC, and DEM tile batch to segment burned ground.",
                requiresAttachment: true
            )
        case .earthTessera:
            return StudioTaskPresentation(
                title: title, systemImage: "square.stack.3d.down.right",
                emptyTitle: "Embed a year of observations.",
                emptyMessage: "Attach raw Sentinel-1/2 observations with their day-of-year tensors.",
                requiresAttachment: true
            )
        case .earthOlmoEarth:
            return StudioTaskPresentation(
                title: title, systemImage: "globe.europe.africa",
                emptyTitle: "Embed multisensor observations.",
                emptyMessage: "Attach a tile bundle with a TIMESTAMPS tensor to encode it.",
                requiresAttachment: true
            )
        case .modelsInstalled, .modelsLocations, .modelsHealth, .modelsBenchmarks, .modelsAdapters:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Your models, on this Mac.",
                emptyMessage: "Install, locate, check, and benchmark the models Studio runs."
            )
        case .serverServing, .serverMusic, .serverVision:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Serve a model locally.",
                emptyMessage: "Start the server and point your tools at it."
            )
        case .runsRuns:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Every run, durable.",
                emptyMessage: "Inspect local and Relay runs and fetch what they produced."
            )
        case .pluginsCatalog:
            return StudioTaskPresentation(
                title: title, systemImage: domain.systemImage,
                emptyTitle: "Extend mere.run.",
                emptyMessage: "Install verified companion tools and run them here."
            )
        case .imageGenerate, .videoGenerate, .musicCompose, .soundGenerate, .voiceSpeak, .chatChat, .chatCode,
             .visionRead, .visionFind, .visionSegment, .visionTrack, .audioTranscribe:
            // Mode-backed; handled above. Listed so a task that loses its mode must say what it shows.
            return StudioTaskPresentation(title: title, systemImage: domain.systemImage, emptyTitle: title, emptyMessage: "")
        }
    }
}
