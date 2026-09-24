import Foundation

/// Reads what a legacy page persisted for a task into the task draft the workspace keeps, once,
/// the first time the workspace opens a task with no draft of its own. Every variant the page
/// kept settings for is imported — the one the task opens on as the draft, the others parked in
/// it — so a page that ran several commands (Audio ▸ Live's transcript and speaker activity,
/// Vision's Faces, 3D's three engines) loses none of them. Pages that kept one `CommandDraft`
/// per command are read directly; the Vision and 3D pages kept scalar keys, and their commands
/// are rebuilt the way the pages built them, camera files and ordered views included. The
/// pages stamped a per-run destination into what they kept; that was never a setting, so it is
/// cleared and routing names fresh ones.
///
/// The import never writes the session store: `taskDraft(for:)` hands the result out until the
/// first edit parks it, so a view reading it mid-render changes nothing. The camera document a
/// page kept is written as the editor's content-named draft file, which is what `--cameras` has
/// to name.
@MainActor
package enum StudioTaskDraftMigration {
    /// The session key a legacy page kept a template's `CommandDraft` under, scoped by the task
    /// the way `@StudioStoredValue` scopes it.
    package static func legacyKey(for templateID: CommandTemplateID) -> String? {
        switch templateID {
        case .speechDiarize: return "Voice.diarizationDraft"
        case .speechSynthesize: return "Voice.synthesisDraft"
        case .speechProfileCreate: return "Voice.profileDraft"
        case .speechListen: return "Voice.listenDraft"
        case .speechDiarizeLive: return "Voice.liveDiarizationDraft"
        case .musicAnalyze: return "MusicTools.analyzeDraft"
        case .musicTranscribe: return "MusicTools.transcribeDraft"
        case .audioEnhance: return "AudioTools.enhanceDraft"
        case .musicSeparate: return "AudioTools.separationDraft"
        case .sfxVideo: return "SFXLab.videoDraft"
        case .sfxConditionText: return "SFXLab.conditionDraft"
        case .sfxAEEncode: return "SFXLab.encodeDraft"
        case .sfxAEDecode: return "SFXLab.decodeDraft"
        case .sfxClapScore: return "SFXLab.scoreDraft"
        case .imageTrainLoRA: return "Training.imageDraft"
        case .textTrainLoRA: return "Training.textDraft"
        case .musicTrainAdapter: return "Training.musicDraft"
        default: return nil
        }
    }

    /// The imported draft for `task`, or nil when no page kept anything for any of its variants.
    package static func imported(for task: StudioTask, from sessions: StudioTaskSessions) -> StudioTaskDraft? {
        let templates = task.variantTemplates
        let pages = templates.compactMap { template in
            pageDraft(for: template.id, task: task, in: sessions).map {
                StudioTaskDraft(templateID: template.id, form: StudioConsoleCommand.seed(template: template, draft: $0))
            }
        }
        guard let first = templates.first, !pages.isEmpty else { return nil }
        var draft = StudioTaskDraft(templateID: first.id)
        for page in pages { draft.adopt(page) }
        draft.switchTemplate(to: pageVariant(for: task, in: sessions) ?? first.id)
        return draft.withoutDestinations()
    }

    /// The Earth page kept one `CommandDraft` per workflow in one dictionary, scoped by the task
    /// like every page key; each Earth task reads its own entry.
    package static let earthPageKey = "GeoLab.drafts"

    /// The page's dictionary key, `StudioGeoTool`, was a `String` enum without
    /// `CodingKeyRepresentable`, so `JSONEncoder` wrote the dictionary as an array of alternating
    /// keys and drafts; decoding through the same kind of key reads that shape back.
    private enum LegacyGeoTool: String, Codable, Hashable {
        case flood
        case fire
        case tessera
        case olmoEarth

        init?(templateID: CommandTemplateID) {
            switch templateID {
            case .geoFlood: self = .flood
            case .geoFire: self = .fire
            case .geoTessera: self = .tessera
            case .geoOlmoEarth: self = .olmoEarth
            default: return nil
            }
        }
    }

    private static func pageDraft(for templateID: CommandTemplateID, task: StudioTask, in sessions: StudioTaskSessions) -> CommandDraft? {
        if let key = legacyKey(for: templateID) {
            return sessions.value(for: task.rawValue + "." + key, default: Optional<CommandDraft>.none)
        }
        if let tool = LegacyGeoTool(templateID: templateID) {
            return sessions.value(for: task.rawValue + "." + earthPageKey, default: [LegacyGeoTool: CommandDraft]())[tool]
        }
        let page = LegacyPage(task: task, sessions: sessions)
        switch task {
        case .threeDFromImage: return page.kept("3DCreation.") ? threeDDraft(for: templateID, page: page) : nil
        case .visionFaces, .visionPose, .visionFlow, .visionDepth, .visionGeometry, .visionLive:
            return page.kept("VisionLab.") ? visionDraft(for: templateID, task: task, page: page) : nil
        default: return nil
        }
    }

    /// The variant the page last had open, where it kept one: 3D's engine.
    private static func pageVariant(for task: StudioTask, in sessions: StudioTaskSessions) -> CommandTemplateID? {
        guard task == .threeDFromImage else { return nil }
        let page = LegacyPage(task: task, sessions: sessions)
        return page.kept("3DCreation.") ? page.engine.templateID : nil
    }

    // MARK: Renoise

    /// The renoise mode Sound ▸ Video Foley's inspector shows: its own (`"<task>.renoiseMode"`),
    /// else the one the SFX Lab page kept beside its Video Foley draft, else automatic.
    package static func renoiseMode(for task: StudioTask, in sessions: StudioTaskSessions) -> StudioRenoise.Mode {
        let key = renoiseModeKey(for: task)
        if sessions.contains(key) { return sessions.value(for: key, default: StudioRenoise.Mode.automatic) }
        return sessions.value(for: task.rawValue + ".SFXLab.videoRenoiseMode", default: StudioRenoise.Mode.automatic)
    }

    package static func renoiseModeKey(for task: StudioTask) -> String {
        task.rawValue + ".renoiseMode"
    }

    // MARK: 3D

    /// The 3D page's engine picker, by its titles; TRELLIS.2 until another was picked.
    private enum LegacyEngine: String, Codable {
        case trellis = "TRELLIS.2"
        case triposr = "TripoSR"
        case instantMesh = "InstantMesh"

        var templateID: CommandTemplateID {
            switch self {
            case .trellis: return .imageReconstruct3DTrellis2
            case .triposr: return .imageReconstruct3D
            case .instantMesh: return .imageReconstruct3DMultiview
            }
        }
    }

    /// The command the 3D page built for an engine from its keys, with its defaults for any it
    /// never wrote. The one model the page kept was the chosen engine's.
    private static func threeDDraft(for templateID: CommandTemplateID, page: LegacyPage) -> CommandDraft? {
        guard let template = CommandCatalog.template(id: templateID) else { return nil }
        var draft = template.defaultDraft()
        draft.inputPath = page.value("3DCreation.sourcePath", "")
        draft.referenceImagePaths = page.value("3DCreation.orderedViews", [String]()).joined(separator: "\n")
        if page.engine.templateID == templateID {
            draft.model = page.value("3DCreation.model", "")
        }
        draft.reconstructionResolution = page.value("3DCreation.resolution", 256)
        draft.densityThreshold = page.value("3DCreation.densityThreshold", 25.0)
        draft.foregroundRatio = page.value("3DCreation.foregroundRatio", 0.85)
        draft.alreadyFramed = page.value("3DCreation.alreadyFramed", false)
        draft.noVertexColors = !page.value("3DCreation.vertexColors", true)
        draft.seed = page.value("3DCreation.seed", "42")
        draft.trellisTextureSeed = page.value("3DCreation.textureSeed", "42")
        draft.maxTokens = page.value("3DCreation.maxTokens", 2_097_152)
        let remesh = page.value("3DCreation.remesh", true)
        draft.trellisNoRemesh = !remesh
        draft.trellisRemeshBand = remesh ? page.value("3DCreation.remeshBand", 1.0) : nil
        draft.trellisSealRadius = remesh ? page.value("3DCreation.sealRadius", 12) : nil
        let preflight = page.value("3DCreation.preflight", false)
        draft.dryRun = preflight
        draft.json = preflight
        if templateID == .imageReconstruct3DMultiview, page.value("3DCreation.suppliesCameras", false) {
            let cameras = page.value("3DCreation.cameras", StudioInstantMeshCameraDocument())
            draft.camerasPath = cameraFile(
                cameras.cameras.isEmpty ? nil : try? cameras.json(),
                chosen: page.value("3DCreation.camerasPath", ""),
                draftPage: "3D Creation"
            )
        }
        return draft
    }

    // MARK: Vision

    /// The command the Vision page built for a variant from its keys (scoped by the task it was
    /// opened as), with its defaults for any it never wrote. The page kept one model for all its
    /// variants; it is carried only where the task's variants share a default model, so a depth
    /// model never lands on the video-depth command.
    private static func visionDraft(for templateID: CommandTemplateID, task: StudioTask, page: LegacyPage) -> CommandDraft? {
        guard let template = CommandCatalog.template(id: templateID) else { return nil }
        var draft = template.defaultDraft()
        let primary = page.value("VisionLab.primaryInput", "")
        let additional = page.value("VisionLab.additionalInputs", [String]())
        draft.inputPath = primary
        draft.visionSecondInputPath = page.value("VisionLab.secondaryInput", "")
        draft.visionAdditionalInputs = additional.joined(separator: "\n")
        if Set(task.variantTemplates.map(\.defaultModel)).count == 1 {
            draft.model = page.value("VisionLab.model", "")
        }
        draft.visionFaceScoreThreshold = page.value("VisionLab.faceThreshold", 0.65)
        draft.visionExecutionProvider = page.value("VisionLab.provider", "auto")
        draft.visionMaxFaces = page.value("VisionLab.maxFaces", 0)
        draft.visionIncludeEmbeddings = page.value("VisionLab.includeEmbeddings", false)
        draft.visionFaceIndex = String(page.value("VisionLab.faceIndex", 0))
        draft.visionReferenceFaceIndex = String(page.value("VisionLab.referenceFaceIndex", 0))
        draft.visionCandidateFaceIndex = String(page.value("VisionLab.candidateFaceIndex", 0))
        draft.visionInputList = page.value("VisionLab.inputListPath", "")
        draft.visionFailFast = page.value("VisionLab.failFast", false)
        draft.visionPoseBody = page.value("VisionLab.poseBody", true)
        draft.visionPoseHands = page.value("VisionLab.poseHands", true)
        draft.visionPoseFace = page.value("VisionLab.poseFace", true)
        draft.visionMaxHands = page.value("VisionLab.maxHands", 2)
        draft.visionMinimumConfidence = page.value("VisionLab.minimumConfidence", 0.1)
        draft.visionFlowAccuracy = page.value("VisionLab.flowAccuracy", "high")
        draft.visionInputSize = page.value("VisionLab.inputSize", 518)
        draft.visionMaxFrames = page.value("VisionLab.maxFrames", 240)
        let native = page.value("VisionLab.depthNative", false)
        draft.visionMaxEdge = native ? nil : page.value("VisionLab.depthMaxEdge", 1_024)
        draft.visionNative = native
        let checkpoint = page.value("VisionLab.depthCheckpoint", "")
        draft.visionCheckpoint = checkpoint.isEmpty ? nil : checkpoint
        draft.visionResolutionLevel = page.value("VisionLab.resolutionLevel", 9)
        draft.visionTokenCount = page.value("VisionLab.tokenCount", 0)
        draft.visionMaxPoints = page.value("VisionLab.maxPoints", 0)
        draft.visionProcessResolution = page.value("VisionLab.processResolution", 504)
        draft.visionReferenceView = page.value("VisionLab.referenceView", "saddle-balanced")
        draft.visionConfidencePercentile = page.value("VisionLab.confidencePercentile", 40.0)
        draft.prompt = page.value("VisionLab.prompts", "a person")
        draft.visionCamera = page.value("VisionLab.camera", 0)
        draft.durationSeconds = page.value("VisionLab.duration", 10.0)
        draft.visionInitFrame = page.value("VisionLab.initFrame", 0)
        draft.visionSeedSearchFrames = page.value("VisionLab.seedSearchFrames", 30)
        draft.visionThreshold = page.value("VisionLab.trackingThreshold", 0.05)
        draft.visionResolution = page.value("VisionLab.trackingResolution", 1008)
        draft.force = page.value("VisionLab.showBoxes", true)
        draft.visionShowLabels = page.value("VisionLab.showLabels", true)
        draft.dryRun = page.value("VisionLab.dryRun", false)
        draft.json = true
        if templateID == .visionGeometryMultiview, page.value("VisionLab.suppliesCameras", false) {
            let cameras = page.value("VisionLab.geometryCameras", StudioGeometryCameraDocument())
            draft.camerasPath = cameraFile(
                cameras.cameras.isEmpty ? nil : try? cameras.json(),
                chosen: page.value("VisionLab.camerasPath", ""),
                draftPage: "Vision Geometry"
            )
        }
        return draft
    }

    // MARK: Cameras

    /// The `--cameras` a page's kept cameras become: the document it edited, written as the
    /// editor's content-named draft file (an incomplete one too, so the run is refused with the
    /// reason rather than run without cameras), else a file chosen before the page edited
    /// cameras itself. Empty when neither exists, or the draft file cannot be written.
    private static func cameraFile(_ document: Data?, chosen: String, draftPage: String) -> String {
        guard let document else { return chosen }
        return (try? StudioCameraDocuments.storeDraft(page: draftPage, content: document).path) ?? chosen
    }

    // MARK: Page keys

    /// One legacy page's keys for a task, read the way `@StudioStoredValue` wrote them.
    @MainActor
    private struct LegacyPage {
        let task: StudioTask
        let sessions: StudioTaskSessions

        func value<Value: Codable>(_ key: String, _ initial: Value) -> Value {
            sessions.value(for: task.rawValue + "." + key, default: initial)
        }

        var engine: LegacyEngine {
            value("3DCreation.engine", LegacyEngine.trellis)
        }

        /// Whether the page wrote any key under `prefix` for this task.
        func kept(_ prefix: String) -> Bool {
            sessions.containsKey(withPrefix: task.rawValue + "." + prefix)
        }
    }
}
