import Combine
import Foundation

/// Sidebar sections. Order here is the navigation order and drives ⌘1…⌘9 / ⌥⌘1… in the Go menu.
package enum StudioDomainGroup: String, CaseIterable, Identifiable {
    case create = "Create"
    case converse = "Converse"
    case understand = "Understand"
    case system = "System"

    package var id: String { rawValue }

    package var domains: [StudioDomain] {
        StudioDomain.allCases.filter { $0.group == self }
    }
}

/// One sidebar row. Every capability lives in exactly one domain, reached the same way as every
/// other; what varies per domain is its list of tasks, shown in the toolbar task control.
package enum StudioDomain: String, CaseIterable, Codable, Identifiable {
    case image
    case video
    case music
    case sound
    case voice
    case threeD
    case chat
    case vision
    case audio
    case text
    case earth
    case models
    case server
    case runs
    case plugins

    package var id: String { rawValue }

    package var group: StudioDomainGroup {
        switch self {
        case .image, .video, .music, .sound, .voice, .threeD: return .create
        case .chat: return .converse
        case .vision, .audio, .text, .earth: return .understand
        case .models, .server, .runs, .plugins: return .system
        }
    }

    package var title: String {
        switch self {
        case .image: return "Image"
        case .video: return "Video"
        case .music: return "Music"
        case .sound: return "Sound"
        case .voice: return "Voice"
        case .threeD: return "3D"
        case .chat: return "Chat"
        case .vision: return "Vision"
        case .audio: return "Audio"
        case .text: return "Text"
        case .earth: return "Earth"
        case .models: return "Models"
        case .server: return "Server"
        case .runs: return "Runs"
        case .plugins: return "Plugins"
        }
    }

    /// One line under the title in the toolbar.
    package var subtitle: String {
        switch self {
        case .image: return "Text or reference to PNG · stays on this Mac"
        case .video: return "Prompt to clip · subject animation"
        case .music: return "Prompt to song · live steering"
        case .sound: return "Prompt to sound effect · Foley and scoring"
        case .voice: return "Text to natural speech · cloned and saved voices"
        case .threeD: return "Single image to mesh"
        case .chat: return "Local model · nothing leaves this Mac"
        case .vision: return "Images and video, understood locally"
        case .audio: return "Transcribe, identify, enhance, and separate"
        case .text: return "Embeddings and anonymization"
        case .earth: return "Native Earth-observation inference"
        case .models: return "Local models, locations, and health"
        case .server: return "The local API and resident engines"
        case .runs: return "Durable runs and Relay jobs"
        case .plugins: return "Companion tools, verified and installed"
        }
    }

    package var systemImage: String {
        switch self {
        case .image: return "photo"
        case .video: return "film"
        case .music: return "music.note"
        case .sound: return "speaker.wave.2"
        case .voice: return "waveform"
        case .threeD: return "cube.transparent"
        case .chat: return "bubble.left.and.bubble.right"
        case .vision: return "eye"
        case .audio: return "waveform.badge.mic"
        case .text: return "text.alignleft"
        case .earth: return "globe.europe.africa"
        case .models: return "shippingbox"
        case .server: return "network"
        case .runs: return "list.bullet.rectangle.portrait"
        case .plugins: return "puzzlepiece.extension"
        }
    }

    package var tasks: [StudioTask] {
        StudioTask.allCases.filter { $0.domain == self }
    }

    package var defaultTask: StudioTask {
        tasks[0]
    }

    package var defaultDestination: StudioDestination {
        StudioDestination(domain: self, task: defaultTask)
    }

    /// Whether one of this domain's tasks is a composer-driven prompt mode. The Library column
    /// itself follows the task (`StudioTask.isPromptTask`), so a Create domain's Project or
    /// Session task still takes the full width.
    package var hasPromptWorkspace: Bool {
        tasks.contains { $0.mode != nil }
    }

    /// The domain a command belongs to, so Library rows land where the work was done rather
    /// than under the prompt mode a specialist view happened to attribute them to.
    package init(templateID: CommandTemplateID) {
        switch templateID {
        case .imageGenerate, .imageTrainLoRA, .imageValidate, .imageDatasetDiscover, .imageRunPlan,
             .imageVisualizeRun:
            self = .image
        case .imageReconstruct3D, .imageReconstruct3DTrellis2, .imageReconstruct3DMultiview:
            self = .threeD
        case .videoGenerate, .videoRetake, .videoDubIt, .videoAnimate, .videoCosmos3, .videoPrepareMasks,
             .videoExportLatents, .videoSession:
            self = .video
        case .musicGenerate, .musicAnalyze, .musicTranscribe, .musicSeparate, .musicRealtime,
             .musicTrainAdapter:
            self = .music
        case .sfxGenerate, .sfxVideo, .sfxAEEncode, .sfxAEDecode, .sfxClapScore, .sfxConditionText,
             .audioGenerate:
            self = .sound
        case .speechSynthesize, .speechProfileList, .speechProfileCreate, .speechProfileDelete:
            self = .voice
        case .textChat, .textCode, .textTrainLoRA:
            self = .chat
        case .visionInspect, .visionEmbed, .visionCaption, .visionOCR, .visionGround, .visionSegment,
             .visionTrack, .visionTrackLive, .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare,
             .visionFaceBatch, .visionPose, .visionFlow, .visionDepthVideo, .visionGeometry,
             .visionGeometryMultiview:
            self = .vision
        case .speechTranscribe, .speechDiarize, .speechListen, .audioEnhance:
            self = .audio
        case .textEmbed, .textAnonymize:
            self = .text
        case .geoFlood, .geoFire, .geoTessera, .geoOlmoEarth:
            self = .earth
        case .adapterList, .adapterPull, .modelList, .modelCapabilities, .modelPull, .modelInfo, .modelRemove, .modelRepairManifests,
             .modelOptimize, .modelStorage, .modelGarbageCollect, .modelRuntimeGet, .modelRuntimeSet,
             .modelLocationList, .modelLocationAdd, .modelLocationRemove, .modelLocationBind,
             .modelLocationUnbind, .qualityGate, .modelBenchmark, .modelBenchmarkLagunaDFlash,
             .modelBenchmarkChat, .modelBenchmarkCode, .modelBenchmarkFused, .modelBenchmarkFusedFixture,
             .modelBenchmarkVLM, .modelBenchmarkToolCalls, .modelBenchmarkToolContinuations,
             .modelBenchmarkGemma4KV, .modelBenchmarkGemma4MTP, .modelBenchmarkParakeetCoreML,
             .modelBenchmarkAPIWorkload:
            self = .models
        case .apiServe, .visionServe, .musicServe, .openWebui, .worldServe, .statusSnapshot, .agentOnboard,
             .agentStatus, .agentInstallPi, .agentStart, .setup, .graphStudio, .nodeConsole, .custom:
            self = .server
        case .runList, .runInspect, .runWatch, .runFetch, .runCancel, .runRetry, .evaluationPackValidate,
             .evaluationRun, .evaluationPromote:
            self = .runs
        case .pluginList, .pluginInstall, .pluginDoctor, .pluginInfo, .pluginRun, .pluginRollback:
            self = .plugins
        }
    }
}

/// One entry in a domain's task control. Tasks backed by a `StudioMode` render the prompt
/// workspace (canvas + composer + Library); every other task re-hosts a former specialist
/// sheet inline in the detail area.
package enum StudioTask: String, CaseIterable, Codable, Identifiable {
    case imageGenerate = "image.generate"
    case imageDatasets = "image.datasets"
    case imageTrain = "image.train"

    case videoGenerate = "video.generate"
    case videoSubjects = "video.subjects"

    case musicCompose = "music.compose"
    case musicRealtime = "music.realtime"
    case musicAnalyze = "music.analyze"
    case musicTranscribe = "music.transcribe"
    case musicSeparate = "music.separate"
    case musicTrain = "music.train"

    case soundGenerate = "sound.generate"
    case soundFoley = "sound.foley"
    case soundCondition = "sound.condition"
    case soundEncode = "sound.encode"
    case soundDecode = "sound.decode"
    case soundScore = "sound.score"

    case voiceSpeak = "voice.speak"
    case voiceClone = "voice.clone"
    case voiceVoices = "voice.voices"

    case threeDFromImage = "threeD.fromImage"

    case chatChat = "chat.chat"
    case chatCode = "chat.code"
    case chatTrain = "chat.train"

    case visionRead = "vision.read"
    case visionFind = "vision.find"
    case visionSegment = "vision.segment"
    case visionTrack = "vision.track"
    case visionDepth = "vision.depth"
    case visionPose = "vision.pose"
    case visionFaces = "vision.faces"
    case visionFlow = "vision.flow"
    case visionGeometry = "vision.geometry"
    case visionLive = "vision.live"

    case audioTranscribe = "audio.transcribe"
    case audioWhoSpoke = "audio.whoSpoke"
    case audioEnhance = "audio.enhance"
    case audioSeparate = "audio.separate"
    case audioLive = "audio.live"

    case textEmbeddings = "text.embeddings"
    case textAnonymize = "text.anonymize"

    case earthFlood = "earth.flood"
    case earthFire = "earth.fire"
    case earthTessera = "earth.tessera"
    case earthOlmoEarth = "earth.olmoEarth"

    case modelsInstalled = "models.installed"
    case modelsLocations = "models.locations"
    case modelsHealth = "models.health"
    case modelsBenchmarks = "models.benchmarks"
    case modelsAdapters = "models.adapters"

    case serverServing = "server.serving"
    case serverMusic = "server.music"

    case runsRuns = "runs.runs"

    case pluginsCatalog = "plugins.catalog"

    package var id: String { rawValue }

    package var domain: StudioDomain {
        switch self {
        case .imageGenerate, .imageDatasets, .imageTrain: return .image
        case .videoGenerate, .videoSubjects: return .video
        case .musicCompose, .musicRealtime, .musicAnalyze, .musicTranscribe, .musicSeparate, .musicTrain:
            return .music
        case .soundGenerate, .soundFoley, .soundCondition, .soundEncode, .soundDecode, .soundScore:
            return .sound
        case .voiceSpeak, .voiceClone, .voiceVoices: return .voice
        case .threeDFromImage: return .threeD
        case .chatChat, .chatCode, .chatTrain: return .chat
        case .visionRead, .visionFind, .visionSegment, .visionTrack, .visionDepth, .visionPose,
             .visionFaces, .visionFlow, .visionGeometry, .visionLive:
            return .vision
        case .audioTranscribe, .audioWhoSpoke, .audioEnhance, .audioSeparate, .audioLive: return .audio
        case .textEmbeddings, .textAnonymize: return .text
        case .earthFlood, .earthFire, .earthTessera, .earthOlmoEarth: return .earth
        case .modelsInstalled, .modelsLocations, .modelsHealth, .modelsBenchmarks, .modelsAdapters:
            return .models
        case .serverServing, .serverMusic: return .server
        case .runsRuns: return .runs
        case .pluginsCatalog: return .plugins
        }
    }

    package var title: String {
        switch self {
        case .imageGenerate, .videoGenerate, .soundGenerate: return "Generate"
        case .imageDatasets: return "Datasets"
        case .imageTrain, .musicTrain, .chatTrain: return "Train"
        case .videoSubjects: return "Subjects"
        case .musicCompose: return "Compose"
        case .musicRealtime: return "Realtime"
        case .musicAnalyze: return "Analyze"
        case .musicTranscribe, .audioTranscribe: return "Transcribe"
        case .musicSeparate, .audioSeparate: return "Separate"
        case .soundFoley: return "Video Foley"
        case .soundCondition: return "Condition"
        case .soundEncode: return "Encode"
        case .soundDecode: return "Decode"
        case .soundScore: return "Score"
        case .voiceSpeak: return "Speak"
        case .voiceClone: return "Clone"
        case .voiceVoices: return "Voices"
        case .threeDFromImage: return "From image"
        case .chatChat: return "Chat"
        case .chatCode: return "Code"
        case .visionRead: return "Read"
        case .visionFind: return "Find"
        case .visionSegment: return "Segment"
        case .visionTrack: return "Track"
        case .visionDepth: return "Depth"
        case .visionPose: return "Pose"
        case .visionFaces: return "Faces"
        case .visionFlow: return "Flow"
        case .visionGeometry: return "Geometry"
        case .visionLive, .audioLive: return "Live"
        case .audioWhoSpoke: return "Who Spoke"
        case .audioEnhance: return "Enhance"
        case .textEmbeddings: return "Embeddings"
        case .textAnonymize: return "Anonymize"
        case .earthFlood: return "Flood"
        case .earthFire: return "Fire"
        case .earthTessera: return "TESSERA"
        case .earthOlmoEarth: return "OlmoEarth"
        case .modelsInstalled: return "Installed"
        case .modelsLocations: return "Locations"
        case .modelsHealth: return "Health"
        case .modelsBenchmarks: return "Benchmarks"
        case .modelsAdapters: return "Adapters"
        case .serverServing: return "Serving"
        case .serverMusic: return "Music server"
        case .runsRuns: return "Runs"
        case .pluginsCatalog: return "Catalog"
        }
    }

    /// The prompt mode this task renders, when it is one of the twelve composer-driven modes.
    package var mode: StudioMode? {
        switch self {
        case .imageGenerate: return .createImage
        case .videoGenerate: return .video
        case .musicCompose: return .music
        case .soundGenerate: return .sfx
        case .voiceSpeak: return .speak
        case .chatChat: return .chat
        case .chatCode: return .code
        case .visionRead: return .readImage
        case .visionFind: return .findObjects
        case .visionSegment: return .segment
        case .visionTrack: return .track
        case .audioTranscribe: return .listen
        default: return nil
        }
    }

    package var destination: StudioDestination {
        StudioDestination(domain: domain, task: self)
    }

    /// Whether this task is one of the composer-driven prompt tasks (Generate, Compose, Chat,
    /// Read, Find, Segment, Track, Transcribe, Speak). Only prompt tasks show the Library column
    /// and the inspector; Project, Session, and Manage tasks take the full content width.
    package var isPromptTask: Bool {
        mode != nil
    }
}

/// Where the window is: one domain and one of its tasks. The raw value is what `@SceneStorage`
/// persists, so a destination survives relaunch as text and decodes strictly.
package struct StudioDestination: Hashable, Codable, RawRepresentable {
    package let domain: StudioDomain
    package let task: StudioTask

    package init(domain: StudioDomain, task: StudioTask) {
        precondition(task.domain == domain, "\(task) does not belong to \(domain)")
        self.domain = domain
        self.task = task
    }

    package init(task: StudioTask) {
        self.init(domain: task.domain, task: task)
    }

    package init?(rawValue: String) {
        let parts = rawValue.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2,
              let domain = StudioDomain(rawValue: parts[0]),
              let task = StudioTask(rawValue: parts[1]),
              task.domain == domain else {
            return nil
        }
        self.domain = domain
        self.task = task
    }

    package var rawValue: String {
        "\(domain.rawValue)/\(task.rawValue)"
    }

    package static let `default` = StudioDomain.image.defaultDestination

    /// Same destination shape as the twelve legacy `StudioMode` values.
    package init(mode: StudioMode) {
        self.init(task: mode.task)
    }
}

extension CommandTemplateID {
    package var studioDomain: StudioDomain {
        StudioDomain(templateID: self)
    }
}

extension StudioLibraryItem {
    /// The domain a row is filed under: its command's domain when the row records one, otherwise
    /// the domain of the mode that created it.
    package var domain: StudioDomain {
        templateID?.studioDomain ?? mode.destination.domain
    }
}

extension StudioMode {
    /// The v2 task that renders this mode. Every mode maps to exactly one task.
    package var task: StudioTask {
        switch self {
        case .createImage: return .imageGenerate
        case .chat: return .chatChat
        case .code: return .chatCode
        case .speak: return .voiceSpeak
        case .listen: return .audioTranscribe
        case .readImage: return .visionRead
        case .findObjects: return .visionFind
        case .segment: return .visionSegment
        case .track: return .visionTrack
        case .music: return .musicCompose
        case .video: return .videoGenerate
        case .sfx: return .soundGenerate
        }
    }

    package var destination: StudioDestination {
        StudioDestination(mode: self)
    }
}

/// How a domain's tasks are presented in the toolbar.
package enum StudioTaskControlStyle: Equatable {
    /// One task: the title alone is enough.
    case none
    /// Every task as a segment of one pill.
    case segmented
    /// The first `visible` tasks as segments, the rest behind a "More" segment that opens a menu.
    case segmentedWithOverflow(visible: Int)

    /// The most segments the pill holds at the default window width with the Library shown.
    package static let segmentedLimit = 6
    /// Segments shown before "More" when a domain has more tasks than `segmentedLimit`.
    package static let overflowVisibleCount = 5

    package static func style(for domain: StudioDomain) -> StudioTaskControlStyle {
        let count = domain.tasks.count
        if count <= 1 { return .none }
        return count <= segmentedLimit ? .segmented : .segmentedWithOverflow(visible: overflowVisibleCount)
    }
}

/// Encodes the set of tasks whose inspector stays open as one `@SceneStorage` string
/// (`studio.inspectorTasks`), comma-separated task ids in a stable order.
package enum StudioInspectorTaskMemory {
    package static func encode(_ tasks: Set<StudioTask>) -> String {
        tasks.map(\.rawValue).sorted().joined(separator: ",")
    }

    package static func decode(_ raw: String) -> Set<StudioTask> {
        Set(raw.split(separator: ",").compactMap { StudioTask(rawValue: String($0)) })
    }
}

/// What a per-task draft is worth keeping across a relaunch: the words the user typed and the
/// file they attached. Sampling settings are not persisted — they come back from the task's own
/// defaults, which is also the baseline the inspector diffs against — so `@SceneStorage` stays
/// small enough to be a scene value rather than a database.
package struct StudioDraftMemoryEntry: Codable, Equatable {
    package var prompt = ""
    package var secondaryText = ""
    package var inputPath = ""

    package var isEmpty: Bool {
        prompt.isBlank && secondaryText.isBlank && inputPath.isBlank
    }
}

/// Encodes the unsent work of every prompt task as one `@SceneStorage` string, keyed by task, so
/// switching domains or tasks — and quitting — never throws away a half-written prompt.
package enum StudioDraftMemory {
    package static func entry(for draft: StudioDraft) -> StudioDraftMemoryEntry {
        StudioDraftMemoryEntry(
            prompt: draft.prompt,
            secondaryText: draft.secondaryText,
            inputPath: draft.inputPath
        )
    }

    package static func encode(_ entries: [StudioTask: StudioDraftMemoryEntry]) -> String {
        let kept = entries.filter { !$0.value.isEmpty }
        guard !kept.isEmpty else { return "" }
        let keyed = Dictionary(uniqueKeysWithValues: kept.map { ($0.key.rawValue, $0.value) })
        guard let data = try? JSONEncoder.mereRunApp.encode(keyed),
              let text = String(data: data, encoding: .utf8) else {
            return ""
        }
        return text
    }

    package static func decode(_ raw: String) -> [StudioTask: StudioDraftMemoryEntry] {
        guard !raw.isBlank, let data = raw.data(using: .utf8),
              let keyed = try? JSONDecoder.mereRunApp.decode([String: StudioDraftMemoryEntry].self, from: data) else {
            return [:]
        }
        var entries: [StudioTask: StudioDraftMemoryEntry] = [:]
        for (key, value) in keyed {
            guard let task = StudioTask(rawValue: key), task.isPromptTask, !value.isEmpty else { continue }
            entries[task] = value
        }
        return entries
    }

    /// Puts a remembered entry back into a fresh draft. An attachment whose file has since moved
    /// or been emptied out of a temporary directory is dropped rather than restored as a dangling
    /// path the composer would show as a chip and the run would fail on.
    package static func apply(
        _ entry: StudioDraftMemoryEntry,
        to draft: inout StudioDraft,
        fileManager: FileManager = .default
    ) {
        if !entry.prompt.isBlank { draft.prompt = entry.prompt }
        if !entry.secondaryText.isBlank { draft.secondaryText = entry.secondaryText }
        if !entry.inputPath.isBlank, fileManager.fileExists(atPath: entry.inputPath) {
            draft.inputPath = entry.inputPath
        }
    }
}
