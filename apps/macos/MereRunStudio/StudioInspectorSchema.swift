import Foundation

// The inspector's declarative surface: which `StudioDraft` fields each prompt mode shows in
// which section, and which fall under "Advanced". Chips, inspector, and the Command view all
// bind to the same draft, so this schema is also what "N changed" and per-section Reset read.

/// One draft field the inspector owns: compared against a baseline draft for the modified
/// count, and restored from it by Reset. Type-erased over the field's value type.
struct StudioInspectorField: Identifiable {
    let id: String
    let isChanged: (StudioDraft, StudioDraft) -> Bool
    let reset: (inout StudioDraft, StudioDraft) -> Void

    init<Value: Equatable>(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Value>) {
        self.id = id
        isChanged = { draft, baseline in draft[keyPath: keyPath] != baseline[keyPath: keyPath] }
        reset = { draft, baseline in draft[keyPath: keyPath] = baseline[keyPath: keyPath] }
    }
}

enum StudioInspectorSectionKind: String, CaseIterable, Identifiable {
    case prompt
    case output
    case model
    case sampling
    case transcript

    var id: String { rawValue }

    var title: String {
        switch self {
        case .prompt: return "Prompt"
        case .output: return "Output"
        case .model: return "Model & adapters"
        case .sampling: return "Sampling"
        case .transcript: return "Transcript"
        }
    }
}

struct StudioInspectorSection: Identifiable {
    let kind: StudioInspectorSectionKind
    let fields: [StudioInspectorField]

    var id: StudioInspectorSectionKind { kind }

    func changedCount(draft: StudioDraft, baseline: StudioDraft) -> Int {
        fields.filter { $0.isChanged(draft, baseline) }.count
    }

    func reset(_ draft: inout StudioDraft, to baseline: StudioDraft) {
        for field in fields { field.reset(&draft, baseline) }
    }
}

enum StudioInspectorSchema {
    /// The sections a mode's inspector shows, in order. Attachment slots (input, references,
    /// audio) live in the composer's well and are not repeated here.
    static func sections(for mode: StudioMode) -> [StudioInspectorSection] {
        switch mode {
        case .createImage:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .output, fields: [.init("width", \.width), .init("height", \.height)]),
                StudioInspectorSection(kind: .model, fields: [
                    .init("model", \.model), .init("loraPath", \.loraPath), .init("loraScale", \.loraScale),
                ]),
                StudioInspectorSection(kind: .sampling, fields: [
                    .init("steps", \.steps), .init("cfgScale", \.cfgScale), .init("seed", \.seed),
                ]),
            ]
        case .video:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .output, fields: [
                    .init("width", \.width), .init("height", \.height), .init("numFrames", \.numFrames),
                    .init("useDuration", \.useDuration), .init("durationSeconds", \.durationSeconds),
                ]),
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
                StudioInspectorSection(kind: .sampling, fields: [
                    .init("steps", \.steps), .init("cfgScale", \.cfgScale), .init("seed", \.seed),
                ]),
            ]
        case .music:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .output, fields: [
                    .init("useDuration", \.useDuration), .init("durationSeconds", \.durationSeconds),
                    .init("musicQuality", \.musicQuality),
                ]),
                StudioInspectorSection(kind: .model, fields: [
                    .init("model", \.model), .init("musicAdapterPaths", \.musicAdapterPaths),
                ]),
                StudioInspectorSection(kind: .sampling, fields: [
                    .init("musicOverrideSteps", \.musicOverrideSteps), .init("steps", \.steps), .init("seed", \.seed),
                ]),
            ]
        case .sfx:
            return [
                StudioInspectorSection(kind: .output, fields: [.init("durationSeconds", \.durationSeconds)]),
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
                StudioInspectorSection(kind: .sampling, fields: [.init("steps", \.steps), .init("seed", \.seed)]),
            ]
        case .speak:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model), .init("voiceMode", \.voiceMode)]),
            ]
        case .chat:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .model, fields: [
                    .init("model", \.model), .init("loraPath", \.loraPath), .init("loraScale", \.loraScale),
                ]),
                StudioInspectorSection(kind: .sampling, fields: [
                    .init("temperature", \.temperature), .init("topP", \.topP), .init("maxTokens", \.maxTokens),
                    .init("thinkingMode", \.thinkingMode), .init("responseFormat", \.responseFormat),
                ]),
            ]
        case .code:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("secondaryText", \.secondaryText)]),
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
                StudioInspectorSection(kind: .sampling, fields: [
                    .init("temperature", \.temperature), .init("topP", \.topP), .init("maxTokens", \.maxTokens),
                ]),
            ]
        case .readImage:
            return [
                StudioInspectorSection(kind: .prompt, fields: [.init("readImageAction", \.readImageAction)]),
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
            ]
        case .findObjects, .segment, .track:
            return [
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
                StudioInspectorSection(kind: .sampling, fields: [.init("visionThreshold", \.visionThreshold)]),
            ]
        case .listen:
            return [
                StudioInspectorSection(kind: .model, fields: [.init("model", \.model)]),
                StudioInspectorSection(kind: .transcript, fields: [
                    .init("language", \.language), .init("timestamps", \.timestamps),
                ]),
            ]
        }
    }

    /// Everything else the mode's command takes, collapsed under "Advanced · N more". The count
    /// is this list's length; Reset restores every one of them.
    static func advancedFields(for mode: StudioMode) -> [StudioInspectorField] {
        switch mode {
        case .createImage:
            return [
                .init("imageMaskPath", \.imageMaskPath),
                .init("imageOutpaintTop", \.imageOutpaintTop), .init("imageOutpaintRight", \.imageOutpaintRight),
                .init("imageOutpaintBottom", \.imageOutpaintBottom), .init("imageOutpaintLeft", \.imageOutpaintLeft),
                .init("imageMaskFeather", \.imageMaskFeather),
                .init("keepOriginalAspect", \.keepOriginalAspect),
                .init("strength", \.strength), .init("sigmaShift", \.sigmaShift),
                .init("imageMaxSequenceLength", \.imageMaxSequenceLength),
                .init("structuredPrompt", \.structuredPrompt),
                .init("structuredPromptModel", \.structuredPromptModel),
                .init("structuredPromptMaxTokens", \.structuredPromptMaxTokens),
                .init("kreaConditioningMultiplier", \.kreaConditioningMultiplier),
                .init("kreaConditioningLayerWeights", \.kreaConditioningLayerWeights),
                .init("kreaBaseQuantizationBits", \.kreaBaseQuantizationBits),
                .init("preflight", \.preflight), .init("preflightJSON", \.preflightJSON),
                .init("progressJSON", \.progressJSON),
            ]
        case .video:
            return [
                .init("fps", \.fps), .init("strength", \.strength),
                .init("videoQuality", \.videoQuality), .init("videoOutputMode", \.videoOutputMode),
                .init("audioStartTime", \.audioStartTime), .init("audioMaxDuration", \.audioMaxDuration),
                .init("endImageStrength", \.endImageStrength),
                .init("a2vSteps", \.a2vSteps), .init("a2vGuidanceScale", \.a2vGuidanceScale),
                .init("videoCFGGuidanceScale", \.videoCFGGuidanceScale),
                .init("audioCFGGuidanceScale", \.audioCFGGuidanceScale),
                .init("v2aGuidanceScale", \.v2aGuidanceScale),
                .init("h3WeightMode", \.h3WeightMode), .init("h3AccelerationMode", \.h3AccelerationMode),
                .init("h3Steps", \.h3Steps), .init("h3ReferenceInputs", \.h3ReferenceInputs),
                .init("preflight", \.preflight), .init("timings", \.timings),
            ]
        case .music:
            return [
                .init("musicTask", \.musicTask),
                .init("musicCoverStrength", \.musicCoverStrength), .init("musicCoverNoiseStrength", \.musicCoverNoiseStrength),
                .init("musicLMMode", \.musicLMMode), .init("musicAnalyzeSourceAudio", \.musicAnalyzeSourceAudio),
                .init("musicLMTemperature", \.musicLMTemperature),
                .init("musicLMRepetitionPenalty", \.musicLMRepetitionPenalty),
                .init("musicLMCFGScale", \.musicLMCFGScale), .init("musicLMNegativePrompt", \.musicLMNegativePrompt),
                .init("musicInstrumental", \.musicInstrumental),
                .init("musicCandidates", \.musicCandidates), .init("musicKeepCandidates", \.musicKeepCandidates),
                .init("musicFlowEdit", \.musicFlowEdit), .init("musicSourceCaption", \.musicSourceCaption),
                .init("musicSourceLyrics", \.musicSourceLyrics),
                .init("musicAdapterKind", \.musicAdapterKind), .init("musicAdapterScales", \.musicAdapterScales),
                .init("musicLRCFile", \.musicLRCFile), .init("musicStems", \.musicStems),
                .init("musicDAWBundle", \.musicDAWBundle), .init("musicExportFormat", \.musicExportFormat),
                .init("musicNoRecipe", \.musicNoRecipe),
            ]
        case .sfx, .readImage, .findObjects, .segment, .track:
            return []
        case .speak:
            return [.init("voiceProfile", \.voiceProfile), .init("saveProfileName", \.saveProfileName)]
        case .chat:
            return [
                .init("reasoningEffort", \.reasoningEffort),
                .init("contextSize", \.contextSize), .init("topK", \.topK), .init("minP", \.minP),
                .init("kvBits", \.kvBits), .init("kvQuantScheme", \.kvQuantScheme),
                .init("kvGroupSize", \.kvGroupSize), .init("quantizedKVStart", \.quantizedKVStart),
                .init("tools", \.tools), .init("sandboxDir", \.sandboxDir),
                .init("toolLoop", \.toolLoop), .init("allowShellExec", \.allowShellExec),
                .init("allowAbsoluteToolPaths", \.allowAbsoluteToolPaths), .init("autoApproveTools", \.autoApproveTools),
                .init("stats", \.stats), .init("preflight", \.preflight), .init("preflightJSON", \.preflightJSON),
                .init("requireInstalled", \.requireInstalled),
            ]
        case .code:
            return [.init("minP", \.minP)]
        case .listen:
            return [.init("backend", \.backend)]
        }
    }

    /// The ids of every field the inspector (sections and Advanced) binds for `mode`.
    static func fieldIDs(for mode: StudioMode) -> Set<String> {
        Set(sections(for: mode).flatMap { $0.fields.map(\.id) } + advancedFields(for: mode).map(\.id))
    }

    /// How many inspector fields differ from the mode's baseline draft; the header badge.
    static func changedCount(mode: StudioMode, draft: StudioDraft, baseline: StudioDraft) -> Int {
        let sectionChanges = sections(for: mode).reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
        let advancedChanges = advancedFields(for: mode).filter { $0.isChanged(draft, baseline) }.count
        return sectionChanges + advancedChanges
    }

    static func resetAdvanced(for mode: StudioMode, _ draft: inout StudioDraft, to baseline: StudioDraft) {
        for field in advancedFields(for: mode) { field.reset(&draft, baseline) }
    }
}

extension StudioComposerChipKind {
    /// The draft fields a chip edits for `mode`. Every one is also an inspector field for that
    /// mode, so the two surfaces edit the same values and can never drift.
    func draftFieldIDs(for mode: StudioMode) -> [String] {
        switch self {
        case .dimensions: return ["width", "height"]
        case .duration:
            switch mode {
            case .video: return ["durationSeconds", "useDuration", "numFrames"]
            case .music: return ["durationSeconds", "useDuration"]
            default: return ["durationSeconds"]
            }
        case .steps: return ["steps"]
        case .seed: return ["seed"]
        case .threshold: return ["visionThreshold"]
        case .readImageAction: return ["readImageAction"]
        case .voiceMode: return ["voiceMode"]
        case .thinking: return ["thinkingMode"]
        case .model: return ["model"]
        }
    }
}

extension StudioAspectPreset {
    /// The four presets the inspector's segmented control offers, in the board's order. A draft
    /// size outside them (a custom size, or a chip preset such as 4:3) selects no segment.
    static func inspectorPresets(for mode: StudioMode) -> [StudioAspectPreset] {
        let wanted = ["1:1", "3:2", "16:9", "9:16"]
        let all = presets(for: mode)
        return wanted.compactMap { label in all.first { $0.label == label } }
    }

    static func inspectorSelection(for draft: StudioDraft, mode: StudioMode) -> StudioAspectPreset? {
        inspectorPresets(for: mode).first { $0.matches(draft) }
    }

    /// Swaps width and height; a landscape preset becomes its portrait twin.
    static func swap(_ draft: inout StudioDraft) {
        let width = draft.width
        draft.width = draft.height
        draft.height = width
    }
}
