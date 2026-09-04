import Foundation

// The inspector's surface comes from the contract: `StudioContractSchema` turns a capability's
// options into sections (by `group`), disclosure (by `tier`), and controls (by `kind`, `range`,
// `choices`, and `depends_on`). This file is what the inspector, the composer's chips, and the
// tests call, so the wiring stays in one named place.

package enum StudioInspectorSchema {
    /// The sections a mode's inspector shows, in the contract's group order. Attachment slots
    /// (input, references, audio) live in the composer's well and are not repeated here.
    package static func sections(for mode: StudioMode, draft: StudioDraft = StudioDraft()) -> [StudioContractSection] {
        StudioContractSchema.sections(for: mode, draft: draft)
    }

    /// Everything else the mode's command takes, collapsed under "Advanced · N more". The count is
    /// this list's length; Reset restores every draft field behind it.
    package static func advancedFields(for mode: StudioMode, draft: StudioDraft = StudioDraft()) -> [StudioContractField<StudioDraft>] {
        StudioContractSchema.expertFields(for: mode, draft: draft)
    }

    /// The ids of every draft field the inspector (sections and Advanced) binds for `mode`.
    package static func fieldIDs(for mode: StudioMode) -> Set<String> {
        StudioContractSchema.draftFieldIDs(for: mode)
    }

    /// How many inspector fields differ from the mode's baseline draft; the header badge.
    package static func changedCount(mode: StudioMode, draft: StudioDraft, baseline: StudioDraft) -> Int {
        StudioContractSchema.changedCount(mode: mode, draft: draft, baseline: baseline)
    }

    package static func resetAdvanced(for mode: StudioMode, _ draft: inout StudioDraft, to baseline: StudioDraft) {
        for field in advancedFields(for: mode, draft: draft) { field.reset(&draft, to: baseline) }
    }

    package static func advancedChanged(mode: StudioMode, draft: StudioDraft, baseline: StudioDraft) -> Bool {
        advancedFields(for: mode, draft: draft).contains { $0.changedCount(draft: draft, baseline: baseline) > 0 }
    }
}

extension StudioComposerChipKind {
    /// The draft fields a chip edits for `mode`. Every one is also an inspector field for that
    /// mode, so the two surfaces edit the same values and can never drift.
    package func draftFieldIDs(for mode: StudioMode) -> [String] {
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
    package static func inspectorPresets(for mode: StudioMode) -> [StudioAspectPreset] {
        let wanted = ["1:1", "3:2", "16:9", "9:16"]
        let all = presets(for: mode)
        return wanted.compactMap { label in all.first { $0.label == label } }
    }

    package static func inspectorSelection(for draft: StudioDraft, mode: StudioMode) -> StudioAspectPreset? {
        inspectorPresets(for: mode).first { $0.matches(draft) }
    }

    /// Swaps width and height; a landscape preset becomes its portrait twin.
    package static func swap(_ draft: inout StudioDraft) {
        let width = draft.width
        draft.width = draft.height
        draft.height = width
    }
}
