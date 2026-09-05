import Foundation
import MereRunContract

// The contract is the app's option schema. `MereRunCapabilityCatalog` declares, for every
// capability, each option's kind, choices, default, group, tier, range, and dependency; this file
// binds those options to the `StudioDraft` fields the app keeps them in, so the inspector and the
// Command view are both rendered from the contract rather than from hand-written per-mode lists.
//
// A binding is the only place a flag and a draft field meet. `StudioCommandAdapter` still builds
// the `CommandDraft` and `CommandTemplate.arguments(from:)` still builds the argv, so nothing here
// can change what a run launches: a control writes the same draft field the old hand-written
// control wrote.

// MARK: - Values

/// What a contract option holds in the Studio draft. `unset` is a draft field the app leaves
/// empty on purpose, so the CLI's own default applies and no flag is emitted.
package enum StudioContractValue: Codable, Equatable, Sendable {
    case text(String)
    case integer(Int)
    case number(Double)
    case flag(Bool)
    case unset

    package var text: String? {
        if case .text(let value) = self { return value }
        return nil
    }

    package var integer: Int? {
        if case .integer(let value) = self { return value }
        return nil
    }

    package var number: Double? {
        if case .number(let value) = self { return value }
        return nil
    }

    package var flag: Bool? {
        if case .flag(let value) = self { return value }
        return nil
    }

    /// The number this value carries whatever its numeric shape, for range clamping and sliders.
    package var numericValue: Double? {
        switch self {
        case .integer(let value): return Double(value)
        case .number(let value): return value
        case .text(let value): return Double(value)
        case .flag, .unset: return nil
        }
    }
}

// MARK: - Bindings

/// One contract option bound to the field the app keeps it in.
///
/// `Draft` is whatever the surface edits: the prompt tasks bind a typed `StudioDraft` property
/// per option, and the Command Console binds a `StudioConsoleDraft` entry per flag, so one
/// option kind renders the same control either way.
package struct StudioContractBinding<Draft> {
    /// The draft property name. It doubles as the inspector's field id, so the modified count and
    /// per-section Reset key off the same identity the hand-written schema used.
    package let fieldID: String
    package let read: (Draft) -> StudioContractValue
    package let write: (inout Draft, StudioContractValue) -> Void

    package func isChanged(_ draft: Draft, _ baseline: Draft) -> Bool {
        read(draft) != read(baseline)
    }

    package func reset(_ draft: inout Draft, to baseline: Draft) {
        write(&draft, read(baseline))
    }
}

extension StudioContractBinding where Draft == StudioDraft {
    package static func text(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, String>) -> Self {
        Self(
            fieldID: id,
            read: { .text($0[keyPath: keyPath]) },
            write: { draft, value in
                guard let text = value.text else { return }
                draft[keyPath: keyPath] = text
            }
        )
    }

    package static func integer(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Int>) -> Self {
        Self(
            fieldID: id,
            read: { .integer($0[keyPath: keyPath]) },
            write: { draft, value in
                guard let integer = value.integer else { return }
                draft[keyPath: keyPath] = integer
            }
        )
    }

    package static func number(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Double>) -> Self {
        Self(
            fieldID: id,
            read: { .number($0[keyPath: keyPath]) },
            write: { draft, value in
                guard let number = value.number else { return }
                draft[keyPath: keyPath] = number
            }
        )
    }

    package static func flag(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Bool>) -> Self {
        Self(
            fieldID: id,
            read: { .flag($0[keyPath: keyPath]) },
            write: { draft, value in
                guard let flag = value.flag else { return }
                draft[keyPath: keyPath] = flag
            }
        )
    }

    /// A flag the CLI states negatively: `--no-timestamps` is on exactly when `timestamps` is off.
    package static func invertedFlag(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Bool>) -> Self {
        Self(
            fieldID: id,
            read: { .flag(!$0[keyPath: keyPath]) },
            write: { draft, value in
                guard let flag = value.flag else { return }
                draft[keyPath: keyPath] = !flag
            }
        )
    }

    /// A string-backed enum the draft stores typed (`LTXVideoQuality`, `TextResponseFormat`, …).
    package static func choice<Choice: RawRepresentable>(
        _ id: String,
        _ keyPath: WritableKeyPath<StudioDraft, Choice>
    ) -> Self where Choice.RawValue == String {
        Self(
            fieldID: id,
            read: { .text($0[keyPath: keyPath].rawValue) },
            write: { draft, value in
                guard let text = value.text, let choice = Choice(rawValue: text) else { return }
                draft[keyPath: keyPath] = choice
            }
        )
    }

    /// A draft field that stores an integer option as free text, blank meaning "let the CLI pick"
    /// (the seed). Blank reads as `unset` so the form never claims a value the argv does not carry.
    package static func integerText(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, String>) -> Self {
        Self(
            fieldID: id,
            read: { draft in
                let raw = draft[keyPath: keyPath].trimmingCharacters(in: .whitespacesAndNewlines)
                guard let value = Int(raw) else { return raw.isEmpty ? .unset : .text(raw) }
                return .integer(value)
            },
            write: { draft, value in
                switch value {
                case .unset: draft[keyPath: keyPath] = ""
                case .integer(let integer): draft[keyPath: keyPath] = String(integer)
                case .number(let number): draft[keyPath: keyPath] = String(Int(number))
                case .text(let text): draft[keyPath: keyPath] = text
                case .flag: break
                }
            }
        )
    }

    /// An optional string the draft leaves nil until the user overrides the CLI's own default.
    package static func optionalText(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, String?>) -> Self {
        Self(
            fieldID: id,
            read: { draft in
                guard let value = draft[keyPath: keyPath], !value.isEmpty else { return .unset }
                return .text(value)
            },
            write: { draft, value in
                switch value {
                case .unset: draft[keyPath: keyPath] = nil
                case .text(let text): draft[keyPath: keyPath] = text.isEmpty ? nil : text
                default: break
                }
            }
        )
    }

    /// An optional integer the draft leaves nil until the user overrides the CLI's own default.
    package static func optionalInteger(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Int?>) -> Self {
        Self(
            fieldID: id,
            read: { draft in
                guard let value = draft[keyPath: keyPath] else { return .unset }
                return .integer(value)
            },
            write: { draft, value in
                switch value {
                case .unset: draft[keyPath: keyPath] = nil
                case .integer(let integer): draft[keyPath: keyPath] = integer
                case .number(let number): draft[keyPath: keyPath] = Int(number.rounded())
                default: break
                }
            }
        )
    }

    /// An optional number the draft leaves nil until the user overrides the CLI's own default.
    package static func optionalNumber(_ id: String, _ keyPath: WritableKeyPath<StudioDraft, Double?>) -> Self {
        Self(
            fieldID: id,
            read: { draft in
                guard let value = draft[keyPath: keyPath] else { return .unset }
                return .number(value)
            },
            write: { draft, value in
                switch value {
                case .unset: draft[keyPath: keyPath] = nil
                case .number(let number): draft[keyPath: keyPath] = number
                case .integer(let integer): draft[keyPath: keyPath] = Double(integer)
                default: break
                }
            }
        )
    }
}

// MARK: - Groups

/// The inspector's and the Command view's sections, in the order both columns show them. The
/// names are the contract's own `MereRunCapabilityOptionGroup` constants; a capability that files
/// an option under any other name lands in Options.
package enum StudioContractGroup: String, CaseIterable, Identifiable, Hashable {
    case prompt = "Prompt"
    case inputs = "Inputs"
    case output = "Output"
    case model = "Model & adapters"
    case sampling = "Sampling"
    case run = "Run"
    case options = "Options"

    package var id: String { rawValue }

    package var title: String { rawValue }

    package init(contractGroup: String?) {
        self = StudioContractGroup(rawValue: contractGroup ?? "") ?? .options
    }

    /// The group a capability files `flag` under. A flag the capability does not declare — one the
    /// app emits that the contract has yet to describe — lands in Options.
    package static func group(forFlag flag: String, in capability: MereRunCommandCapability?) -> StudioContractGroup {
        guard let option = capability?.options.first(where: { $0.flag == flag }) else { return .options }
        return StudioContractGroup(contractGroup: option.group)
    }
}

/// User-language names for the contract's raw choice values. Anything not listed falls back to
/// the raw value with its dashes opened up, so a new choice reads sensibly the day it lands.
package enum StudioContractChoiceTitles {
    package static func title(for choice: String, flag: String) -> String {
        if let title = table["\(flag) \(choice)"] { return title }
        let opened = choice.replacingOccurrences(of: "-", with: " ")
        return opened.prefix(1).uppercased() + opened.dropFirst()
    }

    private static let table: [String: String] = [
        "--mode style": "Preset voice",
        "--mode clone": "Cloned voice",
        "--response-format text": "Text",
        "--response-format json_object": "JSON",
        "--output-mode video-only": "Video",
        "--output-mode audio-video": "Audio + Video",
        "--backend auto": "Auto",
        "--kv-quant-scheme uniform": "Uniform",
        "--kv-quant-scheme polar": "Polar",
        "--kv-quant-scheme turboquant": "Turbo",
        "--krea-base-quantization-bits 4": "4-bit",
        "--krea-base-quantization-bits 8": "8-bit",
        "--h3-weight-mode auto": "Auto",
        "--h3-weight-mode quantized": "Quantized",
        "--h3-weight-mode resident-bf16": "BF16",
        "--h3-acceleration quality": "Exact",
        "--h3-acceleration balanced": "Balanced",
        "--h3-acceleration maximum": "Maximum",
        "--adapter-kind auto": "Auto",
        "--adapter-kind lora": "LoRA",
        "--adapter-kind lokr": "LoKr",
        "--export-format pcm16": "PCM 16",
        "--export-format pcm24": "PCM 24",
        "--export-format float32": "Float 32",
        "--task-type text2music": "Create",
        "--task-type cover-nofsq": "No-FSQ",
    ]
}

// MARK: - Fields and sections

/// The control a contract option is rendered as.
package enum StudioContractControl: Equatable {
    /// A single-line text field: `string`.
    case field
    /// A checkbox: `boolean`.
    case toggle
    /// A segmented control: `choice` with up to four choices.
    case segmented
    /// A pop-up: `choice` with more than four.
    case picker
    /// A file or directory well with Choose and a clear button.
    case path
    /// `integer` or `number` whose contract range has both ends.
    case slider
    /// `integer` or `number` whose contract range is open, or absent.
    case stepper
    /// A composite editor the app draws itself.
    case override
}

/// One row of a contract-driven form: the option that anchors it (its kind, range, choices,
/// group, and tier), the draft fields it writes, and the composite editor that draws it when the
/// contract cannot describe the control.
///
/// A plain option writes exactly one draft field. A composite editor — the aspect pair, the mask
/// canvas, seconds-or-frames — writes several, and every one of them counts toward the modified
/// badge and is restored by Reset, so the row behaves as one control either way.
package struct StudioContractField<Draft>: Identifiable {
    package let option: MereRunCapabilityOption
    package let bindings: [StudioContractBinding<Draft>]
    /// nil renders from the contract; otherwise the caller's override builder draws this row.
    package var overrideID: StudioContractOverrideID?

    package init(
        option: MereRunCapabilityOption,
        bindings: [StudioContractBinding<Draft>],
        overrideID: StudioContractOverrideID? = nil
    ) {
        precondition(!bindings.isEmpty, "\(option.flag) needs at least one draft binding to render")
        self.option = option
        self.bindings = bindings
        self.overrideID = overrideID
    }

    /// The binding the contract-rendered control reads and writes.
    package var binding: StudioContractBinding<Draft> { bindings[0] }

    package var draftFieldIDs: [String] { bindings.map(\.fieldID) }

    package var id: String { option.flag }
    package var flag: String { option.flag }
    package var label: String { option.label }
    package var kind: MereRunCapabilityValueKind { option.kind }
    package var group: StudioContractGroup { StudioContractGroup(contractGroup: option.group) }
    package var tier: MereRunCapabilityOptionTier { option.tier ?? .standard }

    /// Which control the option's `kind` and `range` call for. The form switches on this and the
    /// tests assert it, so the mapping is one decision in one place.
    package var control: StudioContractControl {
        if overrideID != nil { return .override }
        switch option.kind {
        case .boolean:
            return .toggle
        case .choice:
            return option.choices.count <= 4 && !option.choices.isEmpty ? .segmented : .picker
        case .file, .directory:
            return .path
        case .string:
            return .field
        case .integer, .number:
            guard let range = option.range, let minimum = range.min, let maximum = range.max,
                  maximum > minimum else { return .stepper }
            return .slider
        }
    }

    /// The contract's declared default, read as the value the control would sit at.
    package var defaultValue: StudioContractValue {
        guard let raw = option.defaultValue else { return .unset }
        switch option.kind {
        case .integer:
            return Int(raw).map { .integer($0) } ?? .text(raw)
        case .number:
            return Double(raw).map { .number($0) } ?? .text(raw)
        case .boolean:
            return .flag(raw == "true")
        case .string, .file, .directory, .choice:
            return .text(raw)
        }
    }

    package func value(in draft: Draft) -> StudioContractValue {
        binding.read(draft)
    }

    /// Whether the control sits at the contract's declared default (or at nothing at all). A
    /// control at its default emits no flag.
    package func isAtDefault(in draft: Draft) -> Bool {
        let value = binding.read(draft)
        if case .unset = value { return true }
        if case .flag(let on) = value { return !on }
        if case .text(let text) = value, text.isEmpty { return true }
        guard option.defaultValue != nil else {
            // A number the contract declares no default for reads as "let the CLI pick" at zero,
            // and below the declared minimum: `--kv-bits 0` and `--candidates 0` are the app's way
            // of saying it has no opinion, and the argv leaves them out.
            guard let raw = value.numericValue else { return false }
            if raw == 0 { return true }
            if let minimum = option.range?.min, raw < minimum { return true }
            return false
        }
        if value == defaultValue { return true }
        // "1.0" and "1" are the same default; compare numerically when both sides are numbers.
        if let lhs = value.numericValue, let rhs = defaultValue.numericValue { return lhs == rhs }
        return false
    }

    /// Whether the draft gives this option a value the command line would carry.
    package func emits(in draft: Draft) -> Bool {
        !isAtDefault(in: draft)
    }

    /// The value clamped into the contract's declared range, and snapped to its step.
    package func clamped(_ value: StudioContractValue) -> StudioContractValue {
        guard let range = option.range, let raw = value.numericValue else { return value }
        var clamped = raw
        if let step = range.step, step > 0 {
            let origin = range.min ?? 0
            clamped = origin + ((clamped - origin) / step).rounded() * step
        }
        if let minimum = range.min { clamped = max(clamped, minimum) }
        if let maximum = range.max { clamped = min(clamped, maximum) }
        switch value {
        case .integer: return .integer(Int(clamped.rounded()))
        case .number: return .number(clamped)
        default: return value
        }
    }

    package func write(_ value: StudioContractValue, to draft: inout Draft) {
        binding.write(&draft, clamped(value))
    }

    /// How many of the draft fields this row owns differ from the mode's defaults.
    package func changedCount(draft: Draft, baseline: Draft) -> Int {
        bindings.filter { $0.isChanged(draft, baseline) }.count
    }

    package func reset(_ draft: inout Draft, to baseline: Draft) {
        for binding in bindings { binding.reset(&draft, to: baseline) }
    }
}

/// One group of contract fields, in the order the contract declares them.
package struct StudioContractSection: Identifiable {
    package let group: StudioContractGroup
    package let fields: [StudioContractField<StudioDraft>]

    package var id: StudioContractGroup { group }
    package var title: String { group.title }

    package func changedCount(draft: StudioDraft, baseline: StudioDraft) -> Int {
        fields.reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }

    package func reset(_ draft: inout StudioDraft, to baseline: StudioDraft) {
        for field in fields { field.reset(&draft, to: baseline) }
    }
}

// MARK: - Schema

package enum StudioContractSchema {
    /// The capability a mode's current draft runs, which for Read Image depends on the action.
    package static func capability(for mode: StudioMode, draft: StudioDraft) -> MereRunCommandCapability? {
        capability(for: mode, readImageAction: draft.readImageAction)
    }

    package static func capability(
        for mode: StudioMode,
        readImageAction: StudioReadImageAction = .inspect
    ) -> MereRunCommandCapability? {
        let templateID = mode == .readImage ? readImageAction.templateID : mode.defaultTemplateID
        guard let capabilityID = templateID.capabilityID else { return nil }
        return MereRunCapabilityCatalog.command(id: capabilityID)
    }

    /// Every option of the mode's capability the app has a draft field for, in contract order.
    /// Options the contract declares that the Studio draft does not carry are left out: the app
    /// has no state to bind them to, so a control for them could not change the command.
    package static func fields(for mode: StudioMode, draft: StudioDraft) -> [StudioContractField<StudioDraft>] {
        fields(for: mode, readImageAction: draft.readImageAction)
    }

    package static func fields(
        for mode: StudioMode,
        readImageAction: StudioReadImageAction = .inspect
    ) -> [StudioContractField<StudioDraft>] {
        guard let capability = capability(for: mode, readImageAction: readImageAction) else { return [] }
        let bindings = StudioContractBindings.bindings(for: mode)
        var claimed: Set<StudioContractOverrideID> = []
        var fields: [StudioContractField<StudioDraft>] = []
        for option in capability.options {
            guard let override = StudioContractOverrides.override(forFlag: option.flag, mode: mode) else {
                guard let binding = bindings[option.flag] else { continue }
                fields.append(StudioContractField(option: option, bindings: [binding]))
                continue
            }
            // A composite editor renders once, where the first of its flags is declared, and owns
            // every draft field behind it — the other flags' bindings plus the ones with no flag
            // of their own (seconds-or-frames, the outpaint edges).
            guard claimed.insert(override.id).inserted else { continue }
            let owned = override.flags.compactMap { bindings[$0] } + override.companions
            guard !owned.isEmpty else { continue }
            fields.append(StudioContractField(option: option, bindings: owned, overrideID: override.id))
        }
        return uncoveredFields(for: mode) + fields
    }

    /// The fields the inspector edits itself: everything except the ones the composer's prompt and
    /// attachment well already own.
    package static func inspectorFields(for mode: StudioMode, draft: StudioDraft) -> [StudioContractField<StudioDraft>] {
        fields(for: mode, draft: draft).filter { field in
            StudioContractOverrides.override(forFlag: field.flag, mode: mode)?.isExternal != true
        }
    }

    /// Controls the contract has no option for yet, declared here in the contract's own shape so
    /// the form renders them the same way. Read Image's task picker is the only one: it chooses
    /// which capability runs rather than an argument of one.
    package static func uncoveredFields(for mode: StudioMode) -> [StudioContractField<StudioDraft>] {
        guard mode == .readImage else { return [] }
        return [
            StudioContractField(
                option: .init(
                    flag: "read-image-action",
                    label: "Read task",
                    kind: .choice,
                    choices: StudioReadImageAction.allCases.map(\.rawValue),
                    defaultValue: StudioReadImageAction.inspect.rawValue,
                    group: MereRunCapabilityOptionGroup.prompt,
                    tier: .essential
                ),
                bindings: [.choice("readImageAction", \.readImageAction)],
                overrideID: .readImageAction
            )
        ]
    }

    /// The inspector's sections: the mode's `essential` and `standard` fields grouped by the
    /// contract's group, in group order. `expert` fields collapse under Advanced instead.
    ///
    /// Essentials appear here as well as on the composer's chips on purpose: the chip strip is the
    /// two-second edit and the inspector is the considered one, and both bind the same draft, so a
    /// change in either shows in the other.
    package static func sections(for mode: StudioMode, draft: StudioDraft) -> [StudioContractSection] {
        let fields = inspectorFields(for: mode, draft: draft).filter { $0.tier != .expert }
        return StudioContractGroup.allCases.compactMap { group in
            let grouped = fields.filter { $0.group == group }
            guard !grouped.isEmpty else { return nil }
            return StudioContractSection(group: group, fields: grouped)
        }
    }

    /// Everything the mode's command takes that the inspector collapses under "Advanced · N more".
    package static func expertFields(for mode: StudioMode, draft: StudioDraft) -> [StudioContractField<StudioDraft>] {
        inspectorFields(for: mode, draft: draft).filter { $0.tier == .expert }
    }

    /// Whether `field` is reachable: every option it declares a dependency on must carry a value.
    /// A dependency the app does not bind (an attachment the well owns, say) is read from the
    /// draft all the same, so an editor stays hidden until its input exists.
    package static func isVisible(_ field: StudioContractField<StudioDraft>, for mode: StudioMode, in draft: StudioDraft) -> Bool {
        isVisible(field, in: draft, dependencies: dependencies(for: mode, draft: draft))
    }

    /// One entry per option of the mode's capability the app can read a value for: whether the
    /// draft gives it a value, and what it in turn depends on. Every flag is answered, including
    /// the ones a composite editor or the composer's well owns, so a row gated on an attachment
    /// still knows whether the attachment is there.
    package static func dependencies(for mode: StudioMode, draft: StudioDraft) -> [String: (carries: Bool, dependsOn: String?)] {
        var entries: [String: (carries: Bool, dependsOn: String?)] = [:]
        for field in boundFields(for: mode, draft: draft) {
            entries[field.flag] = (field.emits(in: draft), field.option.dependsOn)
        }
        return entries
    }

    /// One field per bound option, before the composite editors fold their flags together. This is
    /// the per-flag view the dependency walk and the Command view need; `fields(for:)` is the
    /// per-row view the forms render.
    package static func boundFields(for mode: StudioMode, draft: StudioDraft = StudioDraft()) -> [StudioContractField<StudioDraft>] {
        guard let capability = capability(for: mode, draft: draft) else { return [] }
        let bindings = StudioContractBindings.bindings(for: mode)
        return capability.options.compactMap { option in
            guard let binding = bindings[option.flag] else { return nil }
            return StudioContractField(option: option, bindings: [binding])
        }
    }

    package static func isVisible<Draft>(
        _ field: StudioContractField<Draft>,
        in draft: Draft,
        dependencies: [String: (carries: Bool, dependsOn: String?)]
    ) -> Bool {
        var flag = field.option.dependsOn
        var seen: Set<String> = [field.flag]
        while let dependency = flag, seen.insert(dependency).inserted {
            // A dependency the app has no state for cannot hide anything.
            guard let entry = dependencies[dependency] else { return true }
            guard entry.carries else { return false }
            flag = entry.dependsOn
        }
        return true
    }

    /// Reads every bound field out of `draft` and writes it straight back. The identity this
    /// establishes is what lets the contract-driven form share the argv builder: a control that
    /// round-trips its own value cannot change the command.
    package static func roundTrip(_ draft: inout StudioDraft, for mode: StudioMode) {
        for binding in fields(for: mode, draft: draft).flatMap(\.bindings) {
            binding.write(&draft, binding.read(draft))
        }
    }

    /// Every draft field the mode's inspector binds, by draft property name.
    package static func draftFieldIDs(for mode: StudioMode, draft: StudioDraft = StudioDraft()) -> Set<String> {
        Set(inspectorFields(for: mode, draft: draft).flatMap(\.draftFieldIDs))
    }

    /// How many of the mode's inspector fields differ from its defaults; the header badge.
    package static func changedCount(mode: StudioMode, draft: StudioDraft, baseline: StudioDraft) -> Int {
        inspectorFields(for: mode, draft: draft)
            .reduce(0) { $0 + $1.changedCount(draft: draft, baseline: baseline) }
    }
}

// MARK: - Overrides

/// A composite editor the app draws itself, because one control spans several flags or because
/// the flag's editor lives outside the inspector.
package enum StudioContractOverrideID: String, CaseIterable, Hashable {
    /// The filtered model picker with its readiness glyph.
    case model
    /// The aspect presets, the width × height pair, and Swap.
    case dimensions
    /// Random / a typed seed / Reuse last.
    case seed
    /// Steps, over the range the mode's models actually use rather than the CLI's whole span:
    /// Music gates it behind "Override preset steps" and MiniMax-H3 stores it separately.
    case steps
    /// Guidance, likewise over the useful span rather than everything the CLI parses.
    case guidance
    /// Seconds or frames, which the draft splits across `useDuration`.
    case duration
    /// The saved voice profiles, which the app reads from the CLI rather than the contract.
    case voiceProfile
    /// The LoRA row: a path and its scale on one line.
    case lora
    /// The mask brush and outpaint edges, edited on the image itself.
    case imageCanvas
    /// The ACE-Step adapter rows and their scales.
    case musicAdapters
    /// Preset / On / Off over the CLI's `--use-lm` and `--no-lm` pair.
    case musicLMMode
    /// Auto / Show / Off over the CLI's `--thinking` and `--no-thinking` pair.
    case thinking
    /// The ordered MiniMax-H3 Ref2VA reference list.
    case orderedReferences
    /// Inspect / OCR / Caption, which picks the capability rather than one of its arguments.
    case readImageAction
    /// An attachment the composer's well owns; the inspector never repeats it.
    case attachment
}

/// One composite editor: the flags it owns, and any draft field behind it that has no flag of its
/// own (seconds-or-frames, the outpaint edges, "override the preset steps").
package struct StudioContractOverride {
    package let id: StudioContractOverrideID
    /// Every flag the editor writes. The editor renders where the first of them is declared.
    package let flags: [String]
    /// Draft fields the editor owns that the contract has no option for.
    package var companions: [StudioContractBinding<StudioDraft>] = []
    /// The editor lives in the composer, not the inspector.
    package var isExternal = false
}

package enum StudioContractOverrides {
    package static func overrides(for mode: StudioMode) -> [StudioContractOverride] {
        var overrides: [StudioContractOverride] = [
            StudioContractOverride(id: .model, flags: ["--model"]),
        ]
        switch mode {
        case .createImage:
            overrides += [
                StudioContractOverride(id: .dimensions, flags: ["--width", "--height"]),
                StudioContractOverride(id: .seed, flags: ["--seed"]),
                StudioContractOverride(id: .steps, flags: ["--steps"]),
                StudioContractOverride(id: .guidance, flags: ["--cfg"]),
                StudioContractOverride(id: .lora, flags: ["--lora", "--lora-scale"]),
                StudioContractOverride(
                    id: .imageCanvas,
                    flags: ["--mask", "--outpaint", "--mask-feather"],
                    companions: [
                        .integer("imageOutpaintTop", \.imageOutpaintTop),
                        .integer("imageOutpaintRight", \.imageOutpaintRight),
                        .integer("imageOutpaintBottom", \.imageOutpaintBottom),
                        .integer("imageOutpaintLeft", \.imageOutpaintLeft),
                    ]
                ),
                StudioContractOverride(id: .attachment, flags: ["--input", "--ref-image"], isExternal: true),
            ]
        case .video:
            overrides += [
                StudioContractOverride(id: .dimensions, flags: ["--width", "--height"]),
                StudioContractOverride(id: .seed, flags: ["--seed"]),
                StudioContractOverride(
                    id: .steps, flags: ["--steps"],
                    companions: [.optionalInteger("h3Steps", \.h3Steps)]
                ),
                StudioContractOverride(id: .guidance, flags: ["--guidance-scale"]),
                StudioContractOverride(
                    id: .duration, flags: ["--duration", "--num-frames"],
                    companions: [.flag("useDuration", \.useDuration)]
                ),
                StudioContractOverride(id: .orderedReferences, flags: ["--reference"]),
                StudioContractOverride(id: .attachment, flags: ["--image", "--end-image", "--audio"], isExternal: true),
            ]
        case .music:
            overrides += [
                StudioContractOverride(id: .seed, flags: ["--seed"]),
                StudioContractOverride(
                    id: .duration, flags: ["--duration"],
                    companions: [.flag("useDuration", \.useDuration)]
                ),
                StudioContractOverride(
                    id: .steps, flags: ["--steps"],
                    companions: [.flag("musicOverrideSteps", \.musicOverrideSteps)]
                ),
                StudioContractOverride(id: .musicAdapters, flags: ["--adapter", "--adapter-scale"]),
                StudioContractOverride(
                    id: .musicLMMode, flags: ["--use-lm", "--no-lm"],
                    companions: [.text("musicLMMode", \.musicLMMode)]
                ),
                StudioContractOverride(
                    id: .attachment, flags: ["--source-audio", "--reference-audio"], isExternal: true
                ),
            ]
        case .sfx:
            overrides += [
                StudioContractOverride(id: .seed, flags: ["--seed"]),
                StudioContractOverride(id: .steps, flags: ["--steps"]),
            ]
        case .speak:
            overrides += [
                StudioContractOverride(id: .voiceProfile, flags: ["--profile"]),
                StudioContractOverride(id: .attachment, flags: ["--ref-audio"], isExternal: true),
            ]
        case .chat:
            overrides += [
                StudioContractOverride(id: .lora, flags: ["--lora", "--lora-scale"]),
                StudioContractOverride(
                    id: .thinking, flags: ["--thinking", "--no-thinking"],
                    companions: [.choice("thinkingMode", \.thinkingMode)]
                ),
                StudioContractOverride(id: .attachment, flags: ["--image"], isExternal: true),
            ]
        case .code, .listen, .readImage, .findObjects, .segment, .track:
            break
        }
        return overrides
    }

    package static func override(forFlag flag: String, mode: StudioMode) -> StudioContractOverride? {
        overrides(for: mode).first { $0.flags.contains(flag) }
    }
}

// MARK: - The binding table

/// Which `StudioDraft` field each contract flag reads and writes, per prompt mode. Only the flags
/// `StudioCommandAdapter` actually forwards are listed: a control for a flag the adapter drops
/// would look live and change nothing.
package enum StudioContractBindings {
    package static func bindings(for mode: StudioMode) -> [String: StudioContractBinding<StudioDraft>] {
        switch mode {
        case .createImage: return image
        case .video: return video
        case .music: return music
        case .sfx: return sfx
        case .speak: return speak
        case .listen: return listen
        case .chat: return chat
        case .code: return code
        case .readImage: return readImage
        case .findObjects: return findObjects
        case .segment, .track: return segmentAndTrack
        }
    }

    private static var model: [String: StudioContractBinding<StudioDraft>] { ["--model": .text("model", \.model)] }

    private static var image: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--negative-prompt": .text("secondaryText", \.secondaryText),
        "--input": .text("inputPath", \.inputPath),
        "--ref-image": .text("referenceImagePaths", \.referenceImagePaths),
        "--width": .integer("width", \.width),
        "--height": .integer("height", \.height),
        "--steps": .integer("steps", \.steps),
        "--seed": .integerText("seed", \.seed),
        "--cfg": .number("cfgScale", \.cfgScale),
        "--strength": .number("strength", \.strength),
        "--sigma-shift": .number("sigmaShift", \.sigmaShift),
        "--mask": .text("imageMaskPath", \.imageMaskPath),
        "--mask-feather": .integer("imageMaskFeather", \.imageMaskFeather),
        "--keep-original-aspect": .flag("keepOriginalAspect", \.keepOriginalAspect),
        "--structured-prompt": .flag("structuredPrompt", \.structuredPrompt),
        "--structured-prompt-model": .text("structuredPromptModel", \.structuredPromptModel),
        "--structured-prompt-max-tokens": .integer("structuredPromptMaxTokens", \.structuredPromptMaxTokens),
        "--max-sequence-length": .integer("imageMaxSequenceLength", \.imageMaxSequenceLength),
        "--lora": .text("loraPath", \.loraPath),
        "--lora-scale": .number("loraScale", \.loraScale),
        "--krea-conditioning-multiplier": .number("kreaConditioningMultiplier", \.kreaConditioningMultiplier),
        "--krea-conditioning-layer-weights": .text("kreaConditioningLayerWeights", \.kreaConditioningLayerWeights),
        "--krea-base-quantization-bits": .text("kreaBaseQuantizationBits", \.kreaBaseQuantizationBits),
        "--preflight": .flag("preflight", \.preflight),
        "--json": .flag("preflightJSON", \.preflightJSON),
        "--progress-json": .flag("progressJSON", \.progressJSON),
    ]) { first, _ in first } }

    private static var video: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--negative-prompt": .text("secondaryText", \.secondaryText),
        "--image": .text("inputPath", \.inputPath),
        "--end-image": .text("endImagePath", \.endImagePath),
        "--audio": .text("audioPath", \.audioPath),
        "--reference": .text("h3ReferenceInputs", \.h3ReferenceInputsText),
        "--width": .integer("width", \.width),
        "--height": .integer("height", \.height),
        "--num-frames": .integer("numFrames", \.numFrames),
        "--duration": .number("durationSeconds", \.durationSeconds),
        "--fps": .integer("fps", \.fps),
        "--steps": .integer("steps", \.steps),
        "--seed": .integerText("seed", \.seed),
        "--guidance-scale": .number("cfgScale", \.cfgScale),
        "--shift": .number("scheduleShift", \.scheduleShift),
        "--quality": .choice("videoQuality", \.videoQuality),
        "--output-mode": .choice("videoOutputMode", \.videoOutputMode),
        "--image-strength": .number("strength", \.strength),
        "--end-image-strength": .number("endImageStrength", \.endImageStrength),
        "--audio-start-time": .number("audioStartTime", \.audioStartTime),
        "--audio-max-duration": .number("audioMaxDuration", \.audioMaxDuration),
        "--a2v-guidance-scale": .number("a2vGuidanceScale", \.a2vGuidanceScale),
        "--video-cfg-guidance-scale": .number("videoCFGGuidanceScale", \.videoCFGGuidanceScale),
        "--audio-cfg-guidance-scale": .number("audioCFGGuidanceScale", \.audioCFGGuidanceScale),
        "--v2a-guidance-scale": .number("v2aGuidanceScale", \.v2aGuidanceScale),
        "--a2v-steps": .integer("a2vSteps", \.a2vSteps),
        "--h3-weight-mode": .optionalText("h3WeightMode", \.h3WeightMode),
        "--h3-acceleration": .optionalText("h3AccelerationMode", \.h3AccelerationMode),
        "--preflight": .flag("preflight", \.preflight),
        "--timings": .flag("timings", \.timings),
        "--timings-output": .text("timingsOutputPath", \.timingsOutputPath),
    ]) { first, _ in first } }

    private static var music: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--lyrics": .text("secondaryText", \.secondaryText),
        "--source-audio": .text("musicSourceAudio", \.musicSourceAudio),
        "--reference-audio": .text("musicReferenceAudioPaths", \.musicReferenceAudioPaths),
        "--lrc-file": .text("musicLRCFile", \.musicLRCFile),
        "--duration": .number("durationSeconds", \.durationSeconds),
        "--quality": .text("musicQuality", \.musicQuality),
        "--steps": .integer("steps", \.steps),
        "--seed": .integerText("seed", \.seed),
        "--task-type": .text("musicTask", \.musicTask),
        "--instrumental": .flag("musicInstrumental", \.musicInstrumental),
        "--analyze-source-audio": .flag("musicAnalyzeSourceAudio", \.musicAnalyzeSourceAudio),
        "--lm-temperature": .number("musicLMTemperature", \.musicLMTemperature),
        "--lm-repetition-penalty": .number("musicLMRepetitionPenalty", \.musicLMRepetitionPenalty),
        "--lm-cfg-scale": .number("musicLMCFGScale", \.musicLMCFGScale),
        "--lm-negative-prompt": .text("musicLMNegativePrompt", \.musicLMNegativePrompt),
        "--candidates": .integer("musicCandidates", \.musicCandidates),
        "--keep-candidates": .flag("musicKeepCandidates", \.musicKeepCandidates),
        "--audio-cover-strength": .number("musicCoverStrength", \.musicCoverStrength),
        "--cover-noise-strength": .number("musicCoverNoiseStrength", \.musicCoverNoiseStrength),
        "--retake-seed": .integerText("musicRetakeSeed", \.musicRetakeSeed),
        "--retake-variance": .number("musicRetakeVariance", \.musicRetakeVariance),
        "--repaint-start": .number("musicRepaintStart", \.musicRepaintStart),
        "--repaint-end": .number("musicRepaintEnd", \.musicRepaintEnd),
        "--repaint-mode": .text("musicRepaintMode", \.musicRepaintMode),
        "--repaint-strength": .number("musicRepaintStrength", \.musicRepaintStrength),
        "--flow-edit": .flag("musicFlowEdit", \.musicFlowEdit),
        "--source-caption": .text("musicSourceCaption", \.musicSourceCaption),
        "--source-lyrics": .text("musicSourceLyrics", \.musicSourceLyrics),
        "--adapter": .text("musicAdapterPaths", \.musicAdapterPaths),
        "--adapter-kind": .text("musicAdapterKind", \.musicAdapterKind),
        "--adapter-scale": .text("musicAdapterScales", \.musicAdapterScales),
        "--stems": .text("musicStems", \.musicStems),
        "--daw-bundle": .text("musicDAWBundle", \.musicDAWBundle),
        "--export-format": .text("musicExportFormat", \.musicExportFormat),
        "--no-recipe": .flag("musicNoRecipe", \.musicNoRecipe),
    ]) { first, _ in first } }

    private static var sfx: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--duration": .number("durationSeconds", \.durationSeconds),
        "--steps": .integer("steps", \.steps),
        "--seed": .integerText("seed", \.seed),
    ]) { first, _ in first } }

    private static var speak: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--voice": .text("secondaryText", \.secondaryText),
        "--mode": .text("voiceMode", \.voiceMode),
        "--profile": .text("voiceProfile", \.voiceProfile),
        "--ref-audio": .text("refAudioPath", \.refAudioPath),
        "--save-profile": .text("saveProfileName", \.saveProfileName),
    ]) { first, _ in first } }

    private static var listen: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--backend": .text("backend", \.backend),
        "--language": .text("language", \.language),
        "--no-timestamps": .invertedFlag("timestamps", \.timestamps),
    ]) { first, _ in first } }

    private static var chat: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--system": .text("secondaryText", \.secondaryText),
        "--image": .text("inputPath", \.inputPath),
        "--temperature": .number("temperature", \.temperature),
        "--top-p": .number("topP", \.topP),
        "--min-p": .number("minP", \.minP),
        "--max-tokens": .integer("maxTokens", \.maxTokens),
        "--context-size": .integer("contextSize", \.contextSize),
        "--top-k": .integer("topK", \.topK),
        "--kv-bits": .integer("kvBits", \.kvBits),
        "--kv-quant-scheme": .text("kvQuantScheme", \.kvQuantScheme),
        "--kv-group-size": .integer("kvGroupSize", \.kvGroupSize),
        "--quantized-kv-start": .integer("quantizedKVStart", \.quantizedKVStart),
        "--response-format": .choice("responseFormat", \.responseFormat),
        "--reasoning-effort": .optionalNumber("reasoningEffort", \.reasoningEffort),
        "--lora": .text("loraPath", \.loraPath),
        "--lora-scale": .number("loraScale", \.loraScale),
        "--stats": .flag("stats", \.stats),
        "--tools": .text("tools", \.tools),
        "--tool-loop": .flag("toolLoop", \.toolLoop),
        "--sandbox-dir": .text("sandboxDir", \.sandboxDir),
        "--allow-shell-exec": .flag("allowShellExec", \.allowShellExec),
        "--allow-absolute-tool-paths": .flag("allowAbsoluteToolPaths", \.allowAbsoluteToolPaths),
        "--auto-approve-tools": .flag("autoApproveTools", \.autoApproveTools),
        "--preflight": .flag("preflight", \.preflight),
        "--json": .flag("preflightJSON", \.preflightJSON),
        "--require-installed": .flag("requireInstalled", \.requireInstalled),
    ]) { first, _ in first } }

    private static var code: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--system": .text("secondaryText", \.secondaryText),
        "--temperature": .number("temperature", \.temperature),
        "--top-p": .number("topP", \.topP),
        "--min-p": .number("minP", \.minP),
        "--max-tokens": .integer("maxTokens", \.maxTokens),
    ]) { first, _ in first } }

    private static var readImage: [String: StudioContractBinding<StudioDraft>] { model }

    private static var findObjects: [String: StudioContractBinding<StudioDraft>] { model }

    private static var segmentAndTrack: [String: StudioContractBinding<StudioDraft>] { model.merging([
        "--threshold": .number("visionThreshold", \.visionThreshold),
    ]) { first, _ in first } }
}

// MARK: - Draft shims

extension StudioDraft {
    /// The MiniMax-H3 reference list as the ordered-editor override reads and writes it: one
    /// `kind:path` entry per line, so a list-valued flag fits the same string binding as the rest.
    package var h3ReferenceInputsText: String {
        get { (h3ReferenceInputs ?? []).joined(separator: "\n") }
        set {
            let entries = newValue.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            h3ReferenceInputs = entries.isEmpty ? nil : entries
        }
    }
}
