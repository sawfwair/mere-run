import AppKit
import MereRunContract
import UniformTypeIdentifiers

// The composer's declarative surface: which attachment slots and which parameter chips each
// prompt mode shows, and how each maps onto `StudioDraft`. The views in `StudioComposer.swift`
// render from these declarations, so a mode's essentials live in one place and are testable
// without SwiftUI.

// MARK: - Attachment slots

/// A draft an attachment well can write into: the prompt tasks' `StudioDraft` (a typed field per
/// slot) and the specialist tasks' `StudioTaskDraft` (a contract flag or positional per slot).
/// The well, the canvas drop target, and the paste handler are written once over this.
package protocol StudioAttachmentDraft {
    /// What the slot's storage holds: one path, or newline-separated paths for a list slot.
    func attachmentText(for storage: StudioAttachmentSlot.Storage) -> String
    mutating func setAttachmentText(_ text: String, for storage: StudioAttachmentSlot.Storage)
    /// Settings that follow an attachment so the slot is never silently ignored (a cloned voice
    /// needs clone mode). Most drafts have none.
    mutating func didAttach(to slot: StudioAttachmentSlot)
}

/// One slot in the composer's attachment well, bound to a draft field, a contract flag, or a
/// positional argument.
package struct StudioAttachmentSlot: Identifiable, Equatable {
    package enum Storage: Equatable {
        /// One path in one draft field; attaching replaces it.
        case path(WritableKeyPath<StudioDraft, String>)
        /// Newline-separated paths in one draft field; attaching appends, the slot previews the first.
        case pathList(WritableKeyPath<StudioDraft, String>)
        /// One path in a contract option of a `StudioTaskDraft` (`--cameras`, `--second`).
        case flag(String)
        /// A repeatable file option, one path per line (`--view`).
        case flagList(String)
        /// One positional argument, by the index the contract declares it at.
        case argument(Int)
        /// A repeatable positional: every argument from `index` on, in order (`images`).
        case argumentList(Int)
    }

    package let id: String
    /// The empty slot's caption ("Reference"); a filled slot shows its file name instead.
    package let label: String
    package let acceptedTypes: [UTType]
    package let storage: Storage
    /// The task cannot run without this slot filled.
    package var isRequired = false
    /// A per-turn attachment (Chat's image): the well stays collapsed to the paperclip until
    /// something is attached, so an empty slot never sits above every message.
    package var isTransient = false

    package init(
        id: String,
        label: String,
        acceptedTypes: [UTType],
        storage: Storage,
        isRequired: Bool = false,
        isTransient: Bool = false
    ) {
        self.id = id
        self.label = label
        self.acceptedTypes = acceptedTypes
        self.storage = storage
        self.isRequired = isRequired
        self.isTransient = isTransient
    }

    package var allowsMultiple: Bool {
        switch storage {
        case .pathList, .flagList, .argumentList: return true
        case .path, .flag, .argument: return false
        }
    }

    /// The paths this slot currently holds, in order.
    package func paths<Draft: StudioAttachmentDraft>(in draft: Draft) -> [String] {
        let text = draft.attachmentText(for: storage)
        guard allowsMultiple else {
            let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? [] : [value]
        }
        return Self.separatedPaths(text)
    }

    package func isFilled<Draft: StudioAttachmentDraft>(in draft: Draft) -> Bool {
        !paths(in: draft).isEmpty
    }

    /// The caption shown beside the slot: the file name when filled, the slot label otherwise.
    package func caption<Draft: StudioAttachmentDraft>(in draft: Draft) -> String {
        let paths = paths(in: draft)
        guard let first = paths.first else { return label }
        let name = URL(fileURLWithPath: first).lastPathComponent
        return paths.count > 1 ? "\(name) +\(paths.count - 1)" : name
    }

    /// Whether a dropped or pasted file belongs in this slot. A folder slot takes any directory;
    /// a file slot with no declared types takes any file.
    package func accepts(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        if acceptedTypes.contains(.folder) {
            return (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
        return acceptedTypes.contains { type.conforms(to: $0) }
    }

    /// Stores `urls` in the slot: a single-path slot keeps the first, a list slot appends them all.
    package func attach<Draft: StudioAttachmentDraft>(_ urls: [URL], to draft: inout Draft) {
        let incoming = urls.filter(accepts).map(\.path)
        guard !incoming.isEmpty else { return }
        if allowsMultiple {
            let existing = Self.separatedPaths(draft.attachmentText(for: storage))
            draft.setAttachmentText(
                (existing + incoming.filter { !existing.contains($0) }).joined(separator: "\n"),
                for: storage
            )
        } else {
            draft.setAttachmentText(incoming[0], for: storage)
        }
        draft.didAttach(to: self)
    }

    package func clear<Draft: StudioAttachmentDraft>(in draft: inout Draft) {
        draft.setAttachmentText("", for: storage)
    }

    package static func separatedPaths(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

extension Array where Element == StudioAttachmentSlot {
    /// The slot a file dropped on the canvas or pasted with ⌘V lands in: the first empty slot
    /// that accepts it, else the first slot that accepts it.
    package func slot<Draft: StudioAttachmentDraft>(for url: URL, in draft: Draft) -> StudioAttachmentSlot? {
        let accepting = filter { $0.accepts(url) }
        return accepting.first { !$0.isFilled(in: draft) } ?? accepting.first
    }

    /// The slot a pasted bitmap (no file on the pasteboard) lands in.
    package func pastedImageSlot<Draft: StudioAttachmentDraft>(in draft: Draft) -> StudioAttachmentSlot? {
        let accepting = filter { slot in slot.acceptedTypes.contains { UTType.image.conforms(to: $0) } }
        return accepting.first { !$0.isFilled(in: draft) } ?? accepting.first
    }
}

extension StudioAttachmentDraft {
    /// Routes each dropped file to the slot it belongs in. Returns whether anything was attached.
    @discardableResult
    package mutating func attach(dropped urls: [URL], slots: [StudioAttachmentSlot]) -> Bool {
        var attached = false
        for url in urls {
            guard let slot = slots.slot(for: url, in: self) else { continue }
            slot.attach([url], to: &self)
            attached = true
        }
        return attached
    }
}

extension StudioDraft: StudioAttachmentDraft {
    package func attachmentText(for storage: StudioAttachmentSlot.Storage) -> String {
        switch storage {
        case .path(let keyPath), .pathList(let keyPath):
            return self[keyPath: keyPath]
        case .flag, .flagList, .argument, .argumentList:
            // The prompt tasks bind typed fields, never contract flags.
            return ""
        }
    }

    package mutating func setAttachmentText(_ text: String, for storage: StudioAttachmentSlot.Storage) {
        switch storage {
        case .path(let keyPath) where keyPath == \StudioDraft.inputPath:
            // The input change clears the prompts drawn on the previous picture.
            replaceInput(text)
        case .path(let keyPath), .pathList(let keyPath):
            self[keyPath: keyPath] = text
        case .flag, .flagList, .argument, .argumentList:
            break
        }
    }
}

extension StudioMode {
    /// The attachment slots this mode's composer declares, in well order.
    package var attachmentSlots: [StudioAttachmentSlot] {
        switch self {
        case .createImage:
            return [
                StudioAttachmentSlot(id: "input", label: "Input", acceptedTypes: [.image], storage: .path(\.inputPath)),
                StudioAttachmentSlot(
                    id: "references", label: "Reference", acceptedTypes: [.image],
                    storage: .pathList(\.referenceImagePaths)
                ),
            ]
        case .video:
            return [
                StudioAttachmentSlot(id: "startFrame", label: "Start frame", acceptedTypes: [.image], storage: .path(\.inputPath)),
                StudioAttachmentSlot(id: "endFrame", label: "End frame", acceptedTypes: [.image], storage: .path(\.endImagePath)),
                StudioAttachmentSlot(id: "audio", label: "Audio", acceptedTypes: [.audio], storage: .path(\.audioPath)),
            ]
        case .music:
            return [
                StudioAttachmentSlot(id: "source", label: "Source", acceptedTypes: [.audio], storage: .path(\.musicSourceAudio)),
                StudioAttachmentSlot(
                    id: "timbre", label: "Timbre reference", acceptedTypes: [.audio],
                    storage: .pathList(\.musicReferenceAudioPaths)
                ),
            ]
        case .speak:
            return [
                StudioAttachmentSlot(id: "referenceAudio", label: "Reference audio", acceptedTypes: [.audio], storage: .path(\.refAudioPath)),
            ]
        case .chat:
            return [
                StudioAttachmentSlot(
                    id: "image", label: "Image", acceptedTypes: [.image], storage: .path(\.inputPath),
                    isTransient: true
                ),
            ]
        case .readImage, .findObjects, .segment:
            return [
                StudioAttachmentSlot(id: "input", label: "Image", acceptedTypes: [.image], storage: .path(\.inputPath), isRequired: true),
            ]
        case .track:
            return [
                StudioAttachmentSlot(
                    id: "input", label: "Video", acceptedTypes: [.movie, .video, .audiovisualContent],
                    storage: .path(\.inputPath), isRequired: true
                ),
            ]
        case .listen:
            return [
                StudioAttachmentSlot(id: "input", label: "Audio", acceptedTypes: [.audio], storage: .path(\.inputPath), isRequired: true),
            ]
        case .code, .sfx:
            return []
        }
    }

    /// Whether the composer shows the well: any non-transient slot, or a transient one that is filled.
    package func showsAttachmentWell(for draft: StudioDraft) -> Bool {
        attachmentSlots.contains { !$0.isTransient || $0.isFilled(in: draft) }
    }

    /// The slot a file dropped on the canvas or pasted with ⌘V lands in: the first empty slot
    /// that accepts it, else the first slot that accepts it.
    package func attachmentSlot(for url: URL, in draft: StudioDraft) -> StudioAttachmentSlot? {
        attachmentSlots.slot(for: url, in: draft)
    }

    /// The slot a pasted bitmap (no file on the pasteboard) lands in.
    package func pastedImageSlot(in draft: StudioDraft) -> StudioAttachmentSlot? {
        attachmentSlots.pastedImageSlot(in: draft)
    }
}

extension StudioDraft {
    /// Routes each dropped file to the slot it belongs in. Returns whether anything was attached.
    @discardableResult
    package mutating func attach(dropped urls: [URL], for mode: StudioMode) -> Bool {
        attach(dropped: urls, slots: mode.attachmentSlots)
    }

    /// Settings that follow an attachment so the slot is never silently ignored: LTX audio
    /// needs the audio+video output, and a cloned voice needs clone mode.
    package mutating func didAttach(to slot: StudioAttachmentSlot) {
        switch slot.id {
        case "audio" where slot.storage == .path(\.audioPath):
            videoQuality = .final
            videoOutputMode = .audioVideo
        case "referenceAudio":
            voiceMode = "clone"
        default:
            break
        }
    }
}

/// Reads attachments off the pasteboard: file URLs first, else a pasted bitmap saved as PNG.
package enum StudioAttachmentPasteboard {
    /// File URLs on the pasteboard that `slot` accepts.
    package static func fileURLs(from pasteboard: NSPasteboard, for slot: StudioAttachmentSlot) -> [URL] {
        let urls = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]
        ) as? [URL] ?? []
        return urls.filter(slot.accepts)
    }

    /// Writes a pasted bitmap to a PNG in the temporary directory; nil when there is no image.
    package static func writePastedImage(from pasteboard: NSPasteboard) throws -> URL? {
        guard let image = NSImage(pasteboard: pasteboard),
              let tiff = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiff),
              let data = bitmap.representation(using: .png, properties: [:]) else {
            return nil
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pasted-\(UUID().uuidString).png")
        try data.write(to: url)
        return url
    }
}

// MARK: - Chips

/// One essential parameter shown as an editable chip under the prompt.
package enum StudioComposerChipKind: String, CaseIterable, Identifiable {
    case dimensions
    case duration
    case steps
    case seed
    case threshold
    case readImageAction
    case voiceMode
    case thinking
    case model

    package var id: String { rawValue }
}

extension StudioMode {
    /// The two to four essentials this mode shows as chips, in strip order. Everything else stays
    /// in the options popover until the inspector replaces it.
    package var composerChips: [StudioComposerChipKind] {
        switch self {
        case .createImage: return [.dimensions, .steps, .seed, .model]
        case .video, .music, .sfx: return [.duration, .steps, .seed, .model]
        case .speak: return [.voiceMode, .model]
        case .chat: return [.model, .thinking]
        case .code, .listen: return [.model]
        case .readImage: return [.readImageAction, .model]
        case .findObjects: return [.model]
        case .segment, .track: return [.threshold, .model]
        }
    }

    /// The `model list` categories whose rows the model chip offers for this mode.
    package var modelCategories: Set<String> {
        switch self {
        case .createImage: return ["image"]
        case .chat: return ["text-chat", "vision-chat", "omni-chat"]
        case .code: return ["text-code", "text-chat"]
        case .speak: return ["speech-tts"]
        case .listen: return ["speech-asr"]
        case .readImage: return ["vision-chat", "omni-chat", "vision-ocr"]
        case .findObjects: return ["vision-ground"]
        case .segment, .track: return ["vision-segment"]
        case .music: return ["music"]
        case .video: return ["video"]
        case .sfx: return ["sfx"]
        }
    }

    /// Inventory rows the model chip lists: this mode's categories, installed first.
    package func modelChoices(from inventory: [StudioModelInventoryRow]) -> [StudioModelInventoryRow] {
        StudioModelScope(mode: self).choices(from: inventory)
    }
}

/// What a model picker needs to know about the surface it picks for: which `model list`
/// categories to offer, what "Auto" resolves to, and the noun for "No image models listed yet".
/// A prompt mode and a specialist template each make one, so the composer chip, the inspector
/// row, and the readiness card share a single picker whichever draft is behind them.
package struct StudioModelScope: Equatable {
    package let noun: String
    package let defaultModelID: String
    /// Empty means every row: a template whose models the inventory does not categorize.
    package let categories: Set<String>

    package init(noun: String, defaultModelID: String, categories: Set<String>) {
        self.noun = noun
        self.defaultModelID = defaultModelID
        self.categories = categories
    }

    package init(mode: StudioMode) {
        self.init(
            noun: mode.title.lowercased(),
            defaultModelID: StudioModelNaming.defaultModelID(for: mode),
            categories: mode.modelCategories
        )
    }

    package init(templateID: CommandTemplateID) {
        let template = CommandCatalog.template(id: templateID)
        self.init(
            noun: template?.title.lowercased() ?? templateID.rawValue,
            defaultModelID: template?.defaultModel ?? "",
            categories: Self.categories(for: templateID)
        )
    }

    /// Inventory rows the picker lists: the scope's categories, installed first.
    package func choices(from inventory: [StudioModelInventoryRow]) -> [StudioModelInventoryRow] {
        inventory
            .filter { categories.isEmpty || categories.contains($0.category) }
            .sorted { lhs, rhs in
                if lhs.isInstalled != rhs.isInstalled { return lhs.isInstalled }
                return lhs.id < rhs.id
            }
    }

    /// The model id a draft actually runs with: its explicit model, else the default.
    package func resolvedModelID(model: String) -> String {
        let current = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return current.isEmpty ? defaultModelID : current
    }

    /// What the picker shows: the resolved model's name, or "Auto" when there is no default.
    package func displayLabel(model: String, titles: StudioModelTitles) -> String {
        let resolved = resolvedModelID(model: model)
        return resolved.isEmpty ? "Auto" : StudioModelNaming.displayName(resolved, titles: titles)
    }

    /// The `model list` categories a specialist template's model chip offers
    /// (`ManagedModelCategory` raw values). A template whose models the inventory does not
    /// categorize on its own offers every row.
    package static func categories(for templateID: CommandTemplateID) -> Set<String> {
        switch templateID {
        case .visionDepth, .visionDepthVideo: return ["vision-depth"]
        case .visionFaceDetect, .visionFaceEmbed, .visionFaceCompare, .visionFaceBatch: return ["vision-face"]
        case .visionGeometry, .visionGeometryMultiview: return ["vision-geometry"]
        case .visionTrackLive: return ["vision-segment"]
        case .speechDiarize, .speechDiarizeLive: return ["speech-diarization"]
        case .speechListen: return ["speech-asr"]
        case .audioEnhance, .audioEdit: return ["audio"]
        case .musicAnalyze, .musicTranscribe, .musicSeparate: return ["music"]
        case .sfxVideo, .sfxConditionText, .sfxAEEncode, .sfxAEDecode, .sfxClapScore: return ["sfx"]
        case .textEmbed: return ["text-embed"]
        case .textAnonymize: return ["text-anonymize"]
        case .textDecide: return ["text-decide"]
        case .geoFlood: return ["vision-flood"]
        case .geoFire: return ["vision-fire"]
        case .geoTessera, .geoOlmoEarth: return ["vision-embed"]
        case .imageReconstruct3D, .imageReconstruct3DTrellis2, .imageReconstruct3DMultiview: return ["image-3d"]
        case .speechSynthesize, .speechProfileCreate: return ["speech-tts"]
        case .imageTrainLoRA: return ["image"]
        case .textTrainLoRA: return ["text-chat"]
        case .musicTrainAdapter: return ["music"]
        default:
            if let mode = StudioMode.allCases.first(where: { $0.defaultTemplateID == templateID }) {
                return mode.modelCategories
            }
            return []
        }
    }
}

/// A named aspect ratio with the pixel size it means for one mode.
package struct StudioAspectPreset: Identifiable, Equatable {
    package let label: String
    package let width: Int
    package let height: Int

    package var id: String { label }

    package func matches(_ draft: StudioDraft) -> Bool {
        draft.width == width && draft.height == height
    }

    package func apply(to draft: inout StudioDraft) {
        draft.width = width
        draft.height = height
    }

    /// Image sizes are the 1024² family; video sizes stay multiples of 32 around LTX's 768×512.
    package static func presets(for mode: StudioMode) -> [StudioAspectPreset] {
        switch mode {
        case .video:
            return [
                StudioAspectPreset(label: "3:2", width: 768, height: 512),
                StudioAspectPreset(label: "16:9", width: 832, height: 480),
                StudioAspectPreset(label: "1:1", width: 512, height: 512),
                StudioAspectPreset(label: "9:16", width: 480, height: 832),
                StudioAspectPreset(label: "2:3", width: 512, height: 768),
            ]
        default:
            return [
                StudioAspectPreset(label: "1:1", width: 1024, height: 1024),
                StudioAspectPreset(label: "3:2", width: 1216, height: 832),
                StudioAspectPreset(label: "2:3", width: 832, height: 1216),
                StudioAspectPreset(label: "16:9", width: 1344, height: 768),
                StudioAspectPreset(label: "9:16", width: 768, height: 1344),
                StudioAspectPreset(label: "4:3", width: 1152, height: 896),
                StudioAspectPreset(label: "3:4", width: 896, height: 1152),
            ]
        }
    }
}

/// How the seed chip reads the draft's free-text seed.
package enum StudioSeedMode: Equatable {
    case random
    case fixed(Int)

    package init(draft: StudioDraft) {
        let trimmed = draft.seed.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = Int(trimmed) {
            self = .fixed(value)
        } else {
            self = .random
        }
    }

    package func apply(to draft: inout StudioDraft) {
        switch self {
        case .random: draft.seed = ""
        case .fixed(let value): draft.seed = String(value)
        }
    }

    package var chipTitle: String {
        switch self {
        case .random: return "Seed random"
        case .fixed(let value): return "Seed \(value)"
        }
    }
}

/// Preset values and titles for the chips whose menus list numbers.
package enum StudioComposerPresets {
    package static func steps(for mode: StudioMode) -> [Int] {
        switch mode {
        case .video: return [20, 30, 40, 50]
        case .music: return [8, 27, 60]
        case .sfx: return [4, 8, 16, 32, 50]
        default: return [1, 2, 4, 8, 12, 20, 28, 50]
        }
    }

    /// Seconds for the duration chip.
    package static func durations(for mode: StudioMode) -> [Double] {
        switch mode {
        case .music: return [30, 60, 120, 180, 240]
        case .video: return [2, 3, 5, 8]
        default: return [2, 5, 10, 15, 30]
        }
    }

    /// Frame counts for Video's duration chip when it counts frames rather than seconds.
    package static let videoFrameCounts = [33, 65, 97, 129, 161]

    package static let thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

    package static func dimensionsTitle(_ draft: StudioDraft) -> String {
        "\(draft.width) × \(draft.height)"
    }

    package static func stepsTitle(_ draft: StudioDraft, mode: StudioMode) -> String {
        if mode == .music, !draft.musicOverrideSteps { return "Preset steps" }
        return draft.steps == 1 ? "1 step" : "\(draft.steps) steps"
    }

    package static func durationTitle(_ draft: StudioDraft, mode: StudioMode) -> String {
        switch mode {
        case .video where !draft.useDuration:
            return "\(draft.numFrames) frames"
        case .music where !draft.useDuration:
            return "Preset length"
        default:
            return "\(secondsText(draft.durationSeconds)) s"
        }
    }

    package static func thresholdTitle(_ draft: StudioDraft) -> String {
        "Threshold \(decimalText(draft.visionThreshold))"
    }

    package static func thinkingTitle(_ mode: TextThinkingMode) -> String {
        switch mode {
        case .automatic: return "Thinking auto"
        case .show: return "Thinking on"
        case .hide: return "Thinking off"
        }
    }

    package static func voiceModeTitle(_ draft: StudioDraft) -> String {
        draft.voiceMode == "clone" ? "Cloned voice" : "Preset voice"
    }

    package static func secondsText(_ seconds: Double) -> String {
        seconds.rounded() == seconds ? String(Int(seconds)) : decimalText(seconds)
    }

    /// A number as the argv carries it: the POSIX locale, no grouping, every fraction digit the
    /// value has (up to fifteen), and a whole number without a fraction. `decimalText` is for
    /// chips and labels only; a learning rate of 0.0001 must reach the CLI as typed, not as "0".
    package static func argumentText(_ value: Double) -> String {
        if value.rounded() == value, abs(value) < 1e15 { return String(Int(value)) }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.usesGroupingSeparator = false
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 15
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }

    package static func decimalText(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 2
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }
}
