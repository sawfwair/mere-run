import AppKit
import MereRunContract
import UniformTypeIdentifiers

// The composer's declarative surface: which attachment slots and which parameter chips each
// prompt mode shows, and how each maps onto `StudioDraft`. The views in `StudioComposer.swift`
// render from these declarations, so a mode's essentials live in one place and are testable
// without SwiftUI.

// MARK: - Attachment slots

/// One slot in the composer's attachment well, bound to a draft field.
struct StudioAttachmentSlot: Identifiable, Equatable {
    enum Storage: Equatable {
        /// One path in one draft field; attaching replaces it.
        case path(WritableKeyPath<StudioDraft, String>)
        /// Newline-separated paths in one draft field; attaching appends, the slot previews the first.
        case pathList(WritableKeyPath<StudioDraft, String>)
    }

    let id: String
    /// The empty slot's caption ("Reference"); a filled slot shows its file name instead.
    let label: String
    let acceptedTypes: [UTType]
    let storage: Storage
    /// The task cannot run without this slot filled.
    var isRequired = false
    /// A per-turn attachment (Chat's image): the well stays collapsed to the paperclip until
    /// something is attached, so an empty slot never sits above every message.
    var isTransient = false

    var allowsMultiple: Bool {
        if case .pathList = storage { return true }
        return false
    }

    /// The paths this slot currently holds, in order.
    func paths(in draft: StudioDraft) -> [String] {
        switch storage {
        case .path(let keyPath):
            let value = draft[keyPath: keyPath].trimmingCharacters(in: .whitespacesAndNewlines)
            return value.isEmpty ? [] : [value]
        case .pathList(let keyPath):
            return Self.separatedPaths(draft[keyPath: keyPath])
        }
    }

    func isFilled(in draft: StudioDraft) -> Bool {
        !paths(in: draft).isEmpty
    }

    /// The caption shown beside the slot: the file name when filled, the slot label otherwise.
    func caption(in draft: StudioDraft) -> String {
        let paths = paths(in: draft)
        guard let first = paths.first else { return label }
        let name = URL(fileURLWithPath: first).lastPathComponent
        return paths.count > 1 ? "\(name) +\(paths.count - 1)" : name
    }

    /// Whether a dropped or pasted file belongs in this slot.
    func accepts(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        guard let type = UTType(filenameExtension: url.pathExtension) else { return false }
        return acceptedTypes.contains { type.conforms(to: $0) }
    }

    /// Stores `urls` in the slot: a single-path slot keeps the first, a list slot appends them all.
    func attach(_ urls: [URL], to draft: inout StudioDraft) {
        let incoming = urls.filter(accepts).map(\.path)
        guard !incoming.isEmpty else { return }
        switch storage {
        case .path(let keyPath):
            draft[keyPath: keyPath] = incoming[0]
        case .pathList(let keyPath):
            let existing = Self.separatedPaths(draft[keyPath: keyPath])
            draft[keyPath: keyPath] = (existing + incoming.filter { !existing.contains($0) })
                .joined(separator: "\n")
        }
        draft.didAttach(to: self)
    }

    func clear(in draft: inout StudioDraft) {
        switch storage {
        case .path(let keyPath), .pathList(let keyPath):
            draft[keyPath: keyPath] = ""
        }
    }

    static func separatedPaths(_ raw: String) -> [String] {
        raw.components(separatedBy: .newlines)
            .flatMap { $0.split(separator: ",", omittingEmptySubsequences: true) }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}

extension StudioMode {
    /// The attachment slots this mode's composer declares, in well order.
    var attachmentSlots: [StudioAttachmentSlot] {
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
    func showsAttachmentWell(for draft: StudioDraft) -> Bool {
        attachmentSlots.contains { !$0.isTransient || $0.isFilled(in: draft) }
    }

    /// The slot a file dropped on the canvas or pasted with ⌘V lands in: the first empty slot
    /// that accepts it, else the first slot that accepts it.
    func attachmentSlot(for url: URL, in draft: StudioDraft) -> StudioAttachmentSlot? {
        let accepting = attachmentSlots.filter { $0.accepts(url) }
        return accepting.first { !$0.isFilled(in: draft) } ?? accepting.first
    }

    /// The slot a pasted bitmap (no file on the pasteboard) lands in.
    func pastedImageSlot(in draft: StudioDraft) -> StudioAttachmentSlot? {
        let accepting = attachmentSlots.filter { slot in slot.acceptedTypes.contains { UTType.image.conforms(to: $0) } }
        return accepting.first { !$0.isFilled(in: draft) } ?? accepting.first
    }
}

extension StudioDraft {
    /// Routes each dropped file to the slot it belongs in. Returns whether anything was attached.
    @discardableResult
    mutating func attach(dropped urls: [URL], for mode: StudioMode) -> Bool {
        var attached = false
        for url in urls {
            guard let slot = mode.attachmentSlot(for: url, in: self) else { continue }
            slot.attach([url], to: &self)
            attached = true
        }
        return attached
    }

    /// Settings that follow an attachment so the slot is never silently ignored: LTX audio
    /// needs the audio+video output, and a cloned voice needs clone mode.
    fileprivate mutating func didAttach(to slot: StudioAttachmentSlot) {
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
enum StudioAttachmentPasteboard {
    /// File URLs on the pasteboard that `slot` accepts.
    static func fileURLs(from pasteboard: NSPasteboard, for slot: StudioAttachmentSlot) -> [URL] {
        let urls = pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]
        ) as? [URL] ?? []
        return urls.filter(slot.accepts)
    }

    /// Writes a pasted bitmap to a PNG in the temporary directory; nil when there is no image.
    static func writePastedImage(from pasteboard: NSPasteboard) throws -> URL? {
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
enum StudioComposerChipKind: String, CaseIterable, Identifiable {
    case dimensions
    case duration
    case steps
    case seed
    case threshold
    case readImageAction
    case voiceMode
    case thinking
    case model

    var id: String { rawValue }
}

extension StudioMode {
    /// The two to four essentials this mode shows as chips, in strip order. Everything else stays
    /// in the options popover until the inspector replaces it.
    var composerChips: [StudioComposerChipKind] {
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
    var modelCategories: Set<String> {
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
    func modelChoices(from inventory: [StudioModelInventoryRow]) -> [StudioModelInventoryRow] {
        inventory
            .filter { modelCategories.contains($0.category) }
            .sorted { lhs, rhs in
                if lhs.isInstalled != rhs.isInstalled { return lhs.isInstalled }
                return lhs.id < rhs.id
            }
    }
}

/// A named aspect ratio with the pixel size it means for one mode.
struct StudioAspectPreset: Identifiable, Equatable {
    let label: String
    let width: Int
    let height: Int

    var id: String { label }

    func matches(_ draft: StudioDraft) -> Bool {
        draft.width == width && draft.height == height
    }

    func apply(to draft: inout StudioDraft) {
        draft.width = width
        draft.height = height
    }

    /// Image sizes are the 1024² family; video sizes stay multiples of 32 around LTX's 768×512.
    static func presets(for mode: StudioMode) -> [StudioAspectPreset] {
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
enum StudioSeedMode: Equatable {
    case random
    case fixed(Int)

    init(draft: StudioDraft) {
        let trimmed = draft.seed.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = Int(trimmed) {
            self = .fixed(value)
        } else {
            self = .random
        }
    }

    func apply(to draft: inout StudioDraft) {
        switch self {
        case .random: draft.seed = ""
        case .fixed(let value): draft.seed = String(value)
        }
    }

    var chipTitle: String {
        switch self {
        case .random: return "Seed random"
        case .fixed(let value): return "Seed \(value)"
        }
    }
}

/// Preset values and titles for the chips whose menus list numbers.
enum StudioComposerPresets {
    static func steps(for mode: StudioMode) -> [Int] {
        switch mode {
        case .video: return [20, 30, 40, 50]
        case .music: return [8, 27, 60]
        case .sfx: return [4, 8, 16, 32, 50]
        default: return [1, 2, 4, 8, 12, 20, 28, 50]
        }
    }

    /// Seconds for the duration chip.
    static func durations(for mode: StudioMode) -> [Double] {
        switch mode {
        case .music: return [30, 60, 120, 180, 240]
        case .video: return [2, 3, 5, 8]
        default: return [2, 5, 10, 15, 30]
        }
    }

    /// Frame counts for Video's duration chip when it counts frames rather than seconds.
    static let videoFrameCounts = [33, 65, 97, 129, 161]

    static let thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]

    static func dimensionsTitle(_ draft: StudioDraft) -> String {
        "\(draft.width) × \(draft.height)"
    }

    static func stepsTitle(_ draft: StudioDraft, mode: StudioMode) -> String {
        if mode == .music, !draft.musicOverrideSteps { return "Preset steps" }
        return draft.steps == 1 ? "1 step" : "\(draft.steps) steps"
    }

    static func durationTitle(_ draft: StudioDraft, mode: StudioMode) -> String {
        switch mode {
        case .video where !draft.useDuration:
            return "\(draft.numFrames) frames"
        case .music where !draft.useDuration:
            return "Preset length"
        default:
            return "\(secondsText(draft.durationSeconds)) s"
        }
    }

    static func thresholdTitle(_ draft: StudioDraft) -> String {
        "Threshold \(decimalText(draft.visionThreshold))"
    }

    static func thinkingTitle(_ mode: TextThinkingMode) -> String {
        switch mode {
        case .automatic: return "Thinking auto"
        case .show: return "Thinking on"
        case .hide: return "Thinking off"
        }
    }

    static func voiceModeTitle(_ draft: StudioDraft) -> String {
        draft.voiceMode == "clone" ? "Cloned voice" : "Preset voice"
    }

    static func secondsText(_ seconds: Double) -> String {
        seconds.rounded() == seconds ? String(Int(seconds)) : decimalText(seconds)
    }

    static func decimalText(_ value: Double) -> String {
        let formatter = NumberFormatter()
        formatter.minimumFractionDigits = 0
        formatter.maximumFractionDigits = 2
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }
}
