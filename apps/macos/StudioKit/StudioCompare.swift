import Foundation

/// What Compare lays side by side. Every pane of one comparison is the same kind.
package enum StudioCompareMedia: String, Codable, Equatable, Sendable {
    case image, audio, video

    package init?(_ kind: StudioOutputFileKind) {
        switch kind {
        case .image: self = .image
        case .audio: self = .audio
        case .video: self = .video
        case .text, .model3D, .other: return nil
        }
    }

    /// The plural the title and the Compare button use: "4 images".
    package func count(_ count: Int) -> String {
        switch self {
        case .image: return count == 1 ? "1 image" : "\(count) images"
        case .audio: return count == 1 ? "1 sound" : "\(count) sounds"
        case .video: return count == 1 ? "1 video" : "\(count) videos"
        }
    }
}

/// The rows a comparison shows, by id, stored per task (`StudioTaskSessions.comparison`) so it
/// survives the page being swapped out the way a focused result does.
package struct StudioCompareSelection: Codable, Equatable, Sendable {
    package let itemIDs: [UUID]

    package init(itemIDs: [UUID]) {
        self.itemIDs = itemIDs
    }
}

/// One recorded setting a pane lists: its seed, its model, or an option whose value is not the
/// same on every pane.
package struct StudioCompareSetting: Identifiable, Equatable, Sendable {
    package enum Kind: Equatable, Sendable {
        case seed, model, command, argument, option, extra
    }

    package let id: String
    package let title: String
    package let value: String
    package let kind: Kind

    package init(id: String, title: String, value: String, kind: Kind) {
        self.id = id
        self.title = title
        self.value = value
        self.kind = kind
    }
}

/// One side of a comparison: a finished row, the file of the compared kind it made, the letter
/// the pane is known by, and what it ran with that the other panes did not.
package struct StudioComparePane: Identifiable, Equatable {
    package let item: StudioLibraryItem
    package let url: URL
    package let letter: String
    package let settings: [StudioCompareSetting]

    package var id: UUID { item.id }
}

/// Compare's rules, kept here so the picking, the diff, and the transport are tested without a
/// view: which rows can be compared, which file of each is shown, and which settings each pane
/// lists — read from the argv each run recorded, never from its current draft.
package enum StudioCompare {
    /// How many rows a hand-picked comparison takes.
    package static let selectionRange = 2...4
    /// How many rows of one variation group a comparison opens with: every run of the largest
    /// "Run variations" count.
    package static let groupLimit = StudioVariationCount.eight.rawValue

    /// The kind a finished row is compared as: its primary output's, else its first picture,
    /// sound, or video. nil for a row that is not finished or made none.
    package static func media(of item: StudioLibraryItem) -> StudioCompareMedia? {
        guard item.status == .completed, !item.isConversation else { return nil }
        return item.allArtifactURLs.lazy.compactMap { StudioCompareMedia(StudioOutputFileKind.classify($0)) }.first
    }

    /// The file a pane shows: the primary output when it is of the compared kind, else the
    /// row's first file of that kind.
    package static func url(of item: StudioLibraryItem, as media: StudioCompareMedia) -> URL? {
        item.allArtifactURLs.first { StudioCompareMedia(StudioOutputFileKind.classify($0)) == media }
    }

    /// The kind a hand-picked set compares as: two to four finished rows that all made the
    /// same kind of media. nil when the set cannot be compared.
    package static func comparableMedia(_ items: [StudioLibraryItem]) -> StudioCompareMedia? {
        guard selectionRange.contains(items.count), let first = items.first.flatMap(media(of:)),
              items.allSatisfy({ media(of: $0) == first }) else { return nil }
        return first
    }

    /// Why a hand-picked set cannot be compared, for the disabled button's help.
    package static func unavailableReason(_ items: [StudioLibraryItem]) -> String? {
        if items.count < selectionRange.lowerBound { return "Select 2 to 4 finished results to compare" }
        if items.count > selectionRange.upperBound { return "Compare takes up to 4 results" }
        if items.contains(where: { media(of: $0) == nil }) { return "Compare takes finished images, sounds, or videos" }
        return comparableMedia(items) == nil ? "Compare results of one kind: images, sounds, or videos" : nil
    }

    /// The finished runs of a variation group Compare opens with, in submission order: those of
    /// the kind its first finished run made, at most `groupLimit`. Empty while fewer than two
    /// have finished.
    package static func groupItems(_ group: UUID, in items: [StudioLibraryItem]) -> [StudioLibraryItem] {
        let finished = StudioVariations.members(of: group, in: items).filter { media(of: $0) != nil }
        guard let kind = finished.first.flatMap(media(of:)) else { return [] }
        let sameKind = Array(finished.filter { media(of: $0) == kind }.prefix(groupLimit))
        return sameKind.count >= selectionRange.lowerBound ? sameKind : []
    }

    /// The rows a stored selection still names, in its order, when they can still be compared
    /// together; nil once fewer than two remain (a row deleted, a file gone from the Library).
    package static func resolve(_ selection: StudioCompareSelection, in items: [StudioLibraryItem]) -> [StudioLibraryItem]? {
        let rows = selection.itemIDs.compactMap { id in items.first { $0.id == id } }
        guard rows.count >= selectionRange.lowerBound, rows.count <= groupLimit,
              let kind = rows.first.flatMap(media(of:)), rows.allSatisfy({ media(of: $0) == kind }) else { return nil }
        return rows
    }

    /// The panes for `items`, lettered A, B, C… in the order given, each with its settings.
    package static func panes(for items: [StudioLibraryItem], source: StudioScopeSource) -> [StudioComparePane] {
        guard let kind = items.first.flatMap(media(of:)) else { return [] }
        let settings = settings(for: items, source: source)
        return items.enumerated().compactMap { offset, item in
            url(of: item, as: kind).map { url in
                StudioComparePane(item: item, url: url, letter: letter(offset), settings: settings[offset])
            }
        }
    }

    /// How many panes a row of the image or video grid holds at `width` points: side by side
    /// while each pane stays usable, else two to a row, and one on a narrow window.
    package static func columns(count: Int, width: Double) -> Int {
        let minimumPane = 280.0
        let fitting = max(1, Int(width / minimumPane))
        let preferred = count <= 3 ? count : (count == 4 ? (fitting >= 4 ? 4 : 2) : 4)
        return max(1, min(preferred, fitting, count))
    }

    package static func letter(_ offset: Int) -> String {
        String(UnicodeScalar(UInt8(65 + offset % 26)))
    }

    /// Each row's settings as its pane lists them: always its seed (when its command takes one)
    /// and its model, then the command, arguments, options, and additional options whose values
    /// are not the same on every row. The values come from the argv each run recorded.
    package static func settings(for items: [StudioLibraryItem], source: StudioScopeSource) -> [[StudioCompareSetting]] {
        let forms = items.map { StudioResultComparison.recordedForm($0, source: source) }
        let capabilities = items.map { $0.templateID?.capability }
        var rows: [(id: String, title: String, kind: StudioCompareSetting.Kind, values: [String])] = []

        func differs(_ values: [String]) -> Bool { Set(values).count > 1 }
        func shown(_ text: String, empty: String = "Default") -> String { text.isEmpty ? empty : text }

        if capabilities.contains(where: { capability in
            capability?.options.contains { $0.flag == StudioVariations.seedFlag } == true
        }) {
            rows.append((StudioVariations.seedFlag, "Seed", .seed,
                         forms.map { shown($0?.text(StudioVariations.seedFlag) ?? "", empty: "Random") }))
        }
        rows.append(("--model", "Model", .model, items.map { $0.recordedModelID ?? "Default" }))

        let commands = capabilities.map { $0?.command.joined(separator: " ") ?? "" }
        if differs(commands) { rows.append(("command", "Command", .command, commands)) }

        let argumentCount = forms.map { $0?.arguments.count ?? 0 }.max() ?? 0
        let argumentLabels = capabilities.compactMap { $0 }.first?.arguments.map(\.label) ?? []
        for index in 0..<argumentCount {
            let values = forms.map { form in
                form.map { $0.arguments.indices.contains(index) ? $0.arguments[index] : "" } ?? ""
            }
            guard differs(values) else { continue }
            let title = argumentLabels.indices.contains(index) ? argumentLabels[index] : "Argument \(index + 1)"
            rows.append(("argument.\(index)", title, .argument, values.map { shown($0) }))
        }

        let labels = Dictionary(capabilities.compactMap { $0 }.flatMap(\.options).map { ($0.flag, $0.label) },
                                uniquingKeysWith: { first, _ in first })
        let flags = Set(forms.compactMap { $0 }.flatMap(\.values.keys))
            .subtracting(StudioResultComparison.unlistedFlags)
            .subtracting([StudioVariations.seedFlag, "--model"])
        // In the order the contract declares them (the prompt before the sampler), then any the
        // contract does not know.
        let declared = capabilities.compactMap { $0 }.flatMap(\.options).map(\.flag)
        let ordered = declared.reduce(into: [String]()) { if flags.contains($1), !$0.contains($1) { $0.append($1) } }
            + flags.subtracting(declared).sorted()
        for flag in ordered {
            // A flag a run left out ran at the contract's default, so a recorded "8" and an
            // omitted default of 8 are the same setting.
            let values = items.indices.map { index in
                let text = forms[index]?.text(flag) ?? ""
                return text.isEmpty ? capabilities[index]?.options.first { $0.flag == flag }?.defaultValue ?? "" : text
            }
            guard differs(values) else { continue }
            rows.append((flag, labels[flag] ?? flag, .option, values.map { shown($0) }))
        }

        let extras = forms.map { ShellWords.split($0?.extraArguments ?? "").maskingSecrets().shellQuoted() }
        if differs(extras) { rows.append(("extraArguments", "Additional options", .extra, extras.map { shown($0, empty: "None") })) }

        return items.indices.map { index in
            rows.map { StudioCompareSetting(id: $0.id, title: $0.title, value: $0.values[index], kind: $0.kind) }
        }
    }
}

/// Compare's shared audio and video transport: one position every pane follows, and which pane
/// is heard. Switching keeps the position, so A/B listening compares the same moment.
package struct StudioCompareTransport: Equatable, Sendable {
    /// Each pane's length in seconds, 0 until its file has loaded.
    package var durations: [Double]
    /// The pane heard: the one playing, or the one that plays next.
    package private(set) var active = 0
    /// Seconds from the start, shared by every pane.
    package private(set) var position: Double = 0
    package var isPlaying = false

    package init(durations: [Double]) {
        self.durations = durations
    }

    /// The longest pane's length, the span the shared scrubber covers.
    package var span: Double { durations.max() ?? 0 }

    /// Hands playback to `pane` at the same moment, or at its end when it is shorter.
    package mutating func select(_ pane: Int) {
        guard durations.indices.contains(pane) else { return }
        active = pane
        position = min(position, durations[pane])
    }

    /// Moves the shared position, kept within the heard pane.
    package mutating func seek(to seconds: Double) {
        let limit = durations.indices.contains(active) ? durations[active] : span
        position = min(max(0, seconds), limit)
    }

    /// Seeks to `fraction` of `pane`'s length — a click on that pane's waveform — and makes it
    /// the pane heard.
    package mutating func seek(fraction: Double, in pane: Int) {
        guard durations.indices.contains(pane) else { return }
        active = pane
        seek(to: min(max(0, fraction), 1) * durations[pane])
    }

    /// How far through `pane` the shared position is, 0…1.
    package func progress(of pane: Int) -> Double {
        guard durations.indices.contains(pane), durations[pane] > 0 else { return 0 }
        return min(1, position / durations[pane])
    }

    /// Follows the heard pane's own clock while it plays; at its end playback stops at the start.
    package mutating func advance(to seconds: Double, stillPlaying: Bool) {
        if stillPlaying {
            seek(to: seconds)
        } else {
            isPlaying = false
            position = 0
        }
    }
}

extension StudioTaskSessions {
    /// The comparison `task`'s page shows in place of its canvas, when the rows it names can
    /// still be compared.
    package func comparison(for task: StudioTask, items: [StudioLibraryItem]) -> [StudioLibraryItem]? {
        value(for: task.rawValue + ".compare", default: Optional<StudioCompareSelection>.none)
            .flatMap { StudioCompare.resolve($0, in: items) }
    }

    /// Opens Compare on `task`'s page for `items`, or closes it with nil. A comparison replaces a
    /// focused result.
    package func setComparison(_ items: [StudioLibraryItem]?, for task: StudioTask) {
        set(items.map { StudioCompareSelection(itemIDs: $0.map(\.id)) }, for: task.rawValue + ".compare")
        if items != nil { setFocus(nil, for: task) }
    }
}
