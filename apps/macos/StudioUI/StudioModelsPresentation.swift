import Foundation
import StudioKit
import SwiftUI

/// The display family a managed model category belongs to: the vocabulary of the Installed
/// page's chip row and the first word of every row's meta line ("Image · 2.1 GB").
enum StudioModelFamily: String, CaseIterable, Identifiable {
    case image
    case video
    case music
    case sound
    case voice
    case audio
    case chat
    case text
    case vision
    case threeD
    case other

    var id: String { rawValue }

    var title: String {
        switch self {
        case .image: return "Image"
        case .video: return "Video"
        case .music: return "Music"
        case .sound: return "Sound"
        case .voice: return "Voice"
        case .audio: return "Audio"
        case .chat: return "Chat"
        case .text: return "Text"
        case .vision: return "Vision"
        case .threeD: return "3D"
        case .other: return "Other"
        }
    }

    /// The CLI's `ManagedModelCategory` raw values, grouped the way the sidebar groups domains.
    static func from(category: String) -> StudioModelFamily {
        switch category.lowercased() {
        case "image": return .image
        case "image-3d": return .threeD
        case "video": return .video
        case "music": return .music
        case "sfx": return .sound
        case "speech-tts": return .voice
        case "speech-asr", "speech-diarization", "audio": return .audio
        case "text-chat", "text-code", "omni-chat": return .chat
        case let value where value.hasPrefix("text-"): return .text
        case let value where value.hasPrefix("vision-"): return .vision
        default: return .other
        }
    }

    /// The LoRA/adapter training template for models of this family, when the app has one.
    var trainingTemplateID: CommandTemplateID? {
        switch self {
        case .image: return .imageTrainLoRA
        case .chat: return .textTrainLoRA
        case .music: return .musicTrainAdapter
        default: return nil
        }
    }
}

/// What the Models toolbar subtitle reports: installed count and the managed store's size.
struct StudioModelInventorySummary: Equatable {
    let installedCount: Int
    let storageBytes: Int64?

    var subtitle: String {
        guard installedCount > 0 else { return "No models installed" }
        let count = "\(installedCount) installed"
        guard let storageBytes, storageBytes > 0 else { return count }
        return "\(count) · \(StudioModelsPresenter.bytes(storageBytes)) on this Mac"
    }
}

/// The state dot in front of a list row.
enum StudioModelRowStatus: Equatable {
    case installed
    case attention
    case pulling(Double?)
    case missing

    var color: Color {
        switch self {
        case .installed: return MereRunTheme.green
        case .attention: return MereRunTheme.yellow
        case .pulling: return MereRunTheme.accent
        case .missing: return MereRunTheme.textMuted
        }
    }

    var accessibilityDescription: String {
        switch self {
        case .installed: return "installed"
        case .attention: return "needs attention"
        case .pulling(let fraction):
            guard let fraction else { return "pulling" }
            return "pulling \(Int((fraction * 100).rounded())) percent"
        case .missing: return "not installed"
        }
    }
}

/// A Models-domain job the page's job bar reports: a pull, an optimization, or a storage
/// clean-up. Pulls started from the composer arrive as running `.modelPull` Library rows; the
/// page's own pulls, optimizations, and clean-ups are tracked locally.
struct StudioModelsJob: Equatable {
    enum Kind: Equatable {
        case pull
        case optimize
        case cleanup

        var verb: String {
            switch self {
            case .pull: return "Pull"
            case .optimize: return "Optimize"
            case .cleanup: return "Clean up"
            }
        }
    }

    let kind: Kind
    let modelID: String?
    let subject: String
    let progress: StudioRunProgress?
    let isCancelling: Bool
    /// The Library row backing a composer-initiated pull, so Cancel routes to the controller.
    let libraryItemID: UUID?

    init(
        kind: Kind,
        modelID: String?,
        subject: String,
        progress: StudioRunProgress? = nil,
        isCancelling: Bool = false,
        libraryItemID: UUID? = nil
    ) {
        self.kind = kind
        self.modelID = modelID
        self.subject = subject
        self.progress = progress
        self.isCancelling = isCancelling
        self.libraryItemID = libraryItemID
    }

    var label: String {
        "Models · \(kind.verb) \(subject)"
    }

    var detail: String? {
        if isCancelling { return "Cancelling…" }
        return progress?.detail
    }

    var fraction: Double? {
        progress?.fractionCompleted
    }
}

/// Facts parsed from `mere.run model info <id> --components` for the detail column.
struct StudioModelInfoFacts: Equatable {
    var root: String?
    var quantization: String?
    var hasManifest: Bool?
    var isValid: Bool?

    /// The "Verified" row: nil until the info output has been read.
    var verifiedLine: String? {
        switch (hasManifest, isValid) {
        case (false, _): return "manifest missing"
        case (true, true): return "manifest ok · validation passed"
        case (true, false): return "manifest ok · validation failed"
        case (nil, true): return "validation passed"
        case (nil, false): return "validation failed"
        case (true, nil), (nil, nil): return nil
        }
    }

    var isEmpty: Bool {
        root == nil && quantization == nil && hasManifest == nil && isValid == nil
    }
}

/// How a model has been used from this Mac, from the Library.
struct StudioModelUsageSummary: Equatable {
    let lastUsed: Date
    let runs: Int
}

/// Pure presentation rules for the Installed page: filtering, the row meta line, chips,
/// Library-derived facts, and the job bar. Everything here is unit-testable without a view.
enum StudioModelsPresenter {
    static func displayName(_ row: StudioModelInventoryRow) -> String {
        row.title ?? row.id
    }

    static func family(of row: StudioModelInventoryRow) -> StudioModelFamily {
        StudioModelFamily.from(category: row.category)
    }

    /// The families that have at least one row, in sidebar order.
    static func families(in rows: [StudioModelInventoryRow]) -> [StudioModelFamily] {
        let present = Set(rows.map(family(of:)))
        return StudioModelFamily.allCases.filter(present.contains)
    }

    static func filter(
        _ rows: [StudioModelInventoryRow],
        family: StudioModelFamily?,
        query: String
    ) -> [StudioModelInventoryRow] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return rows.filter { row in
            if let family, self.family(of: row) != family { return false }
            guard !trimmed.isEmpty else { return true }
            return row.id.lowercased().contains(trimmed)
                || displayName(row).lowercased().contains(trimmed)
                || row.category.lowercased().contains(trimmed)
                || self.family(of: row).title.lowercased().contains(trimmed)
        }
    }

    /// Installed rows plus any row that is being pulled right now, so a pull shows up in the
    /// list the moment it starts.
    static func listRows(
        _ rows: [StudioModelInventoryRow],
        pullingIDs: Set<String>
    ) -> [StudioModelInventoryRow] {
        rows.filter { $0.isInstalled || pullingIDs.contains($0.id) }
    }

    static func status(of row: StudioModelInventoryRow, job: StudioModelsJob?) -> StudioModelRowStatus {
        if let job, job.kind == .pull, job.modelID == row.id {
            return .pulling(job.fraction)
        }
        guard row.isInstalled else { return .missing }
        if row.supported == false { return .attention }
        return .installed
    }

    /// "Image · 2.1 GB", "Vision · pulling 25%", "Chat · 4.8 GB download".
    static func meta(for row: StudioModelInventoryRow, status: StudioModelRowStatus) -> String {
        let family = family(of: row).title
        switch status {
        case .pulling(let fraction):
            guard let fraction else { return "\(family) · pulling…" }
            return "\(family) · pulling \(Int((fraction * 100).rounded()))%"
        case .missing:
            let size = row.displayedSize
            return size == "—" ? "\(family) · not installed" : "\(family) · \(size) download"
        case .installed, .attention:
            let size = row.size
            guard !size.isEmpty, size != "—", size != "not measured" else { return family }
            return "\(family) · \(size)"
        }
    }

    /// "Q4 · 2.1 GB" for the detail chip row; the quantization comes from `model info`.
    static func sizeChip(for row: StudioModelInventoryRow, facts: StudioModelInfoFacts?) -> String? {
        var parts: [String] = []
        if let quantization = facts?.quantization { parts.append(quantization) }
        let size = row.isInstalled ? row.size : row.displayedSize
        if !size.isEmpty, size != "—", size != "not measured" { parts.append(size) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    /// Domain titles whose prompt task defaults to this model ("Default for Image").
    static func defaultDomainTitles(for modelID: String) -> [String] {
        var titles: [String] = []
        for task in StudioTask.allCases {
            guard let mode = task.mode,
                  let template = CommandCatalog.template(id: mode.defaultTemplateID),
                  template.defaultModel == modelID else { continue }
            let title = task.domain.title
            if !titles.contains(title) { titles.append(title) }
        }
        return titles
    }

    /// Runs from the Library that used this model: conversation threads record it directly,
    /// other runs carry it in their draft or `--model` argument.
    static func usage(of modelID: String, in items: [StudioLibraryItem]) -> StudioModelUsageSummary? {
        let matches = items.filter { uses(modelID, item: $0) }
        guard let latest = matches.map(\.updatedAt).max() else { return nil }
        return StudioModelUsageSummary(lastUsed: latest, runs: matches.count)
    }

    static func uses(_ modelID: String, item: StudioLibraryItem) -> Bool {
        if item.model == modelID || item.commandDraft?.model == modelID { return true }
        let tokens = item.commandPreview.split(whereSeparator: \.isWhitespace).map(String.init)
        guard let flag = tokens.firstIndex(where: { $0 == "--model" || $0 == "-m" }),
              flag + 1 < tokens.count else {
            return tokens.contains { $0 == "--model=\(modelID)" }
        }
        return tokens[flag + 1] == modelID
    }

    /// "1:31 PM · 214 runs" today, "Aug 30 · 3 runs" otherwise.
    static func usageLine(_ usage: StudioModelUsageSummary, now: Date = Date(), calendar: Calendar = .current) -> String {
        let when: String
        if calendar.isDate(usage.lastUsed, inSameDayAs: now) {
            when = usage.lastUsed.formatted(date: .omitted, time: .shortened)
        } else {
            when = shortDate(usage.lastUsed)
        }
        return "\(when) · \(usage.runs) \(usage.runs == 1 ? "run" : "runs")"
    }

    /// Wall-clock length of the latest completed run with this model ("3.4 s").
    static func lastRunDuration(for modelID: String, in items: [StudioLibraryItem]) -> String? {
        let completed = items
            .filter { uses(modelID, item: $0) && $0.status == .completed && !$0.isConversation }
            .sorted { $0.updatedAt > $1.updatedAt }
        guard let latest = completed.first else { return nil }
        let seconds = latest.updatedAt.timeIntervalSince(latest.createdAt)
        guard seconds > 0 else { return nil }
        return duration(seconds)
    }

    /// The latest quality-gate run: "Quality gate passed · Aug 30".
    static func gateLine(in items: [StudioLibraryItem]) -> (text: String, ok: Bool)? {
        guard let latest = items
            .filter({ $0.templateID == .qualityGate })
            .sorted(by: { $0.updatedAt > $1.updatedAt })
            .first else { return nil }
        switch latest.status {
        case .completed:
            return ("Quality gate passed · \(shortDate(latest.updatedAt))", true)
        case .failed:
            return ("Quality gate failed · \(shortDate(latest.updatedAt))", false)
        case .running, .queued:
            return ("Quality gate running", true)
        case .cancelled, .interrupted:
            return ("Quality gate " + latest.status.rawValue, false)
        }
    }

    /// The latest completed benchmark: "Aug 30 · Lite suite".
    static func benchmarkLine(in items: [StudioLibraryItem]) -> String? {
        guard let latest = items
            .filter({ $0.templateID == .modelBenchmarkFused || $0.templateID == .modelBenchmarkFusedFixture })
            .filter({ $0.status == .completed })
            .sorted(by: { $0.updatedAt > $1.updatedAt })
            .first else { return nil }
        var line = shortDate(latest.updatedAt)
        if let suite = latest.commandDraft?.benchmarkSuite, !suite.isBlank {
            line += " · \(suite.capitalized) suite"
        }
        return line
    }

    /// Composer-initiated pulls show up as running `.modelPull` Library rows.
    static func libraryPullJob(
        in items: [StudioLibraryItem],
        rows: [StudioModelInventoryRow],
        progressByRequestID: [UUID: StudioRunProgress]
    ) -> StudioModelsJob? {
        guard let item = items.first(where: { $0.templateID == .modelPull && $0.status == .running }) else {
            return nil
        }
        let modelID = item.commandDraft?.model ?? item.model ?? pulledModelID(fromPreview: item.commandPreview)
        let subject = rows.first { $0.id == modelID }.map(displayName) ?? modelID ?? "model"
        return StudioModelsJob(
            kind: .pull,
            modelID: modelID,
            subject: subject,
            progress: progressByRequestID[item.id],
            libraryItemID: item.id
        )
    }

    static func pulledModelID(fromPreview preview: String) -> String? {
        let tokens = preview.split(whereSeparator: \.isWhitespace).map(String.init)
        guard let pull = tokens.firstIndex(of: "pull"), pull + 1 < tokens.count,
              pull > 0, tokens[pull - 1] == "model" else { return nil }
        return tokens[pull + 1]
    }

    static func facts(fromInfo output: String) -> StudioModelInfoFacts {
        var facts = StudioModelInfoFacts()
        var precision: String?
        for rawLine in output.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("Model Root:") {
                let path = line.dropFirst("Model Root:".count).trimmingCharacters(in: .whitespaces)
                if !path.isEmpty { facts.root = path }
            } else if line.hasPrefix("Manifest: (missing)") {
                facts.hasManifest = false
            } else if line.hasPrefix("Manifest (") {
                facts.hasManifest = true
            } else if line.hasPrefix("quantization:"), facts.quantization == nil {
                if let bits = value(of: "bits", in: line) { facts.quantization = "Q\(bits)" }
            } else if line.hasPrefix("precision:"), precision == nil {
                let value = line.dropFirst("precision:".count).trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { precision = value }
            } else if line.hasPrefix("isValid:") {
                let flag = line.dropFirst("isValid:".count).trimmingCharacters(in: .whitespaces)
                facts.isValid = flag == "true"
            }
        }
        // Quantized checkpoints report both; the bit width is the fact users recognize.
        if facts.quantization == nil { facts.quantization = precision }
        return facts
    }

    /// "~/Library/Application Support/MereRun/models" for the Store row.
    static func abbreviatedPath(_ path: String, home: String = NSHomeDirectory()) -> String {
        guard !home.isEmpty, path.hasPrefix(home) else { return path }
        return "~" + path.dropFirst(home.count)
    }

    static func sourceLine(for row: StudioModelInventoryRow) -> String? {
        guard let repository = row.sourceRepository, !repository.isBlank else { return nil }
        return "huggingface.co/\(repository)"
    }

    static func memoryLine(for row: StudioModelInventoryRow) -> String? {
        switch (row.minimumUnifiedMemoryGB, row.recommendedUnifiedMemoryGB) {
        case let (minimum?, recommended?) where recommended > minimum:
            return "\(minimum) GB min · \(recommended) GB recommended"
        case let (minimum?, _):
            return "\(minimum) GB min"
        case let (nil, recommended?):
            return "\(recommended) GB recommended"
        case (nil, nil):
            return nil
        }
    }

    /// "LoRA · 48 MB · v2" under an adapter's name.
    static func adapterMeta(format: String, byteCount: Int64, version: String, installed: Bool) -> String {
        var parts = [format.lowercased() == "lora" ? "LoRA" : format.uppercased()]
        if byteCount > 0 { parts.append(bytes(byteCount)) }
        if !version.isBlank { parts.append("v\(version)") }
        if !installed { parts.append("not pulled") }
        return parts.joined(separator: " · ")
    }

    static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    static func shortDate(_ date: Date) -> String {
        date.formatted(.dateTime.month(.abbreviated).day())
    }

    static func duration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.1f s", seconds) }
        let minutes = Int(seconds) / 60
        let rest = Int(seconds) % 60
        return "\(minutes) min \(rest) s"
    }

    private static func value(of key: String, in line: String) -> String? {
        for token in line.split(separator: " ") {
            let pair = token.split(separator: "=", maxSplits: 1)
            guard pair.count == 2, pair[0] == Substring(key) else { continue }
            return String(pair[1])
        }
        return nil
    }
}
