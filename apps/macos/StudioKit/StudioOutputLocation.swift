import Foundation

/// Where a run's file lands, and what it is called.
///
/// Studio used to write every artifact to
/// `~/Library/Application Support/MereRun/App Outputs/<templateID>-<timestamp>.<ext>` — a folder
/// Finder hides and a name nobody can read. Outputs now go to a user-visible folder chosen by
/// what the file *is*, filed under the domain that made it, and named after the prompt:
///
/// ```
/// ~/Pictures/mere.run/Image/a-ceramic-coffee-mug-in-soft-morning-light-8812.png
/// ~/Music/mere.run/Voice/welcome-aboard-3f9c21.wav
/// ~/Documents/mere.run/Vision/every-coffee-cup-and-what-it-sits-on-a41b02.json
/// ```
///
/// Application Support keeps metadata only (`library.json`, receipts). Existing Library rows keep
/// the paths they recorded; nothing is migrated.
///
/// A user-visible root can be overridden in Settings ▸ General (`mererun.app.outputRoot`); when it
/// is set, every domain is filed under `<root>/<Domain>/` regardless of media. When the
/// destination cannot be created — a sandbox denial, a missing external volume, a read-only
/// home — `preparingDestination` moves the run back to App Outputs and reports why, so a run never
/// fails just because of where it was going to write.
package enum StudioOutputLocation {
    /// The `UserDefaults` key holding the user's chosen root ("" = the per-media defaults).
    package static let rootDefaultsKey = "mererun.app.outputRoot"

    /// Where the chosen root is read from: the app's defaults. A test that files runs somewhere
    /// else points this at a throwaway suite instead of writing into the user's settings.
    nonisolated(unsafe) package static var defaults: UserDefaults = .standard

    /// Longest slug we keep before the identifier suffix.
    package static let maximumSlugLength = 60

    // MARK: - Roots

    /// The folder a file of `kind` for `domain` belongs in, given a configured root (may be blank)
    /// and a home directory. Pure: touches no filesystem.
    package static func directory(
        domain: StudioDomain,
        kind: StudioOutputFileKind,
        configuredRoot: String,
        home: URL
    ) -> URL {
        let root = configuredRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        if !root.isEmpty {
            return URL(fileURLWithPath: NSString(string: root).expandingTildeInPath, isDirectory: true)
                .appendingPathComponent(domain.title, isDirectory: true)
        }
        return home
            .appendingPathComponent(mediaFolder(for: kind), isDirectory: true)
            .appendingPathComponent("mere.run", isDirectory: true)
            .appendingPathComponent(domain.title, isDirectory: true)
    }

    /// Pictures for stills and clips, Music for anything that plays, Documents for the rest.
    package static func mediaFolder(for kind: StudioOutputFileKind) -> String {
        switch kind {
        case .image, .video: return "Pictures"
        case .audio: return "Music"
        case .text, .model3D, .other: return "Documents"
        }
    }

    /// `~/Library/Application Support/MereRun/App Outputs` — the pre-v2 destination, kept as the
    /// fallback when a user-visible folder cannot be written.
    package static func appOutputsRoot(fileManager: FileManager = .default) -> URL {
        let support = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("App Outputs", isDirectory: true)
    }

    private static func configuredRoot() -> String {
        defaults.string(forKey: rootDefaultsKey) ?? ""
    }

    // MARK: - Reservations

    /// The destinations of runs submitted in this process. A run's file is not on disk until the
    /// CLI writes it, so a second proposal made in the same second — the page advancing its own
    /// path right after Submit, or two pages sharing a clock — would name the same file; naming
    /// treats a reserved path like an existing one.
    private static let reservations = Reservations()

    private final class Reservations: @unchecked Sendable {
        private let lock = NSLock()
        private var paths: Set<String> = []

        func insert(_ path: String) {
            lock.lock()
            paths.insert(URL(fileURLWithPath: path).standardizedFileURL.path)
            lock.unlock()
        }

        func contains(_ path: String) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return paths.contains(URL(fileURLWithPath: path).standardizedFileURL.path)
        }
    }

    /// Marks `path` as taken by a submitted run. `preparingDestination` does this for every run
    /// it prepares; it is exposed so a test can prove the naming steps aside.
    package static func reserve(_ path: String) {
        guard !path.isBlank else { return }
        reservations.insert(path)
    }

    private static func isTaken(_ path: String, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: path) || reservations.contains(path)
    }

    // MARK: - Names

    /// A file-name stem from free text: lowercased, diacritics folded, everything that is not a
    /// letter or a digit treated as a word break, words joined by hyphens, and cut at a word
    /// boundary no longer than `limit`. Empty when the text carries no letters or digits.
    package static func slug(_ text: String, limit: Int = maximumSlugLength) -> String {
        let folded = text.folding(options: [.diacriticInsensitive, .widthInsensitive], locale: Locale(identifier: "en_US"))
            .lowercased()
        var words: [String] = []
        var current = ""
        for character in folded.unicodeScalars {
            if CharacterSet.alphanumerics.contains(character), character.isASCII {
                current.unicodeScalars.append(character)
            } else if !current.isEmpty {
                words.append(current)
                current = ""
            }
        }
        if !current.isEmpty { words.append(current) }
        guard !words.isEmpty, limit > 0 else { return "" }

        var slug = ""
        for word in words {
            if slug.isEmpty {
                slug = String(word.prefix(limit))
                continue
            }
            guard slug.count + 1 + word.count <= limit else { break }
            slug += "-" + word
        }
        return slug
    }

    /// The stable part of a name: the prompt's slug, or `fallbackStem` when the prompt has no
    /// usable characters (Transcribe and the other input-first tasks have no prompt at all).
    package static func stem(prompt: String, fallbackStem: String) -> String {
        let promptSlug = slug(prompt)
        if !promptSlug.isEmpty { return promptSlug }
        let fallback = slug(fallbackStem)
        return fallback.isEmpty ? "output" : fallback
    }

    /// The identifier that follows the slug: the seed when the run has one (so a reproducible
    /// picture is named after the number that reproduces it), otherwise a short id derived from
    /// the run's own settings. It is derived rather than random so the path the Command view shows
    /// is the path the run writes — a random id would change on every keystroke.
    package static func identifier(seed: String, fingerprint: String) -> String {
        let seedSlug = slug(seed, limit: 20)
        return seedSlug.isEmpty ? shortIdentifier(for: fingerprint) : seedSlug
    }

    /// Six hex characters of an FNV-1a hash — stable across launches, unlike `Hasher`.
    package static func shortIdentifier(for text: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in Array(text.utf8) {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(String(format: "%016lx", hash).suffix(6))
    }

    /// `<stem>-<identifier>.<ext>`, with `-2`, `-3`… inserted before the extension until `exists`
    /// says the name is free. `exists` is a predicate rather than a `FileManager` call so the
    /// naming rule is testable without touching a disk.
    package static func uniqueFileName(
        stem: String,
        identifier: String,
        fileExtension: String,
        exists: (String) -> Bool
    ) -> String {
        let base = identifier.isEmpty ? stem : "\(stem)-\(identifier)"
        let suffix = fileExtension.isEmpty ? "" : ".\(fileExtension)"
        var candidate = base + suffix
        var counter = 2
        while exists(candidate) {
            candidate = "\(base)-\(counter)\(suffix)"
            counter += 1
            // A pathological directory must not hang a run; fall back to a one-off random id.
            if counter > 999 {
                candidate = "\(base)-\(shortIdentifier(for: UUID().uuidString))\(suffix)"
                break
            }
        }
        return candidate
    }

    // MARK: - Whole destinations

    /// The full path a run should write to. Pure apart from the existence checks that make the
    /// name collision-safe; the directory is created later, by `preparingDestination`.
    package static func outputURL(
        domain: StudioDomain,
        prompt: String,
        seed: String = "",
        fingerprint: String = "",
        fallbackStem: String,
        fileExtension: String,
        identifierOverride: String? = nil,
        configuredRoot: String? = nil,
        home: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true),
        fileManager: FileManager = .default
    ) -> URL {
        let directory = directory(
            domain: domain,
            kind: StudioOutputFileKind.classify(URL(fileURLWithPath: "output.\(fileExtension)")),
            configuredRoot: configuredRoot ?? Self.configuredRoot(),
            home: home
        )
        let name = uniqueFileName(
            stem: stem(prompt: prompt, fallbackStem: fallbackStem),
            identifier: identifierOverride ?? identifier(seed: seed, fingerprint: fingerprint),
            fileExtension: fileExtension,
            exists: { isTaken(directory.appendingPathComponent($0).path, fileManager: fileManager) }
        )
        return directory.appendingPathComponent(name, isDirectory: false)
    }

    /// A directory destination (`--output-dir` commands): `<root>/<Domain>/<stem>-<identifier>`.
    package static func outputDirectoryURL(
        domain: StudioDomain,
        prompt: String,
        fingerprint: String = "",
        fallbackStem: String,
        identifierOverride: String? = nil,
        configuredRoot: String? = nil,
        home: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true),
        fileManager: FileManager = .default
    ) -> URL {
        let directory = directory(
            domain: domain,
            kind: .other,
            configuredRoot: configuredRoot ?? Self.configuredRoot(),
            home: home
        )
        let name = uniqueFileName(
            stem: stem(prompt: prompt, fallbackStem: fallbackStem),
            identifier: identifierOverride ?? shortIdentifier(for: fingerprint),
            fileExtension: "",
            exists: { isTaken(directory.appendingPathComponent($0).path, fileManager: fileManager) }
        )
        return directory.appendingPathComponent(name, isDirectory: true)
    }

    /// The destination a template starts a draft with, before anyone types a prompt: the domain's
    /// user-visible folder and the command's own name plus a timestamp. Drafts are made once per
    /// surface, so the timestamp is stable for as long as the draft is — the Command Console shows
    /// the path the run will actually write.
    package static func templateOutputPath(
        templateID: CommandTemplateID,
        title: String,
        outputKind: CommandOutputKind,
        now: Date = Date()
    ) -> String? {
        let stamp = DateFormatter.mereRunTimestamp.string(from: now)
        switch outputKind {
        case .file(let ext):
            return outputURL(
                domain: templateID.studioDomain,
                prompt: "",
                fallbackStem: title,
                fileExtension: ext,
                identifierOverride: stamp
            ).path
        case .directory:
            return outputDirectoryURL(
                domain: templateID.studioDomain,
                prompt: "",
                fallbackStem: title,
                identifierOverride: stamp
            ).path
        case .none:
            return nil
        }
    }

    /// The destination a specialist page proposes before it runs: the domain's folder wherever
    /// Settings says, named `<name>-<timestamp>`. These pages have no prompt to name a file after
    /// and several write a whole directory, so a timestamp is what keeps two runs apart. It is the
    /// folder `templateOutputPath` sends the same command to from the Command Console, so a page
    /// and the console file one command's work together.
    package static func specialistDirectory(
        domain: StudioDomain,
        name: String,
        now: Date = Date(),
        configuredRoot: String? = nil,
        home: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true),
        fileManager: FileManager = .default
    ) -> URL {
        outputDirectoryURL(
            domain: domain,
            prompt: "",
            fallbackStem: name,
            identifierOverride: DateFormatter.mereRunTimestamp.string(from: now),
            configuredRoot: configuredRoot,
            home: home,
            fileManager: fileManager
        )
    }

    /// One file a specialist page writes, filed the same way: `<name>-<timestamp>.<ext>` in the
    /// media folder the extension calls for, or under the configured root.
    package static func specialistFile(
        domain: StudioDomain,
        name: String,
        fileExtension: String,
        now: Date = Date(),
        configuredRoot: String? = nil,
        home: URL = URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true),
        fileManager: FileManager = .default
    ) -> URL {
        outputURL(
            domain: domain,
            prompt: "",
            fallbackStem: name,
            fileExtension: fileExtension,
            identifierOverride: DateFormatter.mereRunTimestamp.string(from: now),
            configuredRoot: configuredRoot,
            home: home,
            fileManager: fileManager
        )
    }

    /// The destination for one Studio run: the same folder the template already chose, but named
    /// after the prompt. `.none` output kinds (chat, the utility probes) keep `existing`.
    package static func namedOutputPath(
        templateID: CommandTemplateID,
        outputKind: CommandOutputKind,
        prompt: String,
        seed: String,
        fingerprint: String,
        fallbackStem: String,
        existing: String
    ) -> String {
        switch outputKind {
        case .file(let ext):
            return outputURL(
                domain: templateID.studioDomain,
                prompt: prompt,
                seed: seed,
                fingerprint: fingerprint,
                fallbackStem: fallbackStem,
                fileExtension: ext
            ).path
        case .directory:
            return outputDirectoryURL(
                domain: templateID.studioDomain,
                prompt: prompt,
                fingerprint: fingerprint,
                fallbackStem: fallbackStem
            ).path
        case .none:
            return existing
        }
    }

    // MARK: - Preparing a run

    /// The result of making a destination real: the draft to run, and why it had to move.
    package struct Preparation: Equatable {
        package var draft: CommandDraft
        /// nil when the intended destination was created (or already existed).
        package var fallbackReason: String?
    }

    /// Creates the destination directory of `draft`, redirecting the run to App Outputs when that
    /// is impossible. Every path the draft writes that lived beside the output — the vision result
    /// document, the per-detection mask directory, the timings report — moves with it, so a
    /// redirected run still keeps its sidecars together.
    package static func preparingDestination(
        of draft: CommandDraft,
        fileManager: FileManager = .default
    ) -> Preparation {
        guard !draft.outputPath.isBlank else { return Preparation(draft: draft) }
        let output = URL(fileURLWithPath: draft.outputPath)
        // A directory destination is itself the folder to create; a file destination needs its parent.
        let intended = draft.outputPath.hasSuffix("/") ? output : output.deletingLastPathComponent()

        do {
            try fileManager.createDirectory(at: intended, withIntermediateDirectories: true)
            reserve(draft.outputPath)
            return Preparation(draft: draft)
        } catch {
            let fallbackDirectory = appOutputsRoot(fileManager: fileManager)
            guard (try? fileManager.createDirectory(at: fallbackDirectory, withIntermediateDirectories: true)) != nil else {
                // Nowhere to write at all: run as asked and let the CLI report the real failure.
                return Preparation(draft: draft)
            }
            var moved = draft
            let originalDirectory = intended.standardizedFileURL.path
            moved.outputPath = redirect(draft.outputPath, from: originalDirectory, to: fallbackDirectory)
            moved.visionJSONOutputPath = redirect(draft.visionJSONOutputPath, from: originalDirectory, to: fallbackDirectory)
            moved.visionMaskOutputDirectory = redirect(
                draft.visionMaskOutputDirectory, from: originalDirectory, to: fallbackDirectory
            )
            moved.timingsOutputPath = redirect(draft.timingsOutputPath, from: originalDirectory, to: fallbackDirectory)
            reserve(moved.outputPath)
            return Preparation(
                draft: moved,
                fallbackReason: "Could not write to \(abbreviate(intended)): \(error.localizedDescription)"
            )
        }
    }

    /// `preparingDestination` for a whole request — the specialist pages and the Command view
    /// submit one of these rather than a prompt draft. When the draft moves, the `--output` the
    /// request's Command edits carry moves with it, so a redirected run and its recorded argv
    /// agree. The reason, when there is one, is for the shell's banner.
    package static func preparing(
        _ request: StudioRunRequest,
        fileManager: FileManager = .default
    ) -> (request: StudioRunRequest, fallbackReason: String?) {
        let prepared = preparingDestination(of: request.draft, fileManager: fileManager)
        guard prepared.draft != request.draft else { return (request, nil) }
        let flag = request.templateID.capability?.output.flag ?? "--output"
        let moved = StudioRunRequest(
            id: request.id,
            mode: request.mode,
            templateID: request.templateID,
            template: request.template,
            draft: prepared.draft,
            createdAt: request.createdAt,
            conversationID: request.conversationID,
            execution: request.execution?.replacing(flag, with: prepared.draft.outputPath),
            parentID: request.parentID
        )
        return (moved, prepared.fallbackReason)
    }

    /// The one line the shell shows when a run had to move: why, and where it went instead.
    package static func fallbackNotice(_ reason: String) -> String {
        "\(reason) Saving to \(abbreviate(appOutputsRoot())) instead."
    }

    /// Rewrites a path that lived in `directory` so it lives in `replacement` instead. Paths
    /// elsewhere (a user-chosen report location) are left exactly as they are.
    private static func redirect(_ path: String, from directory: String, to replacement: URL) -> String {
        guard !path.isBlank else { return path }
        let url = URL(fileURLWithPath: path)
        guard url.deletingLastPathComponent().standardizedFileURL.path == directory else { return path }
        return replacement.appendingPathComponent(url.lastPathComponent).path
    }

    /// `~/Pictures/mere.run/Image` rather than the full home path, for the banner text.
    package static func abbreviate(_ url: URL, home: String = NSHomeDirectory()) -> String {
        let path = url.standardizedFileURL.path
        guard !home.isEmpty, path == home || path.hasPrefix(home + "/") else { return path }
        return "~" + path.dropFirst(home.count)
    }

    // MARK: - Task drafts

    /// The sidecars `destination(for:)` derives beside a primary output: the flag, the extension
    /// the file takes, and the suffix that keeps it apart from a primary of the same extension
    /// (`--context-output` beside a `.json` transcription is `<stem>-context.json`; the vision
    /// result document keeps the bare `<stem>.json` the Analyze canvas has always read).
    /// `--mask-output-dir` is derived too, as `<stem>-masks`. `StudioExecution.replay` renames
    /// these same flags, the mask directory, and `StudioTaskSchema.chosenOutputFlags` when a run
    /// is replayed, so a replay never writes over the original's documents.
    package static let derivedSidecars: [(flag: String, fileExtension: String, suffix: String)] = [
        ("--json-output", "json", ""), ("--jsonl-output", "jsonl", ""),
        ("--context-output", "json", "-context"), ("--timings-output", "json", "-timings"),
    ]

    /// Every flag `destination(for:)` and `StudioExecution.replay` treat as a destination.
    package static var sidecarFlags: Set<String> {
        Set(derivedSidecars.map(\.flag)).union(["--mask-output-dir"])
    }

    /// The draft with its destination filled the way the composer names a prompt run's: the
    /// capability's output flag set to `namedOutputPath` (the domain's folder, the prompt's slug
    /// or the input's name, a derived identifier), and every sidecar in `derivedSidecars` the
    /// capability declares beside it with the same stem, plus `--mask-output-dir`. A destination
    /// the user pointed outside the app's folder in the Command view is kept, as is a sidecar
    /// pointed anywhere but beside the app's own primary; a sidecar the app named earlier moves
    /// with the primary, and an app-named `--context-output` is dropped once
    /// `--no-musical-context` is on. The identifier is derived from what the run does
    /// (template, prompt, model, inputs, options) and never from where it writes, so calling
    /// this again on its own result names the same files: the Command view's "Will run" and the
    /// run agree. The task runner calls it at submit time, when `reserve` makes two runs in one
    /// second step apart.
    package static func destination(
        for draft: StudioTaskDraft,
        fileManager: FileManager = .default
    ) -> StudioTaskDraft {
        guard let capability = draft.capability, let template = draft.template,
              let flag = capability.output.flag,
              let option = capability.options.first(where: { $0.flag == flag }) else { return draft }
        var named = draft
        let templateID = draft.templateID
        let outputKind: CommandOutputKind
        if option.kind == .directory || capability.output.kind == .directory {
            outputKind = .directory
        } else if let ext = capability.output.fileExtension ?? formatExtension(in: draft) {
            outputKind = .file(ext)
        } else {
            outputKind = .none
        }
        let primaryInput = draft.primaryInputPath
        let existing = draft.text(flag)
        if outputKind != .none, existing.isBlank || isAppChosen(existing, templateID: templateID, kind: outputKind) {
            // The argv without its destinations: the same command pointed at another folder is
            // the same run.
            var bare = draft
            for output in StudioTaskSchema.outputFlags(for: capability) { bare.form.values[output] = nil }
            let path = namedOutputPath(
                templateID: templateID,
                outputKind: outputKind,
                prompt: draft.prompt,
                seed: draft.text("--seed"),
                fingerprint: ([templateID.rawValue] + bare.arguments).joined(separator: "\u{1}"),
                fallbackStem: primaryInput.isBlank
                    ? template.title
                    : URL(fileURLWithPath: primaryInput).deletingPathExtension().lastPathComponent,
                existing: existing
            )
            named.form[flag] = .text(path)
        }
        let output = named.text(flag)
        guard !output.isBlank else { return named }
        let stem = URL(fileURLWithPath: output).deletingPathExtension()
        let declared = Set(capability.options.map(\.flag))
        // A sidecar the app named sits beside the primary it was derived from — the one the
        // draft carried in, or the one just named; anywhere else is the user's choice.
        let appFolders = Set([existing, output].filter { !$0.isBlank }
            .map { URL(fileURLWithPath: $0).deletingLastPathComponent().standardizedFileURL.path })
        func isAppNamed(_ path: String) -> Bool {
            path.isBlank || appFolders.contains(URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL.path)
        }
        let skipsContext = named.form["--no-musical-context"].flag == true
        for sidecar in derivedSidecars where sidecar.flag != flag && declared.contains(sidecar.flag) {
            guard isAppNamed(named.text(sidecar.flag)) else { continue }
            if sidecar.flag == "--context-output", skipsContext {
                named.form[sidecar.flag] = .unset
                continue
            }
            named.form[sidecar.flag] = .text(
                stem.deletingLastPathComponent()
                    .appendingPathComponent(stem.lastPathComponent + sidecar.suffix)
                    .appendingPathExtension(sidecar.fileExtension).path
            )
        }
        if flag != "--mask-output-dir", declared.contains("--mask-output-dir"), isAppNamed(named.text("--mask-output-dir")) {
            named.form["--mask-output-dir"] = .text(
                stem.deletingLastPathComponent().appendingPathComponent("\(stem.lastPathComponent)-masks", isDirectory: true).path
            )
        }
        return named
    }

    /// The extension a command whose output follows one of its switches writes: the chosen
    /// `--format` (`speech diarize`, `music transcribe`), with MIDI's conventional extension, or
    /// JSON versus plain text for `text anonymize --json`; nil when the capability has no such
    /// option.
    private static func formatExtension(in draft: StudioTaskDraft) -> String? {
        if draft.templateID == .textAnonymize {
            return draft.form["--json"] == .flag(true) ? "json" : "txt"
        }
        guard let option = draft.capability?.options.first(where: { $0.flag == "--format" }) else { return nil }
        let chosen = draft.text("--format")
        let format = chosen.isEmpty ? (option.defaultValue ?? option.choices.first ?? "") : chosen
        switch format {
        case "": return nil
        case "midi": return "mid"
        default: return format
        }
    }

    /// Whether `path` is one the app proposed (the template's stamped default, or an earlier
    /// naming) rather than a folder the user picked: it sits directly in the domain's folder.
    private static func isAppChosen(_ path: String, templateID: CommandTemplateID, kind: CommandOutputKind) -> Bool {
        let fileExtension: String
        switch kind {
        case .file(let ext): fileExtension = ext
        case .directory, .none: fileExtension = ""
        }
        let folder = directory(
            domain: templateID.studioDomain,
            kind: StudioOutputFileKind.classify(URL(fileURLWithPath: "output.\(fileExtension)")),
            configuredRoot: configuredRoot(),
            home: URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        )
        return URL(fileURLWithPath: path).deletingLastPathComponent().standardizedFileURL.path
            == folder.standardizedFileURL.path
    }
}
