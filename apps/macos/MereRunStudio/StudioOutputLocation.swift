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
enum StudioOutputLocation {
    /// The `UserDefaults` key holding the user's chosen root ("" = the per-media defaults).
    static let rootDefaultsKey = "mererun.app.outputRoot"

    /// Longest slug we keep before the identifier suffix.
    static let maximumSlugLength = 60

    // MARK: - Roots

    /// The folder a file of `kind` for `domain` belongs in, given a configured root (may be blank)
    /// and a home directory. Pure: touches no filesystem.
    static func directory(
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
    static func mediaFolder(for kind: StudioOutputFileKind) -> String {
        switch kind {
        case .image, .video: return "Pictures"
        case .audio: return "Music"
        case .text, .model3D, .other: return "Documents"
        }
    }

    /// `~/Library/Application Support/MereRun/App Outputs` — the pre-v2 destination, kept as the
    /// fallback when a user-visible folder cannot be written.
    static func appOutputsRoot(fileManager: FileManager = .default) -> URL {
        let support = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("App Outputs", isDirectory: true)
    }

    private static func configuredRoot() -> String {
        UserDefaults.standard.string(forKey: rootDefaultsKey) ?? ""
    }

    // MARK: - Names

    /// A file-name stem from free text: lowercased, diacritics folded, everything that is not a
    /// letter or a digit treated as a word break, words joined by hyphens, and cut at a word
    /// boundary no longer than `limit`. Empty when the text carries no letters or digits.
    static func slug(_ text: String, limit: Int = maximumSlugLength) -> String {
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
    static func stem(prompt: String, fallbackStem: String) -> String {
        let promptSlug = slug(prompt)
        if !promptSlug.isEmpty { return promptSlug }
        let fallback = slug(fallbackStem)
        return fallback.isEmpty ? "output" : fallback
    }

    /// The identifier that follows the slug: the seed when the run has one (so a reproducible
    /// picture is named after the number that reproduces it), otherwise a short id derived from
    /// the run's own settings. It is derived rather than random so the path the Command view shows
    /// is the path the run writes — a random id would change on every keystroke.
    static func identifier(seed: String, fingerprint: String) -> String {
        let seedSlug = slug(seed, limit: 20)
        return seedSlug.isEmpty ? shortIdentifier(for: fingerprint) : seedSlug
    }

    /// Six hex characters of an FNV-1a hash — stable across launches, unlike `Hasher`.
    static func shortIdentifier(for text: String) -> String {
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
    static func uniqueFileName(
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
    static func outputURL(
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
            exists: { fileManager.fileExists(atPath: directory.appendingPathComponent($0).path) }
        )
        return directory.appendingPathComponent(name, isDirectory: false)
    }

    /// A directory destination (`--output-dir` commands): `<root>/<Domain>/<stem>-<identifier>`.
    static func outputDirectoryURL(
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
            exists: { fileManager.fileExists(atPath: directory.appendingPathComponent($0).path) }
        )
        return directory.appendingPathComponent(name, isDirectory: true)
    }

    /// The destination a template starts a draft with, before anyone types a prompt: the domain's
    /// user-visible folder and the command's own name plus a timestamp. Drafts are made once per
    /// surface, so the timestamp is stable for as long as the draft is — the Command Console shows
    /// the path the run will actually write.
    static func templateOutputPath(
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

    /// The destination for one Studio run: the same folder the template already chose, but named
    /// after the prompt. `.none` output kinds (chat, the utility probes) keep `existing`.
    static func namedOutputPath(
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
    struct Preparation: Equatable {
        var draft: CommandDraft
        /// nil when the intended destination was created (or already existed).
        var fallbackReason: String?
    }

    /// Creates the destination directory of `draft`, redirecting the run to App Outputs when that
    /// is impossible. Every path the draft writes that lived beside the output — the vision result
    /// document, the per-detection mask directory, the timings report — moves with it, so a
    /// redirected run still keeps its sidecars together.
    static func preparingDestination(
        of draft: CommandDraft,
        fileManager: FileManager = .default
    ) -> Preparation {
        guard !draft.outputPath.isBlank else { return Preparation(draft: draft) }
        let output = URL(fileURLWithPath: draft.outputPath)
        // A directory destination is itself the folder to create; a file destination needs its parent.
        let intended = draft.outputPath.hasSuffix("/") ? output : output.deletingLastPathComponent()

        do {
            try fileManager.createDirectory(at: intended, withIntermediateDirectories: true)
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
            return Preparation(
                draft: moved,
                fallbackReason: "Could not write to \(abbreviate(intended)): \(error.localizedDescription)"
            )
        }
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
    static func abbreviate(_ url: URL, home: String = NSHomeDirectory()) -> String {
        let path = url.standardizedFileURL.path
        guard !home.isEmpty, path == home || path.hasPrefix(home + "/") else { return path }
        return "~" + path.dropFirst(home.count)
    }
}
