import Foundation

// Music ▸ Train builds the dataset manifest `mere.run music train-adapter --dataset` reads instead of
// asking for a hand-written JSONL file. The record shape mirrors
// `ACEStepAdapterTrainingPlan.ManifestRecord` in `MereRunCore` (which Studio does not import):
// `audio` and `caption` are required, `lyrics` is optional, and `loadManifest` accepts either a
// JSON array or one record per line. Studio writes one record per line.

/// One training clip: an audio file, what it sounds like, and optionally what is sung.
package struct StudioMusicTrainingClip: Codable, Equatable, Identifiable {
    package var id = UUID()
    /// An absolute path. Relative paths in an imported manifest are resolved against that
    /// manifest's folder, the way the trainer resolves them.
    package var audioPath: String
    package var caption: String
    package var lyrics: String

    package init(audioPath: String, caption: String = "", lyrics: String = "") {
        self.audioPath = audioPath
        self.caption = caption
        self.lyrics = lyrics
    }

    package var fileName: String {
        URL(fileURLWithPath: audioPath).lastPathComponent
    }

    package var audioURL: URL {
        URL(fileURLWithPath: NSString(string: audioPath).expandingTildeInPath).standardizedFileURL
    }
}

/// The manifest the Music trainer edits: ordered clips, each with a caption and optional lyrics.
package struct StudioMusicTrainingManifest: Codable, Equatable {
    package var clips: [StudioMusicTrainingClip]

    package init(clips: [StudioMusicTrainingClip] = []) {
        self.clips = clips
    }

    /// What the trainer decodes: everything `AVAudioFile` reads, which it resamples to 48 kHz stereo.
    package static let audioExtensions: Set<String> = ["wav", "aif", "aiff", "caf", "mp3", "m4a", "aac", "flac"]

    package static func isAudioFile(_ url: URL) -> Bool {
        audioExtensions.contains(url.pathExtension.lowercased())
    }

    /// One clip per audio file at the top of `directory`, sorted by name, captioned from a sibling
    /// `.txt` when there is one — the same pairing the image trainer's datasets use.
    package static func clips(scanning directory: URL, fileManager: FileManager = .default) -> [StudioMusicTrainingClip] {
        let urls = (try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls
            .filter(isAudioFile)
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
            .map { url in
                let captionURL = url.deletingPathExtension().appendingPathExtension("txt")
                let caption = (try? String(contentsOf: captionURL, encoding: .utf8))?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                return StudioMusicTrainingClip(audioPath: url.path, caption: caption)
            }
    }

    /// Whether a clip's audio file is there. The default asks the file system; the editor passes a
    /// cache so a long list is not stat-ed on every render.
    package typealias FileCheck = @Sendable (URL) -> Bool

    package static let fileExists: FileCheck = { FileManager.default.fileExists(atPath: $0.path) }

    /// Clips the trainer would accept: an audio file that exists and a caption.
    package func readyClipCount(fileExists: FileCheck = StudioMusicTrainingManifest.fileExists) -> Int {
        clips.filter { clipProblems($0, fileExists: fileExists).isEmpty }.count
    }

    /// What stops the manifest from training, in the order the page shows them; empty when it is
    /// ready. These are the checks `music train-adapter` makes when it loads the manifest ("empty
    /// audio path", "empty caption") and when it opens each clip ("Dataset audio N not found").
    package func problems(fileExists: FileCheck = StudioMusicTrainingManifest.fileExists) -> [String] {
        guard !clips.isEmpty else { return ["Add at least one audio clip."] }
        return clips.enumerated().flatMap { index, clip in
            clipProblems(clip, fileExists: fileExists).map { "Clip \(index + 1) \($0)" }
        }
    }

    /// One clip's problems, without its number: "has no audio file.", "needs a caption."
    package func clipProblems(_ clip: StudioMusicTrainingClip, fileExists: FileCheck = StudioMusicTrainingManifest.fileExists) -> [String] {
        var problems: [String] = []
        let path = clip.audioPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.isEmpty {
            problems.append("has no audio file.")
        } else if !fileExists(clip.audioURL) {
            problems.append("is missing its audio file, \(clip.fileName).")
        }
        if clip.caption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            problems.append("needs a caption.")
        }
        return problems
    }

    /// The JSONL `--dataset` reads: one record per line, absolute audio paths, lyrics only when given.
    /// Line and paragraph separators inside a caption or lyrics (U+2028, U+2029, U+0085, `\r`) become
    /// `\n`: `JSONEncoder` leaves them unescaped, and the trainer splits the file on every newline
    /// character, so any of them would break a record in two.
    package func jsonl() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var lines = Data()
        for clip in clips {
            let lyrics = Self.singleLineBreaks(clip.lyrics).trimmingCharacters(in: .whitespacesAndNewlines)
            let record = Record(
                audio: clip.audioURL.path,
                caption: Self.singleLineBreaks(clip.caption).trimmingCharacters(in: .whitespacesAndNewlines),
                lyrics: lyrics.isEmpty ? nil : lyrics
            )
            lines.append(try encoder.encode(record))
            lines.append(contentsOf: "\n".utf8)
        }
        return lines
    }

    /// Every newline character the trainer's line split recognises, as `\n`.
    static func singleLineBreaks(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.utf8.count)
        for character in text {
            if character == "\n" || !character.isNewline {
                result.append(character)
            } else {
                result.append("\n")
            }
        }
        return result
    }

    /// Where a run's manifest goes: beside the adapter it trains, as `<adapter stem>.dataset.jsonl`
    /// (`-2`, `-3`… when that name is taken by an earlier attempt), so the run folder holds what
    /// produced the adapter and never loses an earlier run's manifest.
    package static func manifestURL(besideOutput outputPath: String, fileManager: FileManager = .default) -> URL {
        let output = URL(fileURLWithPath: NSString(string: outputPath).expandingTildeInPath).standardizedFileURL
        let folder = output.deletingLastPathComponent()
        let name = StudioOutputLocation.uniqueFileName(
            stem: "\(output.deletingPathExtension().lastPathComponent).dataset",
            identifier: "",
            fileExtension: "jsonl",
            exists: { fileManager.fileExists(atPath: folder.appendingPathComponent($0).path) }
        )
        return folder.appendingPathComponent(name)
    }

    /// Where the page keeps the manifest it is editing, so the Command view's Run has a real file to
    /// pass as `--dataset`. The name carries a hash of the content, so a command the Library recorded
    /// keeps pointing at the clips it ran with while the page moves on; `pruneDrafts` keeps the folder
    /// small. Each Start training writes its own copy beside its adapter.
    package static func draftManifestURL(content: Data, fileManager: FileManager = .default) -> URL {
        draftFolder(fileManager: fileManager)
            .appendingPathComponent("dataset-\(StudioOutputLocation.shortIdentifier(for: String(decoding: content, as: UTF8.self))).jsonl")
    }

    /// Removes all but the newest `keeping` manifest drafts, never the one at `current`.
    package static func pruneDrafts(current: URL, keeping: Int = 8, fileManager: FileManager = .default) {
        StudioDraftFiles.prune(in: draftFolder(fileManager: fileManager), matching: "dataset-", current: current, keeping: keeping, fileManager: fileManager)
    }

    /// Whether `url` is one of this page's drafts rather than a manifest the user chose.
    package static func isDraftURL(_ url: URL, fileManager: FileManager = .default) -> Bool {
        url.standardizedFileURL.deletingLastPathComponent() == draftFolder(fileManager: fileManager).standardizedFileURL
    }

    private static func draftFolder(fileManager: FileManager) -> URL {
        StudioOutputLocation.appOutputsRoot(fileManager: fileManager)
            .deletingLastPathComponent()
            .appendingPathComponent("Music Training", isDirectory: true)
    }

    /// Reads a manifest written by hand or by this page, the way the trainer does: a JSON array of
    /// records, or one record per line. Relative audio paths resolve against the manifest's folder.
    package static func importing(_ data: Data, from manifestURL: URL) throws -> StudioMusicTrainingManifest {
        let decoder = JSONDecoder()
        let records: [Record]
        if let array = try? decoder.decode([Record].self, from: data) {
            records = array
        } else {
            let text = String(decoding: data, as: UTF8.self)
            records = try text.split(whereSeparator: \.isNewline).enumerated().map { index, line in
                do {
                    return try decoder.decode(Record.self, from: Data(line.utf8))
                } catch {
                    throw StudioMusicManifestImportError.invalidLine(index + 1)
                }
            }
        }
        guard !records.isEmpty else { throw StudioMusicManifestImportError.empty }
        let folder = manifestURL.deletingLastPathComponent()
        return StudioMusicTrainingManifest(clips: records.map { record in
            let audio = record.audio.trimmingCharacters(in: .whitespacesAndNewlines)
            let path: String
            if audio.hasPrefix("/") || audio.hasPrefix("~") {
                path = NSString(string: audio).expandingTildeInPath
            } else if audio.isEmpty {
                path = ""
            } else {
                path = folder.appendingPathComponent(audio).standardizedFileURL.path
            }
            return StudioMusicTrainingClip(audioPath: path, caption: record.caption, lyrics: record.lyrics ?? "")
        })
    }

    /// The wire shape, `ACEStepAdapterTrainingPlan.ManifestRecord`.
    private struct Record: Codable {
        let audio: String
        let caption: String
        let lyrics: String?
    }
}

package enum StudioMusicManifestImportError: LocalizedError, Equatable {
    case invalidLine(Int)
    case empty

    package var errorDescription: String? {
        switch self {
        case .invalidLine(let line):
            return "Line \(line) is not a clip record; each line needs \"audio\" and \"caption\"."
        case .empty:
            return "This manifest has no clips."
        }
    }
}
