import Foundation

// MARK: - music separate

/// The manifest `mere.run music separate` writes as `separation.json` beside its stems and
/// prints on stdout: the source, the model, and one entry per stem it wrote. The stems view
/// lists and plays the stems from it; the raw document stays on the JSON tab.
package struct StudioSeparationManifest: Decodable, Equatable {
    package struct Stem: Decodable, Equatable, Identifiable {
        package let name: String
        package let path: String

        package init(name: String, path: String) {
            self.name = name
            self.path = path
        }

        package var id: String { path }
        package var url: URL { URL(fileURLWithPath: path) }

        /// "Vocals", "Drums", "Instrumental" — the CLI's lower-case stem names, capitalized.
        package var title: String {
            name.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    package struct Source: Decodable, Equatable {
        package let path: String
        package let sampleRate: Int
        package let channels: Int

        package enum CodingKeys: String, CodingKey {
            case path
            case sampleRate = "sample_rate"
            case channels
        }
    }

    package struct Model: Decodable, Equatable {
        package let id: String
        package let computeType: String

        package enum CodingKeys: String, CodingKey {
            case id
            case computeType = "compute_type"
        }
    }

    package let schemaVersion: Int
    package let source: Source
    package let model: Model
    package let overlap: Int
    package let chunks: Int
    package let elapsedSeconds: Double
    package let stems: [Stem]
    package let manifestPath: String
    /// The CLI's "has no effect" warnings for the run; only the stdout copy carries them.
    package var warnings: [String]?

    package enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case model
        case overlap
        case chunks
        case elapsedSeconds = "elapsed_seconds"
        case stems
        case manifestPath = "manifest_path"
        case warnings
    }

    package init(
        schemaVersion: Int,
        source: Source,
        model: Model,
        overlap: Int,
        chunks: Int,
        elapsedSeconds: Double,
        stems: [Stem],
        manifestPath: String
    ) {
        self.schemaVersion = schemaVersion
        self.source = source
        self.model = model
        self.overlap = overlap
        self.chunks = chunks
        self.elapsedSeconds = elapsedSeconds
        self.stems = stems
        self.manifestPath = manifestPath
    }

    /// "4 stems · 2 chunks" — the result panel's header.
    package var summary: String {
        let stemCount = stems.count == 1 ? "1 stem" : "\(stems.count) stems"
        let chunkCount = chunks == 1 ? "1 chunk" : "\(chunks) chunks"
        return "\(stemCount) · \(chunkCount)"
    }

    package static func decode(_ data: Data) -> StudioSeparationManifest? {
        try? JSONDecoder().decode(StudioSeparationManifest.self, from: data)
    }

    package static func load(from url: URL) -> StudioSeparationManifest? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decode(data)
    }
}

// MARK: - speech diarize --format rttm

extension StudioDiarizationDocument {
    /// `mere.run speech diarize --format rttm`: one `SPEAKER <file> 1 <start> <duration> <NA>
    /// <NA> speaker_<index> <NA> <NA>` line per turn. Read into the same document the JSON
    /// format decodes to, so an RTTM timeline draws the speaker lanes and the turn rows too. The
    /// file names no model or total length, so the model is blank and the length is the last
    /// turn's end. Nil for text with no RTTM line in it.
    package static func rttm(_ text: String) -> StudioDiarizationDocument? {
        var turns: [(speaker: String, start: Double, end: Double)] = []
        for line in text.components(separatedBy: .newlines) {
            let fields = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            guard fields.count >= 8, fields[0] == "SPEAKER",
                  let start = Double(fields[3]), let duration = Double(fields[4]) else { continue }
            // The file carries milliseconds; the sum is rounded back to them so 24.6 + 3.3 reads 27.9.
            turns.append((fields[7], start, ((start + duration) * 1_000).rounded() / 1_000))
        }
        guard !turns.isEmpty else { return nil }
        // Speakers are numbered by the `speaker_<n>` suffix the CLI writes when every label has
        // one, else by first appearance — never a mix, which could give two labels one index.
        let labels = turns.map(\.speaker).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        let numbered = labels.compactMap { $0.split(separator: "_").last.flatMap { Int($0) } }
        let indexBySpeaker: [String: Int]
        if numbered.count == labels.count, Set(numbered).count == labels.count {
            indexBySpeaker = Dictionary(uniqueKeysWithValues: zip(labels, numbered))
        } else {
            indexBySpeaker = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($1, $0) })
        }
        let segments = turns.map { turn in
            Segment(speaker: turn.speaker, speakerIndex: indexBySpeaker[turn.speaker] ?? 0, startSeconds: turn.start, endSeconds: turn.end)
        }
        return StudioDiarizationDocument(
            schemaVersion: 1,
            model: "",
            durationSeconds: segments.map(\.endSeconds).max() ?? 0,
            speakerCount: Set(segments.map(\.speakerIndex)).count,
            segments: segments
        )
    }
}
