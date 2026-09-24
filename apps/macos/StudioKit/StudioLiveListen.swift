import Foundation
import MereRunContract

// What Audio ▸ Live reads and launches: the microphones `speech listen --list-devices` prints,
// the two event streams the live commands write on stdout, and the switches the page turns on
// when it starts a session. All of it is plain data, so the page's parsing is tested here rather
// than through the view.

/// One input device, as `speech listen --list-devices` prints it: `* <uid>\t<name>` for the
/// system default, two spaces for the rest.
package struct StudioListenDevice: Equatable, Identifiable {
    package let uid: String
    package let name: String
    package let isDefault: Bool

    package init(uid: String, name: String, isDefault: Bool) {
        self.uid = uid
        self.name = name
        self.isDefault = isDefault
    }

    package var id: String { uid }

    package static func parseList(_ output: String) -> [StudioListenDevice] {
        output.split(separator: "\n").compactMap { rawLine in
            let line = String(rawLine)
            guard let marker = line.first else { return nil }
            let fields = line.dropFirst()
                .trimmingCharacters(in: .whitespaces)
                .split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard fields.count == 2 else { return nil }
            let uid = String(fields[0]).trimmingCharacters(in: .whitespacesAndNewlines)
            let name = String(fields[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !uid.isEmpty, !name.isEmpty else { return nil }
            return StudioListenDevice(uid: uid, name: name, isDefault: marker == "*")
        }
    }
}

/// The transcript `speech listen --jsonl` streams: partials replace one another per utterance
/// until a commit closes it. Chunks arrive split anywhere, so lines are assembled first.
package struct StudioLiveTranscriptAccumulator: Equatable {
    private struct Event: Decodable {
        let protocolVersion: Int
        let type: String
        let utteranceID: String?
        let revision: Int?
        let text: String?
        let message: String?

        enum CodingKeys: String, CodingKey {
            case protocolVersion = "protocol"
            case type
            case utteranceID = "utteranceId"
            case legacyUtteranceID = "utterance_id"
            case revision
            case text
            case message
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            protocolVersion = try values.decode(Int.self, forKey: .protocolVersion)
            type = try values.decode(String.self, forKey: .type)
            utteranceID = try values.decodeIfPresent(String.self, forKey: .utteranceID)
                ?? values.decodeIfPresent(String.self, forKey: .legacyUtteranceID)
            revision = try values.decodeIfPresent(Int.self, forKey: .revision)
            text = try values.decodeIfPresent(String.self, forKey: .text)
            message = try values.decodeIfPresent(String.self, forKey: .message)
        }
    }

    private var buffer = ""
    private var committedUtteranceIDs: Set<String> = []
    private var latestRevisions: [String: Int] = [:]
    private var committedSegments: [String] = []
    package private(set) var partialText = ""
    package private(set) var errorMessage: String?

    package init() {}

    package var committedText: String { committedSegments.joined(separator: "\n") }

    package var displayText: String {
        [committedText, partialText]
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    package mutating func beginSession() {
        buffer = ""
        partialText = ""
        errorMessage = nil
        latestRevisions.removeAll(keepingCapacity: true)
    }

    package mutating func clear() {
        self = StudioLiveTranscriptAccumulator()
    }

    package mutating func receive(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
            buffer.removeSubrange(...newline)
            receiveLine(line)
        }
    }

    private mutating func receiveLine(_ line: String) {
        guard let data = line.data(using: .utf8),
              let event = try? JSONDecoder().decode(Event.self, from: data),
              event.protocolVersion == 1 else { return }

        switch event.type {
        case "partial":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text else { return }
            let revision = event.revision ?? 0
            guard revision >= latestRevisions[id, default: -1] else { return }
            latestRevisions[id] = revision
            partialText = text
        case "commit":
            guard let id = event.utteranceID,
                  !committedUtteranceIDs.contains(id),
                  let text = event.text?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !text.isEmpty else { return }
            committedUtteranceIDs.insert(id)
            committedSegments.append(text)
            partialText = ""
        case "error":
            errorMessage = event.message ?? "Live transcription failed."
        default:
            break
        }
    }
}

/// The speaker activity `speech diarize-live` streams: every `activity` event's segments, in
/// order, with the running speaker count and the seconds of audio heard so far.
package struct StudioLiveDiarizationAccumulator: Equatable {
    private var buffer = ""
    package private(set) var segments: [DiarizationStreamEvent.Segment] = []
    package private(set) var speakerCount = 0
    package private(set) var audioSeconds = 0.0
    package private(set) var errorMessage: String?

    package init() {}

    package var displayText: String {
        segments.map { segment in
            String(format: "%.2f–%.2f  %@", segment.startSeconds, segment.endSeconds, segment.speaker)
        }.joined(separator: "\n")
    }

    package mutating func beginSession() { self = Self() }
    package mutating func clear() { self = Self() }

    package mutating func receive(_ chunk: String) {
        buffer += chunk
        while let newline = buffer.firstIndex(of: "\n") {
            let line = String(buffer[..<newline])
            buffer.removeSubrange(...newline)
            guard let data = line.data(using: .utf8),
                  let event = try? JSONDecoder().decode(DiarizationStreamEvent.self, from: data),
                  event.schemaVersion == 1 else { continue }
            switch event.type {
            case .activity:
                segments.append(contentsOf: event.segments ?? [])
                speakerCount = event.speakerCount ?? speakerCount
                audioSeconds = event.audioSeconds ?? audioSeconds
            case .final:
                speakerCount = event.speakerCount ?? speakerCount
                audioSeconds = event.audioSeconds ?? audioSeconds
            case .error:
                errorMessage = event.message ?? "Live diarization failed."
            case .ready:
                break
            }
        }
    }
}

extension StudioTaskDraft {
    /// The draft Audio ▸ Live launches: the parked settings plus the switches a session needs on
    /// stdout — `--jsonl` for `speech listen`, whose events are what the transcript reads, and
    /// `--quiet` for both commands, so nothing but events reaches the page. Applied at Start and
    /// never parked, so the inspector does not count them as changed settings.
    package func liveListenLaunch() -> StudioTaskDraft {
        var launch = self
        launch.form["--quiet"] = .flag(true)
        if templateID == .speechListen {
            launch.form["--jsonl"] = .flag(true)
        }
        return launch
    }

    /// The flags Audio ▸ Live owns itself and keeps out of its options popover: the device chip,
    /// the two probing switches, and the two the launch sets.
    package static let liveListenOwnedFlags: Set<String> = ["--device", "--list-devices", "--stdin", "--jsonl", "--quiet"]
}
