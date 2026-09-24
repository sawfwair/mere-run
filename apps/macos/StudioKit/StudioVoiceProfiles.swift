import Foundation

// The voices Voice ▸ Voices manages: the records `speech profile create` writes to the CLI's
// manifest, read directly rather than through `speech profile list`, so the page lists them
// without a process; creating and deleting still run through the task runner.

/// One saved voice, as the CLI's `voice_profiles.json` stores it.
package struct StudioVoiceProfileRecord: Codable, Identifiable, Equatable {
    package let id: UUID
    package let name: String
    package let createdAt: Date
    package let updatedAt: Date
    package let transcript: String
    package let language: String?
    package let referenceAudioRelativePath: String
    package let modelFingerprint: String?

    package init(
        id: UUID,
        name: String,
        createdAt: Date,
        updatedAt: Date,
        transcript: String,
        language: String?,
        referenceAudioRelativePath: String,
        modelFingerprint: String?
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.transcript = transcript
        self.language = language
        self.referenceAudioRelativePath = referenceAudioRelativePath
        self.modelFingerprint = modelFingerprint
    }

    /// The reference recording: a path relative to the voices folder as the CLI stores it, or an
    /// absolute one as it is.
    package var referenceAudioURL: URL {
        if referenceAudioRelativePath.hasPrefix("/") {
            return URL(fileURLWithPath: referenceAudioRelativePath)
        }
        return StudioVoiceProfileStore.voicesDirectory
            .appendingPathComponent(referenceAudioRelativePath, isDirectory: false)
    }
}

package enum StudioVoiceProfileStore {
    package static var voicesDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MereRun", isDirectory: true)
            .appendingPathComponent("voices", isDirectory: true)
    }

    package static var manifestURL: URL {
        voicesDirectory.appendingPathComponent("voice_profiles.json", isDirectory: false)
    }

    /// Every saved voice, by name. Nothing yet, or an unreadable manifest, reads as no voices.
    package static func load() -> [StudioVoiceProfileRecord] {
        guard let data = try? Data(contentsOf: manifestURL),
              let profiles = try? JSONDecoder().decode([StudioVoiceProfileRecord].self, from: data) else {
            return []
        }
        return profiles.sorted {
            $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }
}

extension StudioTaskDraft {
    /// The run that removes one saved voice: `speech profile delete --id <uuid>`.
    package static func deletingVoiceProfile(_ id: UUID) -> StudioTaskDraft {
        var draft = StudioTaskDraft(templateID: .speechProfileDelete)
        draft.form["--id"] = .text(id.uuidString)
        return draft
    }
}
