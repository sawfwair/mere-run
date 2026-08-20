import ArgumentParser
import Foundation
import AudioCore
import AudioTTS
import MereRunCore

struct SpeechProfile: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Manage saved voice clone profiles.",
        subcommands: [
            SpeechProfileList.self,
            SpeechProfileCreate.self,
            SpeechProfileDelete.self,
        ],
        defaultSubcommand: SpeechProfileList.self
    )
}

struct SpeechProfileList: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List saved speech voice profiles."
    )

    func run() async throws {
        let store = VoiceProfileStore()
        let profiles = try await store.listProfiles()
        if profiles.isEmpty {
            print("No speech profiles found.")
            return
        }
        for profile in profiles {
            print("\(profile.id.uuidString)\t\(profile.name)\t\(profile.updatedAt)")
        }
    }
}

struct SpeechProfileCreate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "create",
        abstract: "Create a speech profile from reference audio."
    )

    @Option(name: [.long], help: "Profile name.")
    var name: String

    @Option(name: [.long], help: "Reference audio file.")
    var audio: String

    @Option(name: [.long], help: "Reference transcript override (optional).")
    var text: String?

    @Option(name: [.long], help: "Language code (optional, default: auto).")
    var language: String = "auto"

    @Flag(name: [.short, .long], help: "Quiet mode.")
    var quiet: Bool = false

    func run() async throws {
        let audioURL = URL(fileURLWithPath: audio).standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Audio file not found: \(audioURL.path)")
        }

        let transcript: String
        if let text = normalized(text) {
            transcript = text
        } else {
            if !quiet {
                FileHandle.standardError.write(Data("Transcribing reference audio with the speech transcriber...\n".utf8))
            }
            let request = ASRRequest(
                audioURL: audioURL,
                language: normalizedLanguage(language),
                task: .transcribe,
                maxTokens: 448
            )
            let execution = try await CLIASRRouting.transcribe(
                request: request,
                preferredBackend: .auto,
                modelOverride: nil,
                progressHandler: nil
            )
            let result = execution.result
            let trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else {
                throw ValidationError("Auto-transcription returned empty text. Provide --text.")
            }
            transcript = trimmed
        }

        let store = VoiceProfileStore()
        let created = try await store.createProfile(
            name: name,
            referenceAudioURL: audioURL,
            transcript: transcript,
            language: normalizedLanguage(language)
        )

        if !quiet {
            FileHandle.standardError.write(Data("Created profile '\(created.name)' (\(created.id.uuidString))\n".utf8))
        }
        print(created.id.uuidString)
    }

    private func normalized(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func normalizedLanguage(_ value: String?) -> String? {
        guard let normalized = normalized(value) else { return nil }
        if normalized.lowercased() == "auto" {
            return nil
        }
        return normalized
    }
}

struct SpeechProfileDelete: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "delete",
        abstract: "Delete a speech profile by ID."
    )

    @Option(name: [.long], help: "Profile UUID.")
    var id: String

    func run() async throws {
        guard let uuid = UUID(uuidString: id) else {
            throw ValidationError("Invalid UUID: \(id)")
        }
        let store = VoiceProfileStore()
        try await store.deleteProfile(id: uuid)
        print(uuid.uuidString)
    }
}
