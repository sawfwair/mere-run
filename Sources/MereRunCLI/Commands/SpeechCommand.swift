import ArgumentParser

struct Speech: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "Synthesize, transcribe, and manage voice profiles.",
        subcommands: [
            SpeechSynthesize.self,
            SpeechTranscribe.self,
            SpeechProfile.self,
        ]
    )
}
