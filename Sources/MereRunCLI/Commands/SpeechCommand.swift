import ArgumentParser

struct Speech: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "Synthesize, transcribe, diarize, and manage voice profiles.",
        subcommands: [
            SpeechSynthesize.self,
            SpeechTranscribe.self,
            SpeechDiarize.self,
            SpeechListen.self,
            SpeechProfile.self,
        ]
    )
}
