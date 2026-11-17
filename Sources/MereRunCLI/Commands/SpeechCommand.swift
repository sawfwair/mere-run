import ArgumentParser

struct Speech: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "Synthesize, transcribe, diarize, converse, and manage voice profiles.",
        subcommands: [
            SpeechSynthesize.self,
            SpeechTranscribe.self,
            SpeechDiarize.self,
            SpeechDiarizeLive.self,
            SpeechListen.self,
            SpeechConverse.self,
            SpeechProfile.self,
        ]
    )
}
