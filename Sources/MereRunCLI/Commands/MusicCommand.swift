import ArgumentParser

struct Music: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "music",
        abstract: "Generate, analyze, transcribe, and separate music locally.",
        subcommands: [
            MusicAnalyze.self,
            MusicGenerate.self,
            MusicRealtime.self,
            MusicServe.self,
            MusicSeparate.self,
            MusicTrainAdapter.self,
            MusicTranscribe.self,
        ]
    )
}
