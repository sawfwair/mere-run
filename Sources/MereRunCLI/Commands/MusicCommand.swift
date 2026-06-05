import ArgumentParser

struct Music: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "music",
        abstract: "Generate music locally.",
        subcommands: [
            MusicGenerate.self,
            MusicRealtime.self,
        ]
    )
}
