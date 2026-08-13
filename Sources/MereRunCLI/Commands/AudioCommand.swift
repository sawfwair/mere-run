import ArgumentParser

struct Audio: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio",
        abstract: "Enhance general audio locally.",
        subcommands: [
            AudioGenerate.self,
            AudioEnhance.self,
        ]
    )
}
