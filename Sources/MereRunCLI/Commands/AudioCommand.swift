import ArgumentParser

struct Audio: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "audio",
        abstract: "Generate, edit, and enhance audio locally.",
        subcommands: [
            AudioGenerate.self,
            AudioEnhance.self,
            AudioEdit.self,
        ]
    )
}
