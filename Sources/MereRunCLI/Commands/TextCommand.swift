import ArgumentParser

struct Text: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Run local text chat, code, and embedding workflows.",
        subcommands: [
            TextChat.self,
            TextCode.self,
            TextEmbed.self,
        ]
    )
}
