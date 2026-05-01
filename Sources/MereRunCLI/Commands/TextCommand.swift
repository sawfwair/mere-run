import ArgumentParser

struct Text: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Run local chat, code, embedding, and anonymization workflows.",
        subcommands: [
            TextChat.self,
            TextCode.self,
            TextEmbed.self,
            TextAnonymize.self,
        ]
    )
}
