import ArgumentParser

struct Text: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "text",
        abstract: "Run local chat, code, embedding, anonymization, classification, and decision workflows.",
        subcommands: [
            TextChat.self,
            TextCode.self,
            TextEmbed.self,
            TextAnonymize.self,
            TextDecide.self,
            TextClassify.self,
            TextExtract.self,
            TextTrainLoRA.self,
        ]
    )
}
