import ArgumentParser

struct Image: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Generate and validate image models.",
        subcommands: [
            ImageGenerate.self,
            ImageValidate.self,
        ]
    )
}
