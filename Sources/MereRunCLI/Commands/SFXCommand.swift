import ArgumentParser

struct SFX: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sfx",
        abstract: "Generate sound effects locally.",
        subcommands: [
            SFXAE.self,
            SFXCLAP.self,
            SFXCondition.self,
            SFXGenerate.self,
            SFXVideo.self,
        ]
    )
}
