import ArgumentParser

struct Model: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model",
        abstract: "List, pull, remove, and inspect models.",
        subcommands: [
            ModelList.self,
            ModelPull.self,
            ModelRemove.self,
            ModelInfo.self,
            ModelRepairManifests.self,
        ]
    )
}
