import ArgumentParser

struct Model: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model",
        abstract: "List, pull, quantize, remove, and inspect models.",
        subcommands: [
            ModelList.self,
            ModelPull.self,
            ModelQuantize.self,
            ModelRemove.self,
            ModelInfo.self,
            ModelCapabilities.self,
            ModelRuntime.self,
            ModelBenchmark.self,
            ModelRepairManifests.self,
        ]
    )
}
