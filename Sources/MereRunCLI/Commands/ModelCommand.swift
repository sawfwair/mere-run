import ArgumentParser

struct Model: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model",
        abstract: "List, pull, remove, inspect, optimize, and clean up models.",
        subcommands: [
            ModelList.self,
            ModelPull.self,
            ModelRemove.self,
            ModelStorage.self,
            ModelGarbageCollect.self,
            ModelInfo.self,
            ModelCapabilities.self,
            ModelRuntime.self,
            ModelOptimize.self,
            ModelBenchmark.self,
            ModelRepairManifests.self,
        ]
    )
}
