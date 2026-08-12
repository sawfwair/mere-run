import ArgumentParser

struct Model: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model",
        abstract: "List, pull, locate, remove, inspect, optimize, and clean up models.",
        subcommands: [
            ModelList.self,
            ModelLocation.self,
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
