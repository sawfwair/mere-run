import ArgumentParser

struct Image: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Generate and validate image models.",
        subcommands: [
            ImageDataset.self,
            ImageGenerate.self,
            ImageRunPlan.self,
            ImageTrainLoRA.self,
            ImageVisualizeRun.self,
            ImageValidate.self,
        ]
    )
}
