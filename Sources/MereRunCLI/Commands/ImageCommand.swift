import ArgumentParser

struct Image: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "image",
        abstract: "Generate and validate image models.",
        subcommands: [
            ImageDataset.self,
            ImageGenerate.self,
            ImageReconstruct3D.self,
            ImageReconstruct3DMultiview.self,
            ImageRunPlan.self,
            ImageTrainLoRA.self,
            ImageVisualizeRun.self,
            ImageValidate.self,
        ]
    )
}
