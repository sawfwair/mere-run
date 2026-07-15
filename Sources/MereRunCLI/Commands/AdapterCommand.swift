import ArgumentParser

struct Adapter: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "adapter",
        abstract: "List and pull verified LoRA adapters.",
        subcommands: [
            AdapterList.self,
            AdapterPull.self,
        ]
    )
}
