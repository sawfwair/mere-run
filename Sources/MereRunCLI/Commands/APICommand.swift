import ArgumentParser

struct API: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "api",
        abstract: "Serve local models through API surfaces.",
        subcommands: [
            APIServe.self,
        ]
    )
}
