import ArgumentParser

private let _mereRunCLIModelStoreBootstrap: Void = {
    CLIModelStoreBootstrap.bootstrap(arguments: CommandLine.arguments)
}()

@main
struct MereRunCLI: AsyncParsableCommand {
    @Option(
        name: [.customLong("models-root")],
        help: "Override mere.run model storage root directory (same as MERERUN_MODELS_DIR)."
    )
    var modelsRoot: String?

    static let configuration = CommandConfiguration(
        commandName: "mere.run",
        abstract: "Run local inference workflows with MereRunCore.",
        version: MereRunCLIVersion.current,
        subcommands: [
            GuideCommand.self,
            Image.self,
            Text.self,
            Speech.self,
            Vision.self,
            Music.self,
            Video.self,
            Model.self,
            API.self,
            Setup.self,
            Agent.self,
        ],
    )
}
