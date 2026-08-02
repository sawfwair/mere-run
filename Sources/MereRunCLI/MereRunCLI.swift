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

    mutating func validate() throws {
        if let modelsRoot, !modelsRoot.isEmpty {
            CLIModelStoreBootstrap.applyOverridePath(modelsRoot)
        } else {
            _ = _mereRunCLIModelStoreBootstrap
        }
    }

    static let configuration = CommandConfiguration(
        commandName: "mere.run",
        abstract: "Run local inference workflows with MereRunCore.",
        version: MereRunCLIVersion.current,
        subcommands: [
            GuideCommand.self,
            CatalogCommand.self,
            Image.self,
            Text.self,
            Speech.self,
            Vision.self,
            Audio.self,
            Music.self,
            SFX.self,
            Video.self,
            World.self,
            Graph.self,
            Executor.self,
            Run.self,
            Model.self,
            Adapter.self,
            Status.self,
            Gate.self,
            Config.self,
            API.self,
            OpenWebUI.self,
            Plugin.self,
            Setup.self,
            Agent.self,
        ]
    )
}
