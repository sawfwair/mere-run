import ArgumentParser
import MereRunRelayKit

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
        try validate(arguments: CommandLine.arguments)
    }

    /// ArgumentParser validates the root before it parses the leaf, so this runs first for every
    /// command. `arguments` is the process argv, executable first.
    mutating func validate(
        arguments: [String],
        admit: ([String]) throws -> Void = CLIProcessAdmissionBootstrap.acquireIfNeeded
    ) throws {
        if let modelsRoot, !modelsRoot.isEmpty {
            CLIModelStoreBootstrap.applyOverridePath(modelsRoot)
        } else {
            _ = _mereRunCLIModelStoreBootstrap
        }
        // Before admission, so a run the contract rejects never queues for permits, resolves
        // a model, or downloads one.
        try CLICapabilityGate.check(arguments: arguments)
        try admit(arguments)
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
            Geo.self,
            Audio.self,
            Music.self,
            SFX.self,
            Video.self,
            World.self,
            Graph.self,
            Executor.self,
            Relay.self,
            Run.self,
            EvaluationCommand.self,
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
