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
        // a model, or downloads one. A refusal is a validation error, as the commands' own
        // option checks are: usage under the message and exit status 64.
        // Its warnings print from `main` once every command has validated.
        do {
            try CLICapabilityGate.check(arguments: arguments)
        } catch let rejection as CLICapabilityGate.Rejection {
            throw ValidationError(rejection.messages.joined(separator: "\n"))
        }
        try admit(arguments)
    }

    /// ArgumentParser's entry point, plus the gate's warnings: they print only once the whole
    /// command line has parsed and every command's `validate()` has passed, so a run that ends
    /// in a usage error shows the error alone. The gate already let this command line through
    /// in `validate`, so asking it again only collects the warnings.
    static func main() async {
        do {
            var command = try parseAsRoot()
            for line in try CLICapabilityGate.check(arguments: CommandLine.arguments) {
                CLIStderr.write(line)
            }
            if var asyncCommand = command as? AsyncParsableCommand {
                try await asyncCommand.run()
            } else {
                try command.run()
            }
        } catch {
            exit(withError: error)
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
