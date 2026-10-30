import Foundation
import MereRunCore

enum CLICommandDisplay {
    static var executable: String {
        executable(invokedAs: CommandLine.arguments.first ?? "mere.run")
    }

    /// `swift run` launches the build product by absolute path, so an absolute `.build` path
    /// keeps the `swift run` form. A relative `.build` path is a binary run directly; repeat it
    /// as typed. Anything else is an installed `mere.run`.
    static func executable(invokedAs invocation: String) -> String {
        let underBuild = invocation.hasPrefix(".build/") || invocation.contains("/.build/")
        guard underBuild else { return "mere.run" }
        return invocation.hasPrefix("/") ? "swift run mere.run" : invocation
    }

    static func command(_ arguments: String) -> String {
        "\(executable) \(arguments)"
    }

    static func modelPullCommand(for modelID: String) -> String {
        let acknowledgement = ManagedModelCatalog.spec(for: modelID)?.usageRestriction == nil
            ? ""
            : " --accept-model-license"
        return command("model pull \(modelID)\(acknowledgement)")
    }
}
