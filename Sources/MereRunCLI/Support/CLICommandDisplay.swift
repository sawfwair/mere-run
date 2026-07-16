import Foundation
import MereRunCore

enum CLICommandDisplay {
    static var executable: String {
        let executablePath = Bundle.main.executableURL?.path
            ?? CommandLine.arguments.first
            ?? "mere.run"
        if executablePath.contains("/.build/") {
            return "swift run mere.run"
        }
        return "mere.run"
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
