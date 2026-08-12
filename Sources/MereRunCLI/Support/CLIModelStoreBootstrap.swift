import Foundation
import MereRunCore

enum CLIModelStoreBootstrap {
    static func bootstrap(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaults: UserDefaults = .standard
    ) {
        if let rawPath = modelsRootArgument(in: arguments), !rawPath.isEmpty {
            applyOverridePath(rawPath)
        } else if let rawPath = environment[MereRunModelPaths.modelsDirEnvironmentKey], !rawPath.isEmpty {
            applyOverridePath(rawPath)
        } else if let rawPath = defaults.string(
            forKey: MereRunModelPaths.modelStorageActivePathDefaultsKey
        ), !rawPath.isEmpty {
            applyOverridePath(rawPath, includeRegisteredLocations: true)
        }
    }

    static func applyOverridePath(
        _ rawPath: String,
        includeRegisteredLocations: Bool = false
    ) {
        let expanded = (rawPath as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        MereRunModelPaths.setProcessModelsDirOverride(
            url,
            includeRegisteredLocations: includeRegisteredLocations
        )
        setenv(MereRunModelPaths.modelsDirEnvironmentKey, url.path, 1)
    }

    static func resolvedOverridePath(
        arguments: [String],
        environment: [String: String],
        defaults: UserDefaults
    ) -> String? {
        if let fromArgs = modelsRootArgument(in: arguments), !fromArgs.isEmpty {
            return fromArgs
        }

        if let fromEnv = environment[MereRunModelPaths.modelsDirEnvironmentKey], !fromEnv.isEmpty {
            return fromEnv
        }

        if let local = defaults.string(forKey: MereRunModelPaths.modelStorageActivePathDefaultsKey), !local.isEmpty {
            return local
        }

        return nil
    }

    static func modelsRootArgument(in arguments: [String]) -> String? {
        for (index, argument) in arguments.enumerated() {
            if argument == "--models-root", arguments.indices.contains(index + 1) {
                return arguments[index + 1]
            }

            if argument.hasPrefix("--models-root=") {
                return String(argument.dropFirst("--models-root=".count))
            }
        }
        return nil
    }
}
