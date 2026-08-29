import Foundation

struct PluginInstallationFailureDiagnosis: Equatable {
    let pluginID: String
    let summary: String
    let verificationError: String?
    let repairCommand: String

    var rendered: String {
        var lines = [
            "Plugin \(pluginID) is installed but cannot run its doctor.",
            "Diagnosis: \(summary)",
        ]
        if let verificationError {
            lines.append("Verification error: \(verificationError)")
        }
        lines.append("Repair with:")
        lines.append("  \(repairCommand)")
        return lines.joined(separator: "\n")
    }

    static func make(
        plugin: PluginCatalogEntry,
        install: PluginCatalogInstall,
        inspection: PluginInstallationInspection
    ) -> PluginInstallationFailureDiagnosis {
        make(
            plugin: plugin,
            install: install,
            inspection: inspection,
            missingEditableSourcePath: PluginPipxInstallationMetadata.missingEditableSourcePath(for: plugin)
        )
    }

    static func make(
        plugin: PluginCatalogEntry,
        install: PluginCatalogInstall,
        inspection: PluginInstallationInspection,
        missingEditableSourcePath: String?
    ) -> PluginInstallationFailureDiagnosis {
        let repairArguments = install.manager == "pipx"
            ? "plugin install \(ShellCommand.quote(plugin.id)) --yes --force"
            : "plugin install \(ShellCommand.quote(plugin.id)) --yes"
        if let missingEditableSourcePath {
            return PluginInstallationFailureDiagnosis(
                pluginID: plugin.id,
                summary: "editable source path no longer exists: \(missingEditableSourcePath)",
                verificationError: nil,
                repairCommand: "mere.run \(repairArguments)"
            )
        }
        return PluginInstallationFailureDiagnosis(
            pluginID: plugin.id,
            summary: "plugin manifest verification failed",
            verificationError: conciseVerificationError(inspection.error),
            repairCommand: "mere.run \(repairArguments)"
        )
    }

    private static func conciseVerificationError(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let lines = raw.split(whereSeparator: \Character.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return lines.last.map { String($0.prefix(500)) }
    }
}

enum PluginPipxInstallationMetadata {
    static func backend(for package: String, metadata: Data) throws -> String? {
        let list = try JSONDecoder().decode(PipxList.self, from: metadata)
        return list.venvs[package]?.metadata.backend
    }

    static func missingEditableSourcePath(for plugin: PluginCatalogEntry) -> String? {
        guard PluginProcess.which("pipx") != nil,
              let data = try? PluginProcess.captureExecutable("pipx", arguments: ["list", "--json"]) else {
            return nil
        }
        return try? missingEditableSourcePath(
            for: plugin,
            metadata: data,
            fileExists: { FileManager.default.fileExists(atPath: $0) }
        )
    }

    static func missingEditableSourcePath(
        for plugin: PluginCatalogEntry,
        metadata: Data,
        fileExists: (String) -> Bool
    ) throws -> String? {
        let list = try JSONDecoder().decode(PipxList.self, from: metadata)
        let environment = list.venvs[plugin.package]
            ?? list.venvs[plugin.id]
            ?? list.venvs.values.first { $0.metadata.mainPackage.package == plugin.package }
        guard let package = environment?.metadata.mainPackage,
              package.pipArgs.contains(where: { $0 == "--editable" || $0 == "-e" }),
              let path = localPath(package.packageOrURL),
              !fileExists(path) else {
            return nil
        }
        return path
    }

    private static func localPath(_ raw: String) -> String? {
        if raw.hasPrefix("file://") {
            return URL(string: raw)?.standardizedFileURL.path
        }
        guard raw.hasPrefix("/") || raw.hasPrefix("~") else { return nil }
        return URL(
            fileURLWithPath: NSString(string: raw).expandingTildeInPath
        ).standardizedFileURL.path
    }
}

private struct PipxList: Decodable {
    let venvs: [String: PipxEnvironment]
}

private struct PipxEnvironment: Decodable {
    let metadata: PipxEnvironmentMetadata
}

private struct PipxEnvironmentMetadata: Decodable {
    let mainPackage: PipxMainPackage
    let backend: String?

    enum CodingKeys: String, CodingKey {
        case mainPackage = "main_package"
        case backend
    }
}

private struct PipxMainPackage: Decodable {
    let package: String
    let packageOrURL: String
    let pipArgs: [String]

    enum CodingKeys: String, CodingKey {
        case package
        case packageOrURL = "package_or_url"
        case pipArgs = "pip_args"
    }
}
