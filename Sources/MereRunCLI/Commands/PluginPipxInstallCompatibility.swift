import ArgumentParser
import Foundation

enum PluginPipxInstallCompatibility {
    static func validateForcedInstall(
        package: String,
        capture: (String, [String]) throws -> Data
    ) throws {
        let backend: String?
        do {
            let metadata = try capture("pipx", ["list", "--json"])
            backend = try PluginPipxInstallationMetadata.backend(for: package, metadata: metadata)
        } catch {
            throw ValidationError(
                """
                Cannot inspect pipx installations before a forced reinstall.
                Run pipx list --json to diagnose the installation, then retry.
                No plugin installation was attempted.
                """
            )
        }
        guard backend == "uv" else { return }

        let output = try capture("pipx", ["--version"])
        guard let version = PipxReleaseVersion(String(decoding: output, as: UTF8.self)) else {
            throw ValidationError(
                """
                Cannot verify pipx compatibility with this plugin's uv environment.
                Use a stable pipx release, version 1.16.0 or later, then retry.
                Check the version with: pipx --version
                No plugin installation was attempted.
                """
            )
        }
        guard version.supportsUVForcedInstall else {
            throw ValidationError(
                """
                pipx \(version.description) cannot force-reinstall the uv environment for \(package).
                Upgrade pipx to 1.16.0 or later, then retry the same mere.run command.
                If you installed pipx with Homebrew: brew upgrade pipx
                Otherwise, update pipx using its original installer.
                The plugin has not been changed. Switching pipx backends is not required.
                """
            )
        }
    }
}

private struct PipxReleaseVersion {
    let components: [Int]

    init?(_ output: String) {
        let fields = output.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ".", omittingEmptySubsequences: false)
        guard fields.count == 3,
              fields.allSatisfy({ !$0.isEmpty && $0.utf8.allSatisfy { (48...57).contains($0) } }) else {
            return nil
        }
        let components = fields.compactMap { Int($0) }
        guard components.count == 3 else { return nil }
        self.components = components
    }

    var supportsUVForcedInstall: Bool {
        !components.lexicographicallyPrecedes([1, 16, 0])
    }

    var description: String {
        components.map(String.init).joined(separator: ".")
    }
}
